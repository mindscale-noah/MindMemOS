"""Sanitize and correlate records before any managed destination sees them."""

import json
import logging
import traceback
from datetime import UTC, datetime

from opentelemetry import trace

from .context import _context
from .sanitization import safe_text, structured_text


def log_message(record: logging.LogRecord) -> str:
    """Render a message and optional event fields as one readable string.

    Args:
        record: Sanitized log record from the execution context.

    Returns:
        Message with pipe-separated event details, without a JSON appendix.
    """
    message = record.getMessage()
    body = record.__dict__.get("event_body")
    if isinstance(body, dict) and body:
        details = " ".join(f"{key}={value}" for key, value in body.items())
        message += f" | {details}"
    return message


class RecordFilter(logging.Filter):
    """Prepare one record once for console, task storage and OTel."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach context and sanitize the shared record.

        Args:
            record: Standard Python record containing optional event_body metadata.

        Returns:
            True to retain the prepared record.
        """
        if record.__dict__.get("_skill_prepared"):
            return True
        record.__dict__["_skill_prepared"] = True
        fields = {key: None for key in ("request_id", "task_id", "attempt", "run_id", "trace_id", "span_id")}
        fields.update(service_name="mindmemos-skill", project_id=None)
        fields.update(_context.get())
        span = trace.get_current_span().get_span_context()
        if span.is_valid:
            fields.update(trace_id=trace.format_trace_id(span.trace_id), span_id=trace.format_span_id(span.span_id))
        record.__dict__.update(fields)
        record.msg = safe_text(record.getMessage())
        record.args = ()
        body = record.__dict__.get("event_body")
        if body is not None:
            rendered = structured_text(body)
            try:
                record.__dict__["event_body"] = json.loads(rendered)
            except ValueError:
                record.__dict__["event_body"] = {"content": rendered}
        if record.exc_info:
            record.exc_text = safe_text("".join(traceback.format_exception(*record.exc_info)))
            record.exc_info = None
        if record.stack_info:
            record.stack_info = safe_text(record.stack_info)
        return True


class ContextFormatter(logging.Formatter):
    """Render one readable log with pipe-separated identity and message sections."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record and its structured body.

        Args:
            record: Prepared Python log record.

        Returns:
            Console text with UTC timestamp, correlation and optional traceback.
        """
        timestamp = (
            datetime.fromtimestamp(record.created, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        )
        fields = (
            "project_id",
            "task_id",
            "attempt",
            "run_id",
            "tree_id",
            "parent_id",
            "node_id",
            "group_id",
            "request_id",
            "trace_id",
            "span_id",
        )
        # Retain correlation on records, but display it only when the algorithm starts.
        if record.__dict__.get("run_id") is not None and record.getMessage().split(" | ", 1)[0] != "algorithm.run.started":
            fields = tuple(key for key in fields if key not in {"run_id", "request_id"})
        context = " ".join(f"{key}={record.__dict__[key]}" for key in fields if record.__dict__.get(key) is not None)
        text = f"[{timestamp}] [{record.levelname}] [{record.name}]"
        if context:
            text += f" | {context}"
        text += f" | {log_message(record)}"
        if record.exc_text:
            text += f"\n{record.exc_text.rstrip()}"
        if record.stack_info:
            text += f"\n{record.stack_info.rstrip()}"
        return text
