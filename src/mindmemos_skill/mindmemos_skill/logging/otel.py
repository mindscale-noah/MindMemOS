"""Bridge service-wide Python logging records to OpenTelemetry logs."""

import logging
import sys
import traceback
from collections.abc import Callable, Mapping
from typing import Literal

from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider

from .config import attach_capture_handler, detach_capture_handler
from .sanitization import safe_text


class OpenTelemetryLoggingHandler(logging.Handler):
    """Translate Python records into OTel logs, including records outside algorithm runs."""

    def __init__(
        self,
        provider: LoggerProvider,
        *,
        context: Callable[[], Mapping[str, str] | None],
        capture_level: Literal["info", "debug"],
    ) -> None:
        super().__init__()
        self.provider = provider
        self.context = context
        self.capture_level = capture_level

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a safe correlated log and report bridge failures to the terminal.

        Args:
            record: Prepared Python log record.
        """
        if record.__dict__.get("skip_otlp"):
            return
        try:
            self._emit(record)
        except Exception:
            sys.stderr.write(f"OTel log forwarding failed\n{safe_text(traceback.format_exc())}\n")

    def _emit(self, record: logging.LogRecord) -> None:
        """Translate one prepared record into the provider's log protocol."""
        attributes = self.context()
        if record.levelno < logging.INFO and self.capture_level != "debug":
            return
        severity = {
            logging.DEBUG: SeverityNumber.DEBUG,
            logging.INFO: SeverityNumber.INFO,
            logging.WARNING: SeverityNumber.WARN,
            logging.ERROR: SeverityNumber.ERROR,
            logging.CRITICAL: SeverityNumber.FATAL,
        }.get(record.levelno, SeverityNumber.INFO)
        attributes = dict(attributes or {})
        attributes.update(
            {
                key: record.__dict__[key]
                for key in (
                    "request_id",
                    "project_id",
                    "task_id",
                    "attempt",
                    "run_id",
                    "trace_id",
                    "span_id",
                    "tree_id",
                    "parent_id",
                    "node_id",
                    "group_id",
                )
                if record.__dict__.get(key) is not None
            }
        )
        if record.exc_text:
            attributes["exception.stacktrace"] = record.exc_text
        event_body = record.__dict__.get("event_body")
        structured = isinstance(event_body, Mapping)
        self.provider.get_logger(record.name).emit(
            timestamp=int(record.created * 1_000_000_000),
            severity_number=severity,
            severity_text=record.levelname,
            event_name=record.getMessage() if structured else "python.log",
            body=event_body if structured else {"message": record.getMessage()},
            attributes=dict(attributes),
        )


def attach_logs(
    provider: LoggerProvider,
    *,
    context: Callable[[], Mapping[str, str] | None],
    capture_level: Literal["info", "debug"],
) -> OpenTelemetryLoggingHandler:
    """Forward service-wide package logs to one OTel logger provider.

    Args:
        provider: OTel logger provider receiving Python records.
        context: Optional correlation attributes; None still emits a service log.
        capture_level: Minimum service log capture level.

    Returns:
        Attached handler to remove when the owning service closes.
    """
    handler = OpenTelemetryLoggingHandler(provider, context=context, capture_level=capture_level)
    attach_capture_handler(handler)
    return handler


def detach_logs(handler: OpenTelemetryLoggingHandler) -> None:
    """Stop forwarding logs for a closed service.

    Args:
        handler: Handler returned by attach_logs.
    """
    detach_capture_handler(handler)
