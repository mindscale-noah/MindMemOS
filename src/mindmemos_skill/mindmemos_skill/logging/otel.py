"""Bridge run-scoped Python logging records to OpenTelemetry logs."""

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal

from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from pydantic import BaseModel

from .config import attach_otel_handler, detach_otel_handler

_MAX_DEBUG_CHARS = 32768
_SENSITIVE_KEYS = {"api_key", "password", "token", "secret", "authorization"}


def _sensitive_key(key: Any) -> bool:
    """Recognize common credential field names without inspecting values."""
    name = str(key).lower().replace("-", "_")
    return name in _SENSITIVE_KEYS or name.endswith(("_api_key", "_token", "_password", "_secret"))


def _json_value(value: Any) -> Any:
    """Turn structured debug content into portable JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def _debug_text(value: Any) -> str:
    """Serialize one bounded debug payload with credential-key redaction."""

    def sanitize(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return sanitize(item.model_dump(mode="python"))
        if isinstance(item, Mapping):
            return {str(key): "[REDACTED]" if _sensitive_key(key) else sanitize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [sanitize(child) for child in item]
        return _json_value(item)

    content = json.dumps(sanitize(value), ensure_ascii=False, default=str)
    return content if len(content) <= _MAX_DEBUG_CHARS else content[:_MAX_DEBUG_CHARS] + "...[truncated]"


def debug_fields(body: Mapping[str, Any]) -> dict[str, str]:
    """Redact and bound structured values before logging debug content.

    Args:
        body: Values to emit as a structured debug event.

    Returns:
        Safe serialized fields for a debug log record.
    """
    return {key: '"[REDACTED]"' if _sensitive_key(key) else _debug_text(value) for key, value in body.items()}


class OpenTelemetryLoggingHandler(logging.Handler):
    """Translate Python log records from one active run into OTel log records."""

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
        """Emit a correlated OTel log for a record in this handler's run."""
        attributes = self.context()
        if attributes is None or (record.levelno < logging.INFO and self.capture_level != "debug"):
            return
        severity = {
            logging.DEBUG: SeverityNumber.DEBUG,
            logging.INFO: SeverityNumber.INFO,
            logging.WARNING: SeverityNumber.WARN,
            logging.ERROR: SeverityNumber.ERROR,
            logging.CRITICAL: SeverityNumber.FATAL,
        }.get(record.levelno, SeverityNumber.INFO)
        event_body = record.__dict__.get("event_body")
        structured = isinstance(event_body, Mapping)
        self.provider.get_logger(record.name).emit(
            severity_number=severity,
            severity_text=record.levelname,
            event_name=record.getMessage() if structured else "python.log",
            body=_json_value(event_body) if structured else {"message": record.getMessage()},
            attributes=dict(attributes),
        )


def attach_run_logs(
    provider: LoggerProvider,
    *,
    context: Callable[[], Mapping[str, str] | None],
    capture_level: Literal["info", "debug"],
) -> OpenTelemetryLoggingHandler:
    """Forward current-run package logs to one OTel logger provider.

    Args:
        provider: OTel logger provider receiving Python records.
        context: Correlation attributes for the active owning run, or None.
        capture_level: Minimum algorithm audit detail level.

    Returns:
        Attached handler to remove when the owning service closes.
    """
    handler = OpenTelemetryLoggingHandler(provider, context=context, capture_level=capture_level)
    attach_otel_handler(handler)
    return handler


def detach_run_logs(handler: OpenTelemetryLoggingHandler) -> None:
    """Stop forwarding logs for a closed service.

    Args:
        handler: Handler returned by attach_run_logs.
    """
    detach_otel_handler(handler)
