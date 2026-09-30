"""Capture OpenTelemetry spans and hand portable records to an owning sink."""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor, ReadWriteLogRecord
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def _json_value(value: Any) -> Any:
    """Convert OpenTelemetry attribute containers to portable JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def serialize_span(span: ReadableSpan) -> dict[str, Any] | None:
    """Convert a completed span into a backend-independent record.

    Args:
        span: Completed OpenTelemetry span emitted by the local provider.

    Returns:
        Portable span fields, or None when required timing or identity is absent.
    """
    context = span.context
    if context is None or span.start_time is None or span.end_time is None:
        return None
    return {
        "trace_id": trace.format_trace_id(context.trace_id),
        "span_id": trace.format_span_id(context.span_id),
        "parent_span_id": trace.format_span_id(span.parent.span_id) if span.parent else None,
        "name": span.name,
        "kind": span.kind.name,
        "status": span.status.status_code.name,
        "started_at": datetime.fromtimestamp(span.start_time / 1_000_000_000, tz=UTC),
        "finished_at": datetime.fromtimestamp(span.end_time / 1_000_000_000, tz=UTC),
        "attributes": _json_value(span.attributes or {}),
        "events": [
            {"name": event.name, "timestamp_ns": event.timestamp, "attributes": _json_value(event.attributes or {})}
            for event in span.events
        ],
        "links": [
            {
                "trace_id": trace.format_trace_id(link.context.trace_id),
                "span_id": trace.format_span_id(link.context.span_id),
            }
            for link in span.links
        ],
        "resource": _json_value(span.resource.attributes),
    }


def serialize_log(record: ReadWriteLogRecord) -> dict[str, Any]:
    """Convert an emitted OTel log into portable storage fields.

    Args:
        record: Log delivered by an OpenTelemetry log processor.

    Returns:
        Correlation IDs, timestamp, severity, event name, and JSON-safe body.
    """
    value = record.log_record
    timestamp = value.timestamp or value.observed_timestamp
    return {
        "trace_id": trace.format_trace_id(value.trace_id),
        "span_id": trace.format_span_id(value.span_id),
        "occurred_at": datetime.fromtimestamp(timestamp / 1_000_000_000, tz=UTC),
        "level": (value.severity_text or "INFO").lower(),
        "name": value.event_name or "python.log",
        "body": _json_value(value.body),
    }


class _CallbackSpanProcessor(SpanProcessor):
    """Deliver completed spans synchronously to a caller-owned sink."""

    def __init__(self, on_end: Callable[[ReadableSpan], None], on_error: Callable[[Exception], None]):
        self.on_end_callback = on_end
        self.on_error = on_error

    def on_start(self, span, parent_context=None) -> None:
        """Leave in-progress state to the owning application."""

    def on_end(self, span: ReadableSpan) -> None:
        """Deliver a completed span and report sink failures."""
        try:
            self.on_end_callback(span)
        except Exception as exc:
            self.on_error(exc)

    def shutdown(self) -> None:
        """Leave sink ownership to the caller."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Report completion because delivery is synchronous."""
        return True


class _CallbackLogProcessor(LogRecordProcessor):
    """Deliver emitted logs synchronously to a caller-owned sink."""

    def __init__(self, on_emit: Callable[[ReadWriteLogRecord], None], on_error: Callable[[Exception], None]):
        self.on_emit_callback = on_emit
        self.on_error = on_error

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        """Deliver a log and report sink failures."""
        try:
            self.on_emit_callback(log_record)
        except Exception as exc:
            self.on_error(exc)

    def shutdown(self) -> None:
        """Leave sink ownership to the caller."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Report completion because delivery is synchronous."""
        return True


class OpenTelemetryRecorder:
    """Own isolated tracer and logger providers with caller-supplied sinks."""

    def __init__(
        self,
        *,
        service_name: str,
        instrumentation_name: str,
        on_span_end: Callable[[ReadableSpan], None],
        on_log_emit: Callable[[ReadWriteLogRecord], None],
        on_export_error: Callable[[Exception], None],
        otlp_endpoint: str | None = None,
        otlp_timeout: float = 5.0,
    ) -> None:
        """Configure a local provider without changing the process global provider.

        Args:
            service_name: OpenTelemetry service.name resource value.
            instrumentation_name: Library or component producing the spans.
            on_span_end: Sink callback for every completed span.
            on_log_emit: Sink callback for every emitted OTel log.
            on_export_error: Callback for a sink failure.
            otlp_endpoint: Optional OTLP/HTTP collector base URL.
            otlp_timeout: Timeout in seconds for each OTLP export request.
        """
        resource = Resource.create({SERVICE_NAME: service_name})
        self.provider = TracerProvider(resource=resource)
        self.provider.add_span_processor(_CallbackSpanProcessor(on_span_end, on_export_error))
        if otlp_endpoint is not None:
            base = otlp_endpoint.rstrip("/")
            self.provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces", timeout=otlp_timeout))
            )
        self.tracer = self.provider.get_tracer(instrumentation_name)
        self.logger_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
        self.logger_provider.add_log_record_processor(_CallbackLogProcessor(on_log_emit, on_export_error))
        if otlp_endpoint is not None:
            self.logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{base}/v1/logs", timeout=otlp_timeout))
            )

    def close(self) -> None:
        """Stop the local provider after its active spans have ended."""
        self.logger_provider.shutdown()
        self.provider.shutdown()
