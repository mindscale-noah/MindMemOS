"""Transport-neutral OpenTelemetry span capture primitives."""

from .tracer import OpenTelemetryRecorder, serialize_log, serialize_span

__all__ = ["OpenTelemetryRecorder", "serialize_log", "serialize_span"]
