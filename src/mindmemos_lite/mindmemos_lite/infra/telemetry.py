"""OpenTelemetry tracing infrastructure with configurable storage exporters."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from ..config import ObservabilityConfig
from ..logging import get_logger
from .observability import BackendSpanExporter, ObservabilityBackend, PostgresObservabilityBackend, SQLiteSpanExporter

logger = get_logger(__name__)

_provider: TracerProvider | None = None


def _signal_endpoint(base: str, signal: str) -> str:
    """Build the OTLP HTTP endpoint for one telemetry signal."""

    base = base.rstrip("/")
    if "/v1/" in base:
        base = base.rsplit("/v1/", 1)[0]
    return f"{base}/v1/{signal}"


def setup_tracer_provider(
    config: ObservabilityConfig,
    *,
    backend: ObservabilityBackend | None = None,
) -> TracerProvider | None:
    """Build and install the process-global tracer provider.

    ``backend`` is the composition seam for packages that reuse Lite tracing
    without adopting its SQLite span store. When omitted, configuration keeps
    selecting the built-in SQLite, PostgreSQL, console, or OTLP exporter.
    """

    global _provider
    if not config.enabled:
        return None
    if _provider is not None:
        return _provider

    resource = Resource.create({SERVICE_NAME: config.service_name})
    provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(config.trace_sampling_ratio),
    )
    exporter_name = config.exporter.strip().lower()

    if backend is None and exporter_name == "postgres":
        backend = PostgresObservabilityBackend(config.postgres, retention_days=config.retention_days)
    elif backend is not None:
        exporter_name = "custom_backend"

    if backend is not None:
        exporter = BackendSpanExporter(backend, capture_content=config.capture_content)
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=config.max_queue_size,
                schedule_delay_millis=config.schedule_delay_millis,
                max_export_batch_size=config.max_export_batch_size,
                export_timeout_millis=config.export_timeout_millis,
            )
        )
    elif exporter_name == "sqlite":
        exporter = SQLiteSpanExporter(
            config.sqlite_path,
            retention_days=config.retention_days,
            capture_content=config.capture_content,
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=config.max_queue_size,
                schedule_delay_millis=config.schedule_delay_millis,
                max_export_batch_size=config.max_export_batch_size,
                export_timeout_millis=config.export_timeout_millis,
            )
        )
    elif exporter_name == "console":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        assert config.otlp_endpoint is not None
        exporter = OTLPSpanExporter(
            endpoint=_signal_endpoint(config.otlp_endpoint, "traces"),
            timeout=config.export_timeout_millis / 1000,
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                max_queue_size=config.max_queue_size,
                schedule_delay_millis=config.schedule_delay_millis,
                max_export_batch_size=config.max_export_batch_size,
                export_timeout_millis=config.export_timeout_millis,
            )
        )

    trace.set_tracer_provider(provider)
    _provider = provider
    logger.info(
        "tracer provider installed",
        service_name=config.service_name,
        exporter=exporter_name,
        sqlite_path=config.sqlite_path if exporter_name == "sqlite" else None,
    )
    return provider


def shutdown_tracer_provider() -> None:
    """Flush and shut down the process-global tracer provider."""

    global _provider
    if _provider is None:
        return
    _provider.force_flush()
    _provider.shutdown()
    _provider = None
