"""Process-level tracing configuration for MindMemOS Lite."""

from dataclasses import dataclass, field

from ..errors import InvalidConfigError
from .base import MindMemOSConfig, secret_field
from .validation import join_path, positive_integer, positive_number, range_optional, require_string


@dataclass
class PostgresObservabilityConfig(MindMemOSConfig):
    """Independent PostgreSQL connection settings for trace storage."""

    dsn: str = secret_field(default="")
    schema: str = "observability"
    max_pool_size: int = 2
    pool_timeout_seconds: float = 5.0
    connect_timeout_seconds: float = 5.0
    statement_timeout_seconds: float = 5.0
    create_schema: bool = True

    @classmethod
    def validate_self(cls, value, path: str) -> None:
        require_string(join_path(path, "schema"), value.schema)
        positive_integer(join_path(path, "max_pool_size"), value.max_pool_size)
        positive_number(join_path(path, "pool_timeout_seconds"), value.pool_timeout_seconds)
        positive_number(join_path(path, "connect_timeout_seconds"), value.connect_timeout_seconds)
        positive_number(join_path(path, "statement_timeout_seconds"), value.statement_timeout_seconds)


@dataclass
class ObservabilityConfig(MindMemOSConfig):
    """Configure the local OpenTelemetry span pipeline.

    SQLite remains the zero-configuration default. PostgreSQL can share the
    memory database or use an independent server; console and OTLP are also
    supported.
    """

    enabled: bool = True
    service_name: str = "mindmemos-lite"
    exporter: str = "sqlite"
    sqlite_path: str = ".mindmemos/observability/traces.db"
    postgres: PostgresObservabilityConfig = field(default_factory=PostgresObservabilityConfig)
    otlp_endpoint: str | None = None
    trace_sampling_ratio: float = 1.0
    max_queue_size: int = 2048
    max_export_batch_size: int = 128
    schedule_delay_millis: int = 1000
    export_timeout_millis: int = 5000
    retention_days: int | None = 14
    capture_content: bool = False

    @classmethod
    def validate_self(cls, value, path: str) -> None:
        require_string(join_path(path, "service_name"), value.service_name)
        exporter = value.exporter.strip().lower()
        if exporter not in {"sqlite", "postgres", "console", "otlp"}:
            raise InvalidConfigError(
                join_path(path, "exporter"),
                support="sqlite, postgres, console, or otlp",
            )
        if exporter == "postgres" and value.enabled:
            require_string(join_path(path, "postgres.dsn"), value.postgres.dsn)
        if exporter == "sqlite":
            require_string(join_path(path, "sqlite_path"), value.sqlite_path)
        if exporter == "otlp" and value.enabled:
            require_string(join_path(path, "otlp_endpoint"), value.otlp_endpoint)
        range_optional(
            join_path(path, "trace_sampling_ratio"),
            value.trace_sampling_ratio,
            minimum=0.0,
            maximum=1.0,
        )
        positive_integer(join_path(path, "max_queue_size"), value.max_queue_size)
        positive_integer(
            join_path(path, "max_export_batch_size"),
            value.max_export_batch_size,
        )
        if value.max_export_batch_size > value.max_queue_size:
            raise InvalidConfigError(
                join_path(path, "max_export_batch_size"),
                support="less than or equal to observability.max_queue_size",
            )
        positive_integer(
            join_path(path, "schedule_delay_millis"),
            value.schedule_delay_millis,
        )
        positive_number(
            join_path(path, "export_timeout_millis"),
            value.export_timeout_millis,
        )
        if value.retention_days is not None:
            positive_integer(join_path(path, "retention_days"), value.retention_days)
