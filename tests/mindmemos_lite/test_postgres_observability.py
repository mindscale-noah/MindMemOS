from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from mindmemos_lite.config import ObservabilityConfig, build_config, validate_tree
from mindmemos_lite.config.base import build, safe_dict
from mindmemos_lite.config.observability import PostgresObservabilityConfig
from mindmemos_lite.errors import InvalidConfigError, MissingConfigValueError
from mindmemos_lite.infra import telemetry
from mindmemos_lite.infra.observability import CompletedSpan, PostgresObservabilityBackend, SpanEventRecord
from mindmemos_lite.infra.observability import postgres_backend as module

TRACE_DSN_ENV = "MINDMEMOS_TRACE_POSTGRES_DSN"
TRACE_SCHEMA_ENV = "MINDMEMOS_TRACE_POSTGRES_SCHEMA"
MEMORY_DSN_ENV = "PGVECTOR_DSN"
EXAMPLE_CONFIG = "config/mindmemos_lite/dev.example.yaml"


def make_span(**overrides):
    return replace(
        CompletedSpan(
            trace_id="a" * 32,
            span_id="b" * 16,
            parent_span_id=None,
            name="llm.chat.provider",
            kind="INTERNAL",
            start_time_ns=1_700_000_000_000_000_000,
            end_time_ns=1_700_000_000_000_000_123,
            status_code="OK",
            status_message=None,
            service_name="test",
            instrumentation_scope="test",
            attributes={"run_id": "run-1", "llm.call": True, "llm.usage.total_tokens": 12},
            resource={},
            events=(SpanEventRecord(name="event", timestamp_ns=123, attributes={"value": "测试"}),),
        ),
        **overrides,
    )


@pytest.fixture
def pool_factory(monkeypatch):
    pool = MagicMock()
    connection = MagicMock()
    cursor = MagicMock()
    pool.connection.side_effect = lambda: nullcontext(connection)
    connection.transaction.side_effect = lambda: nullcontext()
    connection.pipeline.side_effect = lambda: nullcontext()
    connection.cursor.side_effect = lambda: nullcontext(cursor)
    factory = MagicMock(return_value=pool)
    monkeypatch.setattr(module, "ConnectionPool", factory)
    return factory, pool, connection, cursor


def test_postgres_backend_uses_qualified_tables_and_atomic_pipeline(pool_factory):
    factory, pool, connection, cursor = pool_factory
    backend = PostgresObservabilityBackend(PostgresObservabilityConfig(dsn="postgresql://logs", schema='trace"test'))
    ddl = "\n".join(
        str(call.args[0].as_string()) for call in cursor.execute.call_args_list if not isinstance(call.args[0], str)
    )
    assert '"trace""test".' in ddl
    assert "JSONB" in ddl
    assert "BIGINT" in ddl
    assert "CREATE EXTENSION" not in ddl
    assert factory.call_args.kwargs["conninfo"] == "postgresql://logs"
    cursor.reset_mock()
    backend.write_spans([make_span()])
    connection.pipeline.assert_called_once()
    statements = [call.args[0].as_string() for call in cursor.execute.call_args_list[1:]]
    assert "LEAST(" in statements[0]
    assert " || " in statements[0]
    assert any("DELETE FROM" in statement for statement in statements)
    backend.force_flush()
    backend.close()
    backend.close()
    pool.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        backend.write_spans([make_span()])
    with pytest.raises(RuntimeError, match="closed"):
        backend.force_flush()


def test_failed_initialization_closes_pool(pool_factory):
    _, pool, _, cursor = pool_factory
    cursor.execute.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        PostgresObservabilityBackend(PostgresObservabilityConfig(dsn="postgresql://logs"))
    pool.close.assert_called_once()


def test_postgres_exporter_configuration_uses_own_database(monkeypatch, pool_factory):
    factory, pool, _, _ = pool_factory
    provider = MagicMock()
    monkeypatch.setattr(telemetry, "_provider", None)
    monkeypatch.setattr(telemetry, "TracerProvider", MagicMock(return_value=provider))
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", MagicMock())
    processor = MagicMock()
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", processor)
    config = ObservabilityConfig(exporter="postgres", postgres=PostgresObservabilityConfig(dsn="postgresql://logs"))
    assert telemetry.setup_tracer_provider(config) is provider
    assert factory.call_args.kwargs["conninfo"] == "postgresql://logs"
    exporter = processor.call_args.args[0]
    exporter.shutdown()
    pool.close.assert_called_once()


def test_example_shares_database_by_default_and_can_split(monkeypatch):
    monkeypatch.delenv(TRACE_DSN_ENV, raising=False)
    monkeypatch.delenv(TRACE_SCHEMA_ENV, raising=False)
    monkeypatch.setenv(MEMORY_DSN_ENV, "postgresql://memory-host/memory")
    config = build_config(config_path=EXAMPLE_CONFIG)
    assert config.observability.postgres.dsn == config.database.pgvector.dsn
    assert config.observability.postgres.schema == "observability"
    monkeypatch.setenv(TRACE_DSN_ENV, "postgresql://log-host/logs")
    monkeypatch.setenv(TRACE_SCHEMA_ENV, "custom_logs")
    config = build_config(config_path=EXAMPLE_CONFIG)
    assert config.observability.postgres.dsn == "postgresql://log-host/logs"
    assert config.observability.postgres.schema == "custom_logs"
    assert config.database.pgvector.dsn == "postgresql://memory-host/memory"
    masked = safe_dict(config)
    assert masked["observability"]["postgres"]["dsn"] == "*****"
    assert masked["database"]["pgvector"]["dsn"] == "*****"


@pytest.mark.parametrize(
    "field", ["max_pool_size", "pool_timeout_seconds", "connect_timeout_seconds", "statement_timeout_seconds"]
)
def test_postgres_timeouts_and_pool_size_are_positive(field):
    config = build(ObservabilityConfig, {"postgres": {field: 0}})
    with pytest.raises(InvalidConfigError):
        validate_tree(config, "observability")


def test_postgres_requires_dsn_only_when_selected_and_enabled():
    for overrides in ({}, {"exporter": "postgres", "enabled": False}):
        validate_tree(build(ObservabilityConfig, overrides), "observability")
    with pytest.raises(MissingConfigValueError):
        validate_tree(build(ObservabilityConfig, {"exporter": "postgres"}), "observability")
