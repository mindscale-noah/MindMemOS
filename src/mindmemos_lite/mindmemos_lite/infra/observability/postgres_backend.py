"""Synchronous PostgreSQL storage for completed OpenTelemetry spans."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from threading import Lock

from psycopg import Cursor, sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from ...config.observability import PostgresObservabilityConfig
from .constants import LLM_CALLS_TABLE, SPAN_EVENTS_TABLE, SPANS_TABLE, TRACES_TABLE
from .models import CompletedSpan
from .serialization import _llm_operation, _optional_int, _optional_text, _truthy

_SCOPE_KEYS = ("run_id", "request_id", "project_id", "api_key_uuid")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS {traces} (
    trace_id TEXT PRIMARY KEY,
    root_span_id TEXT,
    service_name TEXT NOT NULL,
    start_time_ns BIGINT NOT NULL,
    end_time_ns BIGINT NOT NULL,
    run_id TEXT,
    request_id TEXT,
    project_id TEXT,
    api_key_uuid TEXT,
    attributes_json JSONB NOT NULL DEFAULT '{{}}'
);

CREATE TABLE IF NOT EXISTS {spans} (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_time_ns BIGINT NOT NULL,
    end_time_ns BIGINT NOT NULL,
    duration_ns BIGINT NOT NULL,
    status_code TEXT NOT NULL,
    status_message TEXT,
    service_name TEXT NOT NULL,
    instrumentation_scope TEXT,
    attributes_json JSONB NOT NULL,
    resource_json JSONB NOT NULL,
    PRIMARY KEY (trace_id, span_id),
    FOREIGN KEY (trace_id) REFERENCES {traces}(trace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS {span_events} (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    event_index BIGINT NOT NULL,
    name TEXT NOT NULL,
    timestamp_ns BIGINT NOT NULL,
    attributes_json JSONB NOT NULL,
    PRIMARY KEY (trace_id, span_id, event_index),
    FOREIGN KEY (trace_id, span_id) REFERENCES {spans}(trace_id, span_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS {llm_calls} (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    task TEXT,
    model TEXT,
    provider TEXT,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    total_tokens BIGINT,
    duration_ns BIGINT NOT NULL,
    status_code TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id),
    FOREIGN KEY (trace_id, span_id) REFERENCES {spans}(trace_id, span_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_traces_scope
    ON {traces}(project_id, api_key_uuid, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_traces_run
    ON {traces}(run_id, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_spans_name_start
    ON {spans}(name, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_spans_trace_start
    ON {spans}(trace_id, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_spans_status_start
    ON {spans}(status_code, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_span_events_name_time
    ON {span_events}(name, timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_llm_calls_task_model
    ON {llm_calls}(task, model);
"""


class PostgresObservabilityBackend:
    """Store trace batches atomically without requiring the vector extension."""

    def __init__(self, config: PostgresObservabilityConfig, *, retention_days: int | None = 14) -> None:
        """Open an independent pool and initialize the trace tables.

        Args:
            config: Trace database connection, schema and timeout settings.
            retention_days: Startup retention window; None disables deletion.
        """
        self._lock = Lock()
        self._closed = False
        self._statement_timeout_ms = math.ceil(config.statement_timeout_seconds * 1000)
        self._tables = {
            "traces": sql.Identifier(config.schema, TRACES_TABLE),
            "spans": sql.Identifier(config.schema, SPANS_TABLE),
            "span_events": sql.Identifier(config.schema, SPAN_EVENTS_TABLE),
            "llm_calls": sql.Identifier(config.schema, LLM_CALLS_TABLE),
        }
        self._pool = ConnectionPool(
            conninfo=config.dsn,
            min_size=1,
            max_size=config.max_pool_size,
            timeout=config.pool_timeout_seconds,
            kwargs={"connect_timeout": max(1, math.ceil(config.connect_timeout_seconds))},
            open=False,
        )
        try:
            self._pool.open(wait=True, timeout=config.pool_timeout_seconds)
            with self._pool.connection() as connection, connection.transaction(), connection.cursor() as cursor:
                self._set_timeout(cursor)
                # Serialize initialization across workers sharing a schema.
                cursor.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s), hashtext(%s))",
                    ("mindmemos.observability.schema", config.schema),
                )
                if config.create_schema:
                    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(config.schema)))
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        cursor.execute(sql.SQL(statement).format(**self._tables))
                if retention_days is not None:
                    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
                    cursor.execute(
                        sql.SQL("DELETE FROM {traces} WHERE end_time_ns < %s").format(**self._tables),
                        (int(cutoff.timestamp() * 1_000_000_000),),
                    )
        except Exception:
            self._pool.close()
            raise

    def _set_timeout(self, cursor: Cursor) -> None:
        cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(self._statement_timeout_ms),))

    def write_spans(self, spans: Sequence[CompletedSpan]) -> None:
        """Commit one batch atomically, using pipelining to reduce network round trips.

        Args:
            spans: Completed spans; repeated span IDs replace events and projections.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("PostgreSQL observability backend is closed")
            if not spans:
                return
            with self._pool.connection() as connection, connection.transaction(), connection.cursor() as cursor:
                self._set_timeout(cursor)
                with connection.pipeline():
                    # Consistent trace locking order reduces cross-process deadlocks.
                    for span in sorted(spans, key=lambda item: (item.trace_id, item.span_id)):
                        self._write_span(cursor, span)

    def force_flush(self) -> None:
        """Wait for any active batch; each write already commits before returning."""
        with self._lock:
            if self._closed:
                raise RuntimeError("PostgreSQL observability backend is closed")

    def close(self) -> None:
        """Wait for active writes and release the pool exactly once."""
        with self._lock:
            if not self._closed:
                self._pool.close()
                self._closed = True

    def _write_span(self, cursor: Cursor, span: CompletedSpan) -> None:
        attributes = span.attributes
        scope_attributes = {key: _optional_text(attributes.get(key)) for key in _SCOPE_KEYS}
        trace_attributes = {key: value for key, value in scope_attributes.items() if value is not None}

        cursor.execute(
            sql.SQL("""
            INSERT INTO {traces} AS traces (
                trace_id, root_span_id, service_name, start_time_ns, end_time_ns,
                run_id, request_id, project_id, api_key_uuid, attributes_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(trace_id) DO UPDATE SET
                root_span_id = COALESCE(excluded.root_span_id, traces.root_span_id),
                service_name = excluded.service_name,
                start_time_ns = LEAST(traces.start_time_ns, excluded.start_time_ns),
                end_time_ns = GREATEST(traces.end_time_ns, excluded.end_time_ns),
                run_id = COALESCE(excluded.run_id, traces.run_id),
                request_id = COALESCE(excluded.request_id, traces.request_id),
                project_id = COALESCE(excluded.project_id, traces.project_id),
                api_key_uuid = COALESCE(excluded.api_key_uuid, traces.api_key_uuid),
                attributes_json = traces.attributes_json || excluded.attributes_json
            """).format(**self._tables),
            (
                span.trace_id,
                span.span_id if span.parent_span_id is None else None,
                span.service_name,
                span.start_time_ns,
                span.end_time_ns,
                scope_attributes["run_id"],
                scope_attributes["request_id"],
                scope_attributes["project_id"],
                scope_attributes["api_key_uuid"],
                Jsonb(trace_attributes),
            ),
        )
        cursor.execute(
            sql.SQL("""
            INSERT INTO {spans} (
                trace_id, span_id, parent_span_id, name, kind, start_time_ns,
                end_time_ns, duration_ns, status_code, status_message,
                service_name, instrumentation_scope, attributes_json, resource_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(trace_id, span_id) DO UPDATE SET
                parent_span_id = excluded.parent_span_id,
                name = excluded.name,
                kind = excluded.kind,
                start_time_ns = excluded.start_time_ns,
                end_time_ns = excluded.end_time_ns,
                duration_ns = excluded.duration_ns,
                status_code = excluded.status_code,
                status_message = excluded.status_message,
                service_name = excluded.service_name,
                instrumentation_scope = excluded.instrumentation_scope,
                attributes_json = excluded.attributes_json,
                resource_json = excluded.resource_json
            """).format(**self._tables),
            (
                span.trace_id,
                span.span_id,
                span.parent_span_id,
                span.name,
                span.kind,
                span.start_time_ns,
                span.end_time_ns,
                span.duration_ns,
                span.status_code,
                span.status_message,
                span.service_name,
                span.instrumentation_scope,
                Jsonb(attributes),
                Jsonb(span.resource),
            ),
        )

        cursor.execute(
            sql.SQL("DELETE FROM {span_events} WHERE trace_id = %s AND span_id = %s").format(**self._tables),
            (span.trace_id, span.span_id),
        )
        for index, event in enumerate(span.events):
            cursor.execute(
                sql.SQL("""
                INSERT INTO {span_events} (
                    trace_id, span_id, event_index, name, timestamp_ns, attributes_json
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """).format(**self._tables),
                (
                    span.trace_id,
                    span.span_id,
                    index,
                    event.name,
                    event.timestamp_ns,
                    Jsonb(event.attributes),
                ),
            )

        cursor.execute(
            sql.SQL("DELETE FROM {llm_calls} WHERE trace_id = %s AND span_id = %s").format(**self._tables),
            (span.trace_id, span.span_id),
        )
        if _truthy(attributes.get("llm.call")):
            cursor.execute(
                sql.SQL("""
                INSERT INTO {llm_calls} (
                    trace_id, span_id, operation, task, model, provider,
                    prompt_tokens, completion_tokens, total_tokens,
                    duration_ns, status_code
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """).format(**self._tables),
                (
                    span.trace_id,
                    span.span_id,
                    str(attributes.get("llm.operation") or _llm_operation(span.name)),
                    _optional_text(attributes.get("llm.task")),
                    _optional_text(attributes.get("llm.model")),
                    _optional_text(attributes.get("llm.provider")),
                    _optional_int(attributes.get("llm.usage.prompt_tokens")),
                    _optional_int(attributes.get("llm.usage.completion_tokens")),
                    _optional_int(attributes.get("llm.usage.total_tokens")),
                    span.duration_ns,
                    span.status_code,
                ),
            )
