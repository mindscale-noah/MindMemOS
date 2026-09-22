"""SQLite implementation of the observability storage port."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock

from .models import CompletedSpan
from .serialization import _json_dumps, _llm_operation, _optional_int, _optional_text, _truthy

_SCHEMA_VERSION = 1
_SCOPE_KEYS = ("run_id", "request_id", "project_id", "api_key_uuid")

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    root_span_id TEXT,
    service_name TEXT NOT NULL,
    start_time_ns INTEGER NOT NULL,
    end_time_ns INTEGER NOT NULL,
    run_id TEXT,
    request_id TEXT,
    project_id TEXT,
    api_key_uuid TEXT,
    attributes_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS spans (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    parent_span_id TEXT,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_time_ns INTEGER NOT NULL,
    end_time_ns INTEGER NOT NULL,
    duration_ns INTEGER NOT NULL,
    status_code TEXT NOT NULL,
    status_message TEXT,
    service_name TEXT NOT NULL,
    instrumentation_scope TEXT,
    attributes_json TEXT NOT NULL,
    resource_json TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id),
    FOREIGN KEY (trace_id) REFERENCES traces(trace_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS span_events (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    name TEXT NOT NULL,
    timestamp_ns INTEGER NOT NULL,
    attributes_json TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id, event_index),
    FOREIGN KEY (trace_id, span_id) REFERENCES spans(trace_id, span_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_calls (
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    task TEXT,
    model TEXT,
    provider TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    duration_ns INTEGER NOT NULL,
    status_code TEXT NOT NULL,
    PRIMARY KEY (trace_id, span_id),
    FOREIGN KEY (trace_id, span_id) REFERENCES spans(trace_id, span_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_traces_scope
    ON traces(project_id, api_key_uuid, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_traces_run
    ON traces(run_id, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_spans_name_start
    ON spans(name, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_spans_trace_start
    ON spans(trace_id, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_spans_status_start
    ON spans(status_code, start_time_ns);
CREATE INDEX IF NOT EXISTS idx_span_events_name_time
    ON span_events(name, timestamp_ns);
CREATE INDEX IF NOT EXISTS idx_llm_calls_task_model
    ON llm_calls(task, model);
"""


class SQLiteObservabilityBackend:
    """Persist completed spans, events, and LLM projections transactionally."""

    def __init__(
        self,
        path: str | Path,
        *,
        retention_days: int | None = 14,
        busy_timeout_seconds: float = 5.0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._closed = False
        self._connection = sqlite3.connect(
            self.path,
            timeout=busy_timeout_seconds,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_seconds * 1000)}")
        self._connection.executescript(_SCHEMA)
        self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        if retention_days is not None:
            cutoff_ns = int((datetime.now(UTC) - timedelta(days=retention_days)).timestamp() * 1_000_000_000)
            self._connection.execute("DELETE FROM traces WHERE end_time_ns < ?", (cutoff_ns,))
        self._connection.commit()

    def write_spans(self, spans: Sequence[CompletedSpan]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLite observability backend is closed")
            with self._connection:
                for span in spans:
                    self._write_span(span)

    def force_flush(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLite observability backend is closed")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._connection.commit()
            self._connection.close()
            self._closed = True

    def _write_span(self, span: CompletedSpan) -> None:
        attributes = span.attributes
        scope_attributes = {key: _optional_text(attributes.get(key)) for key in _SCOPE_KEYS}
        trace_attributes = {key: value for key, value in scope_attributes.items() if value is not None}

        self._connection.execute(
            """
            INSERT INTO traces (
                trace_id, root_span_id, service_name, start_time_ns, end_time_ns,
                run_id, request_id, project_id, api_key_uuid, attributes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trace_id) DO UPDATE SET
                root_span_id = COALESCE(excluded.root_span_id, traces.root_span_id),
                service_name = excluded.service_name,
                start_time_ns = MIN(traces.start_time_ns, excluded.start_time_ns),
                end_time_ns = MAX(traces.end_time_ns, excluded.end_time_ns),
                run_id = COALESCE(excluded.run_id, traces.run_id),
                request_id = COALESCE(excluded.request_id, traces.request_id),
                project_id = COALESCE(excluded.project_id, traces.project_id),
                api_key_uuid = COALESCE(excluded.api_key_uuid, traces.api_key_uuid),
                attributes_json = json_patch(traces.attributes_json, excluded.attributes_json)
            """,
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
                _json_dumps(trace_attributes),
            ),
        )
        self._connection.execute(
            """
            INSERT INTO spans (
                trace_id, span_id, parent_span_id, name, kind, start_time_ns,
                end_time_ns, duration_ns, status_code, status_message,
                service_name, instrumentation_scope, attributes_json, resource_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            """,
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
                _json_dumps(attributes),
                _json_dumps(span.resource),
            ),
        )

        self._connection.execute(
            "DELETE FROM span_events WHERE trace_id = ? AND span_id = ?",
            (span.trace_id, span.span_id),
        )
        for index, event in enumerate(span.events):
            self._connection.execute(
                """
                INSERT INTO span_events (
                    trace_id, span_id, event_index, name, timestamp_ns, attributes_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    span.trace_id,
                    span.span_id,
                    index,
                    event.name,
                    event.timestamp_ns,
                    _json_dumps(event.attributes),
                ),
            )

        self._connection.execute(
            "DELETE FROM llm_calls WHERE trace_id = ? AND span_id = ?",
            (span.trace_id, span.span_id),
        )
        if _truthy(attributes.get("llm.call")):
            self._connection.execute(
                """
                INSERT INTO llm_calls (
                    trace_id, span_id, operation, task, model, provider,
                    prompt_tokens, completion_tokens, total_tokens,
                    duration_ns, status_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
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
