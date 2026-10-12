"""Persist algorithm runs, correlated spans, and debug logs in one SQL backend."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk._logs import ReadWriteLogRecord
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Link, SpanContext, Status, StatusCode, TraceFlags

from ...infra.database import Database, Filter, TableRegistry
from ...infra.observability import OpenTelemetryRecorder, serialize_log, serialize_span
from ...logging import get_logger, log_context
from ...logging.otel import attach_logs, detach_logs
from ...logging.tracing import bind_tracer
from .constants import OBSERVABILITY_LOGS_TABLE, OBSERVABILITY_RUNS_TABLE, OBSERVABILITY_SPANS_TABLE
from .tables import build_tables
from .typing import AuditLog, AuditRun, AuditSpan

_active_session: ContextVar["AuditSession | None"] = ContextVar("skill_audit_session", default=None)


class AuditSession:
    """Carry run identity and an OTel tracer through async and worker contexts."""

    def __init__(
        self, store: "AlgorithmAuditPersistence", project_id: str, run_id: str, level: Literal["info", "debug"]
    ):
        self.store = store
        self.project_id = project_id
        self.run_id = run_id
        self.level = level
        self.export_failed = False
        self.published_version_id: str | None = None
        self.pending: set[asyncio.Task[None]] = set()

    @contextmanager
    def span(self, name: str, attributes: Mapping[str, str | int | float | bool] | None = None) -> Iterator[Any]:
        """Create an algorithm stage span beneath the current operation.

        Args:
            name: Stable operation name, such as algorithm.tree.expand.
            attributes: Scalar identifiers and measurements, without content bodies.

        Yields:
            Active OpenTelemetry span for bounded stage events.
        """
        if not name:
            raise ValueError("Span name must be nonempty")
        with self.store.telemetry.tracer.start_as_current_span(name) as span:
            span.set_attribute("algorithm.project_id", self.project_id)
            span.set_attribute("algorithm.run_id", self.run_id)
            for key, value in (attributes or {}).items():
                span.set_attribute(key, value)
            try:
                yield span
            except BaseException as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                span.set_status(Status(StatusCode.OK))


class AlgorithmAuditPersistence:
    """Bind algorithm audit tables and a local OpenTelemetry recorder to one database."""

    def __init__(
        self,
        database: Database | None,
        *,
        capture_level: Literal["info", "debug"] = "info",
        service_name: str = "mindmemos-skill",
        otlp_endpoint: str | None = None,
        otlp_timeout: float = 5.0,
    ) -> None:
        self.database = database
        self.capture_level = capture_level
        if database is not None:
            additions = build_tables()
            TableRegistry((*database.tables.values(), *additions.values()))
            for spec in additions.values():
                database.tables.register(spec)
        self.telemetry = OpenTelemetryRecorder(
            service_name=service_name,
            instrumentation_name="mindmemos_skill.algorithm",
            on_span_end=self._write_span,
            on_log_emit=self._write_log,
            on_export_error=self._mark_export_failure,
            otlp_endpoint=otlp_endpoint,
            otlp_timeout=otlp_timeout,
        )

        def log_context() -> dict[str, str] | None:
            """Bind the current log to this store's active run."""
            session = get_log_context()
            if session is None or session.store is not self:
                return None
            return {"algorithm.project_id": session.project_id, "algorithm.run_id": session.run_id}

        self._log_handler = attach_logs(
            self.telemetry.logger_provider, context=log_context, capture_level=capture_level
        )
        self._closed = False
        self._write_lock = asyncio.Lock()

    async def initialize_storage(self) -> None:
        """Create missing observability tables without changing existing tables."""
        if self.database is not None:
            (await self.database.create_tables())

    async def close(self) -> None:
        """Stop the local tracer provider before the owning database closes."""
        if self._closed:
            return
        self._closed = True
        detach_logs(self._log_handler)
        await self.telemetry.close()

    @asynccontextmanager
    async def trace_run(
        self, *, project_id: str, run_id: str, algorithm: str, base_version_id: str, operation: str = "train"
    ) -> AsyncIterator[AuditSession]:
        """Record one algorithm attempt with a root span and durable run status.

        Args:
            project_id: Repository scope of the input Skill.
            run_id: Stable identity shared across recovery attempts.
            algorithm: Configured training pipeline or trainer name.
            base_version_id: Input Skill version.
            operation: Operation suffix for the root span, such as train or evolve.

        Yields:
            Active session used by nested Agent calls.
        """
        previous = (await self.get_run(project_id, run_id)) if self.database is not None else None
        if previous is not None and (previous.algorithm != algorithm or previous.base_version_id != base_version_id):
            raise ValueError("Run identity conflicts with its existing audit record")
        prior_roots = (
            (
                await self.database.select(
                    OBSERVABILITY_SPANS_TABLE,
                    filters={"project_id": project_id, "run_id": run_id, "name": f"algorithm.{operation}"},
                    order_by=["-started_at"],
                    limit=1,
                )
            )
            if previous is not None
            else []
        )
        links = []
        if prior_roots:
            old = prior_roots[0]
            links.append(
                Link(
                    SpanContext(
                        trace_id=int(old["trace_id"], 16),
                        span_id=int(old["span_id"], 16),
                        is_remote=True,
                        trace_flags=TraceFlags(TraceFlags.SAMPLED),
                    )
                )
            )
        session = AuditSession(self, project_id, run_id, self.capture_level)
        session.export_failed = previous is not None and previous.audit_status == "incomplete"
        session.published_version_id = previous.published_version_id if previous is not None else None
        token = _active_session.set(session)
        error_type = None
        interrupted = False
        cancelled = None
        try:
            with (
                log_context(run_id=run_id, project_id=project_id),
                bind_tracer(self.telemetry.tracer),
                self.telemetry.tracer.start_as_current_span(f"algorithm.{operation}", links=links) as span,
            ):
                context = span.get_span_context()
                trace_id = trace.format_trace_id(context.trace_id)
                span.set_attribute("algorithm.project_id", project_id)
                span.set_attribute("algorithm.run_id", run_id)
                span.set_attribute("algorithm.name", algorithm)
                span.set_attribute("algorithm.base_version_id", base_version_id)
                span.set_attribute("algorithm.capture_level", self.capture_level)
                started = datetime.now(UTC)
                if self.database is not None:
                    (
                        await self.database.upsert(
                            OBSERVABILITY_RUNS_TABLE,
                            [
                                AuditRun(
                                    project_id=project_id,
                                    run_id=run_id,
                                    algorithm=algorithm,
                                    base_version_id=base_version_id,
                                    status="running",
                                    audit_status="incomplete" if session.export_failed else "complete",
                                    capture_level=self.capture_level,
                                    trace_id=trace_id,
                                    started_at=started,
                                ).model_dump(mode="python")
                            ],
                            update_fields=[
                                "status",
                                "audit_status",
                                "capture_level",
                                "trace_id",
                                "started_at",
                                "finished_at",
                                "error_type",
                            ],
                        )
                    )
                get_logger(__name__).info(
                    "algorithm.run.started",
                    extra={"event_body": {"algorithm": algorithm, "base_version_id": base_version_id}},
                )
                try:
                    yield session
                except asyncio.CancelledError as exc:
                    cancelled = exc
                    span.set_attribute("status", "interrupted")
                    get_logger(__name__).info("algorithm.run.interrupted")
                except BaseException as exc:
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(Status(StatusCode.ERROR))
                    get_logger(__name__).error(
                        "algorithm.run.failed", extra={"event_body": {"error_type": type(exc).__name__}}
                    )
                    raise
                else:
                    span.set_status(Status(StatusCode.OK))
                    get_logger(__name__).info(
                        "algorithm.run.completed",
                        extra={"event_body": {"published_version_id": session.published_version_id}},
                    )
            if cancelled is not None:
                raise cancelled
        except asyncio.CancelledError:
            interrupted = True
            raise
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            if session.pending:
                await asyncio.gather(*session.pending)
            _active_session.reset(token)
            if self.database is not None:
                (
                    await self.database.update(
                        OBSERVABILITY_RUNS_TABLE,
                        {
                            "status": "running" if interrupted else "failed" if error_type else "completed",
                            "audit_status": "incomplete" if session.export_failed else "complete",
                            "published_version_id": session.published_version_id,
                            "finished_at": None if interrupted else datetime.now(UTC),
                            "error_type": error_type,
                        },
                        filters={"project_id": project_id, "run_id": run_id},
                    )
                )

    def _write_span(self, span: ReadableSpan) -> None:
        """Persist a completed OTel span within the active audit session."""
        session = get_log_context()
        if self.database is None or session is None or session.store is not self:
            return
        payload = serialize_span(span)
        if payload is None:
            return
        record = AuditSpan(
            project_id=session.project_id,
            run_id=session.run_id,
            **payload,
        )
        self._enqueue(
            session, lambda: self.database.upsert(OBSERVABILITY_SPANS_TABLE, [record.model_dump(mode="python")])
        )

    def _enqueue(self, session: AuditSession, operation: Callable[[], Awaitable[int]]) -> None:
        """Queue an async database write and retain it until the run drains."""

        async def write():
            try:
                async with self._write_lock:
                    await operation()
            except Exception:
                session.export_failed = True
                get_logger(__name__).exception("audit.persistence.failed")

        task = asyncio.create_task(write())
        session.pending.add(task)
        task.add_done_callback(session.pending.discard)

    def _mark_export_failure(self, error: Exception) -> None:
        """Mark the active run incomplete after a span sink failure."""
        session = get_log_context()
        if session is not None and session.store is self:
            session.export_failed = True

    def _write_log(self, emitted: ReadWriteLogRecord) -> None:
        """Persist an OTel log when the database sink is configured."""
        session = get_log_context()
        if self.database is None or session is None or session.store is not self:
            return
        payload = serialize_log(emitted)
        if payload["level"] == "debug" and self.capture_level != "debug":
            return
        body = payload["body"]
        record = AuditLog(
            event_id=uuid4().hex,
            project_id=session.project_id,
            run_id=session.run_id,
            trace_id=payload["trace_id"],
            span_id=payload["span_id"],
            occurred_at=payload["occurred_at"],
            level=payload["level"],
            name=payload["name"],
            body=body if isinstance(body, dict) else {"value": body},
        )
        self._enqueue(
            session, lambda: self.database.insert(OBSERVABILITY_LOGS_TABLE, [record.model_dump(mode="python")])
        )

    async def get_run(self, project_id: str, run_id: str) -> AuditRun | None:
        """Read a run only within its project scope.

        Args:
            project_id: Required repository scope.
            run_id: Exact run identity.

        Returns:
            Current run state, or None when absent.
        """
        if self.database is None:
            return None
        row = await self.database.get(OBSERVABILITY_RUNS_TABLE, {"project_id": project_id, "run_id": run_id})
        return AuditRun.model_validate(row) if row is not None else None

    async def latest_evolution(self, project_id: str, version_id: str, algorithms: list[str]) -> datetime | None:
        """Read the latest indexed evolution start without loading audit payloads.

        Args:
            project_id: Authorized project identity.
            version_id: Exact evolution input version.
            algorithms: Configured evolution aliases and implementation names.

        Returns:
            Latest start time, or None without a matching run.
        """
        if self.database is None or not algorithms:
            return None
        rows = await self.database.select(
            OBSERVABILITY_RUNS_TABLE,
            filters=[
                Filter(field="project_id", value=project_id),
                Filter(field="base_version_id", value=version_id),
                Filter(field="algorithm", op="in", value=algorithms),
            ],
            fields=["started_at"],
            order_by=["-started_at"],
            limit=1,
        )
        return rows[0]["started_at"] if rows else None

    async def spans(self, project_id: str, run_id: str, *, limit: int = 500, offset: int = 0) -> list[AuditSpan]:
        """Read one stable page of a run's completed trace tree.

        Args:
            project_id: Required repository scope.
            run_id: Exact run identity.
            limit: Maximum returned spans.
            offset: Number of previous spans to skip.

        Returns:
            Spans ordered by start time.
        """
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid observability page")
        if self.database is None:
            return []
        rows = await self.database.select(
            OBSERVABILITY_SPANS_TABLE,
            filters={"project_id": project_id, "run_id": run_id},
            order_by=["started_at", "span_id"],
            limit=limit,
            offset=offset,
        )
        return [AuditSpan.model_validate(row) for row in rows]

    async def logs(self, project_id: str, run_id: str, *, limit: int = 200, offset: int = 0) -> list[AuditLog]:
        """Read one stable page of trace-correlated audit logs.

        Args:
            project_id: Required repository scope.
            run_id: Exact run identity.
            limit: Maximum returned logs.
            offset: Number of previous logs to skip.

        Returns:
            Logs ordered by event time.
        """
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid observability page")
        if self.database is None:
            return []
        rows = await self.database.select(
            OBSERVABILITY_LOGS_TABLE,
            filters={"project_id": project_id, "run_id": run_id},
            order_by=["occurred_at", "event_id"],
            limit=limit,
            offset=offset,
        )
        return [AuditLog.model_validate(row) for row in rows]


def get_log_context() -> AuditSession | None:
    """Return the active run-scoped audit session, if training is active."""
    return _active_session.get()
