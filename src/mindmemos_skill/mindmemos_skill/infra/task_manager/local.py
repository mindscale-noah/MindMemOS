"""Process-local task execution without durable state."""

import asyncio
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ...logging import get_logger, log_context
from ...logging.tasks import async_task_log_reporter
from ...logging.tracing import bind_tracer, get_tracer, log_failure
from ...logging.typing import TaskLog
from .base import TaskManager
from .dispatcher import TaskDispatcher
from .typing import TaskHandler, TaskSnapshot, TaskSpec, TaskStatus


class LocalTaskManager(TaskManager):
    """Own in-memory task state and running asyncio tasks."""

    def __init__(
        self,
        handler: TaskHandler,
        *,
        max_concurrency: int = 2,
        max_buffered: int = 1000,
        tracer: trace.Tracer | None = None,
    ):
        """Accept the application handler used for each submitted task.

        Args:
            handler: Coroutine invoked with one immutable task request.
            max_concurrency: Maximum simultaneous business executions.
            tracer: Owning service provider for task execution spans.
            max_buffered: Maximum queued and running tasks admitted to the dispatcher.
        """
        self._tracer = tracer or get_tracer()
        self._handler = handler
        self._snapshots: dict[str, TaskSnapshot] = {}
        self._logs: dict[str, list[TaskLog]] = {}
        self._dispatcher = TaskDispatcher(self._run, max_concurrency=max_concurrency, max_buffered=max_buffered)
        self._closed = False

    async def submit(self, spec: TaskSpec) -> TaskSnapshot:
        """Wait for admission and return the initial queued snapshot."""
        if self._closed:
            raise RuntimeError("Task manager is closed")
        if spec.task_id in self._snapshots:
            raise ValueError("Task ID already exists")
        snapshot = TaskSnapshot(
            task_id=spec.task_id,
            project_id=spec.project_id,
            kind=spec.kind,
            status=TaskStatus.QUEUED,
            accepted_at=datetime.now(UTC),
        )
        self._snapshots[spec.task_id] = snapshot
        self._logs[spec.task_id] = []
        try:
            await self._dispatcher.submit(spec.task_id, spec.model_copy(deep=True))
        except BaseException:
            del self._snapshots[spec.task_id]
            del self._logs[spec.task_id]
            raise
        return snapshot.model_copy(deep=True)

    async def _run(self, spec: TaskSpec) -> None:
        """Execute a task and retain its terminal state."""
        snapshot = self._snapshots[spec.task_id]
        snapshot.status = TaskStatus.RUNNING
        snapshot.attempt += 1
        snapshot.started_at = datetime.now(UTC)

        async def append_log(row: dict[str, Any]) -> None:
            """Append the same captured log schema used by durable workers."""
            self._logs[spec.task_id].append(TaskLog(**row, sequence=len(self._logs[spec.task_id]) + 1))

        cancelled = None
        try:
            with (
                bind_tracer(self._tracer),
                log_context(request_id=spec.request_id, service_name="mindmemos-skill"),
                self._tracer.start_as_current_span(
                    "task.execute", context=TraceContextTextMapPropagator().extract(spec.trace_context)
                ) as span,
            ):
                span.set_attribute("task_id", spec.task_id)
                span.set_attribute("project_id", spec.project_id)
                span.set_attribute("request_id", spec.request_id or "")
                span.set_attribute("attempt", snapshot.attempt)
                async with async_task_log_reporter(
                    append_log, project_id=spec.project_id, task_id=spec.task_id, attempt=snapshot.attempt
                ) as flush_logs:
                    try:
                        await self._execute(spec, snapshot, span, flush_logs)
                    except asyncio.CancelledError as exc:
                        cancelled = exc
                        span.set_attribute("status", "cancelled")
            if cancelled is not None:
                raise cancelled
            return
        except asyncio.CancelledError:
            snapshot.status = TaskStatus.CANCELLED
            snapshot.finished_at = datetime.now(UTC)
            raise

    async def _execute(self, spec, snapshot, span, flush_logs) -> None:
        """Record execution outcome and terminal timing while task logging is active."""
        logger = get_logger(__name__)
        logger.info("task.started", extra={"event_body": {"waiting_ms": snapshot.waiting_ms}})
        status = TaskStatus.SUCCEEDED
        try:
            snapshot.result = await self._handler(spec)
        except asyncio.CancelledError:
            logger.info("task.cancelled", extra={"event_body": {"status": "cancelled"}})
            raise
        except Exception as exc:
            snapshot.error_type = type(exc).__name__
            status = TaskStatus.FAILED
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            log_failure(logger, "task.failed", body={"error_type": type(exc).__name__})
        await flush_logs()
        snapshot.finished_at = datetime.now(UTC)
        body = {
            "status": status,
            "waiting_ms": snapshot.waiting_ms,
            "execution_ms": snapshot.execution_ms,
            "end_to_end_ms": snapshot.end_to_end_ms,
        }
        for key, value in body.items():
            span.set_attribute(key, value)
        logger.info("task.finished", extra={"event_body": body})
        await flush_logs()
        snapshot.status = status

    async def get(self, project_id: str, task_id: str) -> TaskSnapshot | None:
        """Return a detached snapshot only for its owning project."""
        snapshot = self._snapshots.get(task_id)
        return snapshot.model_copy(deep=True) if snapshot is not None and snapshot.project_id == project_id else None

    async def logs(self, project_id: str, task_id: str, *, after: int = 0) -> list[TaskLog]:
        """Return captured task logs after the supplied sequence."""
        if await self.get(project_id, task_id) is None:
            return []
        return [row.model_copy(deep=True) for row in self._logs[task_id] if row.sequence > after][:100]

    async def close(self) -> None:
        """Cancel owned work when the process-local manager closes."""
        self._closed = True
        await self._dispatcher.close()
        for snapshot in self._snapshots.values():
            if snapshot.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                snapshot.status = TaskStatus.CANCELLED
                snapshot.finished_at = datetime.now(UTC)
