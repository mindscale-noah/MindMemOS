"""Process-local task execution without durable state."""

import asyncio

from .base import TaskHandler, TaskManager
from .dispatcher import TaskDispatcher
from .progress import EventProgressReporter
from .typing import ProgressEvent, TaskSnapshot, TaskSpec, TaskStatus


class LocalTaskManager(TaskManager):
    """Own in-memory task state and running asyncio tasks."""

    def __init__(self, handler: TaskHandler, *, max_concurrency: int = 2, max_buffered: int = 1000):
        """Accept the application handler used for each submitted task.

        Args:
            handler: Coroutine invoked with one task and its progress reporter.
            max_concurrency: Maximum simultaneous business executions.
            max_buffered: Maximum queued and running tasks admitted to the dispatcher.
        """
        self._handler = handler
        self._snapshots: dict[str, TaskSnapshot] = {}
        self._events: dict[str, list[ProgressEvent]] = {}
        self._dispatcher = TaskDispatcher(self._run, max_concurrency=max_concurrency, max_buffered=max_buffered)
        self._closed = False

    async def submit(self, spec: TaskSpec) -> TaskSnapshot:
        """Wait for admission and return the initial queued snapshot."""
        if self._closed:
            raise RuntimeError("Task manager is closed")
        if spec.task_id in self._snapshots:
            raise ValueError("Task ID already exists")
        snapshot = TaskSnapshot(
            task_id=spec.task_id, project_id=spec.project_id, kind=spec.kind, status=TaskStatus.QUEUED
        )
        self._snapshots[spec.task_id] = snapshot
        self._events[spec.task_id] = []
        try:
            await self._dispatcher.submit(spec.task_id, spec.model_copy(deep=True))
        except BaseException:
            del self._snapshots[spec.task_id]
            del self._events[spec.task_id]
            raise
        return snapshot.model_copy(deep=True)

    async def _run(self, spec: TaskSpec) -> None:
        """Execute a task and retain its terminal state."""
        snapshot = self._snapshots[spec.task_id]
        snapshot.status = TaskStatus.RUNNING
        snapshot.attempt += 1

        async def record(event: ProgressEvent) -> str:
            """Store one local event before returning its identity."""
            event = event.model_copy(update={"sequence": len(self._events[spec.task_id]) + 1})
            self._events[spec.task_id].append(event)
            snapshot.progress = event
            return f"{spec.task_id}:{len(self._events[spec.task_id])}"

        try:
            snapshot.result = await self._handler(spec, EventProgressReporter(record, task_id=spec.task_id))
        except asyncio.CancelledError:
            snapshot.status = TaskStatus.CANCELLED
            raise
        except Exception as exc:
            snapshot.error_type = type(exc).__name__
            snapshot.status = TaskStatus.FAILED
        else:
            snapshot.status = TaskStatus.SUCCEEDED

    async def get(self, project_id: str, task_id: str) -> TaskSnapshot | None:
        """Return a detached snapshot only for its owning project."""
        snapshot = self._snapshots.get(task_id)
        return snapshot.model_copy(deep=True) if snapshot is not None and snapshot.project_id == project_id else None

    async def events(self, project_id: str, task_id: str, *, after: int = 0) -> list[ProgressEvent]:
        """Return local events after a one-based event sequence."""
        if await self.get(project_id, task_id) is None:
            return []
        return [event.model_copy(deep=True) for event in self._events[task_id][after:]]

    async def close(self) -> None:
        """Cancel owned work when the process-local manager closes."""
        self._closed = True
        await self._dispatcher.close()
        for snapshot in self._snapshots.values():
            if snapshot.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                snapshot.status = TaskStatus.CANCELLED
