"""Serialize expensive task admission across Skill service instances."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TypeVar

from anyio import CancelScope

from ....errors import ProjectTaskBusyError
from ....infra.database import Database, Filter
from ....infra.task_manager.typing import EVOLVE_TASK, TaskStatus
from ....logging import get_logger
from ...constants import (
    DEFAULT_MAX_ACTIVE_TASKS_PER_PROJECT,
    SKILL_PROJECT_TASK_LEASES_TABLE,
    SKILL_PROJECT_TASKS_TABLE,
    SKILL_TASKS_TABLE,
)

logger = get_logger(__name__)
LEASE_DURATION = timedelta(minutes=2)
HEARTBEAT_SECONDS = 30
ResultT = TypeVar("ResultT")


class ProjectTaskAdmission:
    """Limit active reservations and durable evolutions together per project."""

    def __init__(self, database: Database, *, max_active_tasks: int = DEFAULT_MAX_ACTIVE_TASKS_PER_PROJECT):
        """Use the shared database as the source of admission state.

        Args:
            database: Borrowed Skill business database shared by service instances.
            max_active_tasks: Maximum queued and running algorithm tasks per project.
        """
        if max_active_tasks < 1:
            raise ValueError("Project task limit must be positive")
        self._database = database
        self._max_active_tasks = max_active_tasks

    @asynccontextmanager
    async def admit(self, project_id: str) -> AsyncIterator[Database]:
        """Hold the project row lock through the check and caller's reservation write.

        Args:
            project_id: Authorized project whose active tasks count toward the limit.

        Yields:
            Open database transaction for committing the admitted task.

        Raises:
            ProjectTaskBusyError: The project has reached its active task limit.
        """
        async with self._database.transaction() as database:
            # A no-op upsert serializes concurrent submissions even for a new project.
            await database.upsert(SKILL_PROJECT_TASKS_TABLE, [{"project_id": project_id}], update_fields=["project_id"])
            project = Filter(field="project_id", value=project_id)
            now = datetime.now(UTC)
            await database.delete(
                SKILL_PROJECT_TASK_LEASES_TABLE,
                filters=[project, Filter(field="expires_at", op="le", value=now)],
            )
            fields = ["task_id", "kind", "status"]
            tasks = await database.select(
                SKILL_PROJECT_TASK_LEASES_TABLE,
                filters=[project, Filter(field="expires_at", op="gt", value=now)],
                fields=fields,
                limit=self._max_active_tasks,
            )
            remaining = self._max_active_tasks - len(tasks)
            if remaining:
                tasks.extend(
                    await database.select(
                        SKILL_TASKS_TABLE,
                        filters=[
                            project,
                            Filter(field="kind", value=EVOLVE_TASK),
                            Filter(field="status", op="in", value=[TaskStatus.QUEUED, TaskStatus.RUNNING]),
                        ],
                        fields=fields,
                        order_by=["created_at"],
                        limit=remaining,
                    )
                )
            if len(tasks) == self._max_active_tasks:
                task = tasks[0]
                raise ProjectTaskBusyError(task["task_id"], task["kind"], task["status"])
            yield database

    async def reserve(self, project_id: str, task_id: str, kind: str, status: TaskStatus) -> "TaskReservation":
        """Reserve the slot for work whose lifecycle is owned by this process.

        Args:
            project_id: Authorized project that owns the slot.
            task_id: Server-generated identity of the incoming task.
            kind: Trusted Skill task name.
            status: Initial queued or running state.

        Returns:
            Reservation that the caller must close when work ends.
        """
        async with self.admit(project_id) as database:
            await database.insert(
                SKILL_PROJECT_TASK_LEASES_TABLE,
                [
                    {
                        "project_id": project_id,
                        "task_id": task_id,
                        "kind": kind,
                        "status": status,
                        "expires_at": datetime.now(UTC) + LEASE_DURATION,
                    }
                ],
            )
        return TaskReservation(self._database, project_id, task_id)


class TaskReservation:
    """Renew a process-owned slot while its task is queued or running."""

    def __init__(self, database: Database, project_id: str, task_id: str):
        """Start renewing the already committed reservation until it is closed.

        Args:
            database: Borrowed Skill business database.
            project_id: Authorized project that owns the slot.
            task_id: Exact task that acquired the slot.
        """
        self._database = database
        self._project_id = project_id
        self._task_id = task_id
        self._runner: asyncio.Task | None = None
        self._pulse = asyncio.create_task(self._renew(), name=f"skill-admission-{task_id}")

    async def _renew(self) -> None:
        """Cancel active work if the database no longer confirms slot ownership."""
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            try:
                if await self._update({"expires_at": datetime.now(UTC) + LEASE_DURATION}):
                    continue
            except Exception:
                logger.exception("task.admission.renewal_failed", extra={"task_id": self._task_id})
            if self._runner is not None:
                self._runner.cancel()
            return

    async def _update(self, values: dict) -> int:
        """Change a live reservation only while the task still owns it."""
        return await self._database.update(
            SKILL_PROJECT_TASK_LEASES_TABLE,
            values,
            filters=[
                Filter(field="project_id", value=self._project_id),
                Filter(field="task_id", value=self._task_id),
                Filter(field="expires_at", op="gt", value=datetime.now(UTC)),
            ],
        )

    async def run(self, operation: Callable[..., Awaitable[ResultT]], *args, **kwargs) -> ResultT:
        """Verify the reservation before executing, including after a local queue wait.

        Args:
            operation: Async pipeline call or local task handler.
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The operation's result.

        Raises:
            RuntimeError: The task no longer owns a valid reservation.
        """
        self._runner = asyncio.current_task()
        try:
            if self._pulse.done() or not await self._update({"status": TaskStatus.RUNNING}):
                raise RuntimeError("Skill task admission lease was lost")
            return await operation(*args, **kwargs)
        finally:
            self._runner = None

    async def close(self) -> None:
        """Stop renewal and release the slot without clearing a newer owner."""
        # ASGI cancellation stays active during cleanup unless this scope is shielded.
        with CancelScope(shield=True):
            self._pulse.cancel()
            await asyncio.gather(self._pulse, return_exceptions=True)
            try:
                await self._database.delete(
                    SKILL_PROJECT_TASK_LEASES_TABLE,
                    filters={"project_id": self._project_id, "task_id": self._task_id},
                )
            except Exception:
                logger.exception("task.admission.release_failed", extra={"task_id": self._task_id})
