"""Durable Skill task state on the service's existing database."""

from datetime import UTC, datetime
from typing import Any

from ....infra.database import Database
from ....infra.database.scoped import ScopedDatabase
from ....infra.database.typing import Filter
from ....infra.task_manager.typing import TaskSpec, TaskStatus
from ....logging.typing import TaskLog
from ...constants import SKILL_TASK_LOGS_TABLE, SKILL_TASKS_TABLE
from .typing import TaskRecord


class TaskDataService:
    """Create project tasks, read their state, and append task-owned logs."""

    def __init__(self, database: Database | ScopedDatabase):
        """Borrow the already configured Skill database.

        Args:
            database: Existing service-owned database, with task tables registered.
        """
        self.database = database

    async def append_log(self, record: dict[str, Any]) -> None:
        """Append one sanitized log without changing task status or progress.

        Args:
            record: log_id, project_id, task_id, attempt, occurred_at and sanitized
                payload supplied by the task logging scope.
        """
        async with self.database.transaction() as database:
            # A no-op update locks the task without changing progress or timestamps.
            changed = await database.update(
                SKILL_TASKS_TABLE,
                {"task_id": record["task_id"]},
                filters={"task_id": record["task_id"], "project_id": record["project_id"]},
            )
            if changed != 1:
                raise ValueError("Task log does not belong to an existing project-owned task")
            await database.insert(SKILL_TASK_LOGS_TABLE, [record])

    async def logs(self, task_id: str, *, after: int = 0, limit: int = 100) -> list[TaskLog]:
        """Read historical logs within the authorized project.

        Args:
            task_id: Durable task identity.
            after: Last observed committed log sequence, or zero for the first page.
            limit: Maximum number of records, ordered by their committed sequence.

        Returns:
            Rows containing identity, attempt, timestamp and sanitized payload.
        """
        rows = await self.database.select(
            SKILL_TASK_LOGS_TABLE,
            filters=[
                Filter(field="task_id", value=task_id),
                Filter(field="sequence", op="gt", value=after),
            ],
            order_by=["sequence"],
            limit=limit,
        )
        return [TaskLog.model_validate(row) for row in rows]

    async def create(self, spec: TaskSpec) -> TaskRecord:
        """Persist an authorized request before any Kafka notification.

        Args:
            spec: Immutable request with a globally unique task ID.

        Returns:
            Newly committed queued task.
        """
        now = datetime.now(UTC)
        record = TaskRecord(
            **spec.model_dump(),
            status=TaskStatus.QUEUED,
            next_notify_at=now,
            created_at=now,
            updated_at=now,
        )
        (await self.database.insert(SKILL_TASKS_TABLE, [record.model_dump(mode="python")]))
        return record

    async def get(self, task_id: str) -> TaskRecord | None:
        """Read a task only inside its project scope.

        Args:
            task_id: Stable task identity.

        Returns:
            Stored task or None when absent from the project.
        """
        rows = await self.database.select(SKILL_TASKS_TABLE, filters={"task_id": task_id}, limit=1)
        return TaskRecord.model_validate(rows[0]) if rows else None
