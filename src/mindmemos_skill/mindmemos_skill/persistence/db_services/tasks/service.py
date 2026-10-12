"""Durable Skill task state on the service's existing database."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, func, or_, select

from ....infra.database import Database
from ....infra.database.typing import Filter
from ....infra.task_manager.typing import TaskSpec, TaskStatus
from ....logging.typing import TaskLog
from ...constants import SKILL_TASK_LOGS_TABLE, SKILL_TASKS_TABLE
from .typing import TaskRecord


class TaskDataService:
    """Perform atomic task state changes without holding a long transaction."""

    def __init__(self, database: Database):
        """Borrow the already configured Skill database.

        Args:
            database: Existing service-owned database, with task tables registered.
        """
        self.database = database
        self._tasks = database.tables.get(SKILL_TASKS_TABLE).table

    async def append_log(self, record: dict[str, Any]) -> None:
        """Append one sanitized log without changing task status or progress.

        Args:
            record: log_id, project_id, task_id, attempt, occurred_at and sanitized
                payload supplied by the task logging scope.
        """
        async with self.database.transaction() as database:
            # Serialize writers for this task so pagination never skips a late commit.
            async with database._connect() as connection:
                owner = (
                    await connection.execute(
                        select(self._tasks.c.task_id)
                        .where(self._tasks.c.task_id == record["task_id"])
                        .where(self._tasks.c.project_id == record["project_id"])
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if owner is None:
                    raise ValueError("Task log does not belong to an existing project-owned task")
            await database.insert(SKILL_TASK_LOGS_TABLE, [record])

    async def logs(self, project_id: str, task_id: str, *, after: int = 0, limit: int = 100) -> list[TaskLog]:
        """Read historical logs within the authorized project.

        Args:
            project_id: Authorized owning project.
            task_id: Durable task identity.
            after: Last observed committed log sequence, or zero for the first page.
            limit: Maximum number of records, ordered by their committed sequence.

        Returns:
            Rows containing identity, attempt, timestamp and sanitized payload.
        """
        rows = await self.database.select(
            SKILL_TASK_LOGS_TABLE,
            filters=[
                Filter(field="project_id", value=project_id),
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

    async def get(self, project_id: str, task_id: str) -> TaskRecord | None:
        """Read a task only inside its project scope.

        Args:
            project_id: Authorized project identity.
            task_id: Stable task identity.

        Returns:
            Stored task or None when absent from the project.
        """
        rows = await self.database.select(
            SKILL_TASKS_TABLE, filters={"project_id": project_id, "task_id": task_id}, limit=1
        )
        return TaskRecord.model_validate(rows[0]) if rows else None

    async def load(self, task_id: str) -> TaskRecord | None:
        """Load a task for the trusted worker from its opaque identity.

        Args:
            task_id: Identity received on the dedicated Skill topic.

        Returns:
            Stored task or None when the notification is invalid.
        """
        row = await self.database.get(SKILL_TASKS_TABLE, task_id)
        return TaskRecord.model_validate(row) if row else None

    async def due(self, *, limit: int = 100) -> list[TaskRecord]:
        """List queued tasks requiring another best-effort notification.

        Args:
            limit: Maximum number of notifications in one pass.

        Returns:
            Queued tasks whose next notification deadline has elapsed.
        """
        rows = await self.database.select(
            SKILL_TASKS_TABLE,
            filters=[
                Filter(field="status", value=TaskStatus.QUEUED),
                Filter(field="next_notify_at", op="le", value=datetime.now(UTC)),
            ],
            order_by=["next_notify_at"],
            limit=limit,
        )
        return [TaskRecord.model_validate(row) for row in rows]

    async def expired(self, *, limit: int = 100) -> list[TaskRecord]:
        """List running tasks whose owning worker no longer holds a live lease.

        Args:
            limit: Maximum number of tasks to inspect in one pass.

        Returns:
            Potential recovery candidates, rechecked atomically by claim.
        """
        rows = await self.database.select(
            SKILL_TASKS_TABLE,
            filters=[
                Filter(field="status", value=TaskStatus.RUNNING),
                Filter(field="lease_expires_at", op="le", value=datetime.now(UTC)),
            ],
            order_by=["lease_expires_at"],
            limit=limit,
        )
        return [TaskRecord.model_validate(row) for row in rows]

    async def counts(self) -> dict[str, dict[str, int]]:
        """Read exact database task counts per project for queue diagnostics.

        Returns:
            Project IDs mapped to status names and the current number of tasks.
        """
        statement = select(self._tasks.c.project_id, self._tasks.c.status, func.count()).group_by(
            self._tasks.c.project_id, self._tasks.c.status
        )
        async with self.database._connect() as connection:
            rows = (await connection.execute(statement)).all()
        result: dict[str, dict[str, int]] = {}
        for project_id, status, count in rows:
            result.setdefault(project_id, {})[status] = count
        return result

    async def defer_notification(self, task_id: str, *, seconds: int = 30) -> None:
        """Delay another notification; an unconfirmed send remains retryable.

        Args:
            task_id: Queued task identity.
            seconds: Delay before the next recovery scan republishes it.
        """
        (
            await self.database.update(
                SKILL_TASKS_TABLE,
                {"next_notify_at": datetime.now(UTC) + timedelta(seconds=seconds)},
                filters={"task_id": task_id, "status": TaskStatus.QUEUED},
            )
        )

    async def claim(self, task_id: str, *, lease_seconds: int = 120) -> tuple[TaskRecord, str] | None:
        """Atomically claim a queued or expired task for one physical attempt.

        Args:
            task_id: Opaque task identity from Kafka or a recovery scan.
            lease_seconds: Initial heartbeat lease duration.

        Returns:
            Claimed task and fencing token, or None when already owned or terminal.
        """
        now = datetime.now(UTC)
        token = uuid4().hex
        statement = (
            self._tasks.update()
            .where(self._tasks.c.task_id == task_id)
            .where(
                or_(
                    self._tasks.c.status == TaskStatus.QUEUED,
                    and_(
                        self._tasks.c.status == TaskStatus.RUNNING,
                        self._tasks.c.lease_expires_at <= now,
                    ),
                )
            )
            .values(
                status=TaskStatus.RUNNING,
                attempt=self._tasks.c.attempt + 1,
                lease_token=token,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
                updated_at=now,
                started_at=now,
                finished_at=None,
            )
            .returning(*self._tasks.c)
        )
        async with self.database._connect() as connection:
            row = (await connection.execute(statement)).mappings().first()
        return (TaskRecord.model_validate(row), token) if row else None

    async def heartbeat(self, task_id: str, token: str, *, lease_seconds: int = 120) -> bool:
        """Renew only the still-owned running attempt.

        Args:
            task_id: Stable task identity.
            token: Current worker fencing token.
            lease_seconds: Lease length measured from this heartbeat.

        Returns:
            Whether the worker still owns the task.
        """
        now = datetime.now(UTC)
        return bool(
            (
                await self.database.update(
                    SKILL_TASKS_TABLE,
                    {"lease_expires_at": now + timedelta(seconds=lease_seconds)},
                    filters={"task_id": task_id, "lease_token": token, "status": TaskStatus.RUNNING},
                )
            )
        )

    async def finish(
        self, task_id: str, token: str, *, result: dict | None = None, error_type: str | None = None
    ) -> bool:
        """Commit one terminal result only from the current fenced attempt.

        Args:
            task_id: Stable task identity.
            token: Current worker fencing token.
            result: Successful JSON-compatible result.
            error_type: Failed exception type; mutually exclusive with result.

        Returns:
            Whether this worker committed the terminal state.
        """
        if (result is None) == (error_type is None):
            raise ValueError("Supply exactly one of result or error_type")
        now = datetime.now(UTC)
        return bool(
            (
                await self.database.update(
                    SKILL_TASKS_TABLE,
                    {
                        "finished_at": now,
                        "status": TaskStatus.FAILED if error_type else TaskStatus.SUCCEEDED,
                        "result": result,
                        "error_type": error_type,
                        "lease_token": None,
                        "lease_expires_at": None,
                        "updated_at": now,
                    },
                    filters={"task_id": task_id, "lease_token": token, "status": TaskStatus.RUNNING},
                )
            )
        )
