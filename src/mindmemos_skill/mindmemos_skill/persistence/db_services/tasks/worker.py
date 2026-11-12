"""Trusted cross-project task scheduling and fenced worker state changes."""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select

from mindmemos_skill.common.ids import generate_id

from ....infra.database import Database, Filter, Scope
from ....infra.task_manager.typing import EVOLVE_TASK, TaskSpec, TaskStatus
from ...constants import SKILL_TASKS_TABLE
from .admission import ProjectTaskAdmission
from .service import TaskDataService
from .typing import TaskRecord


class TaskWorkerService:
    """Expose cross-project scheduling only to a trusted worker or maintenance owner."""

    def __init__(self, database: Database, *, admission: ProjectTaskAdmission | None = None):
        """Borrow an unscoped resource for trusted cross-project scheduling.

        Args:
            database: Service-owned database, not a request or repository handle.
            admission: Shared configured project task limiter.

        Raises:
            TypeError: A scope-bound handle was supplied.
        """
        if not isinstance(database, Database):
            raise TypeError("Task workers require a trusted unscoped database")
        self.database = database
        self._tasks = database.tables.get(SKILL_TASKS_TABLE).table
        self._admission = admission if admission is not None else ProjectTaskAdmission(database)

    def scoped(self, scope: Scope) -> TaskDataService:
        """Create a project repository for trusted task inspection."""
        from ....infra.database.scoped import ScopedDatabase
        from ...store import validate_skill_scope

        validate_skill_scope(scope)
        return TaskDataService(ScopedDatabase(self.database, scope))

    async def create(self, spec: TaskSpec) -> TaskRecord:
        """Admit a trusted task through its captured request scope."""
        scope = Scope(fields={"project_id": spec.project_id})
        if spec.kind == EVOLVE_TASK:
            async with self._admission.admit(spec.project_id) as database:
                with database.scoped(scope) as scoped:
                    return await TaskDataService(scoped).create(spec)
        return await self.scoped(scope).create(spec)

    async def append_log(self, record: dict[str, Any]) -> None:
        """Append a worker event through the task logging scope."""
        await self.scoped(Scope(fields={"project_id": record["project_id"]})).append_log(record)

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
        token = generate_id()
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
