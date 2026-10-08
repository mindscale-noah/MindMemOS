"""Durable Skill task state on the service's existing database."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import and_, or_

from ....infra.database import Database
from ....infra.database.typing import Filter
from ....infra.task_manager.typing import ProgressEvent, TaskSpec, TaskStatus
from ...constants import SKILL_TASK_PROGRESS_TABLE, SKILL_TASKS_TABLE
from .typing import TaskProgressRecord, TaskRecord


class TaskDataService:
    """Perform atomic claims and progress writes without holding a long transaction."""

    def __init__(self, database: Database):
        """Borrow the already configured Skill database.

        Args:
            database: Existing service-owned database, with task tables registered.
        """
        self.database = database
        self._tasks = database.tables.get(SKILL_TASKS_TABLE).table

    def create(self, spec: TaskSpec) -> TaskRecord:
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
        self.database.insert(SKILL_TASKS_TABLE, [record.model_dump(mode="python")])
        return record

    def get(self, project_id: str, task_id: str) -> TaskRecord | None:
        """Read a task only inside its project scope.

        Args:
            project_id: Authorized project identity.
            task_id: Stable task identity.

        Returns:
            Stored task or None when absent from the project.
        """
        rows = self.database.select(SKILL_TASKS_TABLE, filters={"project_id": project_id, "task_id": task_id}, limit=1)
        return TaskRecord.model_validate(rows[0]) if rows else None

    def load(self, task_id: str) -> TaskRecord | None:
        """Load a task for the trusted worker from its opaque identity.

        Args:
            task_id: Identity received on the dedicated Skill topic.

        Returns:
            Stored task or None when the notification is invalid.
        """
        row = self.database.get(SKILL_TASKS_TABLE, task_id)
        return TaskRecord.model_validate(row) if row else None

    def due(self, *, limit: int = 100) -> list[TaskRecord]:
        """List queued tasks requiring another best-effort notification.

        Args:
            limit: Maximum number of notifications in one pass.

        Returns:
            Queued tasks whose next notification deadline has elapsed.
        """
        rows = self.database.select(
            SKILL_TASKS_TABLE,
            filters=[
                Filter(field="status", value=TaskStatus.QUEUED),
                Filter(field="next_notify_at", op="le", value=datetime.now(UTC)),
            ],
            order_by=["next_notify_at"],
            limit=limit,
        )
        return [TaskRecord.model_validate(row) for row in rows]

    def expired(self, *, limit: int = 100) -> list[TaskRecord]:
        """List running tasks whose owning worker no longer holds a live lease.

        Args:
            limit: Maximum number of tasks to inspect in one pass.

        Returns:
            Potential recovery candidates, rechecked atomically by claim.
        """
        rows = self.database.select(
            SKILL_TASKS_TABLE,
            filters=[
                Filter(field="status", value=TaskStatus.RUNNING),
                Filter(field="lease_expires_at", op="le", value=datetime.now(UTC)),
            ],
            order_by=["lease_expires_at"],
            limit=limit,
        )
        return [TaskRecord.model_validate(row) for row in rows]

    def defer_notification(self, task_id: str, *, seconds: int = 30) -> None:
        """Delay another notification; an unconfirmed send remains retryable.

        Args:
            task_id: Queued task identity.
            seconds: Delay before the next recovery scan republishes it.
        """
        self.database.update(
            SKILL_TASKS_TABLE,
            {"next_notify_at": datetime.now(UTC) + timedelta(seconds=seconds)},
            filters={"task_id": task_id, "status": TaskStatus.QUEUED},
        )

    def claim(self, task_id: str, *, lease_seconds: int = 120) -> tuple[TaskRecord, str] | None:
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
            )
            .returning(*self._tasks.c)
        )
        with self.database._connect() as connection:
            row = connection.execute(statement).mappings().first()
        return (TaskRecord.model_validate(row), token) if row else None

    def heartbeat(self, task_id: str, token: str, *, lease_seconds: int = 120) -> bool:
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
            self.database.update(
                SKILL_TASKS_TABLE,
                {"lease_expires_at": now + timedelta(seconds=lease_seconds)},
                filters={"task_id": task_id, "lease_token": token, "status": TaskStatus.RUNNING},
            )
        )

    def append_progress(self, task_id: str, token: str, event: ProgressEvent) -> str:
        """Commit progress only for the current physical attempt.

        Args:
            task_id: Stable task identity.
            token: Current worker fencing token.
            event: Validated non-sensitive progress summary.

        Returns:
            Task-scoped event ID suitable for logging correlation.

        Raises:
            RuntimeError: This worker has lost its task lease.
        """
        with self.database.transaction() as database:
            statement = (
                self._tasks.update()
                .where(self._tasks.c.task_id == task_id)
                .where(self._tasks.c.lease_token == token)
                .where(self._tasks.c.status == TaskStatus.RUNNING)
                .values(
                    progress=event.model_dump(mode="json"),
                    progress_seq=self._tasks.c.progress_seq + 1,
                    updated_at=datetime.now(UTC),
                )
                .returning(self._tasks.c.project_id, self._tasks.c.progress_seq, self._tasks.c.attempt)
            )
            with database._connect() as connection:
                row = connection.execute(statement).first()
            if row is None:
                raise RuntimeError("Task lease was lost before progress could be recorded")
            project_id, sequence, attempt = row
            record = TaskProgressRecord(
                project_id=project_id,
                task_id=task_id,
                sequence=sequence,
                attempt=attempt,
                event=event.model_dump(mode="json"),
                occurred_at=event.occurred_at,
            )
            database.insert(SKILL_TASK_PROGRESS_TABLE, [record.model_dump(mode="python")])
        return f"{task_id}:{sequence}"

    def events(self, project_id: str, task_id: str, *, after: int = 0, limit: int = 100) -> list[ProgressEvent]:
        """Read a bounded page of progress within the owning project.

        Args:
            project_id: Authorized project identity.
            task_id: Stable task identity.
            after: Last seen one-based progress sequence.
            limit: Maximum events in the returned page.

        Returns:
            Validated progress events in sequence order.
        """
        rows = self.database.select(
            SKILL_TASK_PROGRESS_TABLE,
            filters=[
                Filter(field="project_id", value=project_id),
                Filter(field="task_id", value=task_id),
                Filter(field="sequence", op="gt", value=after),
            ],
            order_by=["sequence"],
            limit=limit,
        )
        return [
            ProgressEvent.model_validate(row["event"]).model_copy(update={"sequence": row["sequence"]}) for row in rows
        ]

    def finish(self, task_id: str, token: str, *, result: dict | None = None, error_type: str | None = None) -> bool:
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
        return bool(
            self.database.update(
                SKILL_TASKS_TABLE,
                {
                    "status": TaskStatus.FAILED if error_type else TaskStatus.SUCCEEDED,
                    "result": result,
                    "error_type": error_type,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "updated_at": datetime.now(UTC),
                },
                filters={"task_id": task_id, "lease_token": token, "status": TaskStatus.RUNNING},
            )
        )
