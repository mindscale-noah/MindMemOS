"""Store local rollout attempts and Harness-reported Agent executions."""

import hashlib
import json
from datetime import UTC, datetime, time, timedelta
from typing import overload
from uuid import uuid4
from zoneinfo import ZoneInfo

from ....common.time_utils import utcnow
from ....errors import SkillNotFoundError, SkillVersionNotFoundError
from ....infra.database import Database, Filter
from ...constants import TRAJECTORIES_TABLE, TRAJECTORY_CONSUMPTIONS_TABLE
from ...typing import TrajectoryCounts
from ..skills.service import SkillDataService
from .typing import (
    TrajectoryConsumption,
    TrajectoryDayCounts,
    TrajectoryRecord,
    TrajectoryStatistics,
    TrajectorySummary,
)

DEFAULT_REPORTED_FIELDS = (
    "trajectory_id",
    "task_id",
    "skill_id",
    "skill_version_id",
    "user_id",
    "agent_id",
    "status",
    "started_at",
    "finished_at",
    "received_at",
)


class TrajectoryDataService:
    """Read and store rollout attempts on a borrowed database.

    Args:
        database: Infra database with trajectory tables registered.
    """

    def __init__(self, database: Database):
        self.database = database

    async def add(self, record: TrajectoryRecord) -> None:
        """Insert a rollout attempt without overwriting existing evidence.

        Args:
            record: Complete trajectory to store.
        """
        (await self.database.insert(TRAJECTORIES_TABLE, [record.model_dump(mode="python")]))

    async def get(self, trajectory_id: str) -> TrajectoryRecord | None:
        """Read a stored attempt.

        Args:
            trajectory_id: Local physical attempt identity.

        Returns:
            The stored attempt, or None.
        """
        row = await self.database.get(TRAJECTORIES_TABLE, trajectory_id)
        return TrajectoryRecord.model_validate(row) if row is not None else None

    async def counts(self, skill_version_id: str) -> TrajectoryCounts:
        """Count stored trajectories and their distinct successful evolution uses.

        Args:
            skill_version_id: Exact Skill version in the active project scope.

        Returns:
            Total, consumed and unconsumed counts across local and reported sources.
        """
        filters = {"skill_version_id": skill_version_id}
        async with self.database.transaction() as transaction:
            total = await transaction.count(TRAJECTORIES_TABLE, filters=filters)
            consumed = await transaction.count(TRAJECTORY_CONSUMPTIONS_TABLE, filters=filters)
        return TrajectoryCounts(total=total, consumed=consumed, unconsumed=total - consumed)

    async def mark_consumed(self, skill_version_id: str, trajectory_ids: list[str], run_id: str) -> None:
        """Record first successful use without counting repeated evolution twice.

        Args:
            skill_version_id: Exact input version for the completed evolution call.
            trajectory_ids: Stored attempts actually processed by the evolver.
            run_id: Evolution run that produced a persisted candidate.

        Raises:
            ValueError: Any trajectory is absent or belongs to another version or project.
        """
        if not trajectory_ids:
            return
        identities = set(trajectory_ids)
        async with self.database.transaction() as transaction:
            rows = await transaction.select(
                TRAJECTORIES_TABLE,
                filters=[
                    Filter(field="skill_version_id", value=skill_version_id),
                    Filter(field="trajectory_id", op="in", value=list(identities)),
                ],
                fields=["project_id", "trajectory_id"],
                limit=None,
            )
            if {row["trajectory_id"] for row in rows} != identities:
                raise ValueError("Consumed trajectories must belong to the exact Skill version and project")
            (
                await transaction.upsert(
                    TRAJECTORY_CONSUMPTIONS_TABLE,
                    [
                        TrajectoryConsumption(
                            project_id=row["project_id"],
                            trajectory_id=row["trajectory_id"],
                            skill_version_id=skill_version_id,
                            run_id=run_id,
                        ).model_dump(mode="python")
                        for row in rows
                    ],
                )
            )

    async def statistics(self, skill_version_id: str, *, now: datetime | None = None) -> TrajectoryStatistics:
        """Read indexed all-time counts and a fixed thirty-day SQL aggregation.

        Args:
            skill_version_id: Exact version in the active project scope.
            now: Aware query time; None uses the current UTC time.

        Returns:
            All-time counts and thirty calendar buckets.
        """
        timezone = ZoneInfo("Asia/Shanghai")
        today = (now or utcnow()).astimezone(timezone).date()
        first_day = today - timedelta(days=29)
        start = datetime.combine(first_day, time.min, timezone).astimezone(UTC)
        end = datetime.combine(today + timedelta(days=1), time.min, timezone).astimezone(UTC)
        filters = {"skill_version_id": skill_version_id}
        totals = await self.counts(skill_version_id)
        groups = await self.database.count_by_day(
            TRAJECTORIES_TABLE,
            time_field="created_at",
            filters=[
                Filter(field="skill_version_id", value=skill_version_id),
                Filter(field="created_at", op="ge", value=start),
                Filter(field="created_at", op="lt", value=end),
            ],
            matching_table=TRAJECTORY_CONSUMPTIONS_TABLE,
            join_fields=["project_id", "trajectory_id"],
            matching_filters=filters,
            utc_offset_minutes=480,
        )
        by_day = {group.day: group for group in groups}
        days = []
        for offset in range(30):
            day = first_day + timedelta(days=offset)
            group = by_day.get(day)
            total = group.total if group else 0
            consumed = group.matched if group else 0
            days.append(TrajectoryDayCounts(day=day, total=total, consumed=consumed, unconsumed=total - consumed))
        return TrajectoryStatistics(skill_version_id=skill_version_id, totals=totals, days=days)

    async def has_reported(
        self, skill_id: str, *, user_id: str | None, agent_id: str | None, snapshot_at: datetime
    ) -> bool:
        """Check whether one family has a report matching both actor filters.

        Args:
            skill_id: Exact family identity in the active project.
            user_id: Optional exact business user identity.
            agent_id: Optional exact Agent identity.
            snapshot_at: First-page receipt cutoff for a stable Skill list.

        Returns:
            Whether at least one same report satisfies the supplied actors.
        """
        filters = [
            Filter(field="source", value="reported"),
            Filter(field="skill_id", value=skill_id),
            Filter(field="received_at", op="le", value=snapshot_at),
        ]
        if user_id is not None:
            filters.append(Filter(field="user_id", value=user_id))
        if agent_id is not None:
            filters.append(Filter(field="agent_id", value=agent_id))
        return bool(
            (await self.database.select(TRAJECTORIES_TABLE, filters=filters, fields=["trajectory_id"], limit=1))
        )

    async def reported_skill_ids(
        self, skill_ids: list[str], *, user_id: str | None, agent_id: str | None, snapshot_at: datetime
    ) -> set[str]:
        """Find candidate families with a matching report in the active project.

        Args:
            skill_ids: Candidate family identities from one Skill list batch.
            user_id: Optional exact business user identity.
            agent_id: Optional exact Agent identity on the same report.
            snapshot_at: First-page receipt cutoff for the stable Skill list.

        Returns:
            Candidate family IDs with at least one matching report.
        """
        if not skill_ids:
            return set()
        filters = [
            Filter(field="source", value="reported"),
            Filter(field="skill_id", op="in", value=skill_ids),
            Filter(field="received_at", op="le", value=snapshot_at),
        ]
        if user_id is not None:
            filters.append(Filter(field="user_id", value=user_id))
        if agent_id is not None:
            filters.append(Filter(field="agent_id", value=agent_id))
        return set((await self.database.select_distinct(TRAJECTORIES_TABLE, "skill_id", filters=filters)))

    @overload
    def list_reported(
        self,
        skill_id: str,
        *,
        skill_version_id: str | None,
        status: str | None,
        received_at_from: datetime | None,
        received_at_to: datetime | None,
        snapshot_at: datetime,
        after_received_at: datetime | None,
        after_trajectory_id: str | None,
        limit: int | None,
        fields: None = None,
    ) -> list[TrajectorySummary]: ...

    @overload
    def list_reported(
        self,
        skill_id: str,
        *,
        skill_version_id: str | None,
        status: str | None,
        received_at_from: datetime | None,
        received_at_to: datetime | None,
        snapshot_at: datetime,
        after_received_at: datetime | None,
        after_trajectory_id: str | None,
        limit: int | None,
        fields: list[str],
    ) -> list[dict]: ...

    async def list_reported(
        self,
        skill_id: str,
        *,
        skill_version_id: str | None,
        status: str | None,
        received_at_from: datetime | None,
        received_at_to: datetime | None,
        snapshot_at: datetime,
        after_received_at: datetime | None,
        after_trajectory_id: str | None,
        limit: int | None,
        fields: list[str] | None = None,
    ) -> list[TrajectorySummary] | list[dict]:
        """Read a stable receipt-ordered page of reported attempts.

        Args:
            skill_id: Exact family identity within the active project.
            skill_version_id: Optional exact loaded version.
            status: Optional caller-declared outcome.
            received_at_from: Inclusive lower receipt bound.
            received_at_to: Inclusive upper receipt bound.
            snapshot_at: First-page cutoff excluding later reports.
            after_received_at: Last receipt time from the preceding page.
            after_trajectory_id: Last ID breaking equal receipt times.
            limit: Number of rows to fetch, including any lookahead row.
            fields: Optional database columns; None selects the default report summary.

        Returns:
            Validated summaries by default, or projected rows for custom fields.
        """
        filters = [Filter(field="source", value="reported"), Filter(field="skill_id", value=skill_id)]
        filters.append(Filter(field="received_at", op="le", value=snapshot_at))
        if skill_version_id is not None:
            filters.append(Filter(field="skill_version_id", value=skill_version_id))
        if status is not None:
            filters.append(Filter(field="status", value=status))
        if received_at_from is not None:
            filters.append(Filter(field="received_at", op="ge", value=received_at_from))
        if received_at_to is not None:
            filters.append(Filter(field="received_at", op="le", value=received_at_to))
        selected_fields = list(DEFAULT_REPORTED_FIELDS) if fields is None else fields
        order = ["-received_at", "-trajectory_id"]
        rows = []
        if after_received_at is not None:
            if after_trajectory_id is None:
                raise ValueError("Trajectory cursor requires both receipt time and ID")
            ties = filters + [
                Filter(field="received_at", value=after_received_at),
                Filter(field="trajectory_id", op="lt", value=after_trajectory_id),
            ]
            rows.extend(
                (
                    await self.database.select(
                        TRAJECTORIES_TABLE, filters=ties, fields=selected_fields, order_by=order, limit=limit
                    )
                )
            )
            filters.append(Filter(field="received_at", op="lt", value=after_received_at))
        if limit is None or len(rows) < limit:
            rows.extend(
                (
                    await self.database.select(
                        TRAJECTORIES_TABLE,
                        filters=filters,
                        fields=selected_fields,
                        order_by=order,
                        limit=None if limit is None else limit - len(rows),
                    )
                )
            )
        if fields is not None:
            return rows
        return [TrajectorySummary.model_validate(row) for row in rows]

    async def report(
        self,
        *,
        project_id: str,
        skill_version_id: str,
        user_id: str | None,
        agent_id: str | None,
        messages: list[dict],
        session_id: str | None,
        status: str,
        started_at: datetime | None,
        finished_at: datetime | None,
    ) -> TrajectoryRecord:
        """Convert a sparse Harness report into the shared trajectory record.

        Args:
            project_id: Authorized project owning the report.
            skill_version_id: Exact version loaded during execution.
            user_id: Business user supplied by the Harness, or None when omitted.
            agent_id: Agent supplied by the Harness, or None when omitted.
            messages: Ordered interaction messages.
            session_id: Optional logical Agent session.
            status: Caller-declared execution outcome.
            started_at: Actual start time when supplied.
            finished_at: Actual finish time when supplied.

        Returns:
            Persisted physical attempt with generated task and trajectory IDs.
        """
        async with self.database.transaction() as transaction:
            try:
                version = await SkillDataService(transaction).version_by_id(project_id, skill_version_id)
            except SkillNotFoundError as exc:
                raise SkillVersionNotFoundError("Skill version not found in this project") from exc
            received_at = utcnow()
            trajectory_id = str(uuid4())
            system_prompt = next(
                (_message_text(message["content"]) for message in messages if message["role"] == "system"), None
            )
            user_prompt = next(
                (_message_text(message["content"]) for message in messages if message["role"] == "user"), ""
            )
            encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            record = TrajectoryRecord(
                project_id=project_id,
                trajectory_id=trajectory_id,
                trajectory_hash=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                source="reported",
                task_id=str(uuid4()),
                rollout_id=trajectory_id,
                task_instruction=user_prompt,
                task_system_prompt=system_prompt,
                env_ref="unknown",
                skill_bindings=[
                    {
                        "skill_id": version.skill_id,
                        "version_id": version.version_id,
                        "content_hash": version.content_hash,
                    }
                ],
                agent_type="harness",
                status=status,
                trajectory=messages,
                started_at=started_at,
                finished_at=finished_at,
                n_turn=sum(message["role"] == "assistant" for message in messages),
                created_at=received_at,
                skill_id=version.skill_id,
                skill_version_id=version.version_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                received_at=received_at,
            )
            (await transaction.insert(TRAJECTORIES_TABLE, [record.model_dump(mode="python")]))
        return record


def _message_text(content: object) -> str:
    """Render one prompt content value without losing structured JSON."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, allow_nan=False)
