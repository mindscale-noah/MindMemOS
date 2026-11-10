"""Store local rollout attempts and Harness-reported Agent executions."""

import hashlib
import json
from datetime import UTC, datetime, time, timedelta
from typing import overload
from zoneinfo import ZoneInfo

from mindmemos_skill.common.ids import generate_id

from ....common.time_utils import utcnow
from ....errors import SkillConflictError, SkillNotFoundError, SkillVersionNotFoundError
from ....infra.database import Database, Filter
from ....infra.database.scoped import ScopedDatabase
from ...access import _project_id_from_database
from ...constants import (
    LEGACY_SKILL_SOURCE,
    TRAJECTORIES_TABLE,
    TRAJECTORY_CONSUMPTIONS_TABLE,
    TRAJECTORY_VERSIONS_TABLE,
)
from ...typing import TrajectoryCounts
from ..skills.service import SkillDataService
from .typing import (
    TrajectoryConsumption,
    TrajectoryDayCounts,
    TrajectoryRecord,
    TrajectoryStatistics,
    TrajectorySummary,
    TrajectoryVersionIndex,
    TrajectoryWrite,
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

    def __init__(self, database: Database | ScopedDatabase, *, skills: SkillDataService | None = None):
        """Borrow trajectory storage and an optional granted Skill reader.

        Args:
            database: Active repository or explicit maintenance database handle.
            skills: Session-owned read-only dependency for exact version visibility.
        """
        self.database = database
        self.skills = skills

    async def add(self, record: TrajectoryWrite) -> None:
        """Insert a rollout attempt without overwriting existing evidence.

        Args:
            record: Trajectory input; the active scope supplies its project.
        """
        if not isinstance(record, TrajectoryRecord):
            record = TrajectoryRecord(
                project_id=_project_id_from_database(self.database), **record.model_dump(mode="python")
            )
        async with self.database.transaction() as transaction:
            await transaction.insert(TRAJECTORIES_TABLE, [record.model_dump(mode="python")])
            await self._index_versions(transaction, record)

    async def replicate_legacy(self, record: TrajectoryRecord) -> int:
        """Append resolved legacy bindings inside the caller's replication transaction."""
        key = {"project_id": record.project_id, "trajectory_id": record.trajectory_id}
        inserted = await self.database.upsert(
            TRAJECTORIES_TABLE, [record.model_dump()], conflict_fields=["project_id", "trajectory_id"]
        )
        if inserted:
            await self._index_versions(self.database, record)
            return 1
        locked = await self.database.update(
            TRAJECTORIES_TABLE,
            {"source": "reported"},
            filters=key | {
                "source": "reported",
                "rollout_id": f"{LEGACY_SKILL_SOURCE}:{record.trajectory_id}",
                "trajectory_hash": record.trajectory_hash,
            },
        )
        if not locked:
            raise ValueError("Legacy trajectory conflicts with an existing attempt")
        existing = await self.get(record.trajectory_id)
        assert existing is not None  # The successful update holds the row until commit.
        present = existing.bound_versions
        additions = [
            binding
            for binding in record.skill_bindings
            if isinstance(binding, dict) and binding["version_id"] not in present
        ]
        bindings = [*existing.skill_bindings, *additions]
        original = existing.metadata.get(LEGACY_SKILL_SOURCE)
        incoming = record.metadata[LEGACY_SKILL_SOURCE]
        assert isinstance(incoming, dict)
        provenance = dict(original) if isinstance(original, dict) else {}
        if provenance.get("memory_completed_at") and not incoming.get("memory_completed_at"):
            incoming = {k: v for k, v in incoming.items() if k not in {"memory_completed_at", "add_status"}}
        metadata = {**existing.metadata, LEGACY_SKILL_SOURCE: provenance | incoming}
        if additions or metadata != existing.metadata or record.reward_score != existing.reward_score:
            await self.database.update(
                TRAJECTORIES_TABLE,
                {"skill_bindings": bindings, "metadata": metadata, "reward_score": record.reward_score},
                filters=key,
            )
        if additions:
            merged = existing.model_copy(update={"skill_bindings": bindings})
            await self._index_versions(self.database, merged, existing_ids=set(present))
        return 0

    @staticmethod
    async def _index_versions(
        transaction, record: TrajectoryRecord, *, existing_ids: frozenset[str] | set[str] = frozenset()
    ) -> None:
        """Index every exact version in the same transaction as the shared attempt."""
        seen_families = set()
        rows = []
        for version_id, skill_id in record.bound_versions.items():
            entry = TrajectoryVersionIndex(
                project_id=record.project_id,
                trajectory_id=record.trajectory_id,
                skill_version_id=version_id,
                skill_id=skill_id,
                first_for_family=skill_id not in seen_families,
                task_id=record.task_id,
                source=record.source,
                user_id=record.user_id,
                agent_id=record.agent_id,
                status=record.status,
                started_at=record.started_at,
                finished_at=record.finished_at,
                created_at=record.created_at,
                received_at=record.received_at,
            )
            seen_families.add(skill_id)
            if version_id not in existing_ids:
                rows.append(entry.model_dump(mode="python"))
        if rows:
            await transaction.insert(TRAJECTORY_VERSIONS_TABLE, rows)

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
            total = await transaction.count(TRAJECTORY_VERSIONS_TABLE, filters=filters)
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
                TRAJECTORY_VERSIONS_TABLE,
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
            TRAJECTORY_VERSIONS_TABLE,
            time_field="created_at",
            filters=[
                Filter(field="skill_version_id", value=skill_version_id),
                Filter(field="created_at", op="ge", value=start),
                Filter(field="created_at", op="lt", value=end),
            ],
            matching_table=TRAJECTORY_CONSUMPTIONS_TABLE,
            join_fields=["project_id", "trajectory_id", "skill_version_id"],
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
            (await self.database.select(TRAJECTORY_VERSIONS_TABLE, filters=filters, fields=["trajectory_id"], limit=1))
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
        return set((await self.database.select_distinct(TRAJECTORY_VERSIONS_TABLE, "skill_id", filters=filters)))

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
        else:
            filters.append(Filter(field="first_for_family", value=True))
        if status is not None:
            filters.append(Filter(field="status", value=status))
        if received_at_from is not None:
            filters.append(Filter(field="received_at", op="ge", value=received_at_from))
        if received_at_to is not None:
            filters.append(Filter(field="received_at", op="le", value=received_at_to))
        selected_fields = list(DEFAULT_REPORTED_FIELDS) if fields is None else fields
        payload_fields = [name for name in selected_fields if name not in TrajectoryVersionIndex.model_fields]
        index_fields = [name for name in selected_fields if name in TrajectoryVersionIndex.model_fields]
        if payload_fields and "trajectory_id" not in index_fields:
            index_fields.append("trajectory_id")
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
                        TRAJECTORY_VERSIONS_TABLE, filters=ties, fields=index_fields, order_by=order, limit=limit
                    )
                )
            )
            filters.append(Filter(field="received_at", op="lt", value=after_received_at))
        if limit is None or len(rows) < limit:
            rows.extend(
                (
                    await self.database.select(
                        TRAJECTORY_VERSIONS_TABLE,
                        filters=filters,
                        fields=index_fields,
                        order_by=order,
                        limit=None if limit is None else limit - len(rows),
                    )
                )
            )
        if fields is not None:
            if not payload_fields or not rows:
                return rows
            payload = await self._load_payload([row["trajectory_id"] for row in rows], fields=payload_fields)
            combined = [row | payload[row["trajectory_id"]] for row in rows if row["trajectory_id"] in payload]
            return [{name: row[name] for name in fields} for row in combined]
        return [TrajectorySummary.model_validate(row) for row in rows]

    async def list_for_version(
        self, skill_version_id: str, *, filters: list[Filter], limit: int, offset: int
    ) -> list[TrajectoryRecord]:
        """Read a page of execution payloads associated with an exact version.

        Args:
            skill_version_id: Version identity in the active project.
            filters: Additional conditions on the indexed attempt fields.
            limit: Maximum matching attempts in the page.
            offset: Number of matching attempts to skip.

        Returns:
            Records ordered by creation time and ID, newest first.
        """
        indexed = await self.database.select(
            TRAJECTORY_VERSIONS_TABLE,
            filters=[Filter(field="skill_version_id", value=skill_version_id), *filters],
            fields=["trajectory_id"],
            order_by=["-created_at", "-trajectory_id"],
            limit=limit,
            offset=offset,
        )
        ids = [row["trajectory_id"] for row in indexed]
        payload = await self._load_payload(ids)
        return [TrajectoryRecord.model_validate(payload[identity]) for identity in ids if identity in payload]

    async def _load_payload(self, trajectory_ids: list[str], *, fields: list[str] | None = None) -> dict[str, dict]:
        """Fetch the payload for one bounded page of shared attempt IDs."""
        if not trajectory_ids:
            return {}
        rows = await self.database.select(
            TRAJECTORIES_TABLE,
            filters=[Filter(field="trajectory_id", op="in", value=trajectory_ids)],
            fields=None if fields is None else ["trajectory_id", *fields],
            limit=None,
        )
        return {row["trajectory_id"]: row for row in rows}

    async def report(
        self,
        *,
        skill_version_ids: list[str],
        user_id: str | None,
        agent_id: str | None,
        messages: list[dict],
        session_id: str | None,
        status: str,
        started_at: datetime | None,
        finished_at: datetime | None,
        event_id: str | None = None,
    ) -> TrajectoryRecord:
        """Convert a sparse Harness report into the shared trajectory record.

        Args:
            skill_version_ids: Exact loaded versions, or empty when no version is identified.
            user_id: Business user supplied by the Harness, or None when omitted.
            agent_id: Agent supplied by the Harness, or None when omitted.
            messages: Ordered interaction messages.
            session_id: Optional logical Agent session.
            status: Caller-declared execution outcome.
            started_at: Actual start time when supplied.
            finished_at: Actual finish time when supplied.
            event_id: Stable project-unique identity for an idempotent report.

        Returns:
            Shared physical attempt, including the original ID on replay.

        Raises:
            SkillVersionNotFoundError: A supplied exact version is outside the project or unavailable.
            SkillConflictError: The event ID was already used with different execution inputs.
        """
        project_id = _project_id_from_database(self.database)
        reader = self.skills or SkillDataService(self.database)
        try:
            resolved = [await reader.get(version_id) for version_id in dict.fromkeys(skill_version_ids)]
            versions = list({version.version_id: version for version in resolved}.values())
        except SkillNotFoundError as exc:
            raise SkillVersionNotFoundError("Skill version not found in this project") from exc

        received_at = utcnow()
        trajectory_id = generate_id(hyphenated=True)
        system_prompt = next(
            (_message_text(message["content"]) for message in messages if message["role"] == "system"), None
        )
        user_prompt = next((_message_text(message["content"]) for message in messages if message["role"] == "user"), "")
        encoded = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        record = TrajectoryRecord(
            project_id=project_id,
            trajectory_id=trajectory_id,
            trajectory_hash=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            source="reported",
            task_id=generate_id(hyphenated=True),
            rollout_id=f"reported:{event_id}" if event_id else trajectory_id,
            task_instruction=user_prompt,
            task_system_prompt=system_prompt,
            env_ref="unknown",
            skill_bindings=[
                {
                    "name": version.name,
                    "skill_id": version.skill_id,
                    "version_id": version.version_id,
                    "content_hash": version.content_hash,
                }
                for version in versions
            ],
            agent_type="harness",
            status=status,
            trajectory=messages,
            started_at=started_at,
            finished_at=finished_at,
            n_turn=sum(message["role"] == "assistant" for message in messages),
            created_at=received_at,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            received_at=received_at,
        )
        async with self.database.transaction() as transaction:
            row = record.model_dump(mode="python")
            if event_id:
                inserted = await transaction.upsert(
                    TRAJECTORIES_TABLE, [row], conflict_fields=["project_id", "rollout_id", "attempt_no"]
                )
                if not inserted:
                    existing = await transaction.select(
                        TRAJECTORIES_TABLE, filters={"rollout_id": record.rollout_id, "attempt_no": 0}, limit=1
                    )
                    generated = {"trajectory_id", "task_id", "created_at", "received_at"}
                    if not existing or any(
                        existing[0][key] != value for key, value in row.items() if key not in generated
                    ):
                        raise SkillConflictError("Trajectory event_id is already bound to a different report")
                    return TrajectoryRecord.model_validate(existing[0])
            else:
                await transaction.insert(TRAJECTORIES_TABLE, [row])
            await self._index_versions(transaction, record)
        return record


def _message_text(content: object) -> str:
    """Render one prompt content value without losing structured JSON."""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, allow_nan=False)
