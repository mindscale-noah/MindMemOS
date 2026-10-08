"""Bounded, reusable collection of stored Skill execution evidence."""

from datetime import datetime
from typing import Literal

from ..infra.database import Filter
from ..persistence.constants import TRAJECTORIES_TABLE
from ..persistence.db_services.trajectories.typing import TrajectoryRecord
from ..persistence.store import SkillPersistence


def collect_skill_trajectories(
    persistence: SkillPersistence,
    *,
    skill_id: str,
    limit: int,
    scan_limit: int,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    status: str | None = None,
    source: Literal["local", "reported"] | None = None,
    max_record_bytes: int = 262_144,
    max_total_bytes: int = 8_388_608,
) -> list[TrajectoryRecord]:
    """Collect recent matching attempts under the caller's active project scope.

    Args:
        persistence: Shared, already scoped Skill store.
        skill_id: Exact Skill family identity across all its versions.
        limit: Maximum retained matching attempts.
        scan_limit: Maximum database rows inspected, including unrelated attempts.
        created_from: Inclusive lower creation time, or no lower bound.
        created_to: Inclusive upper creation time, or the current time.
        status: Optional exact execution outcome.
        source: Optional local or Harness-reported origin.
        max_record_bytes: Maximum serialized size of each retained attempt.
        max_total_bytes: Maximum serialized size of all retained attempts.

    Returns:
        Newest matching attempts within the explicit scan and memory bounds.

    Raises:
        ValueError: A bound is invalid or the time interval is reversed.
    """
    if limit < 1 or scan_limit < 1 or max_record_bytes < 1 or max_total_bytes < 1:
        raise ValueError("Trajectory collection bounds must be positive")
    if created_from is not None and created_to is not None and created_to < created_from:
        raise ValueError("Trajectory creation bounds are reversed")
    filters = []
    if created_from is not None:
        filters.append(Filter(field="created_at", op="ge", value=created_from))
    if created_to is not None:
        filters.append(Filter(field="created_at", op="le", value=created_to))
    if status is not None:
        filters.append(Filter(field="status", value=status))
    if source is not None:
        filters.append(Filter(field="source", value=source))
    selected: list[TrajectoryRecord] = []
    scanned = 0
    offset = 0
    retained_bytes = 0
    while scanned < scan_limit and len(selected) < limit:
        page_size = min(20, scan_limit - scanned)
        rows = persistence.database.select(
            TRAJECTORIES_TABLE,
            filters=filters,
            fields=["trajectory_id", "skill_id", "skill_bindings", "injected_skills"],
            order_by=["-created_at", "-trajectory_id"],
            limit=page_size,
            offset=offset,
        )
        if not rows:
            break
        scanned += len(rows)
        offset += len(rows)
        for row in rows:
            if (
                row.get("skill_id") != skill_id
                and not _matches_binding(persistence, row.get("skill_bindings"), skill_id)
                and not _matches_binding(persistence, row.get("injected_skills"), skill_id)
            ):
                continue
            record = persistence.trajectories.get(row["trajectory_id"])
            if record is None:
                continue
            size = len(record.model_dump_json().encode("utf-8"))
            if size > max_record_bytes or retained_bytes + size > max_total_bytes:
                continue
            selected.append(record)
            retained_bytes += size
            if len(selected) == limit:
                break
    return selected


def _matches_binding(persistence: SkillPersistence, bindings: object, skill_id: str) -> bool:
    """Find explicit family or owned historical version in local bindings."""
    if not isinstance(bindings, list):
        return False
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        if binding.get("skill_id") == skill_id:
            return True
        version_id = binding.get("version_id")
        if isinstance(version_id, str):
            version = persistence.skills.get(version_id)
            if version is not None and version.skill_id == skill_id:
                return True
    return False
