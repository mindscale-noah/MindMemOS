"""Bounded, reusable collection of stored Skill execution evidence."""

from datetime import datetime
from typing import Literal

from ..infra.database import Filter
from ..persistence.db_services.trajectories.typing import TrajectoryRecord
from ..persistence.store import SkillPersistence


async def collect_skill_trajectories(
    persistence: SkillPersistence,
    *,
    skill_version_id: str,
    limit: int,
    max_page_size: int = 200,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
    status: str | None = None,
    source: Literal["local", "reported"] | None = None,
) -> list[TrajectoryRecord]:
    """Collect recent matching attempts under the caller's active project scope.

    Args:
        persistence: Shared, already scoped Skill store.
        skill_version_id: Exact loaded Skill version; family IDs are not accepted.
        limit: Maximum retained matching attempts.
        max_page_size: Maximum matching rows fetched per database query; defaults to 200.
        created_from: Inclusive lower creation time, or no lower bound.
        created_to: Inclusive upper creation time, or the current time.
        status: Optional exact execution outcome.
        source: Optional local or Harness-reported origin.

    Returns:
        Newest matching attempts up to the retained-count limit.

    Raises:
        ValueError: A bound is invalid or the time interval is reversed.
    """
    if limit < 1 or max_page_size < 1:
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
    offset = 0
    while len(selected) < limit:
        rows = await persistence.trajectories.list_for_version(
            skill_version_id,
            filters=filters,
            limit=min(max_page_size, limit - len(selected)),
            offset=offset,
        )
        if not rows:
            break
        offset += len(rows)
        selected.extend(rows)
    return selected
