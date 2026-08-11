"""Protocol shared by complete Skill evolution algorithms."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Protocol, runtime_checkable

from ...typing import EvolveInput, EvolveOutput, Trajectory


class EvolveAlgorithmContext(Protocol):
    """Application dependencies exposed to configurable evolution algorithms."""

    models: Mapping[str, Any]
    agents: Mapping[str, Any]


@runtime_checkable
class EvolveAlgorithm(Protocol):
    """A complete algorithm that evolves an input Skill without persisting it."""

    algorithm_name: ClassVar[str]

    async def evolve(self, request: EvolveInput) -> EvolveOutput:
        """Execute the evolution run and return an unpersisted result."""
        ...


def trajectories_from_rollouts(outcomes: Iterable[Any]) -> list[Trajectory]:
    """Return every physical attempt trajectory once, including failed retries."""

    trajectories: list[Trajectory] = []
    seen: set[str] = set()
    for outcome in outcomes:
        for attempt in outcome.attempts:
            trajectory = attempt.trajectory
            if trajectory is not None and trajectory.trajectory_id not in seen:
                seen.add(trajectory.trajectory_id)
                trajectories.append(trajectory)
    return trajectories


__all__ = ["EvolveAlgorithm", "EvolveAlgorithmContext", "trajectories_from_rollouts"]
