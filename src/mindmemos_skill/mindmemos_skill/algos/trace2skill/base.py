"""Minimal protocol for trajectory-to-Skill algorithms."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol, runtime_checkable

from ...typing import Trace2SkillInput, Trace2SkillOutput


@runtime_checkable
class Trace2SkillAlgorithm(Protocol):
    """Transform a Skill from offline or actively collected trajectories."""

    algorithm_name: ClassVar[str]
    algorithm_version: ClassVar[str]

    async def optimize(self, request: Trace2SkillInput) -> Trace2SkillOutput[Any]:
        """Return an unpersisted Skill candidate derived from trajectory evidence."""
        ...


__all__ = ["Trace2SkillAlgorithm"]
