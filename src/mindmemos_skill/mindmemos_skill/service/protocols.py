"""Algorithm capability protocols used by :class:`SkillAlgorithms`."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..typing.operations import (
    EvolveInput,
    EvolveOutput,
    SkillAnalysisRequest,
    SkillAnalysisResult,
    Trace2SkillInput,
    Trace2SkillOutput,
)


@runtime_checkable
class SkillAnalyzer(Protocol):
    """Algorithm capability used by :class:`SkillAlgorithms.analyze`."""

    async def analyze(self, request: SkillAnalysisRequest) -> SkillAnalysisResult:
        """Analyze one Skill and its local execution evidence."""
        ...


@runtime_checkable
class SkillOptimizer(Protocol):
    """Algorithm capability used by :class:`SkillAlgorithms.optimize`."""

    async def optimize(self, request: Trace2SkillInput) -> Trace2SkillOutput[Any]:
        """Produce the selected optimized Skill candidate."""
        ...


@runtime_checkable
class SkillEvolver(Protocol):
    """Algorithm capability used by :class:`SkillAlgorithms.evolve`."""

    async def evolve(self, request: EvolveInput) -> EvolveOutput:
        """Run one complete Skill evolution without application persistence."""
        ...


__all__ = [
    "SkillAnalyzer",
    "SkillEvolver",
    "SkillOptimizer",
]
