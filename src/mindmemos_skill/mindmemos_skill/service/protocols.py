"""Algorithm capability protocols used by :class:`SkillAlgorithms`."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..typing.operations import (
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillOptimizationRequest,
    SkillOptimizationResult,
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

    async def optimize(self, request: SkillOptimizationRequest) -> SkillOptimizationResult:
        """Produce the selected optimized Skill candidate."""
        ...

__all__ = [
    "SkillAnalyzer",
    "SkillOptimizer",
]
