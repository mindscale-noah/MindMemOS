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
from ..typing.skill_manage import (
    SkillManageBuildInput,
    SkillManageBuildPlan,
    SkillManageEmbedInput,
    SkillManageEmbedOutput,
    SkillManageMutationPlan,
    SkillManageRemoveInput,
    SkillManageRetrieveInput,
    SkillManageRetrieveOutput,
    SkillManageUpsertInput,
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


@runtime_checkable
class SkillManageAlgorithm(Protocol):
    """Algorithm capability used by :class:`SkillAlgorithms` for graph index work.

    Methods return pure plans/results only; they never commit storage or mutate
    canonical Skill versions.
    """

    async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
        """Compute a full graph build plan from exact immutable version refs."""
        ...

    async def plan_upsert(self, request: SkillManageUpsertInput) -> SkillManageMutationPlan:
        """Compute a same-family-replacement mutation plan."""
        ...

    async def plan_remove(self, request: SkillManageRemoveInput) -> SkillManageMutationPlan:
        """Compute a graph-only remove mutation plan."""
        ...

    async def embed(self, request: SkillManageEmbedInput) -> SkillManageEmbedOutput:
        """Embed texts through the injected embedding model under one profile."""
        ...

    async def retrieve(self, request: SkillManageRetrieveInput) -> SkillManageRetrieveOutput:
        """Retrieve Skills over a fixed, generation-qualified snapshot."""
        ...


__all__ = [
    "SkillAnalyzer",
    "SkillEvolver",
    "SkillManageAlgorithm",
    "SkillOptimizer",
]
