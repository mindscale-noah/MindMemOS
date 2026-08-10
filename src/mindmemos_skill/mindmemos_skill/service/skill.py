"""Public transport-neutral API for local Skill algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..agents import Agent
from ..errors import SkillCapabilityUnavailableError, SkillConfigurationError
from ..typing.operations import (
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillOptimizationRequest,
    SkillOptimizationResult,
)
from .protocols import SkillAnalyzer, SkillOptimizer


class SkillAlgorithms:
    """Algorithm API together with the Agents available to algorithm components.

    Resource construction and lifecycle remain owned by the application
    runtime.  This object owns the read-only Agent registry shared by the
    configured algorithm capabilities and dispatches analyze/optimize calls.
    """

    def __init__(
        self,
        *,
        analyzer: SkillAnalyzer | None = None,
        optimizer: SkillOptimizer | None = None,
        agents: Mapping[str, Agent[Any]] | None = None,
    ) -> None:
        if analyzer is None and optimizer is None:
            raise SkillConfigurationError("at least one Skill capability must be configured")
        self._analyzer = analyzer
        self._optimizer = optimizer
        self._agents = MappingProxyType(dict(agents or {}))

    @property
    def agents(self) -> Mapping[str, Agent[Any]]:
        """Return the immutable Agent registry available to algorithms."""

        return self._agents

    @property
    def capabilities(self) -> frozenset[str]:
        """Return the operations available in this algorithm set."""

        capabilities: set[str] = set()
        if self._analyzer is not None:
            capabilities.add("analyze")
        if self._optimizer is not None:
            capabilities.add("optimize")
        return frozenset(capabilities)

    async def analyze(self, request: SkillAnalysisRequest) -> SkillAnalysisResult:
        """Analyze a Skill using the configured local analyzer."""

        if self._analyzer is None:
            raise SkillCapabilityUnavailableError("Skill analysis is not configured")
        return await self._analyzer.analyze(request)

    async def optimize(self, request: SkillOptimizationRequest) -> SkillOptimizationResult:
        """Optimize a Skill using the configured local optimizer."""

        if self._optimizer is None:
            raise SkillCapabilityUnavailableError("Skill optimization is not configured")
        return await self._optimizer.optimize(request)


__all__ = [
    "SkillAlgorithms",
]
