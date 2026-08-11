"""Public transport-neutral API for local Skill algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from ..agents import Agent
from ..errors import SkillCapabilityUnavailableError, SkillConfigurationError
from ..typing.operations import (
    EvolveInput,
    EvolveOutput,
    SkillAnalysisRequest,
    SkillAnalysisResult,
    Trace2SkillInput,
    Trace2SkillOutput,
)
from .protocols import SkillAnalyzer, SkillEvolver, SkillOptimizer


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
        evolver: SkillEvolver | None = None,
        analyzers: Mapping[str, SkillAnalyzer] | None = None,
        optimizers: Mapping[str, SkillOptimizer] | None = None,
        evolvers: Mapping[str, SkillEvolver] | None = None,
        agents: Mapping[str, Agent[Any]] | None = None,
    ) -> None:
        self._analyzers = self._with_default(analyzers, analyzer)
        self._optimizers = self._with_default(optimizers, optimizer)
        self._evolvers = self._with_default(evolvers, evolver)
        if not self._analyzers and not self._optimizers and not self._evolvers:
            raise SkillConfigurationError("at least one Skill capability must be configured")
        self._agents = MappingProxyType(dict(agents or {}))

    @staticmethod
    def _with_default(algorithms: Mapping[str, Any] | None, default: Any | None) -> Mapping[str, Any]:
        resolved = dict(algorithms or {})
        if default is not None:
            if resolved:
                raise SkillConfigurationError("default and named algorithms may not be configured together")
            resolved["default"] = default
        return MappingProxyType(resolved)

    @property
    def agents(self) -> Mapping[str, Agent[Any]]:
        """Return the immutable Agent registry available to algorithms."""

        return self._agents

    @property
    def capabilities(self) -> frozenset[str]:
        """Return the operations available in this algorithm set."""

        capabilities: set[str] = set()
        if self._analyzers:
            capabilities.add("analyze")
        if self._optimizers:
            capabilities.add("optimize")
        if self._evolvers:
            capabilities.add("evolve")
        return frozenset(capabilities)

    def algorithm_names(self, capability: str) -> tuple[str, ...]:
        """Return configured component names for one capability."""

        return tuple(sorted(self._algorithms_for(capability)))

    def resolve_name(self, capability: str, algorithm_name: str | None = None) -> str:
        """Resolve an explicit name or the only configured implementation."""

        algorithms = self._algorithms_for(capability)
        if algorithm_name is not None:
            if algorithm_name not in algorithms:
                available = ", ".join(sorted(algorithms)) or "<none>"
                raise SkillCapabilityUnavailableError(
                    f"Skill {capability} algorithm {algorithm_name!r} is not configured; available: {available}"
                )
            return algorithm_name
        if len(algorithms) == 1:
            return next(iter(algorithms))
        if not algorithms:
            operation = {"analyze": "analysis", "optimize": "optimization", "evolve": "evolution"}[capability]
            raise SkillCapabilityUnavailableError(f"Skill {operation} is not configured")
        available = ", ".join(sorted(algorithms)) or "<none>"
        raise SkillCapabilityUnavailableError(
            f"Skill {capability} requires an algorithm name; available: {available}"
        )

    def _algorithms_for(self, capability: str) -> Mapping[str, Any]:
        try:
            return {
                "analyze": self._analyzers,
                "optimize": self._optimizers,
                "evolve": self._evolvers,
            }[capability]
        except KeyError as exc:
            raise ValueError(f"unknown Skill algorithm capability: {capability}") from exc

    async def analyze(
        self,
        request: SkillAnalysisRequest,
        *,
        algorithm_name: str | None = None,
    ) -> SkillAnalysisResult:
        """Analyze a Skill using the configured local analyzer."""

        name = self.resolve_name("analyze", algorithm_name)
        return await self._analyzers[name].analyze(request)

    async def optimize(
        self,
        request: Trace2SkillInput,
        *,
        algorithm_name: str | None = None,
    ) -> Trace2SkillOutput[Any]:
        """Optimize a Skill using the configured local optimizer."""

        name = self.resolve_name("optimize", algorithm_name)
        return await self._optimizers[name].optimize(request)

    async def evolve(
        self,
        request: EvolveInput,
        *,
        algorithm_name: str | None = None,
    ) -> EvolveOutput:
        """Run a configured complete evolution algorithm."""

        name = self.resolve_name("evolve", algorithm_name)
        return await self._evolvers[name].evolve(request)


__all__ = [
    "SkillAlgorithms",
]
