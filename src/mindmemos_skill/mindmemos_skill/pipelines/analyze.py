"""Shared contract for cache-first evaluation pipelines."""

from abc import ABC, abstractmethod

from ..api.typing import (
    EvaluationQueryRequest,
    EvaluationQueryResult,
    SkillAnalyzeRequest,
    TrajectoryAnalyzeRequest,
)
from ..persistence import EvaluationRecord
from ..registry.base import ConfigT
from .base import Pipeline


class AnalyzePipeline(Pipeline[ConfigT], ABC):
    """Run pure analyzers while owning their cache and persistence boundary."""

    @abstractmethod
    async def analyze_skill(self, request: SkillAnalyzeRequest) -> EvaluationRecord:
        """Return a stored or newly computed static Skill evaluation."""

    @abstractmethod
    async def analyze_trajectory(self, request: TrajectoryAnalyzeRequest) -> EvaluationRecord:
        """Return a stored or newly computed trajectory evaluation."""

    @property
    @abstractmethod
    def evaluation_query_limit(self) -> int:
        """Maximum stored evaluations loaded by one recommendation post-process."""

    @abstractmethod
    async def query(self, request: EvaluationQueryRequest) -> EvaluationQueryResult:
        """Read completed evaluations for exact Skill versions."""


__all__ = ["AnalyzePipeline"]
