"""Shared contract for analysis algorithms."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from ...registry.base import ConfigT
from ..base import Algorithm
from .typing import SkillAnalysisResult

RequestT = TypeVar("RequestT", bound=BaseModel)
"""Concrete request model accepted by one analysis implementation."""


class SkillAnalyzer(Algorithm[ConfigT], Generic[ConfigT, RequestT], ABC):
    """Analyze supplied Skill content or execution evidence without persistence."""

    @abstractmethod
    async def analyze(self, request: RequestT) -> SkillAnalysisResult:
        """Produce a transport-neutral analysis result.

        Args:
            request: Implementation-specific immutable input boundary.

        Returns:
            Summary, findings, metrics, and provenance for caller persistence.
        """


__all__ = ["SkillAnalyzer"]
