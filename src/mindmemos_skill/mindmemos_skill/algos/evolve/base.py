"""Contract for passive skill evolution from historical trajectories."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...registry.base import ConfigT
from ...skill import Skill
from ..base import Algorithm
from .typing import SkillTrajectory


class SkillEvolver(Algorithm[ConfigT], ABC):
    """Optimize one skill from existing trajectories without executing tasks.

    The caller collects and selects trajectories before evolution starts.
    Implementations and their dependencies must not access databases, fetch
    additional records, or invoke environments. Model calls for analysis and
    rewriting are allowed. The caller persists and optionally evaluates the
    returned skill. Register implementations with Algorithm.register.
    """

    @abstractmethod
    async def evolve(self, *, skill: Skill, trajectories: Sequence[SkillTrajectory]) -> Skill:
        """Optimize a supplied skill using existing trajectories.

        Args:
            skill: Initial skill snapshot; must not be mutated.
            trajectories: Materialized historical execution records supplied
                before evolution starts; must not be mutated.

        Returns:
            The evolved skill candidate without executing tasks or writing
            to a database.
        """
