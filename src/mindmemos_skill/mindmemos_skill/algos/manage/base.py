"""Contracts for skill organization and task-time recommendation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping

from ...envs.typing import Task
from ...registry.base import ConfigT
from ...skill import Skill
from ..base import Algorithm


class SkillManager(Algorithm[ConfigT], ABC):
    """Organize caller-selected skills and recommend within that scope.

    Implementations may use database storage for this scope's organization,
    but must never query the database to choose or expand the skill scope.
    Graphs, clusters, and other internal representations are implementation
    details. Register implementations with Algorithm.register.
    """

    @abstractmethod
    def organize(self, *, skills: Mapping[str, Skill]) -> None:
        """Replace the current scope and build its organization.

        Args:
            skills: Complete mapping of caller-owned IDs to skill snapshots.
                An empty mapping defines an empty scope, never all stored skills.
                Implementations must retain their own mapping and must not mutate
                the supplied skills. Repeated calls replace, rather than extend,
                the previous scope. Database operations must stay in this scope.
        """

    @abstractmethod
    def recommend(self, *, task: Task, limit: int = 5) -> list[str]:
        """Recommend only skills from the most recently organized scope.

        Args:
            task: Current task used to select relevant skills.
            limit: Maximum number of recommendations; must be positive.

        Returns:
            At most limit unique skill IDs in descending recommendation order.
            Every ID must belong to the current scope; an empty scope returns [].

        Raises:
            RuntimeError: No scope has been organized, or its latest build failed.
            ValueError: The recommendation limit is invalid.
        """
