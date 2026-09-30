"""Algorithm-owned dynamic SkillSet independent of source version identities."""

from collections.abc import Mapping

from ....envs.typing import Task
from ....skill import Skill, SkillSet
from .algorithm import SkillManageAlgorithm


class SkillManageSkillSet(SkillSet):
    """Recommend members through a caller-provided graph manager, then assemble them."""

    def __init__(
        self, skills: Mapping[str, Skill], *, algorithm: SkillManageAlgorithm, limit: int = 5, max_concurrency: int
    ):
        """Keep selection and member assembly independently configurable.

        Args:
            skills: Explicit source objects keyed by caller-selected member IDs.
            algorithm: Dedicated graph manager with externally bound persistence.
            limit: Maximum recommended members per task.
            max_concurrency: Maximum simultaneous member assembly calls.
        """
        super().__init__(skills, max_concurrency=max_concurrency)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        self.algorithm = algorithm
        self.limit = limit

    def config_snapshot(self) -> dict:
        """Record selection settings without inspecting Skill version identities."""
        return {"limit": self.limit, **self.algorithm.config_snapshot()}

    def select(self, task: Task) -> list[str]:
        """Organize the explicit scope and request its task-specific member IDs.

        Args:
            task: Public task passed to the selection algorithm.
        """
        self.algorithm.organize(skills=self.skills)
        return self.algorithm.recommend(task=task, limit=self.limit)
