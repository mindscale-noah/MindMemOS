"""Task-time selection and bounded parallel assembly of Skill collections."""

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import JsonValue

from ..common.asyncio import map_sync_bounded, run_sync
from .skill import Skill
from .typing import SkillSetIdentity

if TYPE_CHECKING:
    from ..envs.typing import Task


class SkillSet:
    """Select collection members without depending on their assembly internals."""

    def __init__(self, skills: Mapping[str, Skill], *, max_concurrency: int = 1):
        """Retain an ordered candidate scope and an explicit assembly limit.

        Args:
            skills: Caller-owned IDs mapped to source Skills in injection order.
            max_concurrency: Maximum simultaneous member assemblies per call.

        Raises:
            ValueError: The concurrency limit or a member ID is invalid.
            TypeError: A candidate is not a Skill.
        """
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")
        if any(not isinstance(key, str) or not key.strip() for key in skills):
            raise ValueError("Skill IDs must be nonempty strings")
        if any(not isinstance(skill, Skill) for skill in skills.values()):
            raise TypeError("SkillSet members must be Skill objects")
        self._skills = dict(skills)
        self._max_concurrency = max_concurrency

    @property
    def skills(self) -> dict[str, Skill]:
        """Return a copy of the candidate mapping, retaining the source objects."""
        return dict(self._skills)

    @property
    def max_concurrency(self) -> int:
        """Return the maximum number of simultaneous assemblies per call."""
        return self._max_concurrency

    def config_snapshot(self) -> dict[str, JsonValue]:
        """Return non-secret selection settings; configurable subclasses override this."""
        return {}

    def identity(self) -> SkillSetIdentity:
        """Describe this collection for input comparison without selecting or assembling."""
        return SkillSetIdentity(
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            skill_ids=list(self._skills),
            max_concurrency=self.max_concurrency,
            config=self.config_snapshot(),
        )

    def select(self, task: "Task") -> list[str]:
        """Select all candidates in order; subclasses may recommend a subset.

        Args:
            task: Public task input used by dynamic selection implementations.

        Returns:
            Ordered, unique candidate IDs to assemble for this task.
        """
        return list(self._skills)

    async def assemble(self, task: "Task", *, selected_ids: list[str] | None = None) -> list[Skill]:
        """Select members and assemble them concurrently in selection order.

        Args:
            task: Public task input passed to every selected Skill.
            selected_ids: Already selected IDs, allowing the caller to record
                selection without invoking it twice. None performs selection here.

        Returns:
            Complete task-local Skills, retaining selection order regardless of
            worker completion order. Empty selections return an empty list.

        Raises:
            ValueError: Selection contains duplicate or out-of-scope IDs.
            TypeError: A member does not return a complete Skill.
            Exception: Selection or assembly fails; no partial result is returned.
        """
        selected = list(await run_sync(self.select, task) if selected_ids is None else selected_ids)
        if len(set(selected)) != len(selected) or any(key not in self._skills for key in selected):
            raise ValueError("Selection must contain unique IDs from the candidate scope")
        if not selected:
            return []
        assembled = await map_sync_bounded(
            lambda key: self._skills[key].assemble(task), selected, concurrency=self.max_concurrency
        )
        if any(not isinstance(skill, Skill) for skill in assembled):
            raise TypeError("Skill.assemble(task) must return a complete Skill")
        return assembled
