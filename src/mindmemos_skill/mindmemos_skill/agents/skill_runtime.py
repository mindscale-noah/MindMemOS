"""Unified Skill injection and binding runtime contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..skill_runtime import SkillRuntimeTask
from ..typing import (
    Skill,
    SkillBinding,
    SkillInjectionMode,
    SkillUsageType,
    Trajectory,
)


@dataclass(slots=True)
class SkillInjection:
    """Normalized artifacts produced by one Skill injection scope."""

    mode: SkillInjectionMode
    system_prompt_suffix: str | None = None
    system_messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    skill_names: set[str] = field(default_factory=set)
    workspace: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillRuntime(ABC):
    """One injection mode together with its matching binding semantics."""

    supported_modes: ClassVar[frozenset[SkillInjectionMode]] = frozenset()

    def __init__(self, mode: SkillInjectionMode) -> None:
        if mode not in self.supported_modes:
            supported = ", ".join(sorted(item.value for item in self.supported_modes)) or "<none>"
            raise ValueError(f"{type(self).__name__} does not support {mode.value!r}; supported modes: {supported}")
        self.mode = mode

    @abstractmethod
    def inject(self, skills: list[Skill]) -> AbstractContextManager[SkillInjection]:
        """Expose immutable Skill snapshots for one execution scope."""

    @abstractmethod
    def bind(self, trajectory: Trajectory) -> list[SkillBinding]:
        """Interpret this runtime's evidence and bind loaded Skill versions."""

    def attach_runtime_task(self, injection: SkillInjection, task: SkillRuntimeTask) -> None:
        """Project dynamic resources into this Agent-family injection shape."""

        injection.metadata["skill_runtime"] = task.trace()

    def _build_bindings(self, trajectory: Trajectory, loaded_names: set[str]) -> list[SkillBinding]:
        """Build canonical bindings after a runtime has interpreted its evidence."""

        return [
            SkillBinding(
                name=skill.name,
                content_hash=skill.content_hash,
                skill_id=skill.skill_id,
                version_id=skill.version_id,
                version_label=skill.version_label,
                usage=(SkillUsageType.INJECTED if skill.name in loaded_names else SkillUsageType.UNUSED),
                injection_mode=self.mode,
            )
            for skill in trajectory.injected_skills
        ]


__all__ = ["SkillInjection", "SkillRuntime"]
