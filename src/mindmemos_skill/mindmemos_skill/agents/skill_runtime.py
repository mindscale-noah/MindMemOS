"""Unified Skill injection and binding runtime contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ClassVar

from ..typing import (
    AgentExecutionRequest,
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


@dataclass(frozen=True, slots=True)
class RoutedSkillSnapshot:
    """Ephemeral routed content for one persisted Skill version."""

    skill_name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SkillRoute:
    """Normalized result returned by a query-aware runtime callback."""

    skills: tuple[RoutedSkillSnapshot, ...]
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

    async def route(self, request: AgentExecutionRequest) -> SkillRoute | None:
        """Optionally prepare query-specific Skill content before injection."""

        del request
        return None

    def inject_routed(
        self,
        request: AgentExecutionRequest,
        route: SkillRoute,
    ) -> AbstractContextManager[SkillInjection]:
        """Inject a prepared route; legacy runtimes delegate to full injection."""

        del route
        return self.inject(request.skills)

    @asynccontextmanager
    async def injection_scope(self, request: AgentExecutionRequest) -> AsyncIterator[SkillInjection]:
        """Resolve an optional route and expose one request-scoped injection."""

        route = await self.route(request)
        manager = self.inject(request.skills) if route is None else self.inject_routed(request, route)
        with manager as injection:
            if route is not None:
                injection.metadata = {**route.metadata, **injection.metadata}
            yield injection

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


def apply_skill_injection(
    messages: list[dict[str, Any]],
    injection: SkillInjection,
) -> list[dict[str, Any]]:
    """Apply normalized Skill messages and suffix to one conversation."""

    merged = [*injection.system_messages, *messages]
    suffix = injection.system_prompt_suffix
    if suffix is None:
        return merged
    for index, message in enumerate(merged):
        if message.get("role") != "system":
            continue
        content = message.get("content")
        base_prompt = content if isinstance(content, str) else ""
        combined = f"{base_prompt.rstrip()}\n\n{suffix}" if base_prompt else suffix
        merged[index] = {**message, "content": combined}
        return merged
    return [{"role": "system", "content": suffix}, *merged]


__all__ = [
    "RoutedSkillSnapshot",
    "SkillInjection",
    "SkillRoute",
    "SkillRuntime",
    "apply_skill_injection",
]
