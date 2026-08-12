"""Base contract shared by all agent implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from datetime import UTC, datetime
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import BaseModel

from ..llm import ChatResponse
from ..typing import (
    AgentExecutionRequest,
    AgentProfile,
    AgentType,
    ExecutionInfo,
    Skill,
    SkillBinding,
    SkillInjectionMode,
    Trajectory,
    TrajectoryStatus,
)
from .config import AgentConfig
from .skill_runtime import SkillInjection, SkillRuntime

AgentConfigT = TypeVar("AgentConfigT", bound=AgentConfig)


class Agent(ABC, Generic[AgentConfigT]):
    """Configured executable agent and owner of its Skill runtime contract.

    Concrete agents declare ``config_type`` so mapping inputs are validated at
    construction time rather than being interpreted ad hoc during execution.
    They also own Skill injection and binding because discovery paths and
    trajectory evidence are runtime-specific (for example, Claude's
    ``.claude/skills`` directory and ``Skill`` tool calls).
    """

    agent_type: ClassVar[AgentType] = AgentType.UNKNOWN
    config_type: type[AgentConfig] = AgentConfig
    skill_runtime_types: ClassVar[Mapping[SkillInjectionMode, type[SkillRuntime]]] = {}

    def __init__(self, config: AgentConfigT | Mapping[str, Any]) -> None:
        raw_config = config.model_dump() if isinstance(config, BaseModel) else config
        self.config = cast(AgentConfigT, self.config_type.model_validate(raw_config))
        self._model_profile: dict[str, Any] = {}
        self._skill_runtimes = {
            mode: self._create_skill_runtime(mode, runtime_type)
            for mode, runtime_type in self.skill_runtime_types.items()
        }
        configured_mode = self.config.skill_injection_mode
        if configured_mode is not None and configured_mode not in self._skill_runtimes:
            supported = ", ".join(sorted(mode.value for mode in self._skill_runtimes)) or "<none>"
            raise ValueError(
                f"{type(self).__name__} does not support {configured_mode.value!r} Skill injection; "
                f"supported modes: {supported}"
            )

    def attach_model_profile(self, profile: Mapping[str, Any]) -> None:
        """Attach one resolved endpoint snapshot before the Agent starts executing."""

        if self._model_profile:
            raise RuntimeError(f"{type(self).__name__} already has a model profile attached")
        normalized = AgentProfile.from_config(agent_type=self.agent_type, config=profile)
        flattened = normalized.model_dump(mode="json", exclude={"agent_type"}, exclude_none=True)
        extensions = flattened.pop("config")
        self._model_profile = {**extensions, **flattened}

    def inject_skills(
        self,
        skills: Sequence[Skill],
        *,
        mode: SkillInjectionMode | None = None,
    ) -> AbstractContextManager[SkillInjection]:
        """Delegate injection to the runtime mounted for the effective mode."""

        runtime = self.get_skill_runtime(mode)
        return runtime.inject(list(skills))

    def inject_skill_request(
        self,
        request: AgentExecutionRequest,
        *,
        mode: SkillInjectionMode | None = None,
    ) -> AbstractAsyncContextManager[SkillInjection]:
        """Resolve query-aware routing and inject Skills for one request."""

        runtime = self.get_skill_runtime(mode)
        return runtime.injection_scope(request)

    def _create_skill_runtime(
        self,
        mode: SkillInjectionMode,
        runtime_type: type[SkillRuntime],
    ) -> SkillRuntime:
        """Construct one runtime; agents may override to inject dependencies."""

        return runtime_type(mode)

    def bind_skills(self, trajectory: Trajectory) -> list[SkillBinding]:
        """Delegate binding to the same runtime mode persisted on the trajectory."""

        runtime = self.get_skill_runtime(trajectory.agent.skill_injection_mode)
        return runtime.bind(trajectory)

    def _build_trajectory(
        self,
        *,
        request: AgentExecutionRequest,
        config: AgentConfig,
        messages: list[dict[str, Any]],
        started_at: float,
        ended_at: float,
        n_turn: int,
        is_success: bool,
        error_info: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Normalize one execution result and attach runtime-specific Skill bindings."""

        trajectory = Trajectory(
            trajectory_id=request.trajectory_id,
            task=request.task,
            rollout=request.rollout,
            environment=request.environment,
            agent=AgentProfile.from_config(
                agent_type=self.agent_type,
                config={**self._model_profile, **config.snapshot()},
            ),
            injected_skills=request.skills,
            events=messages,
            execution=ExecutionInfo(
                status=TrajectoryStatus.SUCCEEDED if is_success else TrajectoryStatus.FAILED,
                started_at=datetime.fromtimestamp(started_at, tz=UTC),
                finished_at=datetime.fromtimestamp(ended_at, tz=UTC),
                n_turn=n_turn,
                error_info=error_info,
            ),
            metadata={**request.metadata, **(metadata or {})},
        )
        trajectory.skill_bindings = self.bind_skills(trajectory)
        return trajectory

    async def respond(
        self,
        request: AgentExecutionRequest,
        messages: list[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = (),
    ) -> ChatResponse:
        """Generate one assistant response for an environment-owned conversation.

        Interactive benchmark environments own simulator turns and therefore
        cannot delegate the whole rollout to :meth:`execute`. Agents that can
        serve those environments override this lower-level message interface.
        """

        del request, messages, tools
        raise TypeError(f"{type(self).__name__} does not support environment-owned conversations")

    def build_trajectory(
        self,
        *,
        request: AgentExecutionRequest,
        messages: list[dict[str, Any]],
        started_at: float,
        ended_at: float,
        n_turn: int,
        is_success: bool,
        error_info: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trajectory:
        """Finalize messages produced by an interactive environment."""

        config = self.config.with_overrides(request.options)
        return self._build_trajectory(
            request=request,
            config=config,
            messages=messages,
            started_at=started_at,
            ended_at=ended_at,
            n_turn=n_turn,
            is_success=is_success,
            error_info=error_info,
            metadata=metadata,
        )

    @property
    def supported_skill_injection_modes(self) -> frozenset[SkillInjectionMode]:
        """Modes for which this Agent mounted a complete SkillRuntime."""

        return frozenset(self._skill_runtimes)

    def get_skill_runtime(self, mode: SkillInjectionMode | None = None) -> SkillRuntime:
        """Return the mounted runtime for an explicit or configured mode."""

        effective_mode = mode or self.config.skill_injection_mode
        if effective_mode is None:
            raise ValueError(f"{type(self).__name__} has no Skill injection mode configured")
        runtime = self._skill_runtimes.get(effective_mode)
        if runtime is None:
            supported = ", ".join(sorted(item.value for item in self._skill_runtimes)) or "<none>"
            raise ValueError(
                f"{type(self).__name__} does not support {effective_mode.value!r} Skill injection; "
                f"supported modes: {supported}"
            )
        return runtime

    @abstractmethod
    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        """Execute one physical attempt and return its canonical trajectory."""


__all__ = ["Agent", "AgentConfigT"]
