"""Pure rollout planning contracts and registry."""

from __future__ import annotations

from typing import Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .....typing import Skill, Task
from ..contracts import RolloutSpec


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FixedGroupPlan(_PlanModel):
    run_id: str
    scope: str
    phase: str
    tasks: list[Task]
    skills: list[Skill]
    sequence_start: int = Field(ge=0)
    group_size: int = Field(ge=1)
    agent_ref: str
    env_ref: str
    seed: int
    temperature: float | None = None
    agent_options: dict[str, JsonValue] = Field(default_factory=dict)
    env_options: dict[str, JsonValue] = Field(default_factory=dict)


class AblationTarget(_PlanModel):
    candidate_id: str
    skill: Skill
    task_ids: list[str]


class PairedAblationPlan(_PlanModel):
    run_id: str
    scope: str
    tasks: list[Task]
    before_skill: Skill
    targets: list[AblationTarget]
    sequence_start: int = Field(ge=0)
    sample_index_start: int = Field(default=1_000_001, ge=0)
    samples_per_case: int = Field(ge=1)
    agent_ref: str
    env_ref: str
    seed: int
    temperature: float | None = None
    agent_options: dict[str, JsonValue] = Field(default_factory=dict)
    env_options: dict[str, JsonValue] = Field(default_factory=dict)


RolloutPlan: TypeAlias = FixedGroupPlan | PairedAblationPlan


class RolloutStrategy(Protocol):
    """A pure planner: no Env calls, tasks, semaphore, or persistence."""

    name: str

    def plan(self, request: RolloutPlan) -> list[RolloutSpec]: ...


class RolloutStrategyRegistry:
    def __init__(self) -> None:
        self._strategies: dict[str, RolloutStrategy] = {}

    def register(self, strategy: RolloutStrategy) -> None:
        if strategy.name in self._strategies:
            raise ValueError(f"rollout strategy {strategy.name!r} is already registered")
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> RolloutStrategy:
        try:
            return self._strategies[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._strategies)) or "<none>"
            raise ValueError(f"unknown rollout strategy {name!r}; available: {available}") from exc

    @classmethod
    def with_builtins(cls) -> RolloutStrategyRegistry:
        from .fixed_group import FixedGroupRolloutStrategy
        from .paired_ablation import PairedAblationRolloutStrategy

        registry = cls()
        registry.register(FixedGroupRolloutStrategy())
        registry.register(PairedAblationRolloutStrategy())
        return registry


__all__ = [
    "AblationTarget",
    "FixedGroupPlan",
    "PairedAblationPlan",
    "RolloutPlan",
    "RolloutStrategy",
    "RolloutStrategyRegistry",
]
