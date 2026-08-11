"""Optional active trajectory collection for trace2skill algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ...agents import Agent
from ...typing import Skill, Task, Trajectory
from ..evolve.skill_grpo_with_replay_buffer.config import RetryConfig, RolloutConfig
from ..evolve.skill_grpo_with_replay_buffer.rollout import (
    FixedGroupPlan,
    FixedGroupRolloutStrategy,
    MappingAgentResolver,
    RegistryEnvFactory,
    RolloutScheduler,
)


class CollectionRetryConfig(BaseModel):
    """Retry physical collection failures that produced no trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(default=1, ge=1)
    backoff_seconds: float = Field(default=0.0, ge=0.0)


class TaskCollectionConfig(BaseModel):
    """Runtime settings for collecting a fixed number of traces per task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_ref: str = Field(min_length=1)
    env_ref: str = Field(min_length=1)
    samples_per_task: int = Field(default=1, ge=1)
    max_concurrent_rollouts: int = Field(default=8, ge=1)
    queue_capacity: int = Field(default=16, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry: CollectionRetryConfig = Field(default_factory=CollectionRetryConfig)
    fail_fast: bool = False
    workspace_root: Path | None = None
    seed: int = 0
    temperature: float | None = None
    agent_options: dict[str, JsonValue] = Field(default_factory=dict)
    env_options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_queue(self) -> TaskCollectionConfig:
        if self.queue_capacity < self.max_concurrent_rollouts:
            raise ValueError("queue_capacity must be at least max_concurrent_rollouts")
        return self


class TrajectoryCollectionResult(BaseModel):
    """Collected trajectories plus rollout-level failure evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    requested_rollout_ids: list[str] = Field(default_factory=list)
    trajectories: list[Trajectory] = Field(default_factory=list)
    failed_rollout_ids: list[str] = Field(default_factory=list)


class TrajectoryCollector(Protocol):
    """Acquire trajectories without knowing how they will optimize a Skill."""

    async def collect(
        self,
        *,
        run_id: str,
        base_skill: Skill,
        tasks: list[Task],
    ) -> TrajectoryCollectionResult: ...


class ScheduledTrajectoryCollector:
    """Collect a fixed sample group through the same Agent/Env path as evolve."""

    def __init__(self, *, agents: Mapping[str, Agent[Any]], config: TaskCollectionConfig) -> None:
        self._config = config
        self._scheduler = RolloutScheduler(
            agent_resolver=MappingAgentResolver(agents),
            env_factory=RegistryEnvFactory(),
            config=RolloutConfig(
                max_concurrent_rollouts=config.max_concurrent_rollouts,
                queue_capacity=config.queue_capacity,
                timeout_seconds=config.timeout_seconds,
                retry=RetryConfig(
                    max_attempts=config.retry.max_attempts,
                    backoff_seconds=config.retry.backoff_seconds,
                ),
                fail_fast=config.fail_fast,
                workspace_root=config.workspace_root,
            ),
        )

    async def collect(
        self,
        *,
        run_id: str,
        base_skill: Skill,
        tasks: list[Task],
    ) -> TrajectoryCollectionResult:
        plan = FixedGroupPlan(
            run_id=run_id,
            scope="trace2skill_collect",
            phase="train",
            tasks=tasks,
            skills=[base_skill],
            sequence_start=0,
            group_size=self._config.samples_per_task,
            agent_ref=self._config.agent_ref,
            env_ref=self._config.env_ref,
            seed=self._config.seed,
            temperature=self._config.temperature,
            agent_options=self._config.agent_options,
            env_options=self._config.env_options,
        )
        specs = FixedGroupRolloutStrategy().plan(plan)
        outcomes = await self._scheduler.run(specs)
        return TrajectoryCollectionResult(
            run_id=run_id,
            requested_rollout_ids=[spec.rollout_id for spec in specs],
            trajectories=[outcome.trajectory for outcome in outcomes if outcome.trajectory is not None],
            failed_rollout_ids=[outcome.spec.rollout_id for outcome in outcomes if outcome.trajectory is None],
        )


__all__ = [
    "CollectionRetryConfig",
    "ScheduledTrajectoryCollector",
    "TaskCollectionConfig",
    "TrajectoryCollectionResult",
    "TrajectoryCollector",
]
