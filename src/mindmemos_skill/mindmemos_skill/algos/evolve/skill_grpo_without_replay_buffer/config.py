"""Strict configuration for the replay-free Skill GRPO evolution loop."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..skill_grpo_with_replay_buffer.config import (
    DatasetRuntimeConfig,
    PatchConfig,
    RetryConfig,
    RolloutStrategyConfig,
)
from ..skill_grpo_with_replay_buffer.config import (
    TrainingConfig as SharedTrainingConfig,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RolloutConfig(_StrictModel):
    """Rollout settings for training, validation and final test."""

    max_concurrent_rollouts: int = Field(default=8, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    fail_fast: bool = True
    workspace_root: Path | None = None
    train: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 4})
    )
    validation: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 1})
    )
    test: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 1})
    )


class ValidationConfig(_StrictModel):
    enabled: bool = False


class ReflectionConfig(_StrictModel):
    """Reflexion-style feedback passed between failed training rollouts."""

    enabled: bool = True
    max_trajectory_chars: int = Field(default=24_000, ge=1)
    max_previous_answer_chars: int = Field(default=8_000, ge=1)
    max_reflection_chars: int = Field(default=4_000, ge=1)
    max_concurrent_reflections: int | None = Field(default=8, ge=1)


class ExperienceConfig(_StrictModel):
    """Density and concurrency for all three mandatory evidence streams."""

    max_experiences_per_task: int = Field(default=3, ge=1)
    max_concurrent_extractions: int | None = Field(default=None, ge=1)


class TrainingConfig(SharedTrainingConfig):
    """Replay-free batch boundaries for rollout and experience extraction."""

    batch_size: int = Field(default=40, ge=1)
    mini_batch_size: int = Field(default=8, ge=1)


class SkillGrpoWithoutReplayBufferConfig(_StrictModel):
    """Algorithm settings; deliberately contains no replay or ablation knobs."""

    version: str = "3"
    prompt_version: str = "skill-grpo-replay-free-three-stream-v2"
    experience: ExperienceConfig = Field(default_factory=ExperienceConfig)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class SkillGrpoWithoutReplayBufferRunConfig(_StrictModel):
    algorithm: SkillGrpoWithoutReplayBufferConfig = Field(default_factory=SkillGrpoWithoutReplayBufferConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    dataset: DatasetRuntimeConfig


__all__ = [
    "ExperienceConfig",
    "ReflectionConfig",
    "RolloutConfig",
    "SkillGrpoWithoutReplayBufferConfig",
    "SkillGrpoWithoutReplayBufferRunConfig",
    "TrainingConfig",
    "ValidationConfig",
]
