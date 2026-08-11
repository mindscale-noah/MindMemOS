"""Strict configuration for experience-validated, replay-free Skill GRPO."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..skill_grpo_with_replay_buffer.config import (
    DatasetRuntimeConfig,
    PatchConfig,
    RetryConfig,
    RolloutStrategyConfig,
)
from ..skill_grpo_with_replay_buffer.config import TrainingConfig as SharedTrainingConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RolloutConfig(_StrictModel):
    """Rollout settings for training, experience re-runs and final test."""

    max_concurrent_rollouts: int = Field(default=8, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    fail_fast: bool = True
    workspace_root: Path | None = None
    train: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 4})
    )
    experience_validation: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 1})
    )
    test: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 1})
    )


class ReflectionConfig(_StrictModel):
    """Reflexion feedback for training and contrast-experience re-runs."""

    enabled: bool = True
    max_trajectory_chars: int = Field(default=24_000, ge=1)
    max_previous_answer_chars: int = Field(default=8_000, ge=1)
    max_reflection_chars: int = Field(default=4_000, ge=1)
    max_concurrent_reflections: int | None = Field(default=8, ge=1)


class ExperienceConfig(_StrictModel):
    """Density and concurrency for extraction and mandatory re-run checks."""

    max_experiences_per_task: int = Field(default=3, ge=1)
    max_concurrent_extractions: int | None = Field(default=None, ge=1)


class TrainingConfig(SharedTrainingConfig):
    batch_size: int = Field(default=40, ge=1)
    mini_batch_size: int = Field(default=8, ge=1)


class SkillGrpoWithExperienceValidationConfig(_StrictModel):
    """Algorithm settings; the experience re-run is the only acceptance gate."""

    version: str = "1"
    prompt_version: str = "skill-grpo-experience-validation-v1"
    experience: ExperienceConfig = Field(default_factory=ExperienceConfig)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)


class SkillGrpoWithExperienceValidationRunConfig(_StrictModel):
    algorithm: SkillGrpoWithExperienceValidationConfig = Field(default_factory=SkillGrpoWithExperienceValidationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    dataset: DatasetRuntimeConfig

    @model_validator(mode="after")
    def validate_experience_validation_strategy(self) -> SkillGrpoWithExperienceValidationRunConfig:
        if self.rollout.experience_validation.name != "fixed_group":
            raise ValueError("experience_validation rollout strategy must be fixed_group")
        return self


__all__ = [
    "ExperienceConfig",
    "ReflectionConfig",
    "RolloutConfig",
    "SkillGrpoWithExperienceValidationConfig",
    "SkillGrpoWithExperienceValidationRunConfig",
    "TrainingConfig",
]
