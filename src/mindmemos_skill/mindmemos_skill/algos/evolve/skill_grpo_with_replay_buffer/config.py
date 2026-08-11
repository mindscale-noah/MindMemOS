"""Strict configuration for :class:`SkillGrpoWithReplayBuffer`."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RolloutStrategyConfig(_StrictModel):
    """One named strategy plus strategy-specific JSON parameters."""

    name: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)
    temperature: float | None = None


class RetryConfig(_StrictModel):
    max_attempts: int = Field(default=1, ge=1)
    backoff_seconds: float = Field(default=0.0, ge=0.0)


class RolloutConfig(_StrictModel):
    """The only concurrency budget used for every rollout phase."""

    max_concurrent_rollouts: int = Field(default=8, ge=1)
    queue_capacity: int = Field(default=16, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    fail_fast: bool = True
    workspace_root: Path | None = None
    train: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 4})
    )
    ablation: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="paired_ablation", params={"samples_per_case": 1})
    )
    validation: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 4})
    )
    test: RolloutStrategyConfig = Field(
        default_factory=lambda: RolloutStrategyConfig(name="fixed_group", params={"group_size": 4})
    )


class TrainingConfig(_StrictModel):
    seed: int = 0
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=8, ge=1)
    success_reward: float = 1.0


class ExperienceConfig(_StrictModel):
    max_experiences_per_task: int = Field(default=3, ge=1)
    max_concurrent_extractions: int | None = Field(default=None, ge=1)
    skip_all_failed_tasks: bool = False


class PatchConfig(_StrictModel):
    max_edits: int = Field(default=6, ge=1)
    max_attempts: int = Field(default=2, ge=1)


class ReplayBufferConfig(_StrictModel):
    use_embeddings: bool = True
    embedding_model_id: str | None = "text-embedding-3-small"
    similarity_threshold: float = Field(default=0.9, ge=-1.0, le=1.0)
    min_cluster_edits: int = Field(default=2, ge=1)
    capacity: int = Field(default=512, ge=0)
    max_uses: int = Field(default=10, ge=0)
    embedding_failure_mode: str = Field(default="exact_text", pattern=r"^exact_text$")

    @model_validator(mode="after")
    def validate_embedding_model(self) -> ReplayBufferConfig:
        if self.use_embeddings and not self.embedding_model_id:
            raise ValueError("embedding_model_id is required when use_embeddings is true")
        return self


class AblationConfig(_StrictModel):
    max_source_cases_per_candidate: int = Field(default=8, ge=1)
    positive_only: bool = True
    improvement_threshold: float | None = None
    commit_topk: int = Field(default=5, ge=1)
    seed: int = 0


class ValidationConfig(_StrictModel):
    every_batches: int = Field(default=0, ge=0)


class SkillGrpoConfig(_StrictModel):
    """Algorithm parameters independent of infrastructure and persistence."""

    version: str = "1"
    prompt_version: str = "skill-grpo-replay-v1"
    experience: ExperienceConfig = Field(default_factory=ExperienceConfig)
    patch: PatchConfig = Field(default_factory=PatchConfig)
    replay: ReplayBufferConfig = Field(default_factory=ReplayBufferConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)


class DatasetRuntimeConfig(_StrictModel):
    """Default component references used when a task does not override them."""

    env_ref: str = Field(min_length=1)
    agent_ref: str = Field(min_length=1)
    env_options: dict[str, JsonValue] = Field(default_factory=dict)
    agent_options: dict[str, JsonValue] = Field(default_factory=dict)


class SkillGrpoRunConfig(_StrictModel):
    """All serializable configuration required by one evolution run."""

    algorithm: SkillGrpoConfig = Field(default_factory=SkillGrpoConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    dataset: DatasetRuntimeConfig

    @model_validator(mode="after")
    def validate_queue(self) -> SkillGrpoRunConfig:
        if self.rollout.queue_capacity < self.rollout.max_concurrent_rollouts:
            raise ValueError("queue_capacity must be at least max_concurrent_rollouts")
        return self


__all__ = [
    "AblationConfig",
    "DatasetRuntimeConfig",
    "ExperienceConfig",
    "PatchConfig",
    "ReplayBufferConfig",
    "RetryConfig",
    "RolloutConfig",
    "RolloutStrategyConfig",
    "SkillGrpoConfig",
    "SkillGrpoRunConfig",
    "TrainingConfig",
    "ValidationConfig",
]
