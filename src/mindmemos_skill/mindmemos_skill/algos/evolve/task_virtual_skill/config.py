"""Configuration for one-batch trajectory-grounded Skill decomposition."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..skill_grpo_with_replay_buffer.config import DatasetRuntimeConfig, RetryConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BatchConfig(_StrictModel):
    batch_size: int = Field(default=40, ge=1)
    rollouts_per_task: int = Field(default=1, ge=1)
    seed: int = 0


class RolloutConfig(_StrictModel):
    max_concurrent_rollouts: int = Field(default=16, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    fail_fast: bool = False
    workspace_root: str | None = None


class SummaryConfig(_StrictModel):
    sample_size: int = Field(default=20, ge=1)
    max_concurrent_summaries: int = Field(default=8, ge=1)
    transcript_max_chars: int = Field(default=24_000, ge=1)


class DecompositionConfig(_StrictModel):
    max_virtual_skills: int = Field(default=12, ge=1, le=64)
    max_initial_components: int = Field(default=3, ge=0)


class RefinementConfig(_StrictModel):
    retry_rounds: int = Field(default=1, ge=1)
    success_reward: float = 1.0
    max_concurrent_reflections: int = Field(default=8, ge=1)
    max_trajectory_chars: int = Field(default=24_000, ge=1)
    max_previous_answer_chars: int = Field(default=8_000, ge=1)
    max_reflection_chars: int = Field(default=4_000, ge=1)
    max_concurrent_changes: int = Field(default=8, ge=1)
    max_concurrent_merges: int = Field(default=4, ge=1)


class TaskVirtualSkillRunConfig(_StrictModel):
    version: str = "1"
    prompt_version: str = "task-virtual-skill-trajectory-grounded-v1"
    batch: BatchConfig = Field(default_factory=BatchConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    dataset: DatasetRuntimeConfig


__all__ = [
    "BatchConfig",
    "DecompositionConfig",
    "RolloutConfig",
    "RefinementConfig",
    "SummaryConfig",
    "TaskVirtualSkillRunConfig",
]
