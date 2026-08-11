"""Configuration for retrieval-augmented trajectory memory evolution."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..skill_grpo_with_replay_buffer.config import DatasetRuntimeConfig, RetryConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrajectoryMemoryAlgorithmConfig(_StrictModel):
    """Memory construction and retrieval settings."""

    version: str = "1"
    prompt_version: str = "trajectory-memory-v1"
    top_k: int = Field(default=3, ge=1)
    success_reward: float = 1.0
    train_rollouts_per_task: int = Field(default=1, ge=1)
    test_rollouts_per_task: int = Field(default=1, ge=1)
    max_examples_per_task: int = Field(default=1, ge=1)
    max_trajectory_chars: int = Field(default=24_000, ge=1)
    max_summary_chars: int = Field(default=2_000, ge=1)
    max_concurrent_summaries: int = Field(default=8, ge=1)
    run_baseline: bool = True


class TrajectoryMemoryRolloutConfig(_StrictModel):
    """Shared rollout budget for train, baseline, and memory-assisted phases."""

    max_concurrent_rollouts: int = Field(default=16, ge=1)
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    fail_fast: bool = False
    workspace_root: Path | None = None


class TrajectoryMemoryRunConfig(_StrictModel):
    algorithm: TrajectoryMemoryAlgorithmConfig = Field(default_factory=TrajectoryMemoryAlgorithmConfig)
    rollout: TrajectoryMemoryRolloutConfig = Field(default_factory=TrajectoryMemoryRolloutConfig)
    dataset: DatasetRuntimeConfig


__all__ = [
    "TrajectoryMemoryAlgorithmConfig",
    "TrajectoryMemoryRolloutConfig",
    "TrajectoryMemoryRunConfig",
]
