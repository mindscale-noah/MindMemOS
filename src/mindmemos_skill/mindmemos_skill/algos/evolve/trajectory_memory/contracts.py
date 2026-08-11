"""Serializable contracts for trajectory-memory construction and evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ....typing import EvolveInput, EvolveOutput, Task
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from .config import TrajectoryMemoryRunConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TrajectorySnapshot(_StrictModel):
    """Portable trajectory evidence accepted from a live or historical run."""

    task: Task
    rollout_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    events: list[dict[str, Any]] = Field(default_factory=list)
    reward_score: float | None = None
    n_turn: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectorySummary(_StrictModel):
    title: str = Field(min_length=1)
    task_summary: str = Field(min_length=1)
    strategy: str = Field(min_length=1)
    key_steps: list[str] = Field(default_factory=list)
    transferable_lessons: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)


class TrajectoryMemoryItem(_StrictModel):
    memory_id: str = Field(min_length=1)
    source_task_id: str = Field(min_length=1)
    source_rollout_id: str = Field(min_length=1)
    retrieval_key: str = Field(min_length=1)
    reward_score: float | None = None
    n_turn: int = Field(ge=0)
    summary: TrajectorySummary
    embedding: list[float] = Field(default_factory=list)


class RetrievedTrajectoryMemory(_StrictModel):
    rank: int = Field(ge=1)
    similarity: float = Field(ge=-1.0, le=1.0)
    item: TrajectoryMemoryItem


class TaskRetrievalRecord(_StrictModel):
    task_id: str
    retrieval_key: str
    memories: list[RetrievedTrajectoryMemory]


class PairedEvaluationMetrics(_StrictModel):
    task_count: int = Field(default=0, ge=0)
    baseline_score: float | None = None
    memory_score: float | None = None
    delta: float | None = None
    improved: int = Field(default=0, ge=0)
    regressed: int = Field(default=0, ge=0)
    unchanged: int = Field(default=0, ge=0)


class TrajectoryMemoryEvolveInput(EvolveInput):
    train_tasks: list[Task] = Field(default_factory=list)
    test_tasks: list[Task] = Field(default_factory=list)
    config: TrajectoryMemoryRunConfig
    precollected_train_trajectories: list[TrajectorySnapshot] = Field(default_factory=list)


class TrajectoryMemoryEvolveResult(EvolveOutput):
    changed: bool = False
    memory_bank: list[TrajectoryMemoryItem]
    retrievals: list[TaskRetrievalRecord]
    train_rollouts: list[RolloutOutcome] = Field(default_factory=list)
    baseline_rollouts: list[RolloutOutcome] = Field(default_factory=list)
    memory_rollouts: list[RolloutOutcome] = Field(default_factory=list)
    metrics: PairedEvaluationMetrics


__all__ = [
    "PairedEvaluationMetrics",
    "RetrievedTrajectoryMemory",
    "TaskRetrievalRecord",
    "TrajectoryMemoryEvolveInput",
    "TrajectoryMemoryEvolveResult",
    "TrajectoryMemoryItem",
    "TrajectorySnapshot",
    "TrajectorySummary",
]
