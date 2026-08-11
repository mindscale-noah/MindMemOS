"""Serializable contracts for replay-free Skill GRPO evolution."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ....typing import EvolveInput, EvolveOutput, Task
from ..skill_grpo_with_replay_buffer.contracts import (
    ExtractedExperience,
    PatchProposalRecord,
    RolloutOutcome,
    SkillTextEdit,
)
from .config import SkillGrpoWithoutReplayBufferRunConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ValidationDecision(StrEnum):
    DISABLED = "disabled"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NO_CANDIDATE = "no_candidate"
    SCORE_UNAVAILABLE = "score_unavailable"


class ExperienceSource(StrEnum):
    """Evidence source and deterministic patch-priority order."""

    CONTRAST = "contrast"
    FAILURE = "failure"
    SUCCESS = "success"


class ReplayFreeExtractedExperience(ExtractedExperience):
    source: ExperienceSource
    task_ids: list[str] = Field(min_length=1)


class BatchEvolutionRecord(_StrictModel):
    epoch: int = Field(ge=0)
    batch_index: int = Field(ge=0)
    task_ids: list[str]
    skill_hash_before: str
    candidate_skill_hash: str | None = None
    skill_hash_after: str
    experiences: list[ReplayFreeExtractedExperience]
    patch: PatchProposalRecord | None = None
    candidate_edits: list[SkillTextEdit] = Field(default_factory=list)
    applied_edits: list[SkillTextEdit] = Field(default_factory=list)
    train_score: float | None = None
    validation_score_before: float | None = None
    validation_score_after: float | None = None
    validation_decision: ValidationDecision


class EvolutionMetrics(_StrictModel):
    train_score_mean: float | None = None
    validation_score: float | None = None
    test_score: float | None = None
    batches_completed: int = Field(default=0, ge=0)
    batches_accepted: int = Field(default=0, ge=0)
    batches_rejected: int = Field(default=0, ge=0)
    rollouts_completed: int = Field(default=0, ge=0)
    rollouts_failed: int = Field(default=0, ge=0)
    edits_applied: int = Field(default=0, ge=0)


class SkillGrpoWithoutReplayBufferEvolveInput(EvolveInput):
    validation_tasks: list[Task] = Field(default_factory=list)
    test_tasks: list[Task] = Field(default_factory=list)
    config: SkillGrpoWithoutReplayBufferRunConfig


class SkillGrpoWithoutReplayBufferEvolveResult(EvolveOutput):
    metrics: EvolutionMetrics
    batches: list[BatchEvolutionRecord]
    rollouts: list[RolloutOutcome]


__all__ = [
    "BatchEvolutionRecord",
    "EvolutionMetrics",
    "ExperienceSource",
    "ReplayFreeExtractedExperience",
    "SkillGrpoWithoutReplayBufferEvolveInput",
    "SkillGrpoWithoutReplayBufferEvolveResult",
    "ValidationDecision",
]
