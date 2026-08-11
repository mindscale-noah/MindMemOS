"""Serializable contracts for experience-validated Skill GRPO."""

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
from .config import SkillGrpoWithExperienceValidationRunConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExperienceSource(StrEnum):
    CONTRAST = "contrast"
    FAILURE = "failure"
    SUCCESS = "success"


class ExtractedExperienceSet(ExtractedExperience):
    source: ExperienceSource
    task_ids: list[str] = Field(min_length=1)


class ExperienceValidationDecision(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class ExperienceValidationRecord(_StrictModel):
    experience_index: int = Field(ge=0)
    source: ExperienceSource
    task_ids: list[str] = Field(min_length=1)
    baseline_success_rate: float = Field(ge=0.0, le=1.0)
    injected_success_rate: float = Field(ge=0.0, le=1.0)
    baseline_first_success_attempt: int | None = Field(default=None, ge=1)
    injected_first_success_attempt: int | None = Field(default=None, ge=1)
    decision: ExperienceValidationDecision
    reason: str
    rollouts: list[RolloutOutcome]


class PatchDecision(StrEnum):
    APPLIED = "applied"
    NO_ACCEPTED_EXPERIENCE = "no_accepted_experience"
    NO_CANDIDATE = "no_candidate"


class BatchEvolutionRecord(_StrictModel):
    epoch: int = Field(ge=0)
    batch_index: int = Field(ge=0)
    task_ids: list[str]
    skill_hash_before: str
    candidate_skill_hash: str | None = None
    skill_hash_after: str
    experiences: list[ExtractedExperienceSet]
    experience_validations: list[ExperienceValidationRecord]
    accepted_experiences: list[ExtractedExperienceSet]
    patch: PatchProposalRecord | None = None
    candidate_edits: list[SkillTextEdit] = Field(default_factory=list)
    applied_edits: list[SkillTextEdit] = Field(default_factory=list)
    train_score: float | None = None
    patch_decision: PatchDecision


class EvolutionMetrics(_StrictModel):
    train_score_mean: float | None = None
    test_score: float | None = None
    batches_completed: int = Field(default=0, ge=0)
    rollouts_completed: int = Field(default=0, ge=0)
    rollouts_failed: int = Field(default=0, ge=0)
    experiences_extracted: int = Field(default=0, ge=0)
    experiences_accepted: int = Field(default=0, ge=0)
    experience_validation_rollouts: int = Field(default=0, ge=0)
    edits_applied: int = Field(default=0, ge=0)


class SkillGrpoWithExperienceValidationEvolveInput(EvolveInput):
    test_tasks: list[Task] = Field(default_factory=list)
    config: SkillGrpoWithExperienceValidationRunConfig


class SkillGrpoWithExperienceValidationEvolveResult(EvolveOutput):
    metrics: EvolutionMetrics
    batches: list[BatchEvolutionRecord]
    rollouts: list[RolloutOutcome]


__all__ = [
    "BatchEvolutionRecord",
    "EvolutionMetrics",
    "ExperienceSource",
    "ExperienceValidationDecision",
    "ExperienceValidationRecord",
    "ExtractedExperienceSet",
    "PatchDecision",
    "SkillGrpoWithExperienceValidationEvolveInput",
    "SkillGrpoWithExperienceValidationEvolveResult",
]
