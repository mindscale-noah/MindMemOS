"""Serializable contracts for the complete replay-buffer evolution run."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ....typing import EvolveInput, EvolveOutput, Skill, Task, Trajectory
from .config import SkillGrpoRunConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RolloutPhase(StrEnum):
    TRAIN = "train"
    ABLATION_BEFORE = "ablation_before"
    ABLATION_AFTER = "ablation_after"
    VALIDATION = "validation"
    TEST = "test"


class SkillTextEdit(_StrictModel):
    find: str
    replace: str


class RolloutSpec(_StrictModel):
    sequence_no: int = Field(ge=0)
    rollout_id: str = Field(min_length=1)
    phase: RolloutPhase
    task: Task
    skills: list[Skill]
    sample_index: int = Field(ge=0)
    candidate_id: str | None = None
    pair_id: str | None = None
    agent_ref: str = Field(min_length=1)
    env_ref: str = Field(min_length=1)
    seed: int | None = None
    temperature: float | None = None
    agent_options: dict[str, JsonValue] = Field(default_factory=dict)
    env_options: dict[str, JsonValue] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RolloutAttempt(_StrictModel):
    attempt_no: int = Field(ge=0)
    trajectory: Trajectory | None = None
    error_type: str | None = None
    error: str | None = None
    started_at: datetime
    finished_at: datetime


class RolloutOutcome(_StrictModel):
    spec: RolloutSpec
    attempts: list[RolloutAttempt]
    trajectory: Trajectory | None = None
    succeeded: bool = False


class ExtractedExperience(_StrictModel):
    task_id: str
    content: str
    rollout_count: int = Field(ge=1)


class EditSupport(_StrictModel):
    edit: SkillTextEdit
    supporting_experience_sets: list[int] = Field(default_factory=list)


class PatchProposalRecord(_StrictModel):
    raw_content: str
    proposed_edit_count: int = Field(ge=0)
    validation_errors: list[str] = Field(default_factory=list)
    edit_support: list[EditSupport] = Field(default_factory=list)
    attempts: int = Field(ge=1)


class ReplayEditRecord(_StrictModel):
    edit: SkillTextEdit
    batch_index: int = Field(ge=0)
    source_task_id: str
    advantage: float = 0.0
    committed: bool = False


class ReplayClusterState(_StrictModel):
    cluster_id: str
    centroid: list[float] = Field(default_factory=list)
    centroid_text: str
    records: list[ReplayEditRecord] = Field(default_factory=list)
    last_seen_batch: int = Field(ge=0)
    committed_replace: str | None = None
    uses: int = Field(default=0, ge=0)


class CandidateCaseResult(_StrictModel):
    task_id: str
    before: float
    after: float
    delta: float


class CandidateEvaluationRecord(_StrictModel):
    candidate_id: str
    cluster_id: str
    edit: SkillTextEdit
    source_task_id: str
    sampled_task_ids: list[str]
    net_effect: float
    per_case: list[CandidateCaseResult]
    chosen: bool = False
    rejection_reason: str | None = None


class BatchEvolutionRecord(_StrictModel):
    epoch: int = Field(ge=0)
    batch_index: int = Field(ge=0)
    task_ids: list[str]
    skill_hash_before: str
    skill_hash_after: str
    experiences: list[ExtractedExperience]
    patch: PatchProposalRecord | None = None
    candidates: list[CandidateEvaluationRecord] = Field(default_factory=list)
    applied_edits: list[SkillTextEdit] = Field(default_factory=list)
    train_score: float | None = None
    validation_score: float | None = None


class ProcessArtifact(_StrictModel):
    name: str
    media_type: str = "application/json"
    content: JsonValue | None = None
    uri: str | None = None
    checksum: str | None = None


class EvolutionMetrics(_StrictModel):
    train_score_mean: float | None = None
    validation_score_mean: float | None = None
    test_score_mean: float | None = None
    batches_completed: int = Field(default=0, ge=0)
    rollouts_completed: int = Field(default=0, ge=0)
    rollouts_failed: int = Field(default=0, ge=0)
    candidates_evaluated: int = Field(default=0, ge=0)
    edits_applied: int = Field(default=0, ge=0)


class EvolutionState(_StrictModel):
    schema_version: str = "2"
    algorithm_name: str = "skill_grpo_with_replay_buffer"
    algorithm_version: str
    run_id: str
    input_fingerprint: str
    config_fingerprint: str
    base_skill_hash: str
    current_skill: Skill
    completed_batch_index: int = -1
    rollout_sequence: int = Field(default=0, ge=0)
    ablation_sample_counter: int = Field(default=0, ge=0)
    ablation_rng_state: list[JsonValue] | None = None
    ablation_rollout_index: int = Field(default=1_000_000, ge=0)
    embedding_model_identity: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=1)
    replay_clusters: list[ReplayClusterState] = Field(default_factory=list)
    completed_rollout_ids: list[str] = Field(default_factory=list)
    rollout_outcomes: list[RolloutOutcome] = Field(default_factory=list)
    final_test_completed: bool = False
    batches: list[BatchEvolutionRecord] = Field(default_factory=list)
    metrics: EvolutionMetrics = Field(default_factory=EvolutionMetrics)


class SkillGrpoEvolveInput(EvolveInput):
    validation_tasks: list[Task] = Field(default_factory=list)
    test_tasks: list[Task] = Field(default_factory=list)
    config: SkillGrpoRunConfig
    resume_state: EvolutionState | None = None


class SkillGrpoEvolveResult(EvolveOutput):
    metrics: EvolutionMetrics
    state: EvolutionState
    batches: list[BatchEvolutionRecord]
    rollouts: list[RolloutOutcome]
    candidates: list[CandidateEvaluationRecord]
    artifacts: list[ProcessArtifact] = Field(default_factory=list)


class EvolutionEvent(_StrictModel):
    run_id: str
    name: str
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "BatchEvolutionRecord",
    "CandidateCaseResult",
    "CandidateEvaluationRecord",
    "EditSupport",
    "EvolutionEvent",
    "EvolutionMetrics",
    "EvolutionState",
    "ExtractedExperience",
    "PatchProposalRecord",
    "ProcessArtifact",
    "ReplayClusterState",
    "ReplayEditRecord",
    "RolloutAttempt",
    "RolloutOutcome",
    "RolloutPhase",
    "RolloutSpec",
    "SkillGrpoEvolveInput",
    "SkillGrpoEvolveResult",
    "SkillTextEdit",
]
