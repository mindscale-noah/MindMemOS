"""Application-facing contracts for running configured Skill algorithms."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...typing import SkillCandidate, Task


class AlgorithmCommitPolicy(StrEnum):
    """Persistence and remote-delivery policy for one algorithm run."""

    DRY_RUN = "dry_run"
    PERSIST = "persist"
    PERSIST_AND_PUSH = "persist_and_push"


class _RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    algorithm_name: str = Field(min_length=1)
    skill_ref: str = Field(min_length=1)
    base_version_id: str | None = Field(default=None, min_length=1)
    commit_policy: AlgorithmCommitPolicy = AlgorithmCommitPolicy.PERSIST


class Trace2SkillRunRequest(_RunRequest):
    """Resolve a base Skill and existing or newly collected trajectory evidence."""

    trajectory_ids: list[str] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_source(self) -> Trace2SkillRunRequest:
        if not self.trajectory_ids and not self.tasks:
            raise ValueError("trace2skill requires trajectory_ids, tasks, or both")
        if len(self.trajectory_ids) != len(set(self.trajectory_ids)):
            raise ValueError("trajectory_ids must be unique")
        return self


class EvolveRunRequest(_RunRequest):
    """Run a configured online evolution algorithm over explicit task splits."""

    train_tasks: list[Task] = Field(min_length=1)
    validation_tasks: list[Task] = Field(default_factory=list)
    test_tasks: list[Task] = Field(default_factory=list)


class SkillAlgorithmRunResult(BaseModel):
    """Stable application result independent of one algorithm's internal report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    algorithm_name: str
    base_version_id: str
    changed: bool
    candidate: SkillCandidate | None = None
    persisted_version_id: str | None = None
    input_trajectory_ids: list[str] = Field(default_factory=list)
    generated_trajectory_ids: list[str] = Field(default_factory=list)
    persisted_trajectory_ids: list[str] = Field(default_factory=list)
    algorithm_log_ids: list[str] = Field(default_factory=list)
    push_operation_id: str | None = None


__all__ = [
    "AlgorithmCommitPolicy",
    "EvolveRunRequest",
    "SkillAlgorithmRunResult",
    "Trace2SkillRunRequest",
]
