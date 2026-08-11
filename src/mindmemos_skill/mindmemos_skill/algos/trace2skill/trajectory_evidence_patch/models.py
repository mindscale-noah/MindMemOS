"""Audit models emitted by ``trajectory_evidence_patch``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ....typing import Trace2SkillOutput
from ..contracts import AnnotationMode


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrajectorySummary(_StrictModel):
    """LLM-produced analytical summary tied to one source trajectory."""

    trajectory_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    score: float | None = None
    annotation_detail: str | None = None
    annotation_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TrajectoryEvidencePatchReport(_StrictModel):
    """Compact audit report for one optimization request."""

    algorithm_name: str = "trajectory_evidence_patch"
    run_id: str = Field(min_length=1)
    algorithm_version: str
    prompt_version: str
    annotation_mode: AnnotationMode
    collection_run_id: str | None = None
    input_task_ids: list[str] = Field(default_factory=list)
    requested_collection_rollout_ids: list[str] = Field(default_factory=list)
    failed_collection_rollout_ids: list[str] = Field(default_factory=list)
    input_trajectory_ids: list[str]
    used_trajectory_ids: list[str] = Field(default_factory=list)
    duplicate_trajectory_ids: list[str] = Field(default_factory=list)
    failed_summary_trajectory_ids: list[str] = Field(default_factory=list)
    summaries: list[TrajectorySummary] = Field(default_factory=list)
    patch_plan: str | None = None
    changed: bool = False
    reason: str | None = None


class TrajectoryEvidencePatchOutput(Trace2SkillOutput[TrajectoryEvidencePatchReport]):
    """Typed output of one trajectory evidence patch transaction."""

    @model_validator(mode="after")
    def validate_report_outcome(self) -> TrajectoryEvidencePatchOutput:
        if self.report.changed != self.changed:
            raise ValueError("trajectory evidence patch report and candidate disagree on changed state")
        return self


__all__ = ["TrajectoryEvidencePatchOutput", "TrajectoryEvidencePatchReport", "TrajectorySummary"]
