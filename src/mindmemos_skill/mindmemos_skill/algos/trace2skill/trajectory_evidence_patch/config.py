"""Typed configuration for ``trajectory_evidence_patch``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..collection import TaskCollectionConfig
from ..contracts import AnnotationMode


class TrajectoryEvidencePatchConfig(BaseModel):
    """Settings for one bounded trajectory evidence-to-patch transaction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm_version: str = Field(default="1", min_length=1)
    prompt_version: str = Field(default="trajectory-evidence-patch-v1", min_length=1)
    min_trajectories: int = Field(default=8, ge=1)
    max_trajectories: int = Field(default=8, ge=1)
    summary_concurrency: int = Field(default=8, ge=1)
    transcript_max_chars: int = Field(default=1500, ge=1)
    annotation_mode: AnnotationMode = AnnotationMode.AUTO
    require_skill_match: bool = True
    collection: TaskCollectionConfig | None = None
    rewrite_skill: bool = False
    summary_task: str = Field(default="trajectory_evidence_summary", min_length=1)
    patch_task: str = Field(default="trajectory_evidence_patch_propose", min_length=1)
    apply_task: str = Field(default="trajectory_evidence_patch_apply", min_length=1)
    rewrite_task: str = Field(default="trajectory_evidence_patch_rewrite", min_length=1)

    @model_validator(mode="after")
    def validate_batch_bounds(self) -> TrajectoryEvidencePatchConfig:
        if self.max_trajectories < self.min_trajectories:
            raise ValueError("max_trajectories must be greater than or equal to min_trajectories")
        return self


__all__ = ["TrajectoryEvidencePatchConfig"]
