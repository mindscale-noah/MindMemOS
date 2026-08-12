"""Strict configuration for TreeSkill Evolution."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..collection import TaskCollectionConfig
from ..contracts import AnnotationMode


class TreeSkillConfig(BaseModel):
    """Reproducible settings for one bounded TreeSkill evolution run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm_version: str = Field(default="1", min_length=1)
    prompt_version: str = Field(default="treeskill-v1", min_length=1)
    min_trajectories: int = Field(default=1, ge=1)
    max_trajectories: int = Field(default=1000, ge=1)
    transcript_max_chars: int = Field(default=20000, ge=1)
    annotation_mode: AnnotationMode = AnnotationMode.REQUIRED
    require_skill_match: bool = True
    success_score_threshold: float = 1.0
    analysis_concurrency: int = Field(default=16, ge=1)
    localization_concurrency: int = Field(default=16, ge=1)
    analysis_temperature: float = Field(default=1.0, ge=0)
    localization_temperature: float = Field(default=0.0, ge=0)
    fusion_temperature: float = Field(default=0.0, ge=0)
    analysis_max_tokens: int = Field(default=4096, ge=1)
    localization_max_tokens: int = Field(default=2048, ge=1)
    fusion_max_tokens: int = Field(default=4096, ge=1)
    collection: TaskCollectionConfig | None = None
    analysis_task: str = Field(default="treeskill_trajectory_analysis", min_length=1)
    localization_task: str = Field(default="treeskill_evidence_localization", min_length=1)
    fusion_task: str = Field(default="treeskill_node_fusion", min_length=1)

    @model_validator(mode="after")
    def validate_batch_bounds(self) -> TreeSkillConfig:
        if self.max_trajectories < self.min_trajectories:
            raise ValueError("max_trajectories must be greater than or equal to min_trajectories")
        return self


__all__ = ["TreeSkillConfig"]
