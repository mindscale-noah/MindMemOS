"""Configuration for cache-first Skill and trajectory analysis."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from ...config import PipelineComponentConfig


class AnalyzePipelineConfig(BaseModel):
    """Select zero or more registered analyzers used by the service."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported cache and analyzer options."""

    skill_judge: PipelineComponentConfig | None = None
    """Optional static Skill Judge implementation and options."""

    trajectory_judge: PipelineComponentConfig | None = None
    """Optional trajectory Judge implementation and options."""

    evaluation_query_limit: int = Field(default=200, ge=1, le=1000)
    """Maximum evaluation rows returned by one recommendation post-process."""

    @model_validator(mode="after")
    def validate_analyzers(self, info: ValidationInfo) -> Self:
        """Validate selected analyzers while preserving pure algorithm construction."""

        from ....algos import Algorithm
        from ....algos.analyze import SkillAnalyzer, SkillJudge, TrajectoryJudge

        if self.skill_judge is not None:
            selected = Algorithm.resolve(self.skill_judge.class_name)
            if not issubclass(selected, SkillAnalyzer) or selected is TrajectoryJudge:
                raise ValueError("skill_judge must select a skill analyzer")
            if not issubclass(selected, SkillJudge):
                raise ValueError("skill_judge must select the static Skill Judge family")
            self.skill_judge.config = selected.config_type.model_validate(self.skill_judge.config, context=info.context)
        if self.trajectory_judge is not None:
            selected = Algorithm.resolve(self.trajectory_judge.class_name)
            if not issubclass(selected, SkillAnalyzer) or selected is SkillJudge:
                raise ValueError("trajectory_judge must select a trajectory analyzer")
            if not issubclass(selected, TrajectoryJudge):
                raise ValueError("trajectory_judge must select the trajectory Judge family")
            self.trajectory_judge.config = selected.config_type.model_validate(
                self.trajectory_judge.config, context=info.context
            )
        return self


__all__ = ["AnalyzePipelineConfig"]
