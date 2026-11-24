"""Configuration for graph construction and pinned retrieval."""

from pydantic import BaseModel, ConfigDict, Field

from ....algos.manage.skill_manage.config import SkillManageConfig


class GraphManagePipelineConfig(BaseModel):
    """Configure the original Scene graph algorithm and project queries."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject obsolete graph identity and publication options."""

    algorithm: SkillManageConfig | None = None
    """Explicit model and algorithm settings; None enables only stored topology queries."""

    scene_concurrency: int = Field(default=4, ge=1)
    """Maximum concurrent Skill Scene cache reads, extractions and writes per request."""
