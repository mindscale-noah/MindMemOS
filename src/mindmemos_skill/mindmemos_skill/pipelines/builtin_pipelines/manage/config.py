"""Skill management pipeline configuration."""

from pydantic import BaseModel, ConfigDict

from ....algos.manage.skill_manage import SkillManageConfig


class GraphManagePipelineConfig(BaseModel):
    """Configure graph publication and recommendation when requested."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject settings outside the algorithm contract."""

    algorithm: SkillManageConfig | None = None
    """Graph algorithm settings; None enables only Skill Set queries."""

    prefixed_graph_ids: bool = True
    """Generate prefixed graph storage IDs; disable while legacy writers remain active."""
