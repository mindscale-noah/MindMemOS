"""Repository pipeline configuration."""

from pydantic import BaseModel, ConfigDict


class RepositoryPipelineConfig(BaseModel):
    """Repository operations currently need only their injected persistence."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported repository options."""
