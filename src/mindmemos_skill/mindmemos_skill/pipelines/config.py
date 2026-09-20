"""Configuration for built-in repository pipelines."""

from pydantic import BaseModel, ConfigDict


class RepositoryPipelineConfig(BaseModel):
    """Repository operations currently need only their injected persistence."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported repository options."""
