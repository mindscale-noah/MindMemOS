"""HTTP-specific pipeline availability settings."""

from pydantic import BaseModel, ConfigDict, Field

from ..config import Name


class FastAPIConfig(BaseModel):
    """Allow only named pipelines that this HTTP deployment can execute."""

    model_config = ConfigDict(extra="forbid")
    """Reject misspelled or unknown HTTP settings."""

    evolution_pipelines: set[Name] = Field(default_factory=set)
    """Evolution pipeline names permitted through the HTTP endpoint."""
