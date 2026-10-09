"""Resolved local dependencies shared by both application configurations."""

from pydantic import BaseModel, Field


class DependencyPlan(BaseModel):
    """Collect containers and reachable application endpoints before startup."""

    services: set[str] = Field(default_factory=set)
    """Compose services required by enabled local application backends."""

    connections: set[tuple[str, int]] = Field(default_factory=set)
    """Local or external TCP endpoints checked before starting application processes."""
