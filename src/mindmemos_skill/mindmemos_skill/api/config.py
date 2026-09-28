"""Explicit service storage and self-contained pipeline definitions."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..persistence.config import SkillPersistenceConfig

Name = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]


class ImplementationConfig(BaseModel):
    """Select an implementation with its complete, locally nested configuration."""

    model_config = ConfigDict(extra="forbid")
    """Reject assembly fields outside the explicit configuration contract."""

    class_name: Name
    """Registered implementation name in the family determined by the owning field."""

    config: dict[str, Any] = Field(default_factory=dict)
    """Implementation-specific options, including any locally owned component definitions."""


class PersistenceConfig(BaseModel):
    """Configure the service's sole Skill persistence and its database."""

    model_config = ConfigDict(extra="forbid")
    """Reject named stores and arbitrary dependency references."""

    config: SkillPersistenceConfig = Field(default_factory=SkillPersistenceConfig)
    """Business storage options for the single SkillPersistence instance."""

    database: ImplementationConfig
    """Database implementation and connection settings owned by this service."""


class ServiceConfig(BaseModel):
    """Define one persistence and complete pipeline configurations at their call sites."""

    model_config = ConfigDict(extra="forbid")
    """Reject obsolete backend graphs and unknown service fields."""

    persistence: PersistenceConfig | None = None
    """Single global store; omission leaves storage unavailable."""

    repository_pipeline: ImplementationConfig | None = None
    """Explicit repository definition; omission disables repository operations."""

    training_pipelines: dict[Name, ImplementationConfig] = Field(default_factory=dict)
    """Request names mapped directly to complete training definitions."""

    evolution_pipelines: dict[Name, ImplementationConfig] = Field(default_factory=dict)
    """Request names mapped directly to complete evolution definitions."""


class LocalServiceConfig(BaseModel):
    """Persist local identity independently of machine name and location."""

    model_config = ConfigDict(extra="forbid")
    """Reject invalid workspace metadata instead of silently resetting it."""

    project_id: Name
    """Internal local scope, unrelated to authenticated cloud projects."""

    database_path: Name = "data/central.db"
    """SQLite path, relative to the workspace home unless absolute."""
