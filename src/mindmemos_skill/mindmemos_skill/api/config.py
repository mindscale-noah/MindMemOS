"""Explicit service storage and self-contained pipeline definitions."""

from typing import Annotated, Any, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, StringConstraints, model_validator

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


class OTLPConfig(BaseModel):
    """Send algorithm logs and spans to an OTLP/HTTP collector."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown exporter settings."""

    endpoint: AnyHttpUrl
    """Collector base URL; the exporter appends /v1/logs or /v1/traces."""

    timeout: float = Field(default=5.0, gt=0)
    """Timeout in seconds for each OTLP/HTTP export request."""


class ObservabilityConfig(BaseModel):
    """Configure optional SQL audit storage and OTLP export for algorithm runs."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported observability settings."""

    backend: ImplementationConfig | None = None
    """Optional SQLite or PostgreSQL database for run state and audit queries."""

    otlp: OTLPConfig | None = None
    """Optional OTLP/HTTP destination for logs and spans."""

    capture_level: Literal["info", "debug"] = "info"
    """Whether agent inputs and outputs are retained alongside call metadata."""

    service_name: Name = "mindmemos-skill"
    """OpenTelemetry service.name attached to emitted logs and spans."""

    @model_validator(mode="after")
    def require_sink(self) -> "ObservabilityConfig":
        """Reject an observability block without any output destination."""
        if self.backend is None and self.otlp is None:
            raise ValueError("observability requires backend or otlp")
        return self


class ServiceConfig(BaseModel):
    """Define one persistence and complete pipeline configurations at their call sites."""

    model_config = ConfigDict(extra="forbid")
    """Reject obsolete backend graphs and unknown service fields."""

    local_project_id: Name | None = None
    """Explicit project bound to all requests in a personal local deployment."""

    persistence: PersistenceConfig | None = None
    """Single global store; omission leaves storage unavailable."""

    observability: ObservabilityConfig | None = None
    """Optional service-wide trace and algorithm audit backend."""

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
