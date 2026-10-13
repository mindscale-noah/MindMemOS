"""Explicit service storage and self-contained pipeline definitions."""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    SecretStr,
    SerializeAsAny,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from ..agents.config import AgentDefinition, AgentName
from ..persistence.config import SkillPersistenceConfig
from ..registry.config import serialize_options

Name = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]


class HttpAuthConfig(BaseModel):
    """Select the identity source for a FastAPI deployment."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown authentication settings."""

    mode: Literal["none", "local", "gateway_jwt"]
    """Credential source; none is valid only for loopback HTTP listening."""

    default_project_id: Name | None = None
    """Required project for unauthenticated requests; forbidden in authenticated modes."""

    api_key_file: str | None = None
    """Private YAML file containing one project ID and API key for local mode."""

    gateway_jwt_secret: SecretStr | None = None
    """Shared HMAC secret used to verify Gateway-issued JWTs."""

    gateway_jwt_issuer: Name = "mindmemos-gateway"
    """Expected issuer of Gateway-issued JWTs."""

    gateway_jwt_audience: Name = "memory-data-plane"
    """Expected audience of Gateway-issued JWTs."""

    @model_validator(mode="after")
    def validate_mode(self) -> "HttpAuthConfig":
        """Require only the fields used by the selected authentication mode.

        Returns:
            Validated authentication settings.
        """
        if self.mode == "none" and self.default_project_id is None:
            raise ValueError("auth.default_project_id is required for none mode")
        if self.mode != "none" and self.default_project_id is not None:
            raise ValueError("auth.default_project_id is only valid for none mode")
        if self.mode == "local" and not self.api_key_file:
            raise ValueError("auth.api_key_file is required for local mode")
        if self.mode == "gateway_jwt" and (
            self.gateway_jwt_secret is None or not self.gateway_jwt_secret.get_secret_value()
        ):
            raise ValueError("auth.gateway_jwt_secret is required for gateway_jwt mode")
        if self.mode != "local" and self.api_key_file is not None:
            raise ValueError("auth.api_key_file is only valid for local mode")
        if self.mode != "gateway_jwt" and self.gateway_jwt_secret is not None:
            raise ValueError("auth.gateway_jwt_secret is only valid for gateway_jwt mode")
        return self


class FastAPIConfig(BaseModel):
    """Configure authentication and HTTP pipeline policy for one deployment."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown HTTP settings."""

    auth: HttpAuthConfig
    """Credential source for protected Skill routes."""

    evolution_pipelines: set[Name] = Field(default_factory=set)
    """Evolution pipeline names permitted through the HTTP endpoint."""


ServiceName = Literal["training",]


class ImplementationConfig(BaseModel):
    """Select an implementation with its complete, locally nested configuration."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject assembly fields outside the explicit configuration contract."""

    class_name: Name
    """Registered implementation name in the family determined by the owning field."""

    config: dict[str, Any] | SerializeAsAny[InstanceOf[BaseModel]] = Field(default_factory=dict)
    """Implementation-owned model after preflight, or unparsed options before validation."""

    serialize_config = field_serializer("config")(serialize_options)
    """Preserve implementation-owned fields and serializers in nested selections."""


class PersistenceConfig(BaseModel):
    """Configure the service's sole Skill persistence and its database."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject named stores and arbitrary dependency references."""

    config: SkillPersistenceConfig = Field(default_factory=SkillPersistenceConfig)
    """Business storage options for the single SkillPersistence instance."""

    database: ImplementationConfig
    """Database implementation and connection settings owned by this service."""


class OTLPConfig(BaseModel):
    """Send algorithm logs and spans to an OTLP/HTTP collector."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown exporter settings."""

    endpoint: AnyHttpUrl
    """Collector base URL; the exporter appends /v1/logs or /v1/traces."""

    timeout: float = Field(default=5.0, gt=0)
    """Timeout in seconds for each OTLP/HTTP export request."""


class ObservabilityConfig(BaseModel):
    """Configure optional SQL audit storage and OTLP export for algorithm runs."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
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


class TaskManagerConfig(BaseModel):
    """Select process-local execution or durable Kafka-backed Skill work."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject undeclared queue settings."""

    mode: Literal["local", "kafka"] = "local"
    """Execution backend; local tasks disappear when the process exits."""

    bootstrap_servers: Name | None = None
    """Shared Kafka cluster address required for cloud execution."""

    topic: Name = "skill.train.jobs"
    """Dedicated Skill notification topic, separate from Memory topics."""

    group_id: Name = "mindmemos-skill-workers"
    """Dedicated Skill consumer group."""

    worker: bool = False
    """Whether this process consumes and executes queued tasks."""

    max_concurrency: int = Field(default=2, ge=1)
    """Maximum simultaneous task executions per manager in local or Kafka mode."""

    max_buffered: int = Field(default=1000, ge=1)
    """Maximum queued and running tasks admitted to the local dispatcher."""

    scan_interval_seconds: float = Field(default=5.0, gt=0)
    """Seconds between database recovery scans."""

    @model_validator(mode="after")
    def require_kafka_address(self) -> "TaskManagerConfig":
        """Require an address when Kafka execution is selected."""
        if self.mode == "kafka" and self.bootstrap_servers is None:
            raise ValueError("Kafka tasks require bootstrap_servers")
        return self


class ServiceConfig(BaseModel):
    """Define shared storage and configured service pipelines."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject obsolete backend graphs and unknown service fields."""

    disabled_services: set[ServiceName] = Field(default_factory=set)
    """Business capabilities omitted while assembling this service."""

    home: Path
    """Service data root; YAML-relative paths resolve against the configuration directory."""

    agents: dict[AgentName, AgentDefinition] = Field(default_factory=dict)
    """Algorithm-scoped Agent presets named <algo>_<llm_name>, resolved before constructing pipelines."""

    persistence: PersistenceConfig | None = None
    """Single global store; omission leaves storage unavailable."""

    task_manager: TaskManagerConfig | None = None
    """Optional asynchronous Skill execution; omitted disables submission."""

    observability: ObservabilityConfig | None = None
    """Optional service-wide trace and algorithm audit backend."""

    repository_pipeline: ImplementationConfig | None = None
    """Explicit repository definition; omission disables repository operations."""

    manage_pipeline: ImplementationConfig | None = None
    """Optional Skill Set query, graph publication, and recommendation definition."""

    training_pipelines: dict[Name, ImplementationConfig] = Field(default_factory=dict)
    """Request names mapped to training algorithms."""

    evolution_pipelines: dict[Name, ImplementationConfig] = Field(default_factory=dict)
    """Request names mapped directly to complete evolution definitions."""

    @field_validator("home", mode="before")
    @classmethod
    def validate_home(cls, value: object) -> object:
        """Reject empty data-root paths instead of selecting the current directory.

        Args:
            value: Explicit service data-root value before path conversion.

        Returns:
            The nonempty path value for normal Pydantic validation.

        Raises:
            ValueError: The supplied string is empty or whitespace.
        """
        if isinstance(value, str) and not value.strip():
            raise ValueError("service.home must not be empty")
        return value


class SkillConfig(BaseModel):
    """Group service resources and required authentication policy in one YAML document."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown top-level configuration sections."""

    service: ServiceConfig
    """Business resources and disabled capabilities shared by all entry points."""

    fastapi: FastAPIConfig
    """HTTP authentication and pipeline policy used only by the FastAPI entry."""
