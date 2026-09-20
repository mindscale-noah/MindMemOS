"""Transport-independent configuration for service assembly."""

from graphlib import TopologicalSorter
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Name = Annotated[str, StringConstraints(min_length=1, pattern=r"^\S(?:.*\S)?$")]


class BackendConfig(BaseModel):
    """Select a registered implementation and its constructor inputs."""

    model_config = ConfigDict(extra="forbid")
    """Reject misspelled assembly fields."""

    module: Literal["infra", "algos", "agents", "logging", "persistence", "envs", "pipelines"]
    """Module family whose registry contains the implementation."""

    class_name: Name
    """Exact name provided to the registration decorator."""

    config: dict[str, Any] = Field(default_factory=dict)
    """Implementation-specific fields validated by its config_type model."""

    dependencies: dict[Name, Name] = Field(default_factory=dict)
    """Constructor keyword names mapped to other backend instance names."""

    @model_validator(mode="after")
    def validate_dependencies(self) -> Self:
        """Require usable constructor keywords and reserve the config argument."""
        for name in self.dependencies:
            if not name.isidentifier() or name == "config":
                raise ValueError(f"Invalid dependency argument: {name!r}")
        return self


class ServiceConfig(BaseModel):
    """Describe the named backends assembled into a Skill service."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown service configuration fields."""

    backends: dict[Name, BackendConfig] = Field(default_factory=dict)
    """Unique instance names mapped to backend construction specifications."""

    persistence_backend: Name | None = None
    """Default repository persistence when no repository pipeline is explicitly selected."""

    repository_pipeline: Name | None = None
    """Repository pipeline instance; inferred when exactly one exists."""

    training_pipeline: Name | None = None
    """Training pipeline instance; required to choose between multiple implementations."""

    evolution_pipeline: Name | None = None
    """Evolution pipeline instance; required to choose between multiple implementations."""

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        """Reject missing dependency references and cyclic construction graphs."""
        for selected in (self.repository_pipeline, self.training_pipeline, self.evolution_pipeline):
            if selected is not None and selected not in self.backends:
                raise ValueError(f"Pipeline references missing backend {selected!r}")
        for name, backend in self.backends.items():
            for dependency in backend.dependencies.values():
                if dependency not in self.backends:
                    raise ValueError(f"Backend {name!r} references missing backend {dependency!r}")
        graph = {name: set(spec.dependencies.values()) for name, spec in self.backends.items()}
        tuple(TopologicalSorter(graph).static_order())
        return self


class LocalServiceConfig(BaseModel):
    """Persist local identity independently of machine name and location."""

    model_config = ConfigDict(extra="forbid")
    """Reject invalid workspace metadata instead of silently resetting it."""

    project_id: Name
    """Internal local scope, unrelated to authenticated cloud projects."""

    database_path: Name = "data/central.db"
    """SQLite path, relative to the workspace home unless absolute."""
