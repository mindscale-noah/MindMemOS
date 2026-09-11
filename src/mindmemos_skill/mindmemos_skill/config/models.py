"""Typed input and compiled configuration for one local Skill application."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr, model_validator

from ..infra.database import DatabaseRequirements
from ..infra.vector_store import BackendRequirements
from ..persistence.enums import SkillInjectionMode
from ..registry import ComponentSpec

SkillConfigSource: TypeAlias = str | Path | Mapping[str, Any] | BaseModel

_SECRET_FIELD_NAMES = {
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "secret",
    "token",
}
_DSN_FIELD_NAMES = {"connection_uri", "database_url", "dsn", "uri", "url"}


def _redact(value: Any, *, field_name: str | None = None) -> Any:
    """Redact secrets and whole DSN values for snapshots and reprs."""

    if isinstance(value, SecretStr):
        return "<redacted>"
    if field_name is not None and field_name.lower() in _SECRET_FIELD_NAMES | _DSN_FIELD_NAMES:
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(key): _redact(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class _StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SkillDatabaseConfig(_StrictConfig):
    """User-selectable database provider and connection options."""

    provider: str = Field(default="sqlite", min_length=1)
    path: Path | None = None
    options: dict[str, JsonValue] = Field(default_factory=dict)


class SkillVectorStoreConfig(_StrictConfig):
    """User-selectable vector backend provider and connection options."""

    provider: str = Field(default="pgvector", min_length=1)
    options: dict[str, JsonValue] = Field(default_factory=dict)


class SkillLocalConfig(_StrictConfig):
    """Local state and large-artifact locations owned by the Skill package."""

    root_dir: Path = Path("~/.mindmemos/skill")
    database: SkillDatabaseConfig = Field(default_factory=SkillDatabaseConfig)
    vector_store: SkillVectorStoreConfig = Field(default_factory=SkillVectorStoreConfig)
    artifacts_dir: Path | None = None


class SkillModelConfig(_StrictConfig):
    """One named LiteLLM model endpoint referenced by agents or algorithms."""

    model: str = Field(min_length=1)
    kind: Literal["chat", "embedding"] = "chat"
    dimension: int | None = Field(default=None, gt=0)
    api_base: str | None = Field(default=None, min_length=1)
    api_key: SecretStr | None = None
    temperature: float | None = Field(default=None, ge=0)
    options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _embedding_requires_dimension(self) -> SkillModelConfig:
        if self.kind == "embedding" and self.dimension is None:
            raise ValueError("embedding models must declare a positive integer dimension")
        return self


class SkillAgentConfig(_StrictConfig):
    """One named agent component and its cross-component references."""

    type: str = Field(min_length=1)
    model_ref: str | None = Field(default=None, min_length=1)
    skill_injection_mode: SkillInjectionMode | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)


class SkillAlgorithmConfig(_StrictConfig):
    """One named algorithm component and the models filling its roles."""

    type: str = Field(min_length=1)
    model_roles: dict[str, str] = Field(default_factory=dict)
    config: dict[str, JsonValue] = Field(default_factory=dict)


class SkillExecutionConfig(_StrictConfig):
    """Application-wide rollout scheduling and retry policy."""

    max_concurrent_rollouts: int = Field(default=8, ge=1)
    attempt_limit: int = Field(default=2, ge=1)
    rollout_timeout_seconds: float | None = Field(default=None, gt=0)


class SkillRuntimeConfig(_StrictConfig):
    """Models, agents, algorithms, and execution policy for one application."""

    models: dict[str, SkillModelConfig] = Field(default_factory=dict)
    agents: dict[str, SkillAgentConfig] = Field(default_factory=dict)
    algorithms: dict[str, SkillAlgorithmConfig] = Field(default_factory=dict)
    execution: SkillExecutionConfig = Field(default_factory=SkillExecutionConfig)


class SkillApplicationConfig(_StrictConfig):
    """Deployment-neutral user intent accepted by :class:`SkillConfigCompiler`."""

    local: SkillLocalConfig = Field(default_factory=SkillLocalConfig)
    runtime: SkillRuntimeConfig = Field(default_factory=SkillRuntimeConfig)


class _CompiledConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    def __repr__(self) -> str:
        safe = _redact(self.model_dump(mode="python"))
        return f"{type(self).__name__}({safe!r})"


class CompiledSkillDatabaseConfig(_CompiledConfig):
    provider: str
    options: dict[str, JsonValue]
    required: DatabaseRequirements


class CompiledSkillVectorStoreConfig(_CompiledConfig):
    provider: str
    options: dict[str, JsonValue]
    required: BackendRequirements


class CompiledSkillLocalConfig(_CompiledConfig):
    root_dir: Path
    database: CompiledSkillDatabaseConfig
    vector_store: CompiledSkillVectorStoreConfig
    artifacts_dir: Path


class CompiledSkillModelConfig(_CompiledConfig):
    """Validated model endpoint with the provider resolved by LiteLLM."""

    provider: str
    model: str
    kind: Literal["chat", "embedding"]
    dimension: int | None
    api_base: str | None
    api_key: SecretStr | None
    temperature: float | None
    options: dict[str, JsonValue]


class CompiledSkillAgentConfig(_CompiledConfig):
    name: str
    type: str
    model_ref: str | None
    component: ComponentSpec = Field(exclude=True)
    config: BaseModel = Field(exclude=True)


class CompiledSkillAlgorithmConfig(_CompiledConfig):
    name: str
    type: str
    model_roles: dict[str, str]
    component: ComponentSpec = Field(exclude=True)
    config: BaseModel = Field(exclude=True)


class CompiledSkillRuntimeConfig(_CompiledConfig):
    models: dict[str, CompiledSkillModelConfig]
    agents: dict[str, CompiledSkillAgentConfig]
    algorithms: dict[str, CompiledSkillAlgorithmConfig]
    execution: SkillExecutionConfig


class CompiledSkillApplicationConfig(_CompiledConfig):
    """Validated, normalized input consumed by ``SkillApplication.from_config``."""

    local: CompiledSkillLocalConfig
    runtime: CompiledSkillRuntimeConfig
    config_snapshot: dict[str, JsonValue]
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


__all__ = [
    "CompiledSkillAgentConfig",
    "CompiledSkillAlgorithmConfig",
    "CompiledSkillApplicationConfig",
    "CompiledSkillDatabaseConfig",
    "CompiledSkillLocalConfig",
    "CompiledSkillModelConfig",
    "CompiledSkillRuntimeConfig",
    "CompiledSkillVectorStoreConfig",
    "SkillAgentConfig",
    "SkillAlgorithmConfig",
    "SkillApplicationConfig",
    "SkillConfigSource",
    "SkillDatabaseConfig",
    "SkillExecutionConfig",
    "SkillLocalConfig",
    "SkillModelConfig",
    "SkillRuntimeConfig",
    "SkillVectorStoreConfig",
]
