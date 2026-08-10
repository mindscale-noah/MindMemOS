"""Local SDK configuration schema.

These models mirror the on-disk ``~/.mindmemos/settings.json`` format described in
``docs/sdk/design.md``. The first version uses a single profile. ``skills`` is kept
as a permissive list so the skill-management feature can populate it later without a
schema migration here.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, Field

CONFIG_SCHEMA_VERSION = 1
DEFAULT_BASE_URL = "https://api.mindmemos.example.com"


class AuthConfigV1(BaseModel):
    """Authentication material used for API calls."""

    api_key: str | None = None


class DefaultsConfig(BaseModel):
    """SDK-wide actor identity defaults injected into resource requests."""

    user_id: str | None = None
    app_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None


class MemoryDefaultsConfig(BaseModel):
    """Persisted defaults for memory request builders."""

    search_top_k: int | None = 10
    search_strategy: Literal["fast", "agentic"] = "fast"
    search_rerank: bool = False
    search_score_threshold: float | None = None
    search_filters: dict[str, Any] = Field(default_factory=dict)
    add_mode: Literal["sync", "async"] = "sync"
    add_default_role: str = "user"
    add_auto_skill_context: bool = True
    get_top_k: int | None = None
    get_filters: dict[str, Any] = Field(default_factory=dict)
    feedback_mode: Literal["sync", "async"] | None = None
    dreaming_mode: Literal["sync", "async"] = "async"


class StorageConfigV1(BaseModel):
    """Local storage locations for skill cache and backups."""

    skill_cache_dir: str = "~/.mindmemos/skills/cache"
    skill_backup_dir: str = "~/.mindmemos/skills/backups"


class NetworkConfigV1(BaseModel):
    """Default HTTP transport tuning."""

    timeout_seconds: int = 30
    max_retries: int = 2


class HttpConnectionConfig(BaseModel):
    """One shared asynchronous HTTP connection."""

    type: Literal["http"] = "http"
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2


ConnectionConfig = HttpConnectionConfig


class ClientConnectionConfigV1(BaseModel):
    """Route one SDK resource client to a named connection."""

    connection: str = "default"


class ClientsConfigV1(BaseModel):
    """Per-resource connection routing."""

    memory: ClientConnectionConfigV1 = Field(default_factory=ClientConnectionConfigV1)
    skills: ClientConnectionConfigV1 = Field(default_factory=ClientConnectionConfigV1)


class ConfigMetadataV1(BaseModel):
    """Bookkeeping timestamps for the config file."""

    created_at: str | None = None
    updated_at: str | None = None


class SDKConfigV1(BaseModel):
    """Top-level SDK settings persisted to ``~/.mindmemos/settings.json``."""

    version: int = CONFIG_SCHEMA_VERSION
    base_url: str = DEFAULT_BASE_URL
    auth: AuthConfigV1 = Field(default_factory=AuthConfigV1)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    storage: StorageConfigV1 = Field(default_factory=StorageConfigV1)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    network: NetworkConfigV1 = Field(default_factory=NetworkConfigV1)
    memory: MemoryDefaultsConfig = Field(default_factory=MemoryDefaultsConfig)
    connections: dict[str, ConnectionConfig] = Field(default_factory=dict)
    clients: ClientsConfigV1 = Field(default_factory=ClientsConfigV1)
    metadata: ConfigMetadataV1 = Field(default_factory=ConfigMetadataV1)


# Public v1 compatibility aliases. Keep these as aliases instead of duplicate
# subclasses so values created through either spelling have identical Pydantic
# validation and ``isinstance`` behavior.
AuthConfig = AuthConfigV1
ConfigMetadata = ConfigMetadataV1
NetworkConfig = NetworkConfigV1
StorageConfig = StorageConfigV1


class SDKConfig(SDKConfigV1):
    """Deprecated compatibility spelling for the SDK v1 configuration."""

    def __init__(self, **data: Any) -> None:
        warnings.warn(
            "SDKConfig is deprecated; use SDKPortalConfigV2 for new code.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
