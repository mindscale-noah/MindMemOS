"""Local SDK configuration schema.

These models mirror the on-disk ``~/.mindmemos/settings.json`` format described in
``docs/sdk/design.md``. The first version uses a single profile. ``skills`` is kept
as a permissive list so the skill-management feature can populate it later without a
schema migration here.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CONFIG_SCHEMA_VERSION = 1
DEFAULT_BASE_URL = "https://api.mindmemos.example.com"


class AuthConfig(BaseModel):
    """Authentication material used for API calls."""

    api_key: str | None = None


class DefaultsConfig(BaseModel):
    """Default user identity injected into requests."""

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


class StorageConfig(BaseModel):
    """Local storage locations for skill cache and backups."""

    skill_cache_dir: str = "~/.mindmemos/skills/cache"
    skill_backup_dir: str = "~/.mindmemos/skills/backups"


class NetworkConfig(BaseModel):
    """Default HTTP transport tuning."""

    timeout_seconds: int = 30
    max_retries: int = 2


class ConfigMetadata(BaseModel):
    """Bookkeeping timestamps for the config file."""

    created_at: str | None = None
    updated_at: str | None = None


class SDKConfig(BaseModel):
    """Top-level SDK settings persisted to ``~/.mindmemos/settings.json``."""

    version: int = CONFIG_SCHEMA_VERSION
    base_url: str = DEFAULT_BASE_URL
    auth: AuthConfig = Field(default_factory=AuthConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    memory: MemoryDefaultsConfig = Field(default_factory=MemoryDefaultsConfig)
    metadata: ConfigMetadata = Field(default_factory=ConfigMetadata)
