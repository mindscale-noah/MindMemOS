"""Local SDK configuration schema.

These models mirror the on-disk ``settings.json`` format described in
``docs/sdk/design.md``. The first version uses a single profile. ``skills`` is kept
as a permissive list so the skill-management feature can populate it later without a
schema migration here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .paths import get_mindmemos_home

CONFIG_SCHEMA_VERSION = 1
DEFAULT_BASE_URL = "https://api.mindmemos.example.com"


class AuthConfig(BaseModel):
    """Authentication material used for API calls."""

    api_key: str | None = None


class DefaultsConfig(BaseModel):
    """Default user identity injected into requests."""

    user_id: str | None = None


class StorageConfig(BaseModel):
    """Local storage locations for skill cache and backups."""

    skill_cache_dir: str = Field(default_factory=lambda: str(get_mindmemos_home() / "skills/cache"))
    """Cache directory under the selected SDK home unless explicitly configured."""

    skill_backup_dir: str = Field(default_factory=lambda: str(get_mindmemos_home() / "skills/backups"))
    """Backup directory under the selected SDK home unless explicitly configured."""


class NetworkConfig(BaseModel):
    """Default HTTP transport tuning."""

    timeout_seconds: int = 30
    max_retries: int = 2


class ConfigMetadata(BaseModel):
    """Bookkeeping timestamps for the config file."""

    created_at: str | None = None
    updated_at: str | None = None


class SDKConfig(BaseModel):
    """Top-level SDK settings persisted to ``settings.json`` in the selected home."""

    version: int = CONFIG_SCHEMA_VERSION
    base_url: str = DEFAULT_BASE_URL
    auth: AuthConfig = Field(default_factory=AuthConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    skills: list[dict[str, Any]] = Field(default_factory=list)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    metadata: ConfigMetadata = Field(default_factory=ConfigMetadata)
