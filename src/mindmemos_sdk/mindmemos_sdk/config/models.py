"""Local SDK settings for Memory and official or self-hosted Skill services."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_SCHEMA_VERSION = 2
DEFAULT_BASE_URL = "https://mindmemos.cn"


def _normalize_base_url(value: str | None) -> str | None:
    """Normalize an optional HTTP service root."""
    if value is None or not value.strip():
        return None
    value = value.strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be an HTTP URL without credentials, query, or fragment")
    return value.rstrip("/")


class AuthConfig(BaseModel):
    """Authentication material used for API calls."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject undeclared credentials and hide secret input in validation errors."""

    api_key: str | None = Field(default=None, repr=False)
    """Bearer API key; None when credentials have not been configured."""


class MemoryServiceConfig(BaseModel):
    """Connection settings for Memory requests."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject undeclared connection settings without exposing credentials."""

    base_url: str | None = None
    """Custom Memory API root; omitted for the official cloud."""

    auth: AuthConfig = Field(default_factory=AuthConfig)
    """Credentials used by Memory and by Skill when no separate Skill URL is set."""

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        """Normalize the optional custom Memory API root.

        Args:
            value: Configured Memory URL, or None for the official endpoint.

        Returns:
            Normalized custom URL or None.
        """
        return _normalize_base_url(value)


class SkillServiceConfig(BaseModel):
    """Reuse Memory unless a separate Skill service is configured."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject undeclared connection settings without exposing credentials."""

    base_url: str | None = None
    """Separate Skill API root; empty means use the resolved Memory API root."""

    auth: AuthConfig | None = None
    """Separate Skill credentials; omitted when Skill shares Memory's connection."""

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        """Normalize the optional separate Skill API root.

        Args:
            value: Configured Skill URL, or None to share Memory's endpoint.

        Returns:
            Normalized separate URL or None.
        """
        return _normalize_base_url(value)

    @model_validator(mode="after")
    def validate_connection(self) -> SkillServiceConfig:
        """Require separate credentials exactly when using a separate URL.

        Returns:
            The validated Skill service settings.
        """
        if self.base_url is None and self.auth is not None:
            raise ValueError("Skill auth requires a separate Skill base_url")
        if self.base_url is not None and (self.auth is None or not self.auth.api_key):
            raise ValueError("A separate Skill base_url requires its own auth.api_key")
        return self


class DefaultsConfig(BaseModel):
    """Default user identity injected into requests."""

    user_id: str | None = None
    """Business user ID used when an operation accepts a default user."""


class NetworkConfig(BaseModel):
    """Default HTTP transport tuning."""

    timeout_seconds: int = 30
    """Per-request timeout in seconds."""

    max_retries: int = 2
    """Maximum retry attempts allowed by the SDK transport."""


class ConfigMetadata(BaseModel):
    """Bookkeeping timestamps for the config file."""

    created_at: str | None = None
    """UTC creation timestamp of the saved settings file."""

    updated_at: str | None = None
    """UTC timestamp of the most recent saved settings change."""


class SDKConfig(BaseModel):
    """Top-level SDK settings persisted to ``settings.json`` in the selected home."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    """Reject obsolete top-level fields and hide credentials in validation errors."""

    version: int = CONFIG_SCHEMA_VERSION
    """On-disk settings schema version."""

    service_type: Literal["official", "custom"] = "official"
    """Official gateway or custom deployment for the configured services."""

    memory_service: MemoryServiceConfig = Field(default_factory=MemoryServiceConfig)
    """Memory endpoint and its credentials; official endpoint is resolved automatically."""

    skill_service: SkillServiceConfig = Field(default_factory=SkillServiceConfig)
    """Skill routing and any credentials independent of the Memory API."""

    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    """Default business identity added to relevant SDK requests."""

    network: NetworkConfig = Field(default_factory=NetworkConfig)
    """HTTP timeout and retry settings."""

    metadata: ConfigMetadata = Field(default_factory=ConfigMetadata)
    """Configuration creation and update timestamps."""

    @model_validator(mode="after")
    def validate_services(self) -> SDKConfig:
        """Keep official endpoints implicit and custom endpoints complete.

        Returns:
            The validated SDK settings.
        """
        if self.version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"Unsupported SDK config version: {self.version}")
        if self.service_type == "official":
            if self.memory_service.base_url is not None or self.skill_service.base_url is not None:
                raise ValueError("Official service settings must omit both base_url fields")
        elif self.memory_service.base_url is None:
            raise ValueError("Custom service settings require memory_service.base_url")
        return self

    def resolve_memory_connection(self) -> tuple[str, str | None]:
        """Return the effective Memory API root and Bearer key.

        Returns:
            Memory API root and optional Bearer key.
        """
        base_url = DEFAULT_BASE_URL if self.service_type == "official" else self.memory_service.base_url
        if base_url is None:
            raise ValueError("Custom Memory service settings are incomplete")
        return base_url, self.memory_service.auth.api_key

    def resolve_skill_connection(
        self, *, memory_base_url: str | None = None, memory_api_key: str | None = None
    ) -> tuple[str, str | None]:
        """Resolve the Skill transport from an optional independent URL.

        Args:
            memory_base_url: Optional runtime override of the Memory API root.
            memory_api_key: Optional runtime override of the Memory API key.

        Returns:
            Skill API root and Bearer key for the configured destination.
        """
        resolved_memory_url, resolved_memory_key = self.resolve_memory_connection()
        if self.skill_service.base_url is None:
            return (
                memory_base_url if memory_base_url is not None else resolved_memory_url,
                memory_api_key if memory_api_key is not None else resolved_memory_key,
            )
        if self.skill_service.auth is None:
            raise ValueError("Self-hosted Skill service settings are incomplete")
        return self.skill_service.base_url, self.skill_service.auth.api_key
