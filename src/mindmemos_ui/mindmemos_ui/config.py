"""Editable SDK settings without registry or credential disclosure."""

from urllib.parse import urlsplit

from mindmemos_sdk.config import DEFAULT_BASE_URL, SDKConfig
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SettingsUpdate(BaseModel):
    """Only fields controlled by the settings form."""

    model_config = ConfigDict(extra="forbid")
    """Reject accidental writes to SDK-owned registry fields."""

    base_url: str
    """HTTP service endpoint."""

    api_key: str | None = None
    """New credential; null or empty preserves the saved credential."""

    clear_api_key: bool = False
    """Explicit request to remove the saved credential."""

    user_id: str | None = None
    """Default SDK user identity."""

    timeout_seconds: int = Field(default=30, gt=0, le=600)
    """Network timeout in seconds."""

    max_retries: int = Field(default=2, ge=0, le=10)
    """Maximum request retries."""

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        """Require an HTTP endpoint without embedded credentials."""
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Expected an HTTP URL without credentials")
        return value.rstrip("/")


def set_memory_connection(config: SDKConfig, *, base_url: str, api_key: str | None, user_id: str | None) -> None:
    """Update the Memory connection in a v2 SDK config.

    Args:
        config: Existing SDK configuration to update in place.
        base_url: Effective Memory API root selected in the UI.
        api_key: Memory credential, or None to leave credentials empty.
        user_id: Optional default business user identity.
    """
    official = base_url == DEFAULT_BASE_URL and config.skill_service.base_url is None
    config.service_type = "official" if official else "custom"
    config.memory_service.base_url = None if official else base_url
    config.memory_service.auth.api_key = api_key
    config.defaults.user_id = user_id


def apply_sdk_settings(config: SDKConfig, body: SettingsUpdate) -> None:
    """Apply editable connection fields while preserving SDK registry and metadata.

    Args:
        config: Existing SDK configuration to update in place.
        body: Validated editable fields; an omitted key preserves stored credentials.
    """
    api_key = config.memory_service.auth.api_key
    if body.clear_api_key:
        api_key = None
    elif body.api_key:
        api_key = body.api_key
    set_memory_connection(config, base_url=body.base_url, api_key=api_key, user_id=body.user_id)
    config.network.timeout_seconds = body.timeout_seconds
    config.network.max_retries = body.max_retries
