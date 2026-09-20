"""Editable SDK settings without registry or credential disclosure."""

from urllib.parse import urlsplit

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

    skill_cache_dir: str
    """SDK content cache directory."""

    skill_backup_dir: str
    """SDK backup directory."""

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
