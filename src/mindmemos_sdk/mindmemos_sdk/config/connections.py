"""Client connection settings, independent of server algorithm assembly."""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MemoryConnectionConfig(BaseModel):
    """Editable Memory connection fields; never include Skill runtime state."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    base_url: str = "http://127.0.0.1:8000"
    api_key: str | None = Field(default=None, repr=False)
    user_id: str | None = None
    timeout_seconds: int = Field(default=21600, gt=0, le=604800)

    @field_validator("api_key", "user_id", mode="before")
    @classmethod
    def empty_optional(cls, value):
        return None if value == "" else value

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Expected an HTTP URL without embedded credentials")
        if url.query or url.fragment:
            raise ValueError("Service URLs cannot contain a query or fragment")
        return value.rstrip("/")


class SkillConnectionConfig(BaseModel):
    """A Skill endpoint and client-side experiment output directory."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    mode: Literal["local", "remote"] = "local"
    base_url: str = "http://127.0.0.1:8001"
    api_key: str | None = Field(default=None, repr=False)
    timeout_seconds: int = Field(default=21600, gt=0, le=604800)
    experiment_dir: str = "experiments"

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return MemoryConnectionConfig.validate_url(value)

    @field_validator("api_key", mode="before")
    @classmethod
    def empty_key(cls, value):
        return None if value == "" else value

    @field_validator("experiment_dir")
    @classmethod
    def nonempty_directory(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Experiment directory cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_mode(self):
        if self.mode == "local" and urlsplit(self.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Local mode requires a loopback service address")
        return self

    @classmethod
    def from_home(cls, home: Path):
        """Load the saved client connection without reading server service.yaml."""
        path = home / "skill-client.yaml"
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))) if path.exists() else cls()

    def experiment_directory(self, home: Path) -> Path:
        """Resolve relative output directories against client HOME."""
        path = Path(self.experiment_dir).expanduser()
        return (path if path.is_absolute() else home / path).resolve()
