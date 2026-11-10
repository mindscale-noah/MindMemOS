"""Portable environment selections sent to the Skill training service."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentConfig(BaseModel):
    """Carry environment settings for validation by the server's registered implementation."""

    model_config = ConfigDict(extra="forbid")
    """Reject fields outside the environment selection envelope."""

    class_name: str = Field(min_length=1)
    """Registered server-side environment implementation name."""

    config: dict[str, Any] = Field(default_factory=dict)
    """Implementation-owned options; the server validates their fields and constraints."""
