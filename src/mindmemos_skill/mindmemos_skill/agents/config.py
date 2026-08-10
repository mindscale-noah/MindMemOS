"""Validated construction-time configuration for built-in agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

from ..persistence.enums import SkillInjectionMode


class AgentConfig(BaseModel):
    """Configuration shared by every agent implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str | None = Field(default=None, min_length=1)
    max_turns: int | None = Field(default=None, ge=1)
    skill_injection_mode: SkillInjectionMode | None = None

    def snapshot(self) -> dict[str, Any]:
        """Return a secret-free, JSON-compatible trajectory snapshot."""
        return self.model_dump(mode="json", exclude_none=True)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Self:
        """Validate per-attempt overrides and return the effective config."""

        if not overrides:
            return self
        return type(self).model_validate({**self.model_dump(), **overrides})


__all__ = ["AgentConfig"]
