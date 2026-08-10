"""Validated configuration for the OpenAI-compatible ReAct family."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from ...persistence.enums import SkillInjectionMode
from ..config import AgentConfig


class ReactAgentConfig(AgentConfig):
    """Configuration for the OpenAI-compatible tool-calling ReAct agent."""

    max_turns: int = Field(default=10, ge=1)
    skill_injection_mode: Literal[
        SkillInjectionMode.TOOL,
        SkillInjectionMode.SYSTEM_PROMPT,
    ] = SkillInjectionMode.TOOL
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: str | None = Field(default=None, min_length=1)
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_model_kwargs(self) -> ReactAgentConfig:
        managed = {"feedback_on_parse_error", "format_parser", "messages", "model", "task", "tools"}
        overlap = managed & self.model_kwargs.keys()
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"model_kwargs cannot override managed ReAct fields: {names}")
        return self


__all__ = ["ReactAgentConfig"]
