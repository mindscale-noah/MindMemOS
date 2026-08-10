"""Validated configuration for the OpenClaw CLI agent."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ...persistence.enums import SkillInjectionMode
from ..config import AgentConfig


class OpenClawAgentConfig(AgentConfig):
    """Configuration for one ``openclaw agent --local`` turn."""

    skill_injection_mode: Literal[SkillInjectionMode.FILESYSTEM] = SkillInjectionMode.FILESYSTEM
    cli_path: str | None = Field(default=None, min_length=1)
    agent_id: str = Field(default="main", min_length=1)
    timeout_seconds: float = Field(default=600.0, gt=0)
    thinking: Literal["off", "minimal", "low", "medium", "high", "xhigh", "adaptive", "max"] | None = None
    verbose: bool | None = None
    config_path: Path | None = None
    state_dir: Path | None = None
    allowed_tools: tuple[str, ...] = ("read", "write", "edit", "exec")


__all__ = ["OpenClawAgentConfig"]
