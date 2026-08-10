"""Validated configuration for the Claude agent family."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ...persistence.enums import SkillInjectionMode
from ..config import AgentConfig


class ClaudeAgentConfig(AgentConfig):
    """Configuration specific to the Claude Code CLI agent."""

    skill_injection_mode: Literal[SkillInjectionMode.FILESYSTEM] = SkillInjectionMode.FILESYSTEM
    cli_path: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(default=300.0, gt=0)
    dangerously_skip_permissions: bool = False


class ClaudeSDKAgentConfig(AgentConfig):
    """Configuration specific to the Claude Agent SDK agent."""

    skill_injection_mode: Literal[SkillInjectionMode.FILESYSTEM] = SkillInjectionMode.FILESYSTEM
    permission_mode: str = Field(default="bypassPermissions", min_length=1)


__all__ = ["ClaudeAgentConfig", "ClaudeSDKAgentConfig"]
