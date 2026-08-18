"""Validated configuration for the DeepSeek Harness agent family."""

from __future__ import annotations

from pydantic import Field

from ...persistence.enums import SkillInjectionMode
from ..config import AgentConfig

_DEFAULT_PROFILE = "headless"


class DeepSeekHarnessAgentConfig(AgentConfig):
    """Configuration specific to the DeepSeek Harness CLI agent.

    The agent wraps ``dsh --profile headless "<task>"``: one fresh Harness
    Agent answers the task and prints its final assistant message before
    exiting. ``model`` and ``provider`` are forwarded through a temporary
    ``--patch`` overlay that overrides the ``agent-default-model`` row, so a
    MindMemOS experiment can select a different model without touching the
    user's ``$DSH_HOME`` settings.

    ``skill_injection_mode`` accepts ``filesystem`` (Skills materialized into
    a temporary ``.dsh/skills`` tree discovered via ``customSkillDirs``) and
    ``system_prompt`` (interactive environments such as LiveMath already embed
    Skill text in the system message they own, so the runtime only records
    the injection instead of materializing files).
    """

    skill_injection_mode: SkillInjectionMode = SkillInjectionMode.FILESYSTEM
    cli_path: str | None = Field(default=None, min_length=1)
    """``dsh`` executable; defaults to resolving ``dsh`` on ``PATH``."""

    profile: str = Field(default=_DEFAULT_PROFILE, min_length=1)
    """Harness profile to boot; the ``headless`` one-shot app by default."""

    model: str | None = Field(default=None, min_length=1)
    """Model id forwarded to the booted profile's ``agent-default-model`` row."""

    provider: str | None = Field(default=None, min_length=1)
    """Provider route (e.g. ``deepseek-official``) for ``agent-default-model``."""

    timeout_seconds: float = Field(default=300.0, gt=0)
    """Wall-clock budget for one ``dsh`` run, matching the ``claude`` family."""

    empty_output_retries: int = Field(default=1, ge=0, le=5)
    """Retries when headless exits zero without a final assistant message."""


__all__ = ["DeepSeekHarnessAgentConfig", "_DEFAULT_PROFILE"]
