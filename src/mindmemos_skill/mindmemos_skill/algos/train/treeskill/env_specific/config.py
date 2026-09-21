"""Environment-specific settings for exact TreeSkill reference execution."""

from typing import Literal

from pydantic import AliasChoices, Field

from .....envs.builtin_envs.spreadsheetbench.config import SpreadsheetBenchConfig


class SpreadsheetBenchReferenceConfig(SpreadsheetBenchConfig):
    """Configure the pinned SpreadsheetBench text policy."""

    interaction_mode: Literal["bash_only"] = "bash_only"
    """Use the shared single-workbook execution loop."""

    tool_agent: Literal["text_action"] = "text_action"
    """Decode text through the TreeSkill observation hook."""

    skill_loading_mode: Literal["system_prompt"] = "system_prompt"
    """Preload assembled Skill text into the pinned prompt."""

    max_turns: int = Field(default=100, ge=1)
    """Maximum text-action turns per task."""

    shell_timeout_seconds: int = Field(default=120, ge=1)
    """Timeout for each local bash command."""

    stagnation_limit: int = Field(
        default=5, ge=2, validation_alias=AliasChoices("stagnation_limit", "reference_stagnation_limit")
    )
    """Consecutive identical interactions allowed by the reference branch."""

    transactional_recalculation: bool = True
    """Include the exact canonical recalculation command and helper."""
