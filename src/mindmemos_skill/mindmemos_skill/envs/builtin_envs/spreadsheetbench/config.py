"""SpreadsheetBench data, local execution, and workspace ReAct settings."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from ....infra.execution import ExecutionConfig


class SpreadsheetBenchDataConfig(BaseModel):
    """Read a released dataset and its workbook case directories."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unknown data settings."""

    data_path: Path
    """Official dataset.json or normalized JSONL file."""

    data_root: Path | None = None
    """Workbook release root; defaults to the dataset file's directory."""


class SpreadsheetBenchExecutionConfig(BaseModel):
    """Shared model and turn limits for the two spreadsheet protocols."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unrecognized options, including obsolete protocol switches."""

    agent: OpenAIChatConfig
    """Recorded model and transport settings."""

    max_turns: int = Field(default=30, ge=1)
    """Maximum model turns per workbook case."""

    stagnation_limit: int | None = Field(default=None, ge=2)
    """Consecutive identical observations allowed; None disables the cutoff."""


class SpreadsheetBenchToolCallingConfig(SpreadsheetBenchExecutionConfig):
    """Configure native structured tool calls over workspace tools."""

    tools: list[dict[str, JsonValue]] | None = None
    """Explicit tool schemas; None exposes the standard workspace tools."""

    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    """Local Python launcher, shell, and command timeout for workspace tools."""

    skill_loading_mode: Literal["tool", "system_prompt"] = "tool"
    """Load Skill instructions through a tool or preload them into the prompt."""


class SpreadsheetBenchTextActionConfig(SpreadsheetBenchExecutionConfig):
    """Configure text Action/Observation messages and bash execution."""

    max_turns: int = Field(default=100, ge=1)
    """Maximum text-action turns per task."""

    shell_timeout_seconds: int = Field(default=120, ge=1)
    """Timeout for each local bash command."""

    stagnation_limit: int = Field(default=5, ge=2)
    """Consecutive identical actions or format errors allowed before stopping."""

    transactional_recalculation: bool = True
    """Include the canonical recalculation command and local helper."""
