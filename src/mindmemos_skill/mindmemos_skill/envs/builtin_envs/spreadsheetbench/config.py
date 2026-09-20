"""SpreadsheetBench data, local execution, and workspace ReAct settings."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from ....infra.execution import ExecutionConfig


class SpreadsheetBenchDataConfig(BaseModel):
    """Read a released dataset and its workbook case directories."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown data settings."""

    data_path: Path
    """Official dataset.json or normalized JSONL file."""

    data_root: Path | None = None
    """Workbook release root; defaults to the dataset file's directory."""


class SpreadsheetBenchConfig(BaseModel):
    """Configure workspace tools or a single-workbook bash action loop."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecorded rollout options."""

    agent: OpenAIChatConfig
    """Recorded model and transport settings."""

    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    """Local Python launcher, shell, and command timeout."""

    interaction_mode: Literal["workspace_tools", "bash_only"] = "workspace_tools"
    """Select the independent interaction loop and available tool set."""

    stagnation_limit: int | None = Field(default=None, ge=2)
    """Optional repeated-action cutoff in bash_only mode; None matches the original unlimited repetition."""

    max_turns: int = Field(default=30, ge=1)
    """Maximum model turns per workbook case."""

    tool_agent: Literal["openai_tool", "text_action"] = "openai_tool"
    """Agent registration name selecting the tool-call wire format."""

    skill_loading_mode: Literal["tool", "system_prompt"] = "tool"
    """Load xlsx guidance through a tool, or inject it into the system prompt."""

    evaluator_version: str = "spreadsheetbench-cached-values-v2"
    """Cell scoring and exact post-execution report protocol."""

    @model_validator(mode="after")
    def validate_interaction_mode(self) -> Self:
        """Reject unsupported bash-only protocol and skill-loading combinations.

        Returns:
            The validated interaction settings.

        Raises:
            ValueError: Bash-only mode uses a non-text protocol or tool-loaded skills.
        """
        if self.interaction_mode == "bash_only":
            if self.tool_agent != "text_action":
                raise ValueError("bash_only requires tool_agent='text_action'")
            if self.skill_loading_mode != "system_prompt":
                raise ValueError("bash_only requires skill_loading_mode='system_prompt'")
        return self
