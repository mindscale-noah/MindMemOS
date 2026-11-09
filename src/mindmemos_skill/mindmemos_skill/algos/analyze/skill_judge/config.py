"""Configuration for the static Skill judge."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ....agents import OpenAIChatConfig
from ....agents.config import AgentConfig


class SkillJudgeAgentConfig(OpenAIChatConfig):
    """Configure the structured chat model used by the static judge."""

    parse_max_attempts: int = Field(default=3, ge=2)
    """Total model calls allowed per prompt when structured output is invalid."""

    parse_retry_mode: Literal["feedback"] = "feedback"
    """Append parser feedback before retrying malformed judge output."""


class SkillJudgeConfig(BaseModel):
    """Configure exhaustive static Skill inspection and sliding-window scoring."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported analysis options."""

    judge: Annotated[
        SkillJudgeAgentConfig,
        AgentConfig(allowed=("openai_chat",), namespace="skill_judge"),
    ]
    """OpenAI-compatible judge model, including token and sampling settings."""

    algorithm_version: str = Field(default="1", min_length=1)
    """Version of the analysis behavior for stored-result provenance."""

    prompt_version: str = Field(default="skill-judge-v1", min_length=1)
    """Prompt family and revision sent as model-call trace identity."""

    per_file_cap: int = Field(default=6000, ge=256)
    """Maximum characters per SKILL.md window or per resource excerpt."""

    total_cap: int = Field(default=30000, ge=1024)
    """Maximum characters packed into one resource-only window."""

    max_md_lines: int = Field(default=2000, ge=1)
    """Maximum SKILL.md lines considered during one audit."""

    max_findings: int = Field(default=20, ge=1)
    """Maximum concerns retained in the normalized result."""

    window_overlap: int = Field(default=500, ge=0)
    """Character overlap between consecutive SKILL.md windows."""

    max_windows: int = Field(default=60, ge=1)
    """Hard cap on windows; exceeding it fails rather than omitting coverage."""


__all__ = ["SkillJudgeAgentConfig", "SkillJudgeConfig"]
