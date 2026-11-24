"""Configuration for trajectory analysis."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ....agents import OpenAIChatConfig
from ....agents.config import AgentConfig


class TrajectoryJudgeAgentConfig(OpenAIChatConfig):
    """Configure the structured chat model used by the trajectory judge."""

    parse_max_attempts: int = Field(default=3, ge=2)
    """Total model calls allowed per prompt when structured output is invalid."""

    parse_retry_mode: Literal["feedback"] = "feedback"
    """Append parser feedback before retrying malformed judge output."""


class TrajectoryJudgeConfig(BaseModel):
    """Configure success evaluation and optional failure attribution."""

    model_config = ConfigDict(extra="forbid")
    """Reject unsupported trajectory analysis options."""

    judge: Annotated[
        TrajectoryJudgeAgentConfig,
        AgentConfig(allowed=("openai_chat",), namespace="trajectory_judge"),
    ]
    """OpenAI-compatible judge model, including token and sampling settings."""

    algorithm_version: str = Field(default="1", min_length=1)
    """Version of the analysis behavior for stored-result provenance."""

    prompt_version: str = Field(default="trajectory-judge-v1", min_length=1)
    """Prompt family and revision sent as model-call trace identity."""

    max_trajectory_chars: int = Field(default=30000, ge=1024)
    """Maximum rendered trajectory or attribution prompt characters."""

    max_event_chars: int = Field(default=4000, ge=128)
    """Maximum characters retained for one rendered event."""

    attribute_failures: bool = False
    """Run the optional attribution model pass for failed trajectories."""

    attribute_on_partial_success: bool = False
    """Also attribute successful trajectories with verifier scores below one."""


__all__ = ["TrajectoryJudgeAgentConfig", "TrajectoryJudgeConfig"]
