"""Options for the built-in trajectory reflection evolver."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from ....agents import OpenAIChatConfig
from ....agents.config import AgentConfig


class SimpleEvolverConfig(BaseModel):
    """Configure parallel batch extraction and hierarchical patch merging."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported evolution options."""

    optimizer: Annotated[OpenAIChatConfig, AgentConfig(allowed=("openai_chat",), namespace="simple_evolve")]
    """OpenAI-compatible model used for patch extraction and merging."""

    batch_size: int = Field(default=8, ge=1, le=40)
    """Maximum trajectories per extraction call."""

    concurrency: int = Field(default=5, ge=1, le=20)
    """Maximum simultaneous model calls across extraction and merge rounds."""

    user_message_chars: int = Field(default=60_000, ge=100, le=500_000)
    """Text budget per user or system message, preserving task instructions."""

    assistant_message_chars: int = Field(default=12_000, ge=100, le=100_000)
    """Text budget per assistant message, including structured tool arguments."""

    tool_message_chars: int = Field(default=6_000, ge=100, le=100_000)
    """Text budget per tool result or unclassified observation."""

    max_skill_chars: int = Field(default=100_000, ge=100, le=500_000)
    """Maximum length of the current and each historical SKILL.md sent to the model."""

    max_edits: int = Field(default=5, ge=1, le=20)
    """Maximum atomic changes accepted from the proposed patch."""
