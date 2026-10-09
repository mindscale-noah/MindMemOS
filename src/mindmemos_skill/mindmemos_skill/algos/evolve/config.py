"""Options for the built-in trajectory reflection evolver."""

from pydantic import BaseModel, ConfigDict, Field

from ...agents import OpenAIChatConfig


class SimpleEvolverConfig(BaseModel):
    """Configure two model calls that reflect and propose one small patch."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")
    """Reject unsupported evolution options."""

    optimizer: OpenAIChatConfig
    """OpenAI-compatible model used for reflection and patch generation."""

    max_trajectory_chars: int = Field(default=12_000, ge=100, le=100_000)
    """Maximum serialized characters from each sampled execution."""

    max_skill_chars: int = Field(default=100_000, ge=100, le=500_000)
    """Maximum length of the current and each historical SKILL.md sent to the model."""

    max_edits: int = Field(default=5, ge=1, le=20)
    """Maximum atomic changes accepted from the proposed patch."""
