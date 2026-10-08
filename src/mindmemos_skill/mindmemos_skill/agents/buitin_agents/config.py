"""Shared settings for locally installed harness executables."""

from pydantic import Field

from ..config import AgentConnectionConfig


class CLIConfig(AgentConnectionConfig):
    """Configure bounded CLI execution with explicit connection settings."""

    timeout_s: float = Field(default=300, gt=0, allow_inf_nan=False)
    """Maximum execution time in seconds, including native tool calls."""

    extra_args: list[str] = Field(default_factory=list)
    """Additional native CLI arguments supplied by the caller."""

    model: str | None = None
    """Model name; None preserves the installed CLI default."""
