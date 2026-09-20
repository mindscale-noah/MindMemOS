"""Settings for the Claude harness."""

from pydantic import Field

from ..config import CLIConfig


class ClaudeConfig(CLIConfig):
    """Configure the installed Claude executable."""

    cli_path: str = Field(default="claude", min_length=1)
    """Executable name or path."""

    reasoning_effort: str | None = None
    """Optional model reasoning effort."""

    max_turns: int | None = Field(default=None, gt=0)
    """Optional limit on native agent turns."""

    max_completion_tokens: int | None = Field(default=None, gt=0)
    """Optional maximum output tokens per model call."""
