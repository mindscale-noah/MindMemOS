"""Settings for the Codex harness."""

from pydantic import Field

from ..config import CLIConfig


class CodexConfig(CLIConfig):
    """Configure the installed Codex executable."""

    cli_path: str = Field(default="codex", min_length=1)
    """Executable name or path."""

    reasoning_effort: str | None = None
    """Optional model reasoning effort."""
