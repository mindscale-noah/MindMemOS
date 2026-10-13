"""Settings for the Codex harness."""

from typing import Literal

from pydantic import Field

from ..config import CLIConfig


class CodexConfig(CLIConfig):
    """Configure the installed Codex executable."""

    class_name: Literal["codex"] = "codex"
    """Registered backend used to construct this Agent from configuration."""

    cli_path: str = Field(default="codex", min_length=1)
    """Executable name or path."""

    reasoning_effort: str | None = None
    """Optional model reasoning effort."""
