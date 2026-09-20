"""Configuration for shared local Python and shell execution."""

import sys

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecutionConfig(BaseModel):
    """Select the local Python environment and command shell."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown execution settings."""

    python_command: list[str] = Field(default_factory=lambda: [sys.executable], min_length=1)
    """Python launcher argv, e.g. an interpreter path or uv run --project PATH python."""

    shell_command: list[str] = Field(
        default_factory=lambda: ["cmd.exe", "/d", "/s", "/c"] if sys.platform == "win32" else ["/bin/sh", "-c"],
        min_length=1,
    )
    """Shell argv prefix; one complete command string is appended when executing."""

    environment: dict[str, str] = Field(default_factory=dict)
    """Explicit child-process environment overrides; never mutate the parent process."""

    timeout_s: float = Field(default=120, gt=0, allow_inf_nan=False)
    """Wall-clock limit in seconds for each generated command."""

    @field_validator("python_command", "shell_command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        """Reject empty command arguments.

        Args:
            value: Executable and literal arguments.

        Returns:
            Validated argv without shell parsing.

        Raises:
            ValueError: An argument is blank.
        """
        if any(not argument.strip() for argument in value):
            raise ValueError("Command arguments must not be blank")
        return value
