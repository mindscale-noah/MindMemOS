"""Settings for the DeepSeekHarness harness."""

from typing import Literal

from pydantic import Field

from ..config import CLIConfig


class DeepSeekHarnessConfig(CLIConfig):
    """Configure the installed DeepSeekHarness executable."""

    class_name: Literal["deepseek_harness"] = "deepseek_harness"
    """Registered backend used to construct this Agent from configuration."""

    cli_path: str = Field(default="dsh", min_length=1)
    """Executable name or path."""

    profile: str = Field(default="headless", min_length=1)
    """Installed harness profile used for execution."""

    provider: str = Field(default="deepseek-official", min_length=1)
    """Native adapter identifier used when selecting a model."""
