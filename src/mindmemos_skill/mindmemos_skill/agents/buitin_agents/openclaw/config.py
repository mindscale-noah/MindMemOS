"""Settings for the OpenClaw local harness."""

from typing import Literal

from pydantic import Field

from ..config import CLIConfig


class OpenClawConfig(CLIConfig):
    """Configure an isolated OpenClaw run with a custom model endpoint."""

    class_name: Literal["openclaw"] = "openclaw"
    """Registered backend used to construct this Agent."""

    cli_path: str = Field(default="openclaw", min_length=1)
    """Installed executable name or absolute path."""

    model: str = Field(min_length=1)
    """Exact model ID accepted by the configured endpoint, without an added provider prefix."""

    api: Literal["openai-completions", "openai-responses", "anthropic-messages"] = "openai-completions"
    """Wire protocol supported by the custom endpoint."""

    reasoning_effort: Literal["off", "minimal", "low", "medium", "high", "xhigh", "adaptive", "max"] | None = None
    """Optional native thinking level; None keeps OpenClaw's default."""

    context_window: int = Field(default=200000, gt=0)
    """Model context capacity in tokens declared to OpenClaw."""

    max_completion_tokens: int = Field(default=8192, gt=0)
    """Maximum model output tokens declared to OpenClaw."""
