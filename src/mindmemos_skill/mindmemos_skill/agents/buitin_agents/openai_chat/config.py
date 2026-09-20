"""Configuration for a single-call chat agent with optional tool callbacks."""

from typing import Annotated, Literal

from pydantic import Field, JsonValue

from ...config import AgentConnectionConfig


class OpenAIChatConfig(AgentConnectionConfig):
    """Configure a chat model without storing credentials in rollout snapshots."""

    model: str = Field(min_length=1)
    """Provider model or deployment name."""

    timeout_s: float = Field(default=180, gt=0, allow_inf_nan=False)
    """HTTP timeout in seconds for each model request attempt."""

    max_retries: int = Field(default=3, ge=0, strict=True)
    """Additional attempts after the initial request; zero disables retries."""

    retry_initial_wait_s: float = Field(default=1, gt=0, allow_inf_nan=False)
    """Initial retry delay in seconds, doubled after each failed attempt."""

    retry_max_wait_s: float = Field(default=60, gt=0, allow_inf_nan=False)
    """Upper bound in seconds for each exponential backoff delay."""

    max_completion_tokens: int | None = Field(default=4096, gt=0)
    """Maximum completion tokens including reasoning; None omits the API limit."""

    temperature: float | None = Field(default=None, ge=0, le=2)
    """Optional sampling temperature; None leaves the provider default unchanged."""

    reasoning_effort: str | None = None
    """Optional provider reasoning budget; None omits it."""

    top_p: float | None = Field(default=None, ge=0, le=1)
    """Optional nucleus sampling threshold."""

    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    """Optional penalty for tokens already present in the completion."""

    seed: int | None = None
    """Optional provider sampling seed."""

    stop: list[str] = Field(default_factory=list)
    """Stop sequences, such as Observation: for textual actions."""

    extra_body: dict[str, JsonValue] = Field(default_factory=dict)
    """Provider-specific body extensions, such as chat_template_kwargs."""

    response_validation: Literal["strict", "content"] = "strict"
    """Require complete nonempty responses, or accept provider content as returned."""

    strip_thinking: bool = False
    """Remove an embedded thinking prefix before parsing the model response."""

    retry_policy: Literal["transient", "all"] = "transient"
    """Retry transient transport/status errors, or all request exceptions."""

    retry_wait: Literal["exponential", "equal_jitter", "schedule"] = "exponential"
    """Select deterministic backoff, half-to-full jitter, or an explicit schedule."""

    retry_waits_s: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        default_factory=lambda: [5, 10, 30], min_length=1
    )
    """Wait schedule used by schedule mode; the final delay repeats if needed."""

    sdk_max_retries: int = Field(default=0, ge=0)
    """SDK retries inside each outer attempt; zero avoids nested retries."""

    invalid_tool_arguments: Literal["error", "empty"] = "error"
    """Return a parsing error or execute malformed JSON tool arguments as an empty object."""
