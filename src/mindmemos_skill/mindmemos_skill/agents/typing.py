"""Shared outputs of single-call and multi-step agent executions."""

from pydantic import BaseModel, ConfigDict, Field, InstanceOf, JsonValue, SerializeAsAny


class AgentResult(BaseModel):
    """Return an agent's final output, execution trace, and aggregate usage."""

    model_config = ConfigDict(extra="forbid")
    """Reject backend-specific top-level fields; keep them in trace records."""

    output: str | SerializeAsAny[InstanceOf[BaseModel]]
    """Final text or a parsed and validated output model.

    The agent handles parsing and any format-correction retries before returning.
    Concrete model fields are preserved during serialization. Deserialization
    of structured output requires the caller to know its concrete model type.
    """

    trajectory: list[dict[str, JsonValue]] = Field(default_factory=list)
    """Ordered messages, tool calls, observations, or backend execution events.

    Each backend preserves its event payloads, including per-call model names,
    response identifiers, and tool details. Empty when no trace is available.
    """

    usage: dict[str, JsonValue] = Field(default_factory=dict)
    """Usage aggregated across this execution, empty when unavailable.

    Backends normalize token totals to input_tokens, output_tokens, and
    total_tokens when reported, and may preserve additional usage breakdowns.
    """


class ToolCall(BaseModel):
    """Represent a tool invocation independently of its wire format."""

    name: str
    """Environment-owned tool to execute."""

    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    """Parsed arguments supplied by the model."""

    call_id: str = ""
    """Provider call ID for matching native tool observations."""

    error: str = ""
    """Argument parsing error; the environment must not execute this call."""


class ToolTurn(BaseModel):
    """Represent one model turn after decoding its tool protocol."""

    text: str
    """Original assistant text, preserved without reformatting."""

    calls: list[ToolCall] = Field(default_factory=list)
    """Tool calls in model order; empty means completion unless error is set."""

    error: str = ""
    """Text protocol parsing feedback to send before another attempt."""
