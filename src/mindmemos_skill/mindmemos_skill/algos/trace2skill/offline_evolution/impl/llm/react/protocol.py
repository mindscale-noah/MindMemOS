"""Provider-neutral message, tool, request, and response contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant", "tool"]


def estimate_token_count(payload: Any) -> int:
    """Conservatively estimate tokens for mixed ASCII/CJK JSON payloads."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    return math.ceil(ascii_count / 4) + non_ascii_count + 16


@dataclass(frozen=True)
class LLMMessage:
    role: MessageRole
    content: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [dict(item) for item in self.tool_calls]
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True)
class LLMToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class LLMGenerationConfig:
    """Generation controls independent of provider parameter names."""

    temperature: float | None = 0.0
    context_window_tokens: int = 200_000
    max_output_tokens: int = 65_536
    # ``None`` leaves the parameter off the wire. Values are not validated
    # locally: unknown levels fail at the API, which names the accepted set.
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be >= 0 or None")
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be >= 1")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be >= 1")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("max_output_tokens must be smaller than context_window_tokens")
        if self.reasoning_effort is not None and not self.reasoning_effort.strip():
            raise ValueError("reasoning_effort must be a non-empty string or None")


@dataclass(frozen=True)
class LLMRequest:
    model: str
    messages: tuple[LLMMessage, ...]
    tools: tuple[LLMToolDefinition, ...] = ()
    generation: LLMGenerationConfig = field(default_factory=LLMGenerationConfig)
    output_schema: dict[str, Any] | None = None

    def message_dicts(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.messages]

    def tool_dicts(self) -> list[dict[str, Any]]:
        return [tool.to_dict() for tool in self.tools]

    def estimated_input_tokens(self) -> int:
        """Conservatively estimate mixed ASCII/CJK request tokens."""
        return estimate_token_count(
            {
                "messages": self.message_dicts(),
                "tools": self.tool_dicts(),
                "output_schema": self.output_schema,
            }
        )

    def assert_within_context_window(self) -> None:
        estimated = self.estimated_input_tokens()
        total = estimated + self.generation.max_output_tokens
        if total > self.generation.context_window_tokens:
            raise ValueError(
                "estimated communication exceeds context window: "
                f"input={estimated}, reserved_output="
                f"{self.generation.max_output_tokens}, total={total}, limit="
                f"{self.generation.context_window_tokens}"
            )


@dataclass
class LLMUsage:
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.input_tokens += other.input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class LLMResponse:
    content: str = ""
    finish_reason: str | None = None
    provider_request_id: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_call: dict[str, str] | None = None


__all__ = [
    "LLMGenerationConfig",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "LLMToolDefinition",
    "LLMUsage",
    "MessageRole",
    "estimate_token_count",
]
