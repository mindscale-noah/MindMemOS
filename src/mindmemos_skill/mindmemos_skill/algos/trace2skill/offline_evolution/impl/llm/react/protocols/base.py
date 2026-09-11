"""Interfaces shared by wire-protocol implementations for ``llm_api``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import LLMRequest, LLMUsage


@dataclass(frozen=True)
class ProtocolToolCall:
    id: str
    name: str
    arguments: str

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass
class ProtocolTurn:
    content: str = ""
    tool_calls: tuple[ProtocolToolCall, ...] = ()
    assistant_message: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    provider_request_id: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    truncated: bool = False
    blocked: bool = False


class ProtocolResponseError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class APIProtocol(ABC):
    """Translate the standard request contract to one provider wire format."""

    name: str

    @abstractmethod
    def create_client(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        max_retries: int,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def initial_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def create_turn(
        self,
        client: Any,
        request: LLMRequest,
        messages: list[dict[str, Any]],
        *,
        output_token_parameter: str,
    ) -> ProtocolTurn:
        raise NotImplementedError

    @abstractmethod
    def tool_result_messages(
        self,
        results: list[tuple[ProtocolToolCall, str]],
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def output_token_parameter(self, configured: str) -> str:
        return configured

    def user_message(self, content: str) -> dict[str, Any]:
        """Return one ordinary user turn in this protocol's wire format."""
        return {"role": "user", "content": content}

    def context_payload(
        self,
        request: LLMRequest,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "messages": messages,
            "tools": request.tool_dicts(),
            "output_schema": request.output_schema,
        }


__all__ = [
    "APIProtocol",
    "ProtocolResponseError",
    "ProtocolToolCall",
    "ProtocolTurn",
]
