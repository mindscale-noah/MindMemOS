"""OpenAI Chat Completions wire protocol for the generic API backend."""

from __future__ import annotations

from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import (
    LLMRequest,
    LLMToolDefinition,
    LLMUsage,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols.base import (
    APIProtocol,
    ProtocolResponseError,
    ProtocolToolCall,
    ProtocolTurn,
)


def _tool_payload(tool: LLMToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class OpenAIProtocol(APIProtocol):
    name = "openai"

    def create_client(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        max_retries: int,
    ) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai SDK is required for llm.protocol=openai") from exc
        kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    def initial_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        return request.message_dicts()

    def create_turn(
        self,
        client: Any,
        request: LLMRequest,
        messages: list[dict[str, Any]],
        *,
        output_token_parameter: str,
    ) -> ProtocolTurn:
        tools = [_tool_payload(tool) for tool in request.tools]
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            output_token_parameter: request.generation.max_output_tokens,
        }
        if request.generation.temperature is not None:
            kwargs["temperature"] = request.generation.temperature
        if request.generation.reasoning_effort is not None:
            kwargs["reasoning_effort"] = request.generation.reasoning_effort
        if request.output_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "final_output",
                    "strict": True,
                    "schema": request.output_schema,
                },
            }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = False
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ProtocolResponseError("response contained no choices")
        choice = choices[0]
        message = choice.message
        calls: list[ProtocolToolCall] = []
        wire_calls: list[dict[str, Any]] = []
        for call in getattr(message, "tool_calls", None) or []:
            normalized = ProtocolToolCall(
                id=str(call.id),
                name=str(call.function.name),
                arguments=str(call.function.arguments or ""),
            )
            calls.append(normalized)
            wire_calls.append(normalized.to_trace_dict())
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
        }
        if wire_calls:
            assistant["tool_calls"] = wire_calls

        usage_obj = getattr(response, "usage", None)
        cached = 0
        if usage_obj is not None:
            cached = int(getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0)
            details = getattr(usage_obj, "prompt_tokens_details", None)
            if details is not None:
                cached = max(
                    cached,
                    int(getattr(details, "cached_tokens", 0) or 0),
                )
        usage = LLMUsage(
            input_tokens=int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj is not None else 0,
            cache_read_input_tokens=cached,
            output_tokens=int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj is not None else 0,
        )
        finish_reason = getattr(choice, "finish_reason", None)
        return ProtocolTurn(
            content=str(assistant["content"]),
            tool_calls=tuple(calls),
            assistant_message=assistant,
            finish_reason=finish_reason,
            provider_request_id=(str(getattr(response, "id", "") or "") or None),
            usage=usage,
            truncated=finish_reason == "length",
            blocked=finish_reason == "content_filter",
        )

    def tool_result_messages(
        self,
        results: list[tuple[ProtocolToolCall, str]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": content,
            }
            for call, content in results
        ]


__all__ = ["OpenAIProtocol"]
