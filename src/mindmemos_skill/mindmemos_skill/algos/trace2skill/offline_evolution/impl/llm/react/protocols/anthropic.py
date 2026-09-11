"""Anthropic Messages wire protocol for the generic API backend."""

from __future__ import annotations

import json
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
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }


def _block_payload(block: Any) -> dict[str, Any]:
    block_type = str(getattr(block, "type", "") or "")
    if block_type == "text":
        return {"type": "text", "text": str(getattr(block, "text", ""))}
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": str(getattr(block, "id", "")),
            "name": str(getattr(block, "name", "")),
            "input": getattr(block, "input", {}) or {},
        }
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        payload = dump(exclude_none=True)
        if isinstance(payload, dict):
            return payload
    raise ProtocolResponseError(
        f"unsupported Anthropic content block: {block_type or type(block).__name__}",
        retryable=False,
    )


class AnthropicProtocol(APIProtocol):
    name = "anthropic"

    def create_client(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        timeout: float,
        max_retries: int,
    ) -> Any:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("anthropic SDK is required for llm.protocol=anthropic") from exc
        kwargs: dict[str, Any] = {
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return Anthropic(**kwargs)

    def initial_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        return [message.to_dict() for message in request.messages if message.role != "system"]

    def create_turn(
        self,
        client: Any,
        request: LLMRequest,
        messages: list[dict[str, Any]],
        *,
        output_token_parameter: str,
    ) -> ProtocolTurn:
        del output_token_parameter
        tools = [_tool_payload(tool) for tool in request.tools]
        system = "\n\n".join(
            message.content for message in request.messages if message.role == "system" and message.content
        )
        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.generation.max_output_tokens,
        }
        if system:
            kwargs["system"] = system
        if request.generation.reasoning_effort is not None:
            # The Messages API has no equivalent parameter; refusing beats
            # silently dropping a setting the caller believes is in effect.
            raise ProtocolResponseError(
                "llm.reasoning_effort is not supported by llm.protocol="
                "anthropic; the Messages API has no reasoning_effort "
                "parameter. Unset it, or use llm.protocol=openai.",
                retryable=False,
            )
        if request.generation.temperature is not None:
            kwargs["temperature"] = request.generation.temperature
        if request.output_schema is not None:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": request.output_schema,
                }
            }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = {
                "type": "auto",
                "disable_parallel_tool_use": True,
            }

        response = client.messages.create(**kwargs)
        blocks = list(getattr(response, "content", None) or [])
        if not blocks:
            raise ProtocolResponseError("response contained no content blocks")
        block_payloads = [_block_payload(block) for block in blocks]
        calls: list[ProtocolToolCall] = []
        text_parts: list[str] = []
        for block in block_payloads:
            if block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
            elif block.get("type") == "tool_use":
                calls.append(
                    ProtocolToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=json.dumps(
                            block.get("input") or {},
                            ensure_ascii=False,
                        ),
                    )
                )

        usage_obj = getattr(response, "usage", None)
        usage = LLMUsage(
            input_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0) if usage_obj is not None else 0,
            cache_creation_input_tokens=int(getattr(usage_obj, "cache_creation_input_tokens", 0) or 0)
            if usage_obj is not None
            else 0,
            cache_read_input_tokens=int(getattr(usage_obj, "cache_read_input_tokens", 0) or 0)
            if usage_obj is not None
            else 0,
            output_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0) if usage_obj is not None else 0,
        )
        finish_reason = getattr(response, "stop_reason", None)
        return ProtocolTurn(
            content="".join(text_parts),
            tool_calls=tuple(calls),
            assistant_message={"role": "assistant", "content": block_payloads},
            finish_reason=finish_reason,
            provider_request_id=(str(getattr(response, "id", "") or "") or None),
            usage=usage,
            truncated=finish_reason
            in {
                "max_tokens",
                "model_context_window_exceeded",
            },
            blocked=finish_reason == "refusal",
        )

    def tool_result_messages(
        self,
        results: list[tuple[ProtocolToolCall, str]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": content,
                    }
                    for call, content in results
                ],
            }
        ]

    def output_token_parameter(self, configured: str) -> str:
        del configured
        return "max_tokens"

    def context_payload(
        self,
        request: LLMRequest,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = super().context_payload(request, messages)
        payload["system"] = [message.content for message in request.messages if message.role == "system"]
        return payload


__all__ = ["AnthropicProtocol"]
