"""Multi-turn chat execution with tool loop and context management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeError
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.compaction import auto_compact_messages
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    estimate_token_count,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols import APIProtocol
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols.base import ProtocolResponseError
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.tools import SandboxedToolRuntime

# Statuses where re-sending the same request cannot succeed; everything else retries.
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 409, 413, 414, 422})


def provider_error(
    protocol: APIProtocol,
    exc: Exception,
    *,
    backend_name: str,
) -> InvokeError:
    """Wrap a provider-side exception as InvokeError; unrecognized errors default to retryable."""
    if isinstance(exc, ProtocolResponseError):
        return InvokeError(
            backend_name,
            f"{protocol.name} response error: {exc}",
            cause=exc,
            retryable=exc.retryable,
        )
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    retryable = not (isinstance(status, int) and status in _NON_RETRYABLE_STATUS)
    return InvokeError(
        backend_name,
        f"{protocol.name} API request failed: {name}: {exc}",
        cause=exc,
        retryable=retryable,
    )


def run_chat(
    protocol: APIProtocol,
    client: Any,
    request: LLMRequest,
    *,
    backend_name: str,
    runtime: SandboxedToolRuntime | None,
    max_tool_turns: int,
    output_token_parameter: str,
    compact_trigger_ratio: float,
    compact_target_ratio: float,
    compact_keep_recent_messages: int,
    compacted_tool_result_chars: int,
    initial_messages: list[dict[str, Any]] | None = None,
) -> LLMResponse:
    if compact_target_ratio >= compact_trigger_ratio:
        raise ValueError("llm.auto_compact.target_ratio must be smaller than llm.auto_compact.trigger_ratio")
    if compacted_tool_result_chars < 512:
        raise ValueError("llm.auto_compact.compacted_tool_result_chars must be >= 512")
    messages = deepcopy(initial_messages) if initial_messages is not None else protocol.initial_messages(request)
    totals = LLMUsage()
    wire_output_parameter = protocol.output_token_parameter(output_token_parameter)
    transcript: list[dict[str, Any]] = [
        {
            "type": "request_config",
            "protocol": protocol.name,
            "model": request.model,
            "temperature": request.generation.temperature,
            "reasoning_effort": request.generation.reasoning_effort,
            "context_window_tokens": request.generation.context_window_tokens,
            "max_output_tokens": request.generation.max_output_tokens,
            "output_token_parameter": wire_output_parameter,
            "tools": [tool.name for tool in request.tools],
            "structured_output": request.output_schema is not None,
            "resumed_session": initial_messages is not None,
            "auto_compact_trigger_ratio": compact_trigger_ratio,
            "auto_compact_target_ratio": compact_target_ratio,
        }
    ]

    for turn_number in range(1, max_tool_turns + 1):
        estimated_input = estimate_token_count(protocol.context_payload(request, messages))
        input_budget = request.generation.context_window_tokens - request.generation.max_output_tokens
        compact_trigger = max(1, int(input_budget * compact_trigger_ratio))
        compact_target = max(1, int(input_budget * compact_target_ratio))
        if estimated_input > compact_trigger:
            changed, before_compact, estimated_input = auto_compact_messages(
                protocol,
                request,
                messages,
                runtime=runtime,
                target_tokens=compact_target,
                keep_recent_messages=compact_keep_recent_messages,
                compacted_result_chars=compacted_tool_result_chars,
            )
            if changed:
                transcript.append(
                    {
                        "type": "context_compaction",
                        "turn": turn_number,
                        "messages_compacted": changed,
                        "estimated_input_tokens_before": before_compact,
                        "estimated_input_tokens_after": estimated_input,
                        "target_input_tokens": compact_target,
                    }
                )
        estimated_total = estimated_input + request.generation.max_output_tokens
        if estimated_total > request.generation.context_window_tokens:
            raise InvokeError(
                backend_name,
                "estimated communication exceeds context window: "
                f"input={estimated_input}, reserved_output="
                f"{request.generation.max_output_tokens}, total="
                f"{estimated_total}, limit="
                f"{request.generation.context_window_tokens}",
                retryable=False,
            )
        try:
            turn = protocol.create_turn(
                client,
                request,
                messages,
                output_token_parameter=output_token_parameter,
            )
        except InvokeError:
            raise
        except Exception as exc:
            raise provider_error(protocol, exc, backend_name=backend_name) from exc

        totals.add(turn.usage)
        trace_calls = [call.to_trace_dict() for call in turn.tool_calls]
        transcript.append(
            {
                "type": "assistant",
                "turn": turn_number,
                "provider_request_id": turn.provider_request_id,
                "finish_reason": turn.finish_reason,
                "content": turn.content,
                "tool_calls": trace_calls,
                "usage": {
                    "input_tokens": turn.usage.input_tokens,
                    "cache_creation_input_tokens": (turn.usage.cache_creation_input_tokens),
                    "cache_read_input_tokens": (turn.usage.cache_read_input_tokens),
                    "output_tokens": turn.usage.output_tokens,
                },
                "estimated_input_tokens": estimated_input,
            }
        )
        if turn.truncated:
            raise InvokeError(
                backend_name,
                "response truncated because the output-token or context limit was reached",
                retryable=True,
            )
        if turn.blocked:
            raise InvokeError(
                backend_name,
                "response stopped by the provider's safety policy",
                retryable=False,
            )

        if turn.tool_calls:
            messages.append(turn.assistant_message)
            results: list[tuple[Any, str]] = []
            for call in turn.tool_calls:
                content = (
                    runtime.execute(call.name, call.arguments)
                    if runtime is not None
                    else f"Error: no runtime is available for tool {call.name}"
                )
                results.append((call, content))
                transcript.append(
                    {
                        "type": "tool",
                        "turn": turn_number,
                        "tool_call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "result": content,
                    }
                )
            messages.extend(protocol.tool_result_messages(results))
            continue

        # No tool calls terminates the loop; with output_schema the provider
        # constrains only this final message to the schema.
        messages.append(turn.assistant_message)
        return LLMResponse(
            content=turn.content,
            finish_reason=turn.finish_reason,
            provider_request_id=turn.provider_request_id,
            usage=totals,
            transcript=transcript,
            messages=messages,
        )

    raise InvokeError(
        backend_name,
        f"tool loop exceeded {max_tool_turns} turns",
        retryable=True,
    )


__all__ = ["run_chat"]
