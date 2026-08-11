"""Chat client backed by :class:`litellm.Router`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ..errors import ModelEndpointNotConfiguredError
from .recording import LLMCallSink, write_llm_call
from .router import Usage, dump_response, get_response_value, litellm_response_headers, usage_tokens

if TYPE_CHECKING:
    from litellm import Router

logger = logging.getLogger(__name__)

_PARSE_FEEDBACK_TEMPLATE = (
    "Your previous reply could not be applied:\n{error}\n\n"
    "Fix exactly that problem and resend the COMPLETE corrected output in the same "
    "format as before. Do not apologize or add commentary."
)


@dataclass(slots=True)
class ChatResponse:
    """Normalized chat response returned by :class:`LLMClient`."""

    finish_reason: str
    content: str
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    parsed: Any = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def to_assistant_message(self) -> dict[str, Any]:
        """Return the OpenAI message shape consumed by tool-calling agents."""

        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


def _normalize_tool_calls(message: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for call in get_response_value(message, "tool_calls", ()) or ():
        dumped = dump_response(call)
        if dumped:
            normalized.append(dumped)
            continue
        function = get_response_value(call, "function", {})
        normalized.append(
            {
                "id": get_response_value(call, "id", "") or "",
                "type": get_response_value(call, "type", "function") or "function",
                "function": {
                    "name": get_response_value(function, "name", "") or "",
                    "arguments": get_response_value(function, "arguments", "{}") or "{}",
                },
            }
        )
    return normalized


class LLMClient:
    """Thin chat wrapper around a pre-built LiteLLM router."""

    ALIAS = "chat"

    def __init__(
        self,
        router: Router,
        *,
        default_model: str | None = ALIAS,
        max_attempts: int = 3,
        call_sink: LLMCallSink | None = None,
    ) -> None:
        self._router = router
        self._default_model = default_model
        self._format_parser_max_attempts = max(1, max_attempts)
        self._call_sink = call_sink

    async def chat(
        self,
        task: str,
        messages: list[dict[str, Any]],
        format_parser: Callable[[str], Any] | None = None,
        *,
        model: str | None = None,
        feedback_on_parse_error: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Call the chat model and optionally parse its text response."""
        target = model or self._default_model
        if target is None:
            raise ModelEndpointNotConfiguredError("chat")

        conversation = list(messages)
        max_attempts = self._format_parser_max_attempts if format_parser is not None else 1
        last_parse_error: Exception | None = None

        for attempt in range(max_attempts):
            request_payload = {"model": target, "messages": conversation, **kwargs}
            started_at = datetime.now(UTC)
            started_counter = perf_counter()
            try:
                response = await self._router.acompletion(model=target, messages=conversation, **kwargs)
            except Exception as exc:
                finished_at = datetime.now(UTC)
                latency_ms = (perf_counter() - started_counter) * 1000
                await write_llm_call(
                    self._call_sink,
                    call_type="chat",
                    task=task,
                    request=request_payload,
                    response=None,
                    model=target,
                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    started_at=started_at,
                    finished_at=finished_at,
                    latency_ms=latency_ms,
                )
                logger.exception(
                    "LiteLLM chat call failed for task=%s model=%s after %.2fms",
                    task,
                    target,
                    latency_ms,
                )
                raise

            finished_at = datetime.now(UTC)
            latency_ms = (perf_counter() - started_counter) * 1000
            raw_response = dump_response(response)
            usage = usage_tokens(getattr(response, "usage", None))
            model_name = get_response_value(response, "model", target) or target
            await write_llm_call(
                self._call_sink,
                call_type="chat",
                task=task,
                request=request_payload,
                response=raw_response,
                model=model_name,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                status="succeeded",
                error=None,
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=latency_ms,
            )
            choice = response.choices[0]
            content = getattr(choice.message, "content", "") or ""
            parsed: Any = None
            if format_parser is not None:
                try:
                    parsed = format_parser(content)
                except Exception as exc:
                    last_parse_error = exc
                    if feedback_on_parse_error and attempt < max_attempts - 1:
                        conversation.extend(
                            [
                                {"role": "assistant", "content": content},
                                {"role": "user", "content": _PARSE_FEEDBACK_TEMPLATE.format(error=str(exc))},
                            ]
                        )
                    continue

            headers = litellm_response_headers(response)
            logger.info(
                "LiteLLM chat call completed for task=%s model=%s in %.2fms retries=%s/%s",
                task,
                target,
                latency_ms,
                headers.get("x-litellm-attempted-retries"),
                headers.get("x-litellm-max-retries"),
            )
            return ChatResponse(
                finish_reason=getattr(choice, "finish_reason", "") or "",
                content=content,
                model=model_name,
                usage=usage,
                parsed=parsed,
                tool_calls=_normalize_tool_calls(choice.message),
                raw_response=raw_response,
            )

        assert last_parse_error is not None
        raise last_parse_error
