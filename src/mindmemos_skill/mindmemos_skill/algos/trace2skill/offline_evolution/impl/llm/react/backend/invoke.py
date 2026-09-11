"""Public invoke interface for the LLM API backend."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeError, InvokeResult
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.chat import run_chat
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.client import get_client
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.config import (
    _nonnegative_setting,
    _positive_setting,
    _ratio_setting,
    generation_config,
    model,
    output_token_parameter,
    protocol,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.backend.session import (
    commit_session,
    create_session,
    load_session,
    session_signature,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolDefinition,
)
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols.base import ProtocolToolCall
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.tools import SandboxedToolRuntime

BACKEND_NAME = "react"


def _invoke_result(
    response: LLMResponse,
    *,
    session_id: str,
) -> InvokeResult:
    return InvokeResult(
        text=response.content,
        channel=BACKEND_NAME,
        input_tokens=response.usage.input_tokens,
        cache_creation_input_tokens=response.usage.cache_creation_input_tokens,
        cache_read_input_tokens=response.usage.cache_read_input_tokens,
        output_tokens=response.usage.output_tokens,
        raw=None,
        finish_reason=response.finish_reason,
        provider_request_id=response.provider_request_id,
        transcript=response.transcript,
        session_id=session_id,
    )


def invoke(
    system: str,
    user: str,
    *,
    tools: Optional[list[str]] = None,
    cwd: Optional[str] = None,
    read_roots: Optional[list[str]] = None,
    write_roots: Optional[list[str]] = None,
    result_store: Optional[str] = None,
    session_id: Optional[str] = None,
    output_schema: Optional[dict[str, Any]] = None,
    **_ignored: Any,
) -> InvokeResult:
    """Run one standard LLM request using the selected wire protocol."""
    requested_tools = list(tools or [])
    runtime: SandboxedToolRuntime | None = None
    tool_definitions: tuple[LLMToolDefinition, ...] = ()
    if requested_tools:
        if not cwd:
            raise InvokeError(
                BACKEND_NAME,
                "local tools require cwd",
                retryable=False,
            )
        try:
            runtime = SandboxedToolRuntime(
                cwd=cwd,
                read_roots=read_roots or (),
                write_roots=write_roots or (),
                result_store=result_store,
                tool_result_max_chars=_positive_setting("llm.tool_result_max_chars"),
            )
            tool_definitions = runtime.definitions(requested_tools)
        except ValueError as exc:
            raise InvokeError(
                BACKEND_NAME,
                str(exc),
                cause=exc,
                retryable=False,
            ) from exc

    try:
        selected_protocol = protocol()
        messages = ((LLMMessage(role="system", content=system),) if system.strip() else ()) + (
            LLMMessage(role="user", content=user),
        )
        request = LLMRequest(
            model=model(),
            messages=messages,
            tools=tool_definitions,
            generation=generation_config(),
            output_schema=output_schema,
        )
        signature = session_signature(
            selected_protocol,
            request,
            cwd=cwd,
            read_roots=read_roots,
            write_roots=write_roots,
        )
        client = get_client(selected_protocol, backend_name=BACKEND_NAME)
        session = None
        if session_id:
            session = load_session(session_id, backend_name=BACKEND_NAME)
            session.lock.acquire()
            if session.signature != signature:
                session.lock.release()
                raise InvokeError(
                    BACKEND_NAME,
                    "react session cannot be resumed with a different "
                    "protocol, model, system prompt, tool contract, or sandbox",
                    retryable=False,
                    session_id=session_id,
                )
        try:
            initial_messages: list[dict[str, Any]] | None = None
            if session is not None:
                initial_messages = deepcopy(session.messages)
                pending = session.pending_tool_call
                if pending is not None:
                    call = ProtocolToolCall(
                        id=pending["id"],
                        name=pending["name"],
                        arguments=pending["arguments"],
                    )
                    initial_messages.extend(selected_protocol.tool_result_messages([(call, user)]))
                else:
                    initial_messages.append(selected_protocol.user_message(user))
            response = run_chat(
                selected_protocol,
                client,
                request,
                backend_name=BACKEND_NAME,
                runtime=runtime,
                max_tool_turns=_positive_setting("llm.max_tool_turns"),
                output_token_parameter=output_token_parameter(),
                compact_trigger_ratio=_ratio_setting("llm.auto_compact.trigger_ratio"),
                compact_target_ratio=_ratio_setting("llm.auto_compact.target_ratio"),
                compact_keep_recent_messages=_nonnegative_setting("llm.auto_compact.keep_recent_messages"),
                compacted_tool_result_chars=_positive_setting("llm.auto_compact.compacted_tool_result_chars"),
                initial_messages=initial_messages,
            )
            if session is None:
                resolved_session_id = create_session(signature, response)
            else:
                resolved_session_id = session_id
                commit_session(session_id, session, response)
        finally:
            if session is not None:
                session.lock.release()
    except InvokeError as exc:
        if session_id and not exc.session_id:
            exc.session_id = session_id
        raise
    except Exception as exc:
        raise InvokeError(
            BACKEND_NAME,
            f"configuration or request construction failed: {exc}",
            cause=exc,
            retryable=False,
        ) from exc
    return _invoke_result(response, session_id=resolved_session_id)


__all__ = [
    "BACKEND_NAME",
    "invoke",
]
