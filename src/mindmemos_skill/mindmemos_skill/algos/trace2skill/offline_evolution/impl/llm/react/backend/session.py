"""Session state management for multi-turn LLM conversations."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.invocation import InvokeError
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocol import LLMRequest, LLMResponse
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.llm.react.protocols import APIProtocol

_SESSION_LOCK = threading.RLock()
_MAX_SESSIONS = 256


@dataclass
class _APISession:
    signature: tuple[Any, ...]
    messages: list[dict[str, Any]]
    pending_tool_call: dict[str, str] | None
    lock: threading.Lock = field(default_factory=threading.Lock)


_SESSIONS: OrderedDict[str, _APISession] = OrderedDict()


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def session_signature(
    protocol: APIProtocol,
    request: LLMRequest,
    *,
    cwd: str | None,
    read_roots: list[str] | None,
    write_roots: list[str] | None,
) -> tuple[Any, ...]:
    """Bind a session to one model, contract, system, and sandbox."""
    systems = tuple(message.content for message in request.messages if message.role == "system")
    return (
        protocol.name,
        request.model,
        systems,
        _stable_json(request.tool_dicts()),
        _stable_json(request.output_schema),
        str(cwd or ""),
        tuple(str(root) for root in (read_roots or [])),
        tuple(str(root) for root in (write_roots or [])),
    )


def load_session(session_id: str, *, backend_name: str) -> _APISession:
    with _SESSION_LOCK:
        session = _SESSIONS.get(session_id)
        if session is None:
            raise InvokeError(
                backend_name,
                f"unknown or expired react session: {session_id}",
                retryable=False,
            )
        _SESSIONS.move_to_end(session_id)
        return session


def create_session(
    signature: tuple[Any, ...],
    response: LLMResponse,
) -> str:
    session_id = uuid4().hex
    session = _APISession(
        signature=signature,
        messages=deepcopy(response.messages),
        pending_tool_call=deepcopy(response.pending_tool_call),
    )
    with _SESSION_LOCK:
        _SESSIONS[session_id] = session
        while len(_SESSIONS) > _MAX_SESSIONS:
            _SESSIONS.popitem(last=False)
    return session_id


def commit_session(
    session_id: str,
    session: _APISession,
    response: LLMResponse,
) -> None:
    session.messages = deepcopy(response.messages)
    session.pending_tool_call = deepcopy(response.pending_tool_call)
    with _SESSION_LOCK:
        if _SESSIONS.get(session_id) is session:
            _SESSIONS.move_to_end(session_id)


__all__ = [
    "session_signature",
    "load_session",
    "create_session",
    "commit_session",
]
