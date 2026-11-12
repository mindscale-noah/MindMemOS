"""Sync/async memory clients share the same request core."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from mindmemos_sdk.memory import (
    AsyncMemoryClient,
    DialogueMessage,
    MemoryClient,
    MemorySearchHit,
    SkillContext,
    TextMessage,
)
from mindmemos_sdk.transport import AsyncHttpTransport, HttpTransport


@pytest.mark.asyncio
async def test_legacy_skill_context_has_the_same_memory_wire_format_for_both_clients():
    """Typed and dictionary references preserve old Memory binding fields."""
    sync_calls = []
    async_calls = []
    sync_client = _sync_client(sync_calls)
    async_client = _async_client(async_calls)
    fields = {
        "name": "legacy",
        "content_hash": "hash",
        "base_version_id": "v-old",
        "version_label": "v1",
        "usage": "modified",
    }
    sync_client.add([TextMessage(text="hello")], skill_context=[SkillContext(**fields)])
    await async_client.add([TextMessage(text="hello")], skill_context=[fields])
    assert sync_calls == async_calls
    assert sync_calls[0]["path"] == "/v1/memory/add"
    assert sync_calls[0]["body"]["skill_context"] == [fields]


def _response_for(path: str) -> httpx.Response:
    if path.endswith("/add"):
        data = {"memories": [{"operation": "add", "content": "hello", "memory_id": "m1"}]}
    elif path.endswith(("/search", "/get")):
        data = {"memories": [{"id": "m1", "memory": "hello"}]}
    else:
        data = None
    return httpx.Response(200, json={"code": "ok", "message": "done", "request_id": "req-1", "data": data})


def _sync_client(captured: list[dict[str, Any]]) -> MemoryClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"path": request.url.path, "body": json.loads(request.content)})
        return _response_for(request.url.path)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return MemoryClient(
        HttpTransport(base_url="https://api.test", api_key="mk_test", client=client),
        default_user_id="u-default",
        default_app_id="app-default",
    )


def _async_client(captured: list[dict[str, Any]]) -> AsyncMemoryClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"path": request.url.path, "body": json.loads(request.content)})
        return _response_for(request.url.path)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AsyncMemoryClient(
        AsyncHttpTransport(base_url="https://api.test", api_key="mk_test", client=client),
        default_user_id="u-default",
        default_app_id="app-default",
    )


@pytest.mark.asyncio
async def test_sync_and_async_memory_clients_share_request_contract():
    sync_calls: list[dict[str, Any]] = []
    async_calls: list[dict[str, Any]] = []
    sync_client = _sync_client(sync_calls)
    async_client = _async_client(async_calls)

    sync_ops: list[Callable[[MemoryClient], Any]] = [
        lambda c: c.add(
            [DialogueMessage(role="user", content="hello")],
            mode="async",
            agent_id="agent-1",
            session_id="session-1",
            metadata={"source": "test"},
        ),
        lambda c: c.search(
            "hello",
            top_k=3,
            search_strategy="agentic",
            rerank=True,
            score_threshold=0.2,
            token_budget=1500,
            filters={"memory_type": "fact"},
            agent_id="agent-1",
            session_id="session-1",
        ),
        lambda c: c.get(filters={"app_id": "app-default"}, top_k=2),
        lambda c: c.update("m1", "new content"),
        lambda c: c.delete("m1"),
        lambda c: c.feedback(
            feedback="good",
            mode="sync",
            messages=[TextMessage(text="note")],
            recalled_memories=[MemorySearchHit(id="m1", memory="hello")],
            agent_id="agent-1",
            session_id="session-1",
        ),
        lambda c: c.dreaming(mode="sync", agent_id="agent-1", session_id="session-1"),
    ]
    async_ops = [
        lambda c: c.add(
            [DialogueMessage(role="user", content="hello")],
            mode="async",
            agent_id="agent-1",
            session_id="session-1",
            metadata={"source": "test"},
        ),
        lambda c: c.search(
            "hello",
            top_k=3,
            search_strategy="agentic",
            rerank=True,
            score_threshold=0.2,
            token_budget=1500,
            filters={"memory_type": "fact"},
            agent_id="agent-1",
            session_id="session-1",
        ),
        lambda c: c.get(filters={"app_id": "app-default"}, top_k=2),
        lambda c: c.update("m1", "new content"),
        lambda c: c.delete("m1"),
        lambda c: c.feedback(
            feedback="good",
            mode="sync",
            messages=[TextMessage(text="note")],
            recalled_memories=[MemorySearchHit(id="m1", memory="hello")],
            agent_id="agent-1",
            session_id="session-1",
        ),
        lambda c: c.dreaming(mode="sync", agent_id="agent-1", session_id="session-1"),
    ]

    sync_results = [op(sync_client).model_dump() for op in sync_ops]
    async_results = [(await op(async_client)).model_dump() for op in async_ops]

    assert sync_calls == async_calls
    assert sync_results == async_results
