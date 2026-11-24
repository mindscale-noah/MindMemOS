from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from mindmemos.infra.kafka import ConsumedMessage
from mindmemos.workers import memory_add, memory_dreaming, memory_feedback


def _message(topic: str, body: dict) -> ConsumedMessage:
    return ConsumedMessage(
        topic=topic,
        partition=0,
        offset=1,
        key=None,
        value=json.dumps(body).encode("utf-8"),
    )


def _context() -> dict:
    return {
        "request_id": "req-provider-context",
        "account_id": "acc-1",
        "project_id": "proj-1",
        "api_key_uuid": "key-1",
        "memory_algorithm": "vanilla",
        "user_id": "user-1",
        "session_id": "session-1",
    }


class ContextTracker:
    def __init__(self) -> None:
        self.active = False
        self.contexts = []

    async def resolve(self, context):
        self.contexts.append(context)
        return self.scope()

    @contextmanager
    def scope(self):
        assert not self.active
        self.active = True
        try:
            yield
        finally:
            self.active = False


@pytest.mark.asyncio
async def test_memory_add_worker_restores_provider_context(monkeypatch) -> None:
    tracker = ContextTracker()

    class Pipeline:
        async def add_sync(self, inp, context, *, add_record_id=None):
            assert tracker.active
            return SimpleNamespace(memories=[])

    monkeypatch.setattr(memory_add, "provider_config_context", tracker.resolve, raising=False)
    monkeypatch.setattr(
        memory_add,
        "create_pipeline",
        lambda **kwargs: (tracker.active and Pipeline()) or pytest.fail("pipeline built outside provider context"),
    )

    await memory_add.handle_memory_add(
        _message(
            memory_add.TOPIC,
            {
                "context": _context(),
                "input": {"messages": [{"text": "remember this"}], "mode": "async"},
            },
        )
    )

    assert tracker.contexts[0].user_id == "user-1"
    assert tracker.active is False


@pytest.mark.asyncio
async def test_feedback_worker_restores_provider_context(monkeypatch) -> None:
    tracker = ContextTracker()

    class Pipeline:
        async def feedback_sync(self, inp, context):
            assert tracker.active

    monkeypatch.setattr(memory_feedback, "provider_config_context", tracker.resolve, raising=False)
    monkeypatch.setattr(
        memory_feedback,
        "get_config",
        lambda: SimpleNamespace(pipelines=SimpleNamespace(feedback="default_feedback")),
    )
    monkeypatch.setattr(
        memory_feedback,
        "create_pipeline",
        lambda **kwargs: (tracker.active and Pipeline()) or pytest.fail("pipeline built outside provider context"),
    )

    await memory_feedback.handle_memory_feedback(
        _message(
            memory_feedback.TOPIC,
            {"context": _context(), "input": {"feedback": "useful", "mode": "async"}},
        )
    )

    assert tracker.contexts[0].project_id == "proj-1"


@pytest.mark.asyncio
async def test_dreaming_worker_restores_provider_context(monkeypatch) -> None:
    tracker = ContextTracker()

    class Pipeline:
        async def dream_sync(self, inp, context):
            assert tracker.active

    monkeypatch.setattr(memory_dreaming, "provider_config_context", tracker.resolve, raising=False)
    monkeypatch.setattr(
        memory_dreaming,
        "get_config",
        lambda: SimpleNamespace(pipelines=SimpleNamespace(dreaming="default_dreaming")),
    )
    monkeypatch.setattr(
        memory_dreaming,
        "create_pipeline",
        lambda **kwargs: (tracker.active and Pipeline()) or pytest.fail("pipeline built outside provider context"),
    )

    await memory_dreaming.handle_memory_dreaming(
        _message(memory_dreaming.TOPIC, {"context": _context(), "input": {"user_id": "user-1"}})
    )

    assert tracker.contexts[0].api_key_uuid == "key-1"
