"""Tests for the feedback-evo task-end collect API service method."""

from __future__ import annotations

import pytest

from mindmemos.api.schemas import AuthContext, FeedbackEvoCollectRequest
from mindmemos.api.services import memory_service as ms_module
from mindmemos.config import init_config
from mindmemos.typing import FeedbackEvoEvent


def _auth() -> AuthContext:
    return AuthContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        memory_algorithm="feedback_evo",
    )


@pytest.mark.asyncio
async def test_collect_feedback_evo_persists_event(monkeypatch):
    init_config(config_path="config/mindmemos/dev.example.yaml")

    async def _fake_collect(
        context,
        *,
        task_messages,
        task_id=None,
        session_id=None,
    ) -> FeedbackEvoEvent:
        del context, task_messages, task_id, session_id
        return FeedbackEvoEvent(
            event_id="evt-1",
            account_id="acc-1",
            project_id="proj-1",
            api_key_uuid="key-1",
            task_id="task-1",
            signals=[
                {"round_index": 0, "evolvable_path": "search_config.top_k", "reason": "too few memories returned"}
            ],
        )

    monkeypatch.setattr(ms_module.FeedbackEvoCollector, "collect", staticmethod(_fake_collect))
    service = ms_module.get_memory_service()
    result = await service.collect_feedback_evo(
        _auth(),
        FeedbackEvoCollectRequest(
            task_messages=[{"role": "user", "content": "The TechPhone is defective"}],
            task_id="task-1",
        ),
    )

    assert result.status == "ok"
    assert result.event_id == "evt-1"
    assert result.signal_count == 1
    assert result.signals[0]["evolvable_path"] == "search_config.top_k"


@pytest.mark.asyncio
async def test_collect_feedback_evo_empty_signals(monkeypatch):
    init_config(config_path="config/mindmemos/dev.example.yaml")

    async def _fake_collect(
        context,
        *,
        task_messages,
        task_id=None,
        session_id=None,
    ) -> FeedbackEvoEvent:
        del context, task_messages, task_id, session_id
        return FeedbackEvoEvent(
            event_id="evt-2",
            account_id="acc-1",
            project_id="proj-1",
            api_key_uuid="key-1",
            signals=[],
        )

    monkeypatch.setattr(ms_module.FeedbackEvoCollector, "collect", staticmethod(_fake_collect))
    service = ms_module.get_memory_service()
    result = await service.collect_feedback_evo(
        _auth(),
        FeedbackEvoCollectRequest(
            task_messages=[{"role": "assistant", "content": "Everything is fine."}]
        ),
    )

    assert result.status == "ok"
    assert result.signal_count == 0
    assert "0 signal" in result.message
