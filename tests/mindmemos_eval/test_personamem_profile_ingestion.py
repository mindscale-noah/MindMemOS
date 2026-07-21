"""Regression tests for PersonaMem profile ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from mindmemos_eval.memory.envs.personamem.env import (
    _PERSONAMEM_EPOCH_MS,
    PersonaMemEnv,
    PersonaMemScope,
)


def _ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


class _RecordingMemory:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, Any]] = []

    async def add(self, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        self.add_calls.append({"messages": messages, **kwargs})


class _ContextStore:
    def __init__(self, context: list[dict[str, Any]]) -> None:
        self.context = context

    def load(self, shared_context_id: str) -> list[dict[str, Any]]:
        assert shared_context_id == "context"
        return list(self.context)

    def visible(self, scope: PersonaMemScope) -> list[dict[str, Any]]:
        return list(self.context[: scope.end_index])


def _scope(context: list[dict[str, Any]]) -> PersonaMemScope:
    return PersonaMemScope(
        shared_context_id="context",
        end_index=len(context),
        scope_id=f"context:{len(context)}",
        user_id="personamem-user",
        session_id="personamem-session",
    )


def _env(memory: _RecordingMemory, context: list[dict[str, Any]]) -> PersonaMemEnv:
    return PersonaMemEnv(
        memory,
        answer_llm=object(),
        context_store=_ContextStore(context),
        evaluation_mode="memory_rag",
        add_batch_size=10,
    )


@pytest.mark.asyncio
async def test_build_scope_ingests_repeated_persona_once_with_profile_metadata() -> None:
    profile = "Current user persona: Likes quiet outdoor activities."
    context = [
        {"role": "system", "content": profile},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": profile},
        {"role": "user", "content": "u2"},
    ]
    memory = _RecordingMemory()

    summary = await _env(memory, context)._build_scope(_scope(context))

    assert len(memory.add_calls) == 2
    assert memory.add_calls[0]["messages"] == [
        {
            "role": "user",
            "content": profile,
            "timestamp": _PERSONAMEM_EPOCH_MS,
        }
    ]
    assert memory.add_calls[0]["metadata"] == {
        "benchmark": "personamem",
        "shared_context_id": "context",
        "end_index_in_shared_context": len(context),
        "source": "personamem_persona",
        "content_type": "profile",
    }
    assert memory.add_calls[1]["messages"] == [
        {"role": "user", "content": "u1", "timestamp": _ms(2026, 1, 1)},
        {"role": "assistant", "content": "a1", "timestamp": _ms(2026, 1, 1)},
        {"role": "user", "content": "u2", "timestamp": _ms(2026, 2, 1)},
    ]
    assert memory.add_calls[1]["metadata"] == {
        "benchmark": "personamem",
        "shared_context_id": "context",
        "end_index_in_shared_context": len(context),
    }
    assert summary.total_messages == 4
    assert summary.added_messages == 4
    assert summary.add_calls == 2
    assert summary.error is None


@pytest.mark.asyncio
async def test_build_scope_ignores_non_profile_system_markers() -> None:
    context = [
        {"role": "system", "content": "Session 1"},
        {"role": "user", "content": "u1"},
    ]
    memory = _RecordingMemory()

    summary = await _env(memory, context)._build_scope(_scope(context))

    assert len(memory.add_calls) == 1
    assert memory.add_calls[0]["messages"] == [
        {"role": "user", "content": "u1", "timestamp": _PERSONAMEM_EPOCH_MS}
    ]
    assert "source" not in memory.add_calls[0]["metadata"]
    assert "content_type" not in memory.add_calls[0]["metadata"]
    assert summary.total_messages == 1
    assert summary.added_messages == 1
    assert summary.add_calls == 1
    assert summary.error is None
