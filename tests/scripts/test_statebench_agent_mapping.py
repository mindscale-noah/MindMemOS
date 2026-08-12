"""Tests for the installed STATE-Bench MindMemOSAgent message mapping."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

AGENT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "statebench" / "mindmemos_agent.py"


def _load_agent_module(stub_agent_cls: type | None = None) -> types.ModuleType:
    """Load the agent file with a stubbed ``state_bench`` import."""

    state_bench_pkg = types.ModuleType("state_bench")
    agents_pkg = types.ModuleType("state_bench.agents")
    state_bench_mod = types.ModuleType("state_bench.agents.state_bench")
    state_bench_mod.StateBenchAgent = stub_agent_cls or type("StateBenchAgent", (), {})
    agents_pkg.state_bench = state_bench_mod
    state_bench_pkg.agents = agents_pkg
    sys.modules["state_bench"] = state_bench_pkg
    sys.modules["state_bench.agents"] = agents_pkg
    sys.modules["state_bench.agents.state_bench"] = state_bench_mod

    spec = importlib.util.spec_from_file_location("_mindmemos_agent_test", AGENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_add_messages_strips_tool_calls_and_keeps_content():
    module = _load_agent_module()
    conversation = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "The phone is defective."},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_order", "arguments": {"id": 1}}]},
        {"role": "assistant", "content": "I will check.", "tool_calls": [{"name": "get_order", "arguments": {"id": 1}}]},
        {"role": "user", "content": ""},
    ]

    messages = module._to_add_messages(conversation)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "assistant"]
    assert messages[2]["content"].startswith('{"tool_calls":')
    assert messages[3]["content"] == "I will check."
    assert messages[2].keys() == {"role", "content"}


class _StubStateBenchAgent:
    """Minimal StateBenchAgent stand-in for prepare_conversation tests."""

    retrieve_learnings_top_k = 3

    def prepare_conversation(self, conversation):
        return conversation

    def inject_system_message(self, conversation, content, *, before_last_user=True):
        system_item = {"role": "system", "content": content}
        if not before_last_user or not conversation:
            return [*conversation, system_item]
        return [*conversation[:-1], system_item, conversation[-1]]


def test_prepare_conversation_retrieves_on_each_new_user_message(monkeypatch):
    module = _load_agent_module(_StubStateBenchAgent)
    monkeypatch.setenv("MINDMEMOS_API_KEY", "test-key")
    agent = module.MindMemOSAgent()

    calls: list[tuple[str, int]] = []

    def fake_retrieve(query: str, top_k: int = 3) -> list[str]:
        calls.append((query, top_k))
        return [f"learning about {query}"]

    agent.retrieve_learnings = fake_retrieve

    first_turn = [{"role": "user", "content": "opening"}]
    prepared1 = agent.prepare_conversation(list(first_turn))

    assert calls == [("opening", 3)]
    assert prepared1[-1] == first_turn[-1]
    assert prepared1[-2]["role"] == "system"
    assert "opening" in prepared1[-2]["content"]

    second_turn = [
        *first_turn,
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "my order arrived damaged"},
    ]
    prepared2 = agent.prepare_conversation(list(second_turn))

    assert calls == [("opening", 3), ("my order arrived damaged", 3)]
    assert prepared2[-1] == second_turn[-1]
    assert prepared2[-2]["role"] == "system"
    assert "my order arrived damaged" in prepared2[-2]["content"]

    # Re-preparing the same conversation must not trigger another retrieval.
    agent.prepare_conversation(list(second_turn))
    assert calls == [("opening", 3), ("my order arrived damaged", 3)]
