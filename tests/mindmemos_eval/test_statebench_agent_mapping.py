"""Tests for the installed STATE-Bench MindMemOSAgent message mapping."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

AGENT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mindmemos_eval"
    / "mindmemos_eval"
    / "memory"
    / "envs"
    / "statebench"
    / "mindmemos_agent.py"
)


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


def test_to_add_messages_preserves_tool_calls_and_empty_messages():
    module = _load_agent_module()
    conversation = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "The phone is defective."},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_order", "arguments": {"id": 1}}]},
        {
            "role": "assistant",
            "content": "I will check.",
            "tool_calls": [{"name": "get_order", "arguments": {"id": 1}}],
        },
        {"role": "user", "content": ""},
    ]

    messages = module._to_add_messages(conversation)

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "assistant", "user"]
    assert '"tool_calls"' in messages[2]["content"]
    assert '"get_order"' in messages[2]["content"]
    assert messages[3]["content"].startswith("I will check.")
    assert "[STATE-BENCH-RAW-MESSAGE]" in messages[3]["content"]
    assert '"content":""' in messages[4]["content"]
    assert messages[2].keys() == {"role", "content"}


def test_to_add_messages_preserves_unknown_fields_and_non_dict_values():
    module = _load_agent_module()
    messages = module._to_add_messages(
        [
            {"role": "assistant", "content": "done", "metadata": {"score": 0.5}},
            {"role": "user", "content": ["structured", 1]},
            None,
        ]
    )

    assert '"metadata":{"score":0.5}' in messages[0]["content"]
    assert '"content":["structured",1]' in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert "null" in messages[2]["content"]


def test_shared_mapping_module_matches_agent_mapping():
    module = _load_agent_module()
    from mindmemos_eval.memory.envs.statebench.trajectory_mapping import mapping_stats, to_add_messages

    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "lookup"}]},
    ]
    assert module._to_add_messages(conversation) == to_add_messages(conversation)
    stats = mapping_stats(conversation, to_add_messages(conversation))
    assert stats["raw_message_count"] == 2
    assert stats["mapped_message_count"] == 2
    assert stats["structured_message_count"] == 1
    assert stats["mapped_content_chars"] >= stats["raw_content_chars"]
    assert stats["content_expansion_chars"] > 0


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

    def fake_retrieve(query: str, top_k: int, *, source: str) -> list[str]:
        calls.append((query, top_k, source))
        return [f"learning about {query}"]

    agent._retrieve_learnings = fake_retrieve

    first_turn = [{"role": "user", "content": "opening"}]
    prepared1 = agent.prepare_conversation(list(first_turn))

    assert calls == [("opening", 3, "auto_prepare")]
    assert prepared1[-1] == first_turn[-1]
    assert prepared1[-2]["role"] == "system"
    assert "opening" in prepared1[-2]["content"]

    second_turn = [
        *first_turn,
        {"role": "assistant", "content": "Let me check."},
        {"role": "user", "content": "my order arrived damaged"},
    ]
    prepared2 = agent.prepare_conversation(list(second_turn))

    assert calls == [
        ("opening", 3, "auto_prepare"),
        ("my order arrived damaged", 3, "auto_prepare"),
    ]
    assert prepared2[-1] == second_turn[-1]
    assert prepared2[-2]["role"] == "system"
    assert "my order arrived damaged" in prepared2[-2]["content"]

    # Re-preparing the same conversation must not trigger another retrieval.
    agent.prepare_conversation(list(second_turn))
    assert calls == [
        ("opening", 3, "auto_prepare"),
        ("my order arrived damaged", 3, "auto_prepare"),
    ]


def test_search_metadata_preserves_retrieval_provenance(monkeypatch):
    module = _load_agent_module(_StubStateBenchAgent)
    monkeypatch.setenv("MINDMEMOS_API_KEY", "test-key")
    agent = module.MindMemOSAgent()

    def fake_request(path, payload):
        assert path == "/v1/memory/search"
        return {
            "data": {
                "memories": [
                    {
                        "id": "memory-1",
                        "memory": "Ask for approval before making the change.",
                        "memory_type": "procedure",
                        "last_update_at": "2026-09-16 12:00:00",
                        "lineage": {"role": "current", "derived_from_memory_ids": []},
                    }
                ]
            }
        }

    agent._request = fake_request
    assert agent.retrieve_learnings("change my order", top_k=3) == ["Ask for approval before making the change."]

    search = agent._searches[0]
    assert search["provenance_version"] == "statebench-search-provenance-v1"
    assert search["source"] == "agent_tool"
    assert search["query"] == "change my order"
    assert search["query_sha256"]
    assert search["requested_top_k"] == 3
    assert search["score_available"] is False
    assert search["results"] == [
        {
            "rank": 1,
            "memory_id": "memory-1",
            "memory": "Ask for approval before making the change.",
            "memory_sha256": search["results"][0]["memory_sha256"],
            "score": None,
            "memory_type": "procedure",
            "last_update_at": "2026-09-16 12:00:00",
            "event_time": None,
            "source_timestamp": None,
            "lineage": {"role": "current", "derived_from_memory_ids": []},
        }
    ]


def test_ingest_metadata_maps_added_memory_back_to_training_task(monkeypatch):
    module = _load_agent_module(_StubStateBenchAgent)
    monkeypatch.setenv("MINDMEMOS_API_KEY", "test-key")
    monkeypatch.setenv("MINDMEMOS_ROLE", "train")
    agent = module.MindMemOSAgent()
    trajectory = types.SimpleNamespace(
        task_id="train-task-1",
        user_id="cust-1",
        conversation=[{"role": "user", "content": "Please ask before acting."}],
        metadata={},
    )

    def fake_request(path, payload):
        assert path == "/v1/memory/add"
        assert payload["task_id"] == "train-task-1"
        return {
            "data": {
                "memories": [
                    {
                        "operation": "add",
                        "memory_id": "memory-1",
                        "content": "The user wants approval before changes.",
                        "memory_type": "preference",
                        "confidence": 0.95,
                        "related_memory_ids": [],
                        "graph_edge_count": 1,
                    }
                ]
            }
        }

    agent._request = fake_request
    agent.ingest_trajectory(trajectory)

    add = trajectory.metadata["mindmemos"]["add"]
    assert add["status"] == "ok"
    assert add["provenance_version"] == "statebench-add-provenance-v1"
    assert add["memory_event_count"] == 1
    assert add["memory_events"][0]["memory_id"] == "memory-1"
    assert add["memory_events"][0]["memory"] == "The user wants approval before changes."
    assert add["memory_events"][0]["memory_sha256"]
