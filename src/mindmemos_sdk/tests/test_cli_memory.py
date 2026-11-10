"""Tests for the new ``mindmemos memory`` CLI subcommands.

These cover argument wiring, the status-line formatter, and handler dispatch with
a fake client so no network or local config is touched.
"""

from __future__ import annotations

import io
import json
import sys

import pytest
from mindmemos_sdk.memory import AddResult, DialogueMessage, GetResult, MemorySearchHit, SearchResult, StatusResult

from mindmemos_sdk import cli


class _FakeMemory:
    """Records the last call and returns a canned result."""

    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, method: str, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return self._result

    def add(self, *args, **kwargs):
        return self._record("add", *args, **kwargs)

    def search(self, *args, **kwargs):
        return self._record("search", *args, **kwargs)

    def get(self, *args, **kwargs):
        return self._record("get", *args, **kwargs)

    def update(self, *args, **kwargs):
        return self._record("update", *args, **kwargs)

    def delete(self, *args, **kwargs):
        return self._record("delete", *args, **kwargs)

    def feedback(self, *args, **kwargs):
        return self._record("feedback", *args, **kwargs)

    def dreaming(self, *args, **kwargs):
        return self._record("dreaming", *args, **kwargs)


class _FakeClient:
    def __init__(self, result: object) -> None:
        self.memory = _FakeMemory(result)

    def __enter__(self) -> _FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def fake_client(monkeypatch):
    """Patch ``_build_client`` and expose the fake for assertions."""

    holder: dict[str, _FakeClient] = {}

    def factory(result: object) -> _FakeClient:
        client = _FakeClient(result)
        holder["client"] = client
        monkeypatch.setattr(cli, "_build_client", lambda: client)
        return client

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _run(argv: list[str]) -> int:
    return cli.main(argv)


def test_status_line_includes_target_message_and_request_id():
    result = StatusResult(code="ok", request_id="req-9", message="done")
    line = cli._status_line("Updated", "m1", result)
    assert line == "Updated m1. done (request_id=req-9)"


def test_status_line_without_target_or_extras():
    assert cli._status_line("Dreaming triggered", None, StatusResult()) == "Dreaming triggered."


def test_memory_add_builds_dialogue_message(fake_client):
    fake_client(AddResult(code="ok"))
    rc = _run(
        [
            "memory",
            "add",
            "--content",
            "我喜欢咖啡",
            "--role",
            "assistant",
            "--user-id",
            "u1",
            "--app-id",
            "app1",
            "--agent-id",
            "agent1",
            "--session-id",
            "s1",
            "--metadata-json",
            '{"source": "cli"}',
        ]
    )
    assert rc == 0
    calls = fake_client.holder["client"].memory.calls
    assert len(calls) == 1
    name, _args, kwargs = calls[0]
    assert name == "add"
    assert kwargs["mode"] == "sync"
    assert kwargs["user_id"] == "u1"
    assert kwargs["app_id"] == "app1"
    assert kwargs["agent_id"] == "agent1"
    assert kwargs["session_id"] == "s1"
    assert kwargs["metadata"] == {"source": "cli"}
    (message,) = kwargs["messages"]
    assert isinstance(message, DialogueMessage)
    assert message.role == "assistant"
    assert message.content == "我喜欢咖啡"
    assert message.timestamp > 0


def test_memory_add_defaults_role_to_user(fake_client):
    fake_client(AddResult(code="ok"))
    rc = _run(["memory", "add", "--content", "hi", "--async"])
    assert rc == 0
    _name, _args, kwargs = fake_client.holder["client"].memory.calls[0]
    assert kwargs["mode"] == "async"
    assert kwargs["messages"][0].role == "user"


def test_memory_add_accepts_messages_json(fake_client):
    fake_client(AddResult(code="queued", request_id="req-add"))
    messages = [{"role": "user", "content": "hi", "timestamp": 1700000000000}]

    rc = _run(["memory", "add", "--messages-json", json.dumps(messages), "--async", "--json"])

    assert rc == 0
    _name, _args, kwargs = fake_client.holder["client"].memory.calls[0]
    assert kwargs["mode"] == "async"
    assert kwargs["messages"] == messages


def test_memory_add_rejects_invalid_metadata_json(capsys):
    rc = _run(["memory", "add", "--content", "hi", "--metadata-json", "[1, 2]"])
    assert rc == 2
    assert "--metadata-json must be a JSON object" in capsys.readouterr().out


def test_memory_add_requires_content_or_messages_json(capsys):
    rc = _run(["memory", "add"])
    assert rc == 2
    assert "either --content or --messages-json is required" in capsys.readouterr().out


def test_memory_add_rejects_invalid_messages_json(capsys):
    rc = _run(["memory", "add", "--messages-json", '{"role": "user"}'])
    assert rc == 2
    assert "--messages-json must be a JSON array" in capsys.readouterr().out


def test_memory_search_json_output(fake_client, capsys):
    fake_client(SearchResult(memories=[MemorySearchHit(id="m1", memory="likes tea")]))

    rc = _run(
        [
            "memory",
            "search",
            "tea",
            "--top-k",
            "4",
            "--user-id",
            "u1",
            "--app-id",
            "app1",
            "--agent-id",
            "agent1",
            "--session-id",
            "s1",
            "--search-strategy",
            "agentic",
            "--rerank",
            "--filter",
            '{"memory_type": "semantic"}',
            "--json",
        ]
    )

    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [
        (
            "search",
            ("tea",),
            {
                "top_k": 4,
                "user_id": "u1",
                "search_strategy": "agentic",
                "rerank": True,
                "filters": {"memory_type": "semantic"},
                "app_id": "app1",
                "agent_id": "agent1",
                "session_id": "s1",
            },
        )
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["memories"][0]["id"] == "m1"


def test_memory_search_invalid_filter_json_fails_fast(capsys):
    rc = _run(["memory", "search", "tea", "--filter", "{not json}"])
    assert rc == 2
    assert "invalid --filter JSON" in capsys.readouterr().out


def test_memory_update_invokes_client(fake_client, capsys):
    fake_client(StatusResult(code="ok", request_id="req-up"))
    rc = _run(["memory", "update", "m1", "--content", "new text"])
    assert rc == 0
    client = fake_client.holder["client"]
    assert client.memory.calls == [("update", ("m1", "new text"), {})]
    assert "Updated m1." in capsys.readouterr().out


def test_memory_delete_requires_confirmation(monkeypatch, fake_client, capsys):
    fake_client(StatusResult())
    monkeypatch.setattr(cli, "_prompt", lambda _msg: "n")
    rc = _run(["memory", "delete", "m1"])
    assert rc == 1
    # Declining the prompt must not hit the client.
    assert fake_client.holder["client"].memory.calls == []
    assert "Aborted." in capsys.readouterr().out


def test_memory_delete_with_yes_skips_prompt(fake_client, capsys):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "delete", "m1", "--yes"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("delete", ("m1",), {})]
    assert "Deleted m1." in capsys.readouterr().out


def test_memory_feedback_passes_text_and_messages(fake_client):
    fake_client(StatusResult(code="ok"))
    messages = [{"role": "user", "content": "wrong coffee preference", "timestamp": 1700000000000}]

    rc = _run(["memory", "feedback", "--text", "good", "--messages-json", json.dumps(messages)])

    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("feedback", (), {"feedback": "good", "messages": messages})]


def test_memory_feedback_passes_recalled_memories(fake_client):
    fake_client(StatusResult(code="ok"))
    messages = [{"role": "user", "content": "wrong coffee preference"}]
    recalled_memories = [{"id": "m1", "memory": "User prefers hot coffee."}]

    rc = _run(
        [
            "memory",
            "feedback",
            "--text",
            "good",
            "--messages-json",
            json.dumps(messages),
            "--recalled-memories-json",
            json.dumps(recalled_memories),
        ]
    )

    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [
        (
            "feedback",
            (),
            {"feedback": "good", "messages": messages, "recalled_memories": recalled_memories},
        )
    ]


def test_memory_feedback_reads_messages_json_from_stdin(monkeypatch, fake_client):
    fake_client(StatusResult(code="ok"))
    messages = [{"role": "user", "content": "wrong coffee preference"}]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(messages)))

    rc = _run(["memory", "feedback", "--text", "good", "--messages-json-file", "-"])

    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("feedback", (), {"feedback": "good", "messages": messages})]


def test_memory_feedback_text_requires_messages(capsys):
    rc = _run(["memory", "feedback", "--text", "good"])

    assert rc == 2
    assert "--text requires --messages-json or --messages-json-file" in capsys.readouterr().out


def test_memory_feedback_rejects_invalid_messages_json(capsys):
    rc = _run(["memory", "feedback", "--text", "good", "--messages-json", '{"role": "user"}'])

    assert rc == 2
    assert "--messages-json must be a JSON array" in capsys.readouterr().out


def test_memory_feedback_rejects_context_without_text(capsys):
    rc = _run(["memory", "feedback", "--messages-json", '[{"role": "user", "content": "hi"}]'])

    assert rc == 2
    assert "context options require --text" in capsys.readouterr().out


def test_memory_feedback_without_text_runs_implicit_flow(fake_client):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "feedback"])

    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("feedback", (), {})]


def test_memory_dreaming_invokes_client(fake_client, capsys):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "dreaming"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("dreaming", (), {"mode": "async"})]
    assert "Dreaming triggered." in capsys.readouterr().out


def test_memory_dreaming_sync_passes_mode(fake_client):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "dreaming", "--sync"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("dreaming", (), {"mode": "sync"})]


def test_memory_dreaming_rejects_conflicting_modes(fake_client, capsys):
    fake_client(StatusResult(code="ok"))
    rc = _run(["memory", "dreaming", "--sync", "--async"])
    assert rc == 2
    assert fake_client.holder["client"].memory.calls == []
    assert "--sync and --async are mutually exclusive" in capsys.readouterr().err


def test_memory_get_passes_filter_and_top_k(fake_client):
    fake_client(GetResult(memories=[MemorySearchHit(id="m1", memory="cat")]))
    rc = _run(["memory", "get", "--filter", '{"app_id": "a1"}', "--top-k", "3"])
    assert rc == 0
    assert fake_client.holder["client"].memory.calls == [("get", (), {"filters": {"app_id": "a1"}, "top_k": 3})]


def test_memory_get_invalid_filter_json_fails_fast(capsys):
    rc = _run(["memory", "get", "--filter", "{not json}"])
    assert rc == 2
    assert "invalid --filter JSON" in capsys.readouterr().out


def test_memory_get_non_object_filter_rejected(capsys):
    rc = _run(["memory", "get", "--filter", "[1, 2]"])
    assert rc == 2
    assert "must be a JSON object" in capsys.readouterr().out
