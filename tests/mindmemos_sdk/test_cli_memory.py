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
from mindmemos_sdk.skills import (
    ExportSkillResult,
    LocalSkillManifest,
    LocalSkillVersionMetadata,
    PublishLocalResult,
    PushVersionResult,
    RegisterLocalResult,
    SkillDiffResult,
    SkillRecord,
)
from mindmemos_sdk.skills.models import (
    HashState,
    LocalSkillSyncState,
    SkillOrigin,
)

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


class _FakeSkills:
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[tuple[str, tuple, dict]] = []

    def _record(self, method: str, *args, **kwargs):
        self.calls.append((method, args, kwargs))
        return self._result[method] if isinstance(self._result, dict) else self._result

    @property
    def local_repository(self):
        return self

    def register_local(self, *args, **kwargs):
        return self._record("register_local", *args, **kwargs)

    def publish_local(self, *args, **kwargs):
        return self._record("publish_local", *args, **kwargs)

    def list_local(self):
        return self._record("list_local")

    def show_local(self, *args, **kwargs):
        return self._record("show_local", *args, **kwargs)

    def get_local_version(self, *args, **kwargs):
        return self._record("get_local_version", *args, **kwargs)

    def local_history(self, *args, **kwargs):
        return self._record("local_history", *args, **kwargs)

    def pull_local(self, *args, **kwargs):
        return self._record("pull_local", *args, **kwargs)

    def push_local(self, *args, **kwargs):
        return self._record("push_local", *args, **kwargs)

    def sync_local(self, *args, **kwargs):
        return self._record("sync_local", *args, **kwargs)

    def export_local(self, *args, **kwargs):
        return self._record("export_local", *args, **kwargs)

    def diff_local(self, *args, **kwargs):
        return self._record("diff_local", *args, **kwargs)

    def register(self, *args, **kwargs):
        return self._record("register", *args, **kwargs)

    def list(self):
        return self._record("list")

    def show(self, *args, **kwargs):
        return self._record("show", *args, **kwargs)

    def pull(self, *args, **kwargs):
        return self._record("pull", *args, **kwargs)

    def push(self, *args, **kwargs):
        return self._record("push", *args, **kwargs)

    def plan_update(self, *args, **kwargs):
        return self._record("plan_update", *args, **kwargs)

    def history(self, *args, **kwargs):
        return self._record("history", *args, **kwargs)

    def plan_rollback(self, *args, **kwargs):
        return self._record("plan_rollback", *args, **kwargs)

    def apply_checkout(self, *args, **kwargs):
        return self._record("apply_checkout", *args, **kwargs)

    def diff(self, *args, **kwargs):
        return self._record("diff", *args, **kwargs)

    def unregister(self, *args, **kwargs):
        return self._record("unregister", *args, **kwargs)


@pytest.fixture
def fake_skills(monkeypatch):
    holder: dict[str, _FakeSkills] = {}

    def factory(result: object) -> _FakeSkills:
        manager = _FakeSkills(result)
        holder["manager"] = manager
        monkeypatch.setattr(cli, "_build_skill_manager", lambda *, require_api_key: manager)
        return manager

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _skill_record() -> SkillRecord:
    return SkillRecord(
        skill_id="sk_1",
        alias="demo-main",
        path="/tmp/demo",
        skill_name="demo",
        cloud_skill_id="cloud-1",
        base_version_id="v1",
        content_hash="hash-1",
        hash_state=HashState.CONFIRMED,
        version_label="1.0.0",
        updated_at="2026-06-16T00:00:00Z",
    )


def _skill_record_at(path: str) -> SkillRecord:
    return _skill_record().model_copy(update={"path": path})


def _local_manifest(*, latest_version_id: str = "v1") -> LocalSkillManifest:
    return LocalSkillManifest(
        skill_id="00000000-0000-4000-8000-000000000001",
        alias="demo-main",
        name="demo",
        cloud_skill_id=None,
        latest_version_id=latest_version_id,
        version_ids=["v1", "v2"],
        created_at="2026-06-16T00:00:00Z",
        updated_at="2026-06-16T00:00:00Z",
    )


def _local_version(version_id: str = "v1") -> LocalSkillVersionMetadata:
    return LocalSkillVersionMetadata(
        version_id=version_id,
        skill_id="00000000-0000-4000-8000-000000000001",
        parent_version_ids=[] if version_id == "v1" else ["v1"],
        skill_name="demo",
        content_hash=f"hash-{version_id}",
        local_snapshot_hash=f"snapshot-{version_id}",
        version_label="1.0.0" if version_id == "v1" else "1.1.0",
        commit_message="Version message",
        origin=SkillOrigin.LOCAL,
        sync_state=LocalSkillSyncState.PENDING,
        created_at="2026-06-16T00:00:00Z",
    )


def test_status_line_includes_target_message_and_request_id():
    result = StatusResult(code="ok", request_id="req-9", message="done")
    line = cli._status_line("Updated", "m1", result)
    assert line == "Updated m1. done (request_id=req-9)"


def test_status_line_without_target_or_extras():
    assert cli._status_line("Dreaming triggered", None, StatusResult()) == "Dreaming triggered."


def test_skill_register_prints_saved_record(fake_skills, capsys):
    manifest = _local_manifest()
    fake_skills(
        {
            "register_local": RegisterLocalResult(
                action="created",
                skill_id=manifest.skill_id,
                version_id="v1",
                latest_version_id="v1",
            ),
            "show_local": manifest,
            "get_local_version": _local_version(),
        }
    )

    rc = _run(
        ["skill", "register", "/tmp/demo/SKILL.md", "--name", "demo2", "--alias", "demo-main", "--version", "2.0.0"]
    )

    assert rc == 0
    name, args, kwargs = fake_skills.holder["manager"].calls[0]
    assert name == "register_local"
    assert kwargs == {}
    assert args[0].source_path == "/tmp/demo/SKILL.md"
    assert args[0].name == "demo2"
    assert args[0].version_label == "2.0.0"
    assert args[0].alias == "demo-main"
    out = capsys.readouterr().out
    assert "Registered demo" in out
    assert "alias:          demo-main" in out


def test_skill_list_and_show(fake_skills, capsys):
    fake_skills([_local_manifest()])

    assert _run(["skill", "list"]) == 0
    out = capsys.readouterr().out
    assert "skill_id" in out
    assert "latest_version_id" in out
    assert "demo-main" in out

    fake_skills({"show_local": _local_manifest(), "get_local_version": _local_version()})
    assert _run(["skill", "show", "demo-main"]) == 0
    out = capsys.readouterr().out
    assert "skill_id:       00000000-0000-4000-8000-000000000001" in out
    assert "alias:          demo-main" in out
    assert "latest_version: v1" in out
    assert "sync_state:     pending" in out


def test_skill_pull_and_history(fake_skills, capsys):
    version = _local_version("v2")
    fake_skills([version])

    assert _run(["skill", "pull", "sk_1"]) == 0
    assert fake_skills.holder["manager"].calls == [("pull_local", ("sk_1",), {})]
    assert "Pulled 1 version(s)." in capsys.readouterr().out

    fake_skills([_local_version("v2")])
    assert _run(["skill", "history", "sk_1"]) == 0
    assert "v2 parents=v1 sync=pending" in capsys.readouterr().out


def test_skill_push_prints_new_version(fake_skills, capsys):
    manifest = _local_manifest()
    manifest = manifest.model_copy(
        update={
            "cloud_skill_id": "00000000-0000-4000-8000-000000000010",
        }
    )
    fake_skills(
        {
            "push_local": PushVersionResult(
                cloud_skill_id=manifest.cloud_skill_id,
                version_id="v2",
                content_hash="hash-v2",
                status="draft",
                created_at="2026-06-16T00:00:00Z",
                received_at="2026-06-16T00:00:01Z",
            ),
            "show_local": manifest,
        }
    )

    rc = _run(["skill", "push", "demo-main", "--version", "v2"])

    assert rc == 0
    assert fake_skills.holder["manager"].calls == [
        ("push_local", ("demo-main",), {"version_id": "v2"}),
        ("show_local", ("demo-main",), {}),
    ]
    out = capsys.readouterr().out
    assert f"Pushed demo ({manifest.skill_id}) version v2." in out
    assert "content_hash:   hash-v2" in out


def test_skill_update_is_compatibility_alias_for_sync(fake_skills, capsys):
    synced = _local_manifest(latest_version_id="v2")
    manager = fake_skills({"sync_local": synced})

    rc = _run(["skill", "update", "sk_1", "--yes"])

    assert rc == 0
    assert manager.calls == [("sync_local", ("sk_1",), {})]
    assert "Synced demo" in capsys.readouterr().out


def test_skill_pointer_commands_are_not_exposed():
    for command in ("promote", "rollback", "switch"):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["skill", command])


def test_skill_publish_and_export_use_central_repository(fake_skills, capsys):
    manifest = _local_manifest(latest_version_id="v2")
    manager = fake_skills(
        {
            "publish_local": PublishLocalResult(
                skill_id=manifest.skill_id,
                version_id="v2",
                latest_version_id="v2",
                local_snapshot_hash="snapshot-v2",
            ),
            "show_local": manifest,
        }
    )

    assert (
        _run(
            [
                "skill",
                "publish",
                "demo-main",
                "--from",
                "/tmp/demo",
                "--base",
                "v1",
                "--version",
                "1.1.0",
                "--message",
                "Improve guidance",
            ]
        )
        == 0
    )
    name, args, _kwargs = manager.calls[0]
    assert name == "publish_local"
    assert args[0].skill_id == "demo-main"
    assert args[0].source_path == "/tmp/demo"
    assert args[0].base_version_id == "v1"
    assert args[0].commit_message == "Improve guidance"
    assert "Published local version v2" in capsys.readouterr().out

    fake_skills(
        {
            "export_local": ExportSkillResult(
                skill_id=manifest.skill_id,
                version_id="v1",
                target_path="/tmp/exported",
                exported_files=["SKILL.md", "references/api.md"],
                local_snapshot_hash="snapshot-v1",
            )
        }
    )
    assert (
        _run(
            [
                "skill",
                "export",
                "demo-main",
                "--to",
                "/tmp/exported",
                "--version",
                "v1",
                "--no-replace",
            ]
        )
        == 0
    )
    name, args, _kwargs = fake_skills.holder["manager"].calls[0]
    assert name == "export_local"
    assert args[0].skill_id == "demo-main"
    assert args[0].version_id == "v1"
    assert args[0].replace is False
    assert "Exported" in capsys.readouterr().out


def test_skill_diff_prints_unified_diff(fake_skills, capsys):
    fake_skills(
        SkillDiffResult(
            skill_id="sk_1",
            from_version_id="v1",
            to_version_id="v2",
            diff="--- v1/SKILL.md\n+++ v2/SKILL.md\n+new\n",
        )
    )

    rc = _run(["skill", "diff", "sk_1", "--from", "v1", "--to", "v2"])

    assert rc == 0
    assert fake_skills.holder["manager"].calls == [
        ("diff_local", ("sk_1",), {"from_version_id": "v1", "to_version_id": "v2"}),
    ]
    assert "+new" in capsys.readouterr().out


def test_skill_unregister_requires_confirmation(monkeypatch, fake_skills, capsys):
    fake_skills(_skill_record())
    monkeypatch.setattr(cli, "_prompt", lambda _msg: "n")

    rc = _run(["skill", "unregister", "sk_1"])

    assert rc == 1
    assert fake_skills.holder["manager"].calls == []
    assert "Aborted." in capsys.readouterr().out


def test_skill_unregister_with_yes(fake_skills, capsys):
    fake_skills(_skill_record())

    rc = _run(["skill", "unregister", "sk_1", "--yes"])

    assert rc == 0
    assert fake_skills.holder["manager"].calls == [("unregister", ("sk_1",), {})]
    assert "Unregistered demo" in capsys.readouterr().out


def test_skill_unregister_delete_files_with_yes(fake_skills, tmp_path, capsys):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("name: demo\n", encoding="utf-8")
    fake_skills(_skill_record_at(str(skill_dir)))

    rc = _run(["skill", "unregister", "sk_1", "--delete-files", "--yes"])

    assert rc == 2
    assert skill_dir.exists()
    assert fake_skills.holder["manager"].calls == []
    assert "no longer supported" in capsys.readouterr().out


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


def test_memory_add_accepts_skill_context_json(fake_client):
    fake_client(AddResult(code="ok"))
    context = [{"name": "demo", "content_hash": "hash-1", "version_id": "v1"}]

    rc = _run(["memory", "add", "--content", "hi", "--skill-context-json", json.dumps(context)])

    assert rc == 0
    _name, _args, kwargs = fake_client.holder["client"].memory.calls[0]
    assert kwargs["skill_context"] == context


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
