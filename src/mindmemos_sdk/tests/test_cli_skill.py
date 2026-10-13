"""Verify Skill CLI request wiring, output, selection, and local file boundaries."""

import asyncio
import json

import httpx
import pytest
from mindmemos_sdk.client import MindMemOSClient
from mindmemos_sdk.config import SDKConfig
from mindmemos_sdk.skills import AsyncSkillClient, TaskProgressRequest
from mindmemos_sdk.skills import cli as skill_cli
from mindmemos_sdk.skills.cli_files import export_source
from mindmemos_sdk.transport import AsyncHttpTransport, HttpTransport

from mindmemos_sdk import cli

CONTENT = "---\nname: demo\ndescription: Example\n---\n# Demo\n"
SOURCE = {"content": "custom", "name": "demo", "description": "Example", "standard_format": False}
SAMPLE = {"task": {"id": "t1", "instruction": "Do the task"}, "reward_data": {}}
TRAJECTORY = {
    "user_id": "u",
    "agent_id": "a",
    "messages": [{"role": "user", "content": "Task"}],
    "skill_version_id": "skv-a",
}


def test_task_log_displays_algorithm_ids_only_at_start(capsys):
    event = {
        "project_id": "p",
        "task_id": "t",
        "attempt": 1,
        "payload": {
            "run_id": "run-a",
            "request_id": "request-a",
            "node_id": "node-a",
            "message": "algorithm.run.started | algorithm=treeskill",
        },
    }
    skill_cli._print_task_log(event)
    for message in ("sampling progress", "sample failed", "algorithm.run.completed"):
        event["payload"]["message"] = message
        skill_cli._print_task_log(event)
    text = capsys.readouterr().out
    assert text.count("run_id=run-a") == text.count("request_id=request-a") == 1
    assert text.count("node_id=node-a") == text.count("task_id=t") == 4
    assert event["payload"]["run_id"] == "run-a"
    assert event["payload"]["request_id"] == "request-a"


@pytest.fixture
def remote(monkeypatch):
    """Use real SDK serialization over recorded in-memory HTTP responses."""
    calls = []

    def response(request):
        route = request.url.path.removeprefix("/v1/skills/")
        body = json.loads(request.content)
        calls.append((route, body, dict(request.url.params)))
        data = {"route": route}
        if route == "skill-resolve":
            reference = body["skill_ref"]
            family = "sk-b" if reference in ("b", "skv-b") else "sk-a"
            data = {"skill_id": family, "skill_version_id": "skv-b" if family == "sk-b" else "skv-a"}
        elif route == "show":
            data = {
                "skill": {"skill_id": "sk-a"},
                "version": {"skill_version_id": "skv-a", "content": CONTENT, "resources": {"data.bin": "AP8="}},
            }
        elif route == "history":
            data = {"items": [{"skill_version_id": "skv-a", "version_label": "v1.2.3"}], "has_more": False}
        elif route == "list":
            data = {"items": [{"skill_id": "sk-a"}], "has_more": False}
        elif route == "graph-resolve":
            data = {"graph_id": "skg-a", "graph_version_id": "skgv-a"}
        elif route == "tasks/show":
            data = {"status": "succeeded", "task_id": "task-a"}
        elif route == "tasks/progress":
            data = (
                []
                if request.url.params["after"] == "1"
                else [
                    {
                        "sequence": 1,
                        "occurred_at": "2026-10-10T10:00:00+08:00",
                        "payload": {"name": "train", "body": {"stage": "done"}, "message": "Done"},
                    }
                ]
            )
        return httpx.Response(
            202 if route in ("algos/train", "algos/evolve") else 200,
            json={"code": "accepted" if route in ("algos/train", "algos/evolve") else "ok", "data": data},
        )

    http = httpx.Client(transport=httpx.MockTransport(response))
    config = SDKConfig()
    monkeypatch.setattr(skill_cli.ConfigManager, "load", lambda self: config)

    def root_factory(*, config):
        transport = HttpTransport(base_url="http://skill.test", api_key=None, client=http, require_api_key=False)
        return MindMemOSClient(config=config, skill_transport=transport)

    monkeypatch.setattr(skill_cli, "MindMemOSClient", root_factory)
    yield calls
    http.close()


@pytest.mark.parametrize(
    ("arguments", "route"),
    [
        (["register", "--json-skill", json.dumps(SOURCE)], "register"),
        (["list", "--filter", '{"name":"demo*"}'], "list"),
        (["show", "skv-a"], "show"),
        (["resolve", "a"], "skill-resolve"),
        (["history", "a", "--offset", "2"], "history"),
        (["diff", "skv-a", "b"], "diff"),
        (["activate", "a", "--version", "v1.2.3"], "activate"),
        (["activate", "a", "--version-id", "skv-a"], "activate"),
        (["update-info", "a", "--clear-alias"], "update-info"),
        (["delete", "skv-a", "--yes"], "delete"),
        (["evolve", "a"], "algos/evolve"),
        (
            [
                "train",
                "a",
                "--work-dir",
                "/run",
                "--pipeline",
                "algo",
                "--environment-json",
                '{"class_name":"env","config":{}}',
                "--sample-json",
                json.dumps(SAMPLE),
                "--train-id",
                "t1",
            ],
            "algos/train",
        ),
        (["task", "show", "task-a"], "tasks/show"),
        (["task", "progress", "task-a"], "tasks/progress"),
        (["trajectory", "report", "--json-data", json.dumps(TRAJECTORY)], "trajectories/report"),
        (["trajectory", "list", "a"], "trajectories/list"),
        (["trajectory", "show", "trajectory-a"], "trajectories/show"),
        (["graph", "list"], "graphs/list"),
        (["graph", "resolve", "graph"], "graph-resolve"),
        (["graph", "history", "graph"], "graphs/snapshots"),
        (["graph", "show", "skgv-a"], "graphs/snapshot"),
        (["graph", "publish", "--alias", "graph", "--skill", "a"], "algos/organize"),
        (["graph", "recommend", "Task", "--by", "graph"], "algos/recommend"),
    ],
)
def test_remote_commands(remote, capsys, arguments, route):
    assert cli.main(["skill", *arguments]) == 0
    output = capsys.readouterr()
    if route == "tasks/progress":
        assert output.out == "[2026-10-10T02:00:00.000000Z] [INFO] [train] | Done\n"
    else:
        json.loads(output.out)
    assert not output.err
    assert remote[-1][0] == route
    assert all("cursor" not in body for _, body, _ in remote)


@pytest.mark.parametrize("resources", [{}, {"asset.bin": "AP8="}])
def test_json_registration_and_commit_preserve_complete_input(remote, resources):
    source = {
        **SOURCE,
        "content": "custom\r\n",
        "resources": resources,
        "metadata": {"category": "example"},
        "commit_version": "v2.3.4",
        "commit_message": "Publish complete source",
    }
    assert cli.main(["skill", "register", "--json-skill", json.dumps(source), "--alias", "custom"]) == 0
    assert remote[-1][1] == {"skill": source, "alias": "custom"}
    assert cli.main(["skill", "commit", "skv-a", "--json-skill", json.dumps(source), "--activate"]) == 0
    assert remote[-1][1] == {
        "skill_id": "sk-a",
        "skill": source,
        "base_version_id": "skv-a",
        "active": True,
        "inherit_resources": False,
    }


def test_json_commit_defaults_replace_resources_and_keep_activation_opt_in(remote):
    source = {"content": CONTENT, "commit_message": "Update", "version_bump": "minor"}
    assert cli.main(["skill", "commit", "a", "--json-skill", json.dumps(source)]) == 0
    assert remote[-1][1]["skill"] == source
    assert remote[-1][1]["inherit_resources"] is False
    assert remote[-1][1]["active"] is False


@pytest.mark.parametrize(
    ("command", "options"),
    [
        ("register", ["--version", "v0.0.1"]),
        ("register", ["--message", "Initial registration"]),
        ("register", ["--metadata-json", "{}"]),
        ("register", ["--include", "**/*"]),
        ("register", ["--exclude", "**/.env"]),
        ("commit", ["--message", "Change"]),
        ("commit", ["--version", "v2.0.0"]),
        ("commit", ["--bump", "patch"]),
        ("commit", ["--resources-dir", "/unused"]),
        ("commit", ["--clear-resources"]),
    ],
)
def test_json_input_rejects_local_options_before_remote_calls(remote, capsys, command, options):
    arguments = ["skill", command]
    if command == "commit":
        arguments.append("a")
    arguments.extend(["--json-skill", json.dumps({**SOURCE, "commit_message": "Change"}), *options])
    assert cli.main(arguments) == 2
    assert not remote
    assert json.loads(capsys.readouterr().err)["code"] == "invalid_argument"


@pytest.mark.parametrize("message", [None, "", "   "])
def test_commit_requires_message_from_selected_input(remote, capsys, message):
    source = {**SOURCE, "commit_message": message}
    assert cli.main(["skill", "commit", "a", "--json-skill", json.dumps(source)]) == 2
    local = ["skill", "commit", "a", "--from", "/unused"]
    if message is not None:
        local.extend(["--message", message])
    assert cli.main(local) == 2
    assert not remote
    capsys.readouterr()


@pytest.mark.parametrize(
    "source",
    [
        {"content": "missing frontmatter", "commit_message": "Change"},
        {**SOURCE, "commit_message": "Change", "resources": {"../escape": "AA=="}},
        {**SOURCE, "commit_message": "Change", "resources": {"asset.bin": "invalid"}},
        {**SOURCE, "commit_message": "Change", "unknown": True},
    ],
)
def test_invalid_json_commit_fails_before_resolving_target(remote, capsys, source):
    assert cli.main(["skill", "commit", "a", "--json-skill", json.dumps(source)]) == 2
    assert not remote
    capsys.readouterr()


def test_selection_defaults_and_pinned_requests(remote):
    assert cli.main(["skill", "diff", "skv-a", "b"]) == 0
    assert remote[-1][1] == {
        "skill_id": "sk-a",
        "from_skill_version_id": "skv-a",
        "to_skill_id": "sk-b",
        "to_skill_version_id": "skv-b",
    }
    assert cli.main(["skill", "trajectory", "list", "a"]) == 0
    assert remote[-1][1]["skill_version_id"] == "skv-a"
    assert cli.main(["skill", "evolve", "a"]) == 0
    assert remote[-1][1]["auto_accept"] is False
    assert cli.main(["skill", "graph", "recommend", "Task", "--by", "graph"]) == 0
    assert remote[-1][1]["graph_version_id"] == "skgv-a"


def test_source_registration_commit_and_export(remote, tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_bytes(CONTENT.replace("\n", "\r\n").encode())
    (source / "data.bin").write_bytes(b"\x00\xff")
    (source / ".env").write_text("SECRET=private")
    assert cli.main(["skill", "register", str(source), "--include", "**/*", "--exclude", "**/.env"]) == 0
    body = remote[-1][1]
    assert body["alias"] == "demo"
    assert body["skill"]["commit_version"] == "v0.0.1"
    assert body["skill"]["content"].startswith("---\r\n")
    assert body["skill"]["resources"] == {"data.bin": "AP8="}
    assert cli.main(["skill", "commit", "skv-a", "--from", str(source), "--message", "Change"]) == 0
    body = remote[-1][1]
    assert body["base_version_id"] == "skv-a"
    assert body["inherit_resources"] is True
    assert body["active"] is False
    target = tmp_path / "export"
    assert cli.main(["skill", "export", "skv-a", "--to", str(target)]) == 0
    assert (target / "data.bin").read_bytes() == b"\x00\xff"
    assert cli.main(["skill", "export", "skv-a", "--to", str(target)]) == 2
    assert (target / "SKILL.md").read_bytes() == CONTENT.encode()
    assert cli.main(["skill", "export", "skv-a", "--to", str(target), "--overwrite"]) == 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["list", "--cursor", "secret-token"],
        ["show", "a", "--version-id", "skv-a"],
        ["activate", "a"],
        ["activate", "a", "--version", "v1", "--version-id", "skv-a"],
        ["activate", "a", "--version-id", "skv-a", "--expected-active-version-id", "skv-old"],
        ["history", "a", "--cursor", "token"],
        ["trajectory", "list", "a", "--cursor", "token"],
        ["diff", "a", "--to", "b"],
        ["register", "--json-skill", "[1]"],
        ["register"],
        ["register", "/unused", "--json-skill", json.dumps(SOURCE)],
        ["register-structured", "--json", json.dumps(SOURCE)],
        ["register", "--json", json.dumps(SOURCE)],
        ["commit", "a"],
        ["commit", "a", "--from", "/unused", "--json-skill", json.dumps(SOURCE)],
        ["commit", "a", "--json-skill", "[1]"],
    ],
)
def test_invalid_syntax_is_json(arguments, capsys, remote):
    with pytest.raises(SystemExit) as caught:
        cli.main(["skill", *arguments])
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["code"] == "invalid_argument"
    assert not remote


def test_duplicate_filter_wrong_family_and_cancellation(remote, capsys, monkeypatch):
    assert cli.main(["skill", "list", "--filter", '{"name":"a"}', "--name", "b"]) == 2
    assert cli.main(["skill", "activate", "a", "--version-id", "skv-b"]) == 2
    assert not any(route == "activate" for route, _, _ in remote)
    monkeypatch.setattr("builtins.input", lambda: "")
    assert cli.main(["skill", "delete", "a"]) == 1
    assert not any(route == "delete" for route, _, _ in remote)
    capsys.readouterr()


def test_progress_prints_text_and_follow_uses_event_sequence(remote, capsys):
    assert cli.main(["skill", "task", "progress", "task-a", "--after", "1"]) == 0
    assert capsys.readouterr().out == ""
    assert cli.main(["skill", "task", "progress", "task-a", "--follow"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["[2026-10-10T02:00:00.000000Z] [INFO] [train] | Done"]
    assert remote[-1][2] == {"after": "1"}


@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize(
    "traceback", [None, "", 'Traceback (most recent call last):\n  File "train.py", line 3\nValueError: Failed\n']
)
def test_progress_prints_optional_traceback(remote, monkeypatch, capsys, follow, traceback):
    from mindmemos_sdk.skills import SkillClient

    def progress(self, request):
        if request.after >= 1:
            return []
        return [
            {
                "sequence": 1,
                "occurred_at": "2026-10-10T10:00:00+08:00",
                "payload": {"name": "train", "message": "Training failed", "traceback": traceback},
            }
        ]

    monkeypatch.setattr(SkillClient, "task_progress", progress)
    arguments = ["skill", "task", "progress", "task-a"]
    if follow:
        arguments.append("--follow")
    assert cli.main(arguments) == 0
    output = capsys.readouterr()
    expected = "[2026-10-10T02:00:00.000000Z] [INFO] [train] | Training failed\n"
    if traceback:
        expected += traceback
    assert output.out == expected
    assert not output.err


def test_symlink_resource_and_export_traversal_are_rejected(remote, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(CONTENT)
    (source / "link").symlink_to(tmp_path / "private")
    assert cli.main(["skill", "register", str(source)]) == 2
    assert not remote
    target = tmp_path / "export"
    target.mkdir()
    (target / "keep").write_text("original")
    with pytest.raises(ValueError):
        export_source({"content": "new", "resources": {"../escape": "AA=="}}, str(target), True)
    assert (target / "keep").read_text() == "original"


def test_async_accepted_and_array_progress():
    async def exercise():
        def response(request):
            return httpx.Response(202, json={"code": "accepted", "data": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as http:
            transport = AsyncHttpTransport(
                base_url="http://skill.test", api_key=None, client=http, require_api_key=False
            )
            assert await AsyncSkillClient(transport).task_progress(TaskProgressRequest(task_id="task-a")) == []

    asyncio.run(exercise())


def test_export_replacement_failure_rolls_back_original(tmp_path, monkeypatch):
    target = tmp_path / "export"
    target.mkdir()
    (target / "original").write_bytes(b"keep")
    original_rename = type(target).rename

    def fail_staging(self, destination):
        if self.name.startswith(".skill-export-"):
            raise OSError("Replacement failed")
        return original_rename(self, destination)

    monkeypatch.setattr(type(target), "rename", fail_staging)
    with pytest.raises(OSError, match="Replacement failed"):
        export_source({"skill_version_id": "skv-a", "content": CONTENT, "resources": {}}, str(target), True)
    assert (target / "original").read_bytes() == b"keep"
    assert list(tmp_path.iterdir()) == [target]


def test_trajectory_report_file_and_stdin(remote, tmp_path, monkeypatch):
    import io

    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(TRAJECTORY))
    assert cli.main(["skill", "trajectory", "report", "--file", str(path)]) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(TRAJECTORY)))
    assert cli.main(["skill", "trajectory", "report", "--file", "-"]) == 0
    assert remote[-1][1] == TRAJECTORY


def test_invalid_structured_content_and_training_split_fail_before_submission(remote, capsys):
    assert cli.main(["skill", "register", "--json-skill", '{"content":"missing frontmatter"}']) == 2
    assert not remote
    assert (
        cli.main(
            [
                "skill",
                "train",
                "a",
                "--pipeline",
                "algo",
                "--environment-json",
                '{"class_name":"env","config":{}}',
                "--work-dir",
                "/run",
                "--sample-json",
                json.dumps(SAMPLE),
                "--train-id",
                "t1",
                "--val-id",
                "t1",
            ]
        )
        == 2
    )
    assert not any(route == "algos/train" for route, _, _ in remote)
    output = capsys.readouterr()
    assert not output.out
    assert all(json.loads(line)["code"] == "invalid_argument" for line in output.err.splitlines())


def test_training_reads_environment_config_file(remote, tmp_path):
    """Transmit a YAML environment selection as a structured per-run configuration."""
    environment = tmp_path / "environment.yaml"
    environment.write_text("class_name: searchqa\nconfig:\n  agent:\n    model: test-model\n  max_turns: 2\n")
    assert (
        cli.main(
            [
                "skill",
                "train",
                "a",
                "--pipeline",
                "algo",
                "--environment",
                str(environment),
                "--work-dir",
                "/run",
                "--sample-json",
                json.dumps(SAMPLE),
                "--train-id",
                "t1",
            ]
        )
        == 0
    )
    body = remote[-1][1]
    assert body["env"] == {"class_name": "searchqa", "config": {"agent": {"model": "test-model"}, "max_turns": 2}}


def test_training_rejects_environment_name_without_config(remote, capsys):
    """Reject the obsolete string contract before making any HTTP request."""
    assert (
        cli.main(
            [
                "skill",
                "train",
                "a",
                "--pipeline",
                "algo",
                "--environment-json",
                '{"class_name":"env","unknown":true}',
                "--work-dir",
                "/run",
                "--sample-json",
                json.dumps(SAMPLE),
                "--train-id",
                "t1",
            ]
        )
        == 2
    )
    assert not remote
    assert "invalid_argument" in capsys.readouterr().err


def test_config_validation_errors_do_not_echo_secrets(remote, monkeypatch, capsys):
    from mindmemos_sdk.errors import ConfigValidationError

    def invalid_config(self):
        raise ConfigValidationError("api_key input_value=private-token")

    monkeypatch.setattr(skill_cli.ConfigManager, "load", invalid_config)
    assert cli.main(["skill", "list"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "private-token" not in output.err
    assert json.loads(output.err)["code"] == "config_error"


def test_api_errors_preserve_code_and_request_id(remote, monkeypatch, capsys):
    from mindmemos_sdk.errors import ApiError
    from mindmemos_sdk.skills import SkillClient

    def fail(self, request):
        raise ApiError("Missing Skill", code="skill_not_found", request_id="req-cli", status_code=404)

    monkeypatch.setattr(SkillClient, "skill_resolve", fail)
    assert cli.main(["skill", "show", "missing"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == {"code": "skill_not_found", "message": "Missing Skill", "request_id": "req-cli"}


def test_follow_drains_all_terminal_log_pages(capsys):
    from argparse import Namespace

    class Client:
        def __init__(self):
            self.observed = False

        def task_progress(self, request):
            if request.after == 0 and not self.observed:
                self.observed = True
                return []
            return [
                {"sequence": index, "payload": {"message": f"log {index}"}}
                for index in range(request.after + 1, min(request.after + 100, 205) + 1)
            ]

        def show_task(self, request):
            return {"status": "succeeded"}

    assert skill_cli._task_progress(Client(), Namespace(task_id="t", after=0, follow=True)) is None
    assert capsys.readouterr().out.splitlines() == [f"[-] [INFO] [-] | log {index}" for index in range(1, 206)]
