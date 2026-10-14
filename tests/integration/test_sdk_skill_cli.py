"""Run the public SDK CLI against the actual standalone Skill FastAPI app."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mindmemos_sdk.client import MindMemOSClient
from mindmemos_sdk.config import SDKConfig
from mindmemos_sdk.skills import cli as skill_cli
from mindmemos_sdk.transport import HttpTransport
from mindmemos_skill.algos.manage.skill_manage.extraction import SceneExtractor
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.service import SkillService
from mindmemos_skill.config.config import FastAPIConfig
from mindmemos_skill.scene.typing import ProvidedSceneRecord

from mindmemos_sdk import cli
from mindmemos_skill import HttpAuthConfig, ServiceConfig, SkillConfig, create_service

CONTENT = "---\nname: cli-demo\ndescription: CLI integration example\n---\n# Initial\n"


@pytest.fixture
def cli_server(tmp_path, monkeypatch, capsys):
    """Connect CLI commands to an isolated real service through its HTTP app."""

    service_config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'service.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
            "manage_pipeline": {
                "class_name": "graph_manage",
                "config": {
                    "algorithm": {
                        "embedding": {"model": "fake", "base_url": "https://example.invalid", "dimensions": 3},
                        "chat": {"model": "fake"},
                        "llm_linking": False,
                    }
                },
            },
        }
    )
    service = asyncio.run(
        create_service(
            SkillConfig(
                service=service_config,
                fastapi=FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id="cli-project")),
            )
        )
    )
    record = ProvidedSceneRecord.model_validate(
        json.loads(
            (Path(__file__).resolve().parents[2] / "src/mindmemos_skill/tests/fixtures/scene_graph_d0.json").read_text()
        )["build_input"]["provided"][0]["record"]
    )

    async def extract(self, document):
        return record.model_copy(deep=True)

    monkeypatch.setattr(SceneExtractor, "extract", extract)
    pipeline = service._manage_pipeline

    class Embeddings:
        config = pipeline.config.algorithm.embedding

        async def embed(self, texts):
            return [[1.0, 0.2, 0.1] for _ in texts]

    from mindmemos_skill.infra.embedding import EmbeddingModel

    monkeypatch.setattr(EmbeddingModel, "from_config", lambda config: Embeddings())
    config = SDKConfig()
    monkeypatch.setattr(skill_cli.ConfigManager, "load", lambda self: config)
    with TestClient(
        create_app(
            service,
            api_config=FastAPIConfig(
                auth=HttpAuthConfig(mode="none", default_project_id="cli-project"), evolution_pipelines={"probe"}
            ),
        )
    ) as http:

        def factory(*, config):
            transport = HttpTransport(base_url="http://testserver", api_key=None, client=http, require_api_key=False)
            return MindMemOSClient(config=config, skill_transport=transport)

        monkeypatch.setattr(skill_cli, "MindMemOSClient", factory)

        def run(*arguments):
            code = cli.main(["skill", *arguments])
            output = capsys.readouterr()
            assert code == 0, output.err
            assert not output.err
            if arguments[:2] == ("task", "progress"):
                return output.out
            return json.loads(output.out)

        yield run


def test_cli_version_lifecycle_resources_and_trajectories(cli_server, tmp_path):
    run = cli_server
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(CONTENT)
    (source / "resource.bin").write_bytes(b"\x00\xff")
    (source / ".env").write_text("secret=excluded")
    initial = run("register", str(source), "--exclude", "**/.env")
    family, first = initial["skill_id"], initial["skill_version_id"]
    assert run("resolve", "cli-demo")["skill_version_id"] == first
    assert run("list", "--filter", '{"name":"cli-*"}')["items"][0]["skill_id"] == family
    (source / "SKILL.md").write_text(CONTENT.replace("Initial", "Changed"))
    second = run("commit", first, "--from", str(source), "--message", "Second", "--version", "v0.0.2")
    second_id = second["skill_version_id"]
    assert run("show", "cli-demo")["version"]["skill_version_id"] == first
    shown = run("show", second_id)
    assert shown["version"]["content"].endswith("# Changed\n")
    assert shown["version"]["resources"] == {"resource.bin": "AP8="}
    run("diff", first, second_id)
    assert len(run("history", second_id)["items"]) == 2
    run("activate", family, "--version", "v0.0.2")
    assert run("resolve", family)["skill_version_id"] == second_id
    run("activate", family, "--version-id", first)
    exported = tmp_path / "exported"
    exported = Path(run("export", second_id, "--to", str(exported))["directory"])
    assert (exported / "resource.bin").read_bytes() == b"\x00\xff"
    report = {
        "skill_version_id": second_id,
        "user_id": "u",
        "agent_id": "a",
        "messages": [{"role": "user", "content": "Task"}],
    }
    trajectory = run("trajectory", "report", "--json-data", json.dumps(report))
    assert run("trajectory", "list", family)["items"] == []
    assert len(run("trajectory", "list", second_id)["items"]) == 1
    assert run("trajectory", "show", trajectory["trajectory_id"])["skill_version_id"] == second_id
    run("update-info", second_id, "--alias", "renamed")
    assert run("resolve", "renamed")["skill_id"] == family
    run("delete", second_id, "--yes")
    assert run("list")["items"] == []


def test_cli_nonstandard_registration_preserves_api_defaults(cli_server):
    source = {
        "content": "Nonstandard source",
        "standard_format": False,
        "name": "custom",
        "description": "Custom instructions",
    }
    result = cli_server("register", "--json-skill", json.dumps(source))
    shown = cli_server("show", result["skill_version_id"])
    assert shown["version"]["version_label"] == "v1.0.0"
    assert shown["skill"]["alias"] is None
    assert shown["version"]["content"] == source["content"]


def test_cli_graph_snapshots_and_pinned_recommendations(cli_server):
    run = cli_server
    source = {"content": "---\nname: csv\ndescription: CSV processing\n---\nUse CSV tooling\n"}
    skill = run("register", "--json-skill", json.dumps(source), "--alias", "csv")
    graph = run("graph", "publish", "--alias", "tools", "--skill", skill["skill_version_id"])
    graph_id = graph["graph_id"]
    first = run("graph", "resolve", "tools")["graph_version_id"]
    assert len(run("graph", "list")["items"]) == 1
    run("graph", "show", first)
    run("graph", "recommend", "Process CSV", "--by", first)
    run("graph", "publish", "--graph-id", graph_id, "--empty")
    current = run("graph", "resolve", graph_id)["graph_version_id"]
    assert current != first
    assert len(run("graph", "history", graph_id)["items"]) == 2
    run("graph", "show", first)
    run("graph", "recommend", "Process CSV", "--by", first)
    published = run("graph", "publish", "--graph-id", graph_id, "--all-skills")
    assert published["skill_version_ids"] == [skill["skill_version_id"]]
    assert run("graph", "resolve", graph_id)["graph_version_id"] == published["graph_version_id"]


def test_cli_json_commit_preserves_content_and_replaces_resources(cli_server):
    run = cli_server
    initial = run("register", "--json-skill", json.dumps({"content": CONTENT, "resources": {"old.bin": "AP8="}}))
    first = initial["skill_version_id"]
    source = {
        "content": "Updated custom source\r\n",
        "name": "cli-demo",
        "description": "Updated instructions",
        "standard_format": False,
        "resources": {"new.bin": "AQI="},
        "commit_message": "Replace complete source",
        "version_bump": "minor",
    }
    second = run("commit", first, "--json-skill", json.dumps(source))
    shown = run("show", second["skill_version_id"])["version"]
    assert shown["content"] == source["content"]
    assert shown["resources"] == source["resources"]
    assert shown["parent_skill_version_ids"] == [first]
    assert shown["commit_message"] == source["commit_message"]
    assert shown["version_label"] == "v1.1.0"
    assert run("resolve", initial["skill_id"])["skill_version_id"] == first
    cleared = run(
        "commit",
        second["skill_version_id"],
        "--json-skill",
        json.dumps({"content": CONTENT, "commit_message": "Clear resources"}),
        "--activate",
    )
    assert run("show", cleared["skill_version_id"])["version"]["resources"] == {}
    assert run("resolve", initial["skill_id"])["skill_version_id"] == cleared["skill_version_id"]


def test_cli_async_endpoints_and_text_progress(cli_server, monkeypatch):
    from mindmemos_skill.demos.environment import DemoEnvironment

    run = cli_server
    submissions = []

    async def submit_evolution(self, project_id, version_id, *, pipeline, auto_accept):
        submissions.append({"project_id": project_id, "version_id": version_id, "auto_accept": auto_accept})
        return {"task_id": "task-cli", "status": "queued"}

    async def submit_training(self, request):
        submissions.append(request)
        return {"task_id": "task-cli", "status": "queued"}

    async def task_status(self, project_id, task_id):
        return {"task_id": task_id, "status": "succeeded"}

    async def task_progress(self, project_id, task_id, *, after=0):
        return (
            []
            if after >= 1
            else [
                {
                    "sequence": 1,
                    "occurred_at": "2026-10-10T10:00:00+08:00",
                    "payload": {"name": "train", "body": {"stage": "done"}, "message": "Complete"},
                }
            ]
        )

    monkeypatch.setattr(SkillService, "submit_evolution", submit_evolution)
    monkeypatch.setattr(SkillService, "submit_training", submit_training)
    monkeypatch.setattr(SkillService, "task_status", task_status)
    monkeypatch.setattr(SkillService, "task_progress", task_progress)
    registered = run("register", "--json-skill", json.dumps({"content": CONTENT}), "--alias", "async")
    assert run("evolve", "async")["status"] == "queued"
    assert submissions[-1]["auto_accept"] is False
    sample = {"task": {"id": "t1", "instruction": "Do the task"}, "reward_data": {}}
    assert (
        run(
            "train",
            "async",
            "--work-dir",
            "/run",
            "--pipeline",
            "algo",
            "--environment-json",
            json.dumps({"class_name": "demo_offline", "config": DemoEnvironment.config_type().model_dump()}),
            "--sample-json",
            json.dumps(sample),
            "--train-id",
            "t1",
        )["task_id"]
        == "task-cli"
    )
    assert submissions[-1]["init_skill_version_id"] == registered["skill_version_id"]
    assert submissions[-1]["project_id"] == "cli-project"
    assert run("task", "show", "task-cli")["status"] == "succeeded"
    assert run("task", "progress", "task-cli") == "[2026-10-10T02:00:00.000000Z] [INFO] [train] | Complete\n"
    assert run("task", "progress", "task-cli", "--after", "1") == ""
    assert (
        run("task", "progress", "task-cli", "--follow") == "[2026-10-10T02:00:00.000000Z] [INFO] [train] | Complete\n"
    )
