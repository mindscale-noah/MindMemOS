"""Behavioral checks for credentials, previews, and graph isolation."""

from datetime import datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_ui.app import create_app


@pytest_asyncio.fixture
async def client(tmp_path):
    manager = ConfigManager(tmp_path / "sdk")
    config = manager.load_or_default()
    config.auth.api_key = "private-secret-value"
    config.skills = [{"skill_name": "original", "path": str(tmp_path / "original"), "skill_id": "local-1"}]
    manager.save(config)
    with TestClient(await create_app(manager)) as client:
        client.headers["x-workbench-token"] = client.get("/api/session").json()["token"]
        yield client, manager


def test_config_redaction_and_registry_preservation(client):
    http, manager = client
    response = http.get("/api/config")
    assert "private-secret" not in response.text
    body = response.json()
    body.pop("path")
    body.pop("api_key_set")
    body["api_key"] = ""
    body["base_url"] = "https://example.org"
    assert http.put("/api/config", json=body).status_code == 200
    saved = manager.load()
    assert saved.auth.api_key == "private-secret-value"
    assert saved.skills[0]["skill_id"] == "local-1"
    assert saved.base_url == "https://example.org"
    body["clear_api_key"] = True
    assert http.put("/api/config", json=body).status_code == 200
    assert manager.load().auth.api_key is None


def test_session_origin_and_validation_do_not_leak_secrets(client):
    http, _ = client
    assert http.post("/api/config/test", headers={"x-workbench-token": "bad"}).status_code == 403
    assert http.get("/api/session", headers={"Origin": "https://evil.example"}).status_code == 403
    assert http.get("/api/session", headers={"Host": "evil.example"}).status_code == 400
    response = http.put("/api/config", json={"api_key": "secret-not-to-echo"})
    assert response.status_code == 422
    assert "secret-not-to-echo" not in response.text


def test_graph_queries_resolve_versions_and_isolate_projects(tmp_path):
    from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
    from mindmemos_skill.persistence import create_persistence, ensure_collection
    from mindmemos_skill.persistence.constants import (
        SKILL_GRAPH_EDGES_TABLE,
        SKILL_GRAPH_NODES_TABLE,
        SKILL_GRAPH_VERSIONS_TABLE,
        SKILL_GRAPHS_TABLE,
        SKILL_VERSIONS_TABLE,
    )
    from mindmemos_skill.persistence.db_services.skill_management.service import GraphQueryService
    from mindmemos_skill.persistence.db_services.skill_management.typing import (
        SkillGraphEdgeRecord,
        SkillGraphNodeRecord,
        SkillGraphRecord,
        SkillGraphVersionRecord,
    )
    from mindmemos_skill.persistence.db_services.skills.typing import SkillVersionRecord

    db = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'graphs.db'}"))
    try:
        store = create_persistence(db, include_remote=False)
        ensure_collection(store)
        snapshot = uuid4()
        scope = {"project_id": "project-a", "graph_id": "g"}
        db.insert(SKILL_GRAPHS_TABLE, [SkillGraphRecord(**scope, name="Graph A").model_dump()])
        db.insert(
            SKILL_GRAPH_VERSIONS_TABLE, [SkillGraphVersionRecord(**scope, graph_version_id=snapshot).model_dump()]
        )
        for i in (1, 2):
            record = SkillVersionRecord(
                project_id="project-a",
                version_id=f"version-{i}",
                skill_id=f"skill-{i}",
                name=f"Skill {i}",
                bundle="content",
                content_hash="hash",
            )
            db.insert(SKILL_VERSIONS_TABLE, [record.model_dump()])
            node = SkillGraphNodeRecord(
                **scope, graph_version_id=snapshot, node_id=f"n{i}", skill_version_id=f"version-{i}", name=f"Skill {i}"
            )
            db.insert(SKILL_GRAPH_NODES_TABLE, [node.model_dump()])
        edge = SkillGraphEdgeRecord(
            **scope,
            graph_version_id=snapshot,
            edge_id=uuid4(),
            source_node_id="n1",
            target_node_id="n2",
            edge_type="flow",
        )
        db.insert(SKILL_GRAPH_EDGES_TABLE, [edge.model_dump()])
        service = GraphQueryService(db, "project-a")
        result = service.snapshot("g", snapshot)
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["skill_version_id"] == "version-1"
        assert result["edges"][0]["edge_type"] == "flow"
        other = GraphQueryService(db, "project-b")
        assert other.list_graphs() == []
        with pytest.raises(ValueError):
            other.snapshot("g", snapshot)
        with pytest.raises(ValueError):
            other.version("version-1")
    finally:
        db.close()


def register(http, directory, name="Example"):
    preview = http.post("/api/register/preview", json={"path": str(directory), "name": name})
    assert preview.status_code == 200, preview.text
    result = http.post("/api/register/confirm", json={"plan_id": preview.json()["plan_id"]})
    assert result.status_code == 200, result.text
    return result.json()


def test_registration_preview_rejects_changed_resources(client, tmp_path):
    http, _ = client
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("original")
    (source / "asset.bin").write_bytes(b"old")
    preview = http.post("/api/register/preview", json={"path": str(source), "name": "Example"})
    assert preview.status_code == 200
    assert http.get("/api/skills").json() == []
    (source / "asset.bin").write_bytes(b"changed")
    assert http.post("/api/register/confirm", json={"plan_id": preview.json()["plan_id"]}).status_code == 409
    assert http.get("/api/skills").json() == []


def test_local_workflow_needs_no_cloud(client, tmp_path, monkeypatch):
    http, manager = client

    def no_network(*args, **kwargs):
        raise AssertionError("Local operations must not contact the cloud")

    monkeypatch.setattr("mindmemos_sdk.transport.HttpTransport._send_with_retries", no_network)
    config = manager.load()
    config.auth.api_key = None
    manager.save(config)
    before = manager.config_path.read_bytes()
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").write_text("original")
    (root / "asset.bin").write_bytes(b"\x00asset")
    result = register(http, root)
    skill_id = result["skill"]["skill_id"]
    first = result["version"]["version_id"]
    assert http.get("/api/skills").json()[0]["current_version_id"] == first
    (root / "SKILL.md").write_text("updated")
    commit = {"skill_id": skill_id, "path": str(root), "base_version_id": first, "message": "Changed"}
    second = http.post("/api/commit", json=commit)
    assert second.status_code == 200, second.text
    latest = second.json()["version"]["version_id"]
    assert http.post("/api/commit", json=commit).status_code == 409
    assert len(http.get("/api/versions", params={"skill_id": skill_id}).json()) == 2
    ref = {"source": "local", "skill_id": skill_id, "version_id": first}
    content = http.get("/api/content", params=ref).json()
    assert content == {"files": {"SKILL.md": "original"}, "resources": ["asset.bin"], "binary_files": ["asset.bin"]}
    diff = http.post("/api/diff", json={"before": ref, "after": {**ref, "version_id": latest}})
    assert diff.status_code == 200
    assert "updated" in diff.json()["diff"]
    plan = http.post("/api/switch/preview", json={"skill_id": skill_id, "version_id": first}).json()
    assert http.post("/api/switch/confirm", json={"plan_id": plan["plan_id"]}).status_code == 200
    assert http.get("/api/skills").json()[0]["current_version_id"] == first
    assert (root / "SKILL.md").read_text() == "updated"
    target = tmp_path / "export"
    preview = http.post("/api/export/preview", json={"version": ref, "directory": str(target)})
    assert preview.status_code == 200, preview.text
    assert not target.exists()
    confirm = {"plan_id": preview.json()["plan_id"]}
    assert http.post("/api/export/confirm", json=confirm).status_code == 200
    assert http.post("/api/export/confirm", json=confirm).status_code == 409
    assert (target / "SKILL.md").read_text() == "original"
    assert (target / "asset.bin").read_bytes() == b"\x00asset"
    assert http.delete(f"/api/skills/{skill_id}").status_code == 200
    assert http.get("/api/skills").json() == []
    assert target.exists() and root.exists()
    assert manager.config_path.read_bytes() == before


def test_export_detects_destination_changes_and_protects_home(client, tmp_path):
    http, manager = client
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("new")
    result = register(http, source)
    ref = {"source": "local", "skill_id": result["skill"]["skill_id"], "version_id": result["version"]["version_id"]}
    target = tmp_path / "destination"
    target.mkdir()
    (target / "keep.bin").write_bytes(b"original")
    body = {"version": ref, "directory": str(target)}
    plan = http.post("/api/export/preview", json=body).json()
    assert plan["overwrite"] is True
    (target / "keep.bin").write_bytes(b"changed")
    assert http.post("/api/export/confirm", json={"plan_id": plan["plan_id"]}).status_code == 409
    assert (target / "keep.bin").read_bytes() == b"changed"
    plan = http.post("/api/export/preview", json=body).json()
    assert http.post("/api/export/confirm", json={"plan_id": plan["plan_id"]}).status_code == 200
    assert not (target / "keep.bin").exists()
    assert (target / "SKILL.md").read_text() == "new"
    assert http.post("/api/export/preview", json={**body, "directory": str(manager.config_dir)}).status_code == 400
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    assert http.post("/api/export/preview", json={**body, "directory": str(link)}).status_code == 400


@pytest.mark.asyncio
async def test_reopening_home_preserves_identity_and_skills(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("MINDMEMOS_HOME", str(home))
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text("persistent")
    with TestClient(await create_app()) as http:
        session = http.get("/api/session").json()
        http.headers["x-workbench-token"] = session["token"]
        original = register(http, source)
    with TestClient(await create_app()) as http:
        assert http.get("/api/session").json()["project_id"] == session["project_id"]
        assert http.get("/api/skills").json()[0]["skill_id"] == original["skill"]["skill_id"]
    assert not (home / "service.yaml").exists()


def test_connected_graph_and_static_entrypoint(client):
    http, _ = client
    assert http.get("/api/graphs").status_code == 200
    response = http.get("/")
    assert response.status_code == 200
    assert "/assets/" in response.text


def test_cloud_routes_are_not_used_for_central_management(client):
    http, _ = client
    assert http.get("/api/cloud-skills").status_code == 404
    assert http.delete("/api/cloud-skills/old-id").status_code == 404
    assert http.get("/api/versions", params={"source": "cloud", "skill_id": "old-id"}).status_code == 400


def test_cli_without_database_or_project_flags(tmp_path, monkeypatch):
    from mindmemos_ui.cli import main

    monkeypatch.setenv("MINDMEMOS_HOME", str(tmp_path / "default-home"))
    monkeypatch.setattr("sys.argv", ["mindmemos-ui"])
    calls = []

    async def serve(server):
        with TestClient(server.config.app) as client:
            assert client.get("/api/session").json()["graphs_connected"] is True
            assert client.get("/api/graphs").json() == []
            assert client.get("/api/skills").json() == []
        calls.append((server.config.host, server.config.port))

    monkeypatch.setattr("mindmemos_ui.cli.uvicorn.Server.serve", serve)
    main()
    assert calls == [("127.0.0.1", 8765)]
    assert (tmp_path / "default-home/data/central.db").is_file()


@pytest.mark.asyncio
async def test_graph_navigation_and_borrowed_service_lifetime(tmp_path, monkeypatch):
    from mindmemos_skill.api import create_service
    from mindmemos_skill.persistence.constants import (
        SKILL_GRAPH_NODES_TABLE,
        SKILL_GRAPH_VERSIONS_TABLE,
        SKILL_GRAPHS_TABLE,
    )
    from mindmemos_skill.persistence.db_services.skill_management.typing import (
        SkillGraphNodeRecord,
        SkillGraphRecord,
        SkillGraphVersionRecord,
    )

    manager = ConfigManager(tmp_path / "home")
    service = await create_service(home=manager.config_dir)
    calls = []
    original_close = service.close
    monkeypatch.setattr(service, "close", lambda: calls.append("closed"))
    try:
        graph = await service.graph_queries()
        with TestClient(
            await create_app(manager, service=service, project_id=service.local_project_id, graph_service=graph)
        ) as http:
            http.headers["x-workbench-token"] = http.get("/api/session").json()["token"]
            source = tmp_path / "source"
            source.mkdir()
            (source / "SKILL.md").write_text("graph content")
            result = register(http, source)
            snapshot = uuid4()
            scope = {"project_id": service.local_project_id, "graph_id": "example"}
            service.backends["database"].insert(
                SKILL_GRAPHS_TABLE, [SkillGraphRecord(**scope, name="Example").model_dump()]
            )
            service.backends["database"].insert(
                SKILL_GRAPH_VERSIONS_TABLE, [SkillGraphVersionRecord(**scope, graph_version_id=snapshot).model_dump()]
            )
            service.backends["database"].insert(
                SKILL_GRAPH_NODES_TABLE,
                [
                    SkillGraphNodeRecord(
                        **scope,
                        graph_version_id=snapshot,
                        node_id="node",
                        name="Example",
                        skill_version_id=result["version"]["version_id"],
                    ).model_dump()
                ],
            )
            nodes = http.get(f"/api/graphs/example/snapshots/{snapshot}").json()["nodes"]
            assert nodes[0]["skill_id"] == result["skill"]["skill_id"]
            reference = {
                "source": "graph",
                "skill_id": nodes[0]["skill_id"],
                "version_id": nodes[0]["skill_version_id"],
            }
            assert http.get("/api/content", params=reference).json()["files"]["SKILL.md"] == "graph content"
            assert http.delete(f"/api/skills/{nodes[0]['skill_id']}").status_code == 409
        assert calls == []
        other = tmp_path / "other"
        with TestClient(await create_app(ConfigManager(other))) as http:
            assert http.get("/api/skills").json() == []
            assert http.get("/api/content", params=reference).status_code == 404
    finally:
        await original_close()


def test_bundle_content_and_per_file_diff(client, tmp_path):
    """Expose auxiliary text and preserve additions, deletions, and binary changes."""
    http, _ = client
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "references").mkdir()
    (root / "SKILL.md").write_text("unchanged\n")
    (root / "references" / "guide.txt").write_text("before\n")
    (root / "removed.txt").write_text("removed\n")
    (root / "empty.txt").write_text("")
    (root / "asset.bin").write_bytes(b"\xff")
    result = register(http, root)
    first = result["version"]["version_id"]
    ref = {"source": "local", "skill_id": result["skill"]["skill_id"], "version_id": first}
    content = http.get("/api/content", params=ref).json()
    assert content["files"]["references/guide.txt"] == "before\n"
    assert content["files"]["empty.txt"] == ""
    assert content["binary_files"] == ["asset.bin"]
    (root / "references" / "guide.txt").write_text("after\n")
    (root / "removed.txt").unlink()
    (root / "added.txt").write_text("")
    (root / "asset.bin").write_bytes(b"\x00new")
    response = http.post(
        "/api/commit",
        json={
            "skill_id": ref["skill_id"],
            "path": str(root),
            "base_version_id": first,
        },
    )
    assert response.status_code == 200, response.text
    after = {**ref, "version_id": response.json()["version"]["version_id"]}
    response = http.post("/api/diff", json={"before": ref, "after": after})
    assert response.status_code == 200, response.text
    changes = {file["path"]: file for file in response.json()["files"]}
    assert set(changes) == {"references/guide.txt", "removed.txt", "added.txt", "asset.bin"}
    assert "-before" in changes["references/guide.txt"]["diff"]
    assert "+after" in changes["references/guide.txt"]["diff"]
    assert changes["removed.txt"]["change"] == "deleted"
    assert changes["added.txt"]["change"] == "added"
    assert changes["asset.bin"]["binary"] is True
    assert http.post("/api/diff", json={"before": after, "after": after}).json()["files"] == []


def test_ui_publishes_semantic_labels_and_reserves_algorithm_suffixes(client, tmp_path):
    http, _ = client
    root = tmp_path / "labeled"
    root.mkdir()
    (root / "SKILL.md").write_text("first")
    body = {"path": str(root), "name": "Labeled", "version_label": "v1.2.0"}
    preview = http.post("/api/register/preview", json=body)
    assert preview.status_code == 200
    result = http.post("/api/register/confirm", json={"plan_id": preview.json()["plan_id"]}).json()
    assert result["version"]["version_label"] == "v1.2.0"
    rows = http.get("/api/skills").json()
    assert rows[0]["current_version_label"] == "v1.2.0"
    assert rows[0]["latest_version_label"] == "v1.2.0"
    assert datetime.fromisoformat(rows[0]["created_at"]) == datetime.fromisoformat(result["skill"]["created_at"])
    assert rows[0]["cloud_sync_status"] == "disabled"
    commit = {
        "skill_id": result["skill"]["skill_id"],
        "path": str(root),
        "base_version_id": result["version"]["version_id"],
        "version_bump": "minor",
    }
    next_version = http.post("/api/commit", json=commit)
    assert next_version.status_code == 200, next_version.text
    assert next_version.json()["version"]["version_label"] == "v1.3.0"
    plan = http.post(
        "/api/switch/preview",
        json={
            "skill_id": result["skill"]["skill_id"],
            "version_id": result["version"]["version_id"],
        },
    ).json()
    assert http.post("/api/switch/confirm", json={"plan_id": plan["plan_id"]}).status_code == 200
    row = http.get("/api/skills").json()[0]
    assert row["current_version_label"] == "v1.2.0"
    assert row["latest_version_label"] == "v1.3.0"
    assert datetime.fromisoformat(row["created_at"]) == datetime.fromisoformat(result["skill"]["created_at"])
    assert http.post("/api/register/preview", json={**body, "version_label": "v1.02.0"}).status_code == 422
    assert http.post("/api/register/preview", json={**body, "version_label": "v1.2.0-grpo"}).status_code == 422
    assert http.post("/api/commit", json={**commit, "algorithm": "grpo"}).status_code == 422


@pytest.mark.asyncio
async def test_task_assembly_runs_real_skill_contract_and_compares_versions(tmp_path, monkeypatch):
    """Exercise real TreeSkill assembly with only the provider response replaced."""
    import json

    from mindmemos_skill.agents.buitin_agents.openai_chat import OpenAIChatAgent
    from mindmemos_skill.agents.buitin_agents.openai_chat.config import OpenAIChatConfig
    from mindmemos_skill.agents.constants import AGENT_API_KEY_ENV
    from mindmemos_skill.agents.typing import AgentResult
    from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
    from mindmemos_skill.api import create_service
    from mindmemos_skill.pipelines.typing import CommitRequest, RegisterRequest
    from mindmemos_skill.skill import Skill, SkillBundle, SkillResources

    monkeypatch.setenv(AGENT_API_KEY_ENV, "runtime-credential")
    calls = []

    def route(self, *, messages):
        assert self.config.api_key.get_secret_value() == "runtime-credential"
        user = messages[-1]["content"]
        calls.append(json.loads(user.split("Task:\n", 1)[1].split("\n\nSkill tree:", 1)[0]))
        return AgentResult(output='{"selected_subtree_ids": ["002"]}')

    monkeypatch.setattr(OpenAIChatAgent, "run", route)
    manager = ConfigManager(tmp_path / "home")
    service = await create_service(home=manager.config_dir)
    bundle = SkillBundle(files={"SKILL.md": "# Root\nContext\n## Sales\nSum\n## Clean\nRemove duplicates\n"})
    source = Skill(bundle=bundle, resources=SkillResources(files={"reference.txt": b"preserved"}))
    first = await service.register_skill(RegisterRequest(source=source.snapshot(), name="Example"))
    dynamic = TreeSkill(
        TreeSkillConfig(router=OpenAIChatConfig(model="test-model", api_key="not-real")),
        bundle=bundle,
        resources=source.resources,
    )
    second = await service.commit_skill(CommitRequest(skill_ref=first.skill.skill_id, source=dynamic.snapshot()))
    changed = dynamic.with_bundle(
        SkillBundle(files={"SKILL.md": bundle.files["SKILL.md"].replace("Sum", "Sum monthly")})
    )
    third = await service.commit_skill(CommitRequest(skill_ref=first.skill.skill_id, source=changed.snapshot()))
    refs = [
        {"source": "local", "skill_id": first.skill.skill_id, "version_id": item.version.version_id}
        for item in (first, second, third)
    ]
    try:
        with TestClient(await create_app(manager, service=service, project_id=service.local_project_id)) as http:
            http.headers["x-workbench-token"] = http.get("/api/session").json()["token"]
            for index, ref in enumerate(refs):
                info = http.get("/api/implementation", params=ref).json()
                assert info["dynamic"] == (index > 0)
                assert "tree" not in info and "config" not in info
                assert http.get("/api/content", params=ref).status_code == 200
            assert calls == []
            assert http.get("/api/skills").json()[0]["implementation"] == "treeskill"
            body = {"versions": refs[1:], "instruction": "Summarize monthly sales"}
            result = http.post("/api/assemble", json=body)
            assert result.status_code == 200, result.text
            response = result.json()
            assert len(calls) == 2 and calls[0] == calls[1] == response["task"]
            assert "Clean" not in response["contents"][0]["files"]["SKILL.md"]
            assert response["contents"][0]["files"]["reference.txt"] == "preserved"
            assert "+Sum monthly" in response["comparison"]["files"][0]["diff"]
            calls.clear()
            same = http.post("/api/assemble", json={**body, "versions": [refs[1], refs[1]]}).json()
            assert len(calls) == 1 and same["comparison"]["files"] == []
            calls.clear()
            mixed = http.post("/api/assemble", json={**body, "versions": refs[:2]}).json()
            assert len(calls) == 1
            assert mixed["contents"][0]["files"]["SKILL.md"] == bundle.files["SKILL.md"]
            assert len(http.get("/api/versions", params={"skill_id": first.skill.skill_id}).json()) == 3
            assert http.get("/api/content", params=refs[1]).json()["files"]["SKILL.md"] == bundle.files["SKILL.md"]
            assert http.post("/api/assemble", json={**body, "instruction": "   "}).status_code == 422
            assert http.post("/api/assemble", json=body, headers={"x-workbench-token": "bad"}).status_code == 403
            invalid = {**refs[1], "skill_id": "missing"}
            assert http.post("/api/assemble", json={**body, "versions": [invalid]}).status_code == 404
            assert http.post("/api/treeskill/preview", json={}).status_code == 404

            def fail(*args, **kwargs):
                raise RuntimeError("secret-provider-key")

            monkeypatch.setattr(OpenAIChatAgent, "run", fail)
            failure = http.post("/api/assemble", json=body)
            assert failure.status_code == 502 and "secret-provider-key" not in failure.text
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unknown_implementation_is_dynamic_but_never_silently_static(tmp_path):
    from mindmemos_skill.api import create_service
    from mindmemos_skill.pipelines.typing import RegisterRequest
    from mindmemos_skill.skill import Skill, SkillBundle

    manager = ConfigManager(tmp_path / "home")
    service = await create_service(home=manager.config_dir)
    source = Skill(bundle=SkillBundle(files={"SKILL.md": ""})).snapshot()
    source.implementation = "custom-dynamic"
    record = await service.register_skill(RegisterRequest(source=source, name="Custom"))
    ref = {"source": "local", "skill_id": record.skill.skill_id, "version_id": record.version.version_id}
    try:
        with TestClient(await create_app(manager, service=service, project_id=service.local_project_id)) as http:
            http.headers["x-workbench-token"] = http.get("/api/session").json()["token"]
            assert http.get("/api/implementation", params=ref).json()["dynamic"] is True
            assert http.get("/api/content", params=ref).json()["files"] == {"SKILL.md": ""}
            assert http.post("/api/assemble", json={"versions": [ref], "instruction": "Test"}).status_code == 501
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_custom_runtime_loader_and_absent_bundle(tmp_path, monkeypatch):
    """Generic adapters can restore new implementations; empty source stays empty."""
    from mindmemos_skill.api import create_service
    from mindmemos_skill.pipelines.typing import RegisterRequest
    from mindmemos_skill.skill import Skill, SkillBundle

    class CustomSkill(Skill):
        def assemble(self, task):
            return self.with_bundle(SkillBundle(files={"SKILL.md": task.instruction}))

    manager = ConfigManager(tmp_path / "home")
    service = await create_service(home=manager.config_dir)
    snapshot = Skill(bundle=SkillBundle(files={"SKILL.md": "source"})).snapshot()
    snapshot.implementation = "custom"
    record = await service.register_skill(RegisterRequest(source=snapshot, name="Custom"))
    ref = {"source": "local", "skill_id": record.skill.skill_id, "version_id": record.version.version_id}
    try:
        with TestClient(
            await create_app(
                manager,
                service=service,
                project_id=service.local_project_id,
                skill_loader=CustomSkill.from_snapshot,
            )
        ) as http:
            http.headers["x-workbench-token"] = http.get("/api/session").json()["token"]
            result = http.post("/api/assemble", json={"versions": [ref], "instruction": "Actual task"})
            assert result.status_code == 200
            assert result.json()["contents"][0]["files"]["SKILL.md"] == "Actual task"
            for missing in ("", "[]", None):
                absent = record.model_copy(update={"version": record.version.model_copy(update={"bundle": missing})})

                async def show_absent(request):
                    return absent

                monkeypatch.setattr(service, "show_skill", show_absent)
                assert http.get("/api/content", params=ref).json()["files"] == {}
                assert http.get("/api/implementation", params=ref).json()["dynamic"] is True
    finally:
        await service.close()
