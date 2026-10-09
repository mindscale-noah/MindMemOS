"""Behavioral checks for credentials, previews, and graph isolation."""

from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.errors import ApiError
from mindmemos_sdk.memory.models import MemoryPageResult, MemorySearchHit, SearchResult
from mindmemos_skill.errors import GraphNotFoundError
from mindmemos_skill.skill import content_hash, skill_hash
from mindmemos_ui.app import create_app


@pytest_asyncio.fixture
async def client(tmp_path):
    manager = ConfigManager(tmp_path / "sdk")
    config = manager.load_or_default()
    config.memory_service.auth.api_key = "private-secret-value"
    manager.save(config)
    with TestClient(await create_app(manager)) as client:
        client.headers["x-workbench-token"] = client.get("/api/session").json()["token"]
        yield client, manager


def test_config_redaction_and_connection_preservation(client):
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
    assert saved.memory_service.auth.api_key == "private-secret-value"
    assert saved.memory_service.base_url == "https://example.org"
    body["clear_api_key"] = True
    assert http.put("/api/config", json=body).status_code == 200
    assert manager.load().memory_service.auth.api_key is None


def test_session_origin_and_validation_do_not_leak_secrets(client):
    http, _ = client
    assert http.post("/api/config/test", headers={"x-workbench-token": "bad"}).status_code == 403
    assert http.get("/api/session", headers={"Origin": "https://evil.example"}).status_code == 403
    assert http.get("/api/session", headers={"Host": "evil.example"}).status_code == 400
    response = http.put("/api/config", json={"api_key": "secret-not-to-echo"})
    assert response.status_code == 422
    assert "secret-not-to-echo" not in response.text


@pytest.mark.parametrize(
    "status,message,body,expected",
    [
        (401, "Invalid credential", None, "Invalid credential"),
        (422, "Invalid Skill version", None, "Invalid Skill version"),
        (500, "Database unavailable", None, "Database unavailable"),
        (422, "API error (HTTP 422)", {"detail": "Invalid source path"}, "Invalid source path"),
    ],
)
def test_sdk_http_error_preserves_status_and_message(client, monkeypatch, status, message, body, expected):
    """Expose the upstream HTTP failure without replacing its useful explanation."""
    http, _ = client

    class FailingSkills:
        def list(self, request):
            raise ApiError(message, status_code=status, response_body=body)

    class FailingSDK:
        def __init__(self, *, config):
            self.skills = FailingSkills()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr("mindmemos_ui.app.MindMemOSClient", FailingSDK)
    response = http.post("/api/config/test")
    assert response.status_code == status
    assert response.json() == {"detail": expected}


def test_connections_check_memory_and_skill_independently(client, monkeypatch):
    http, manager = client
    calls = []

    class MemoryProbe:
        def get(self, *, top_k):
            calls.append(("memory", top_k))
            raise ApiError("Invalid credential", status_code=401)

    class SkillProbe:
        def list(self, request):
            calls.append(("skill", request.limit))
            return {"items": []}

    class SDKProbe:
        def __init__(self, *, config):
            assert config.network.timeout_seconds == 5
            assert config.network.max_retries == 0
            self.memory = MemoryProbe()
            self.skills = SkillProbe()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr("mindmemos_ui.app.MindMemOSClient", SDKProbe)
    response = http.get("/api/connections")
    assert response.status_code == 200
    assert response.json()["memory"]["status"] == "unauthorized"
    assert response.json()["skill"]["status"] == "connected"
    assert calls == [("memory", 1), ("skill", 1)]
    assert "private-secret-value" not in response.text

    config = manager.load()
    config.memory_service.auth.api_key = None
    manager.save(config)
    calls.clear()
    response = http.get("/api/connections")
    assert response.json()["memory"]["status"] == "unconfigured"
    assert response.json()["skill"]["status"] == "unconfigured"
    assert calls == []


def test_memories_page_uses_read_only_sdk_list_and_search(client, monkeypatch):
    """Keep browser memory requests scoped to the configured SDK client."""
    http, manager = client
    calls = []
    memory = MemorySearchHit(id="m-1", memory="Example memory", user_id="cyq", memory_type="fact")
    config = manager.load()
    config.defaults.user_id = "cyq"
    manager.save(config)

    class MemoryClient:
        def list(self, *, page, page_size, user_id):
            calls.append(("list", page, page_size, user_id))
            return MemoryPageResult(memories=[memory], page=page, page_size=page_size, total=1)

        def search(self, query, *, top_k, user_id):
            calls.append(("search", query, top_k, user_id))
            return SearchResult(memories=[memory])

    class FakeClient:
        def __init__(self, *, config):
            assert config.memory_service.auth.api_key == "private-secret-value"
            self.memory = MemoryClient()

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    monkeypatch.setattr("mindmemos_ui.app.MindMemOSClient", FakeClient)
    listed = http.get("/api/memories?page=2&page_size=5")
    searched = http.post("/api/memories/search", json={"query": "Example"})
    filtered = http.post("/api/memories/search", json={"query": "Example", "top_k": 5, "user_id": " other "})

    assert listed.status_code == 200
    assert listed.json()["memories"][0]["memory"] == "Example memory"
    assert listed.json()["memories"][0]["user_id"] == "cyq"
    assert searched.status_code == 200
    assert searched.json()["memories"][0]["id"] == "m-1"
    assert filtered.status_code == 200
    assert calls == [("list", 2, 5, "cyq"), ("search", "Example", 20, "cyq"), ("search", "Example", 5, "other")]
    assert "private-secret-value" not in listed.text + searched.text + filtered.text
    assert http.get("/api/memories?page=0").status_code == 422
    assert http.post("/api/memories/search", json={"query": "   "}).status_code == 422
    assert http.post("/api/memories/search", json={"query": "Example", "top_k": 0}).status_code == 422
    assert http.post("/api/memories/search", json={"query": "Example", "user_id": "   "}).status_code == 422


def test_graph_queries_resolve_versions_and_isolate_projects(tmp_path):
    from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
    from mindmemos_skill.persistence import create_persistence, ensure_collection
    from mindmemos_skill.persistence.constants import (
        SKILL_GRAPH_EDGES_TABLE,
        SKILL_GRAPH_NODES_TABLE,
        SKILL_GRAPH_VERSIONS_TABLE,
        SKILL_GRAPHS_TABLE,
        SKILL_VERSIONS_TABLE,
        SKILLS_TABLE,
    )
    from mindmemos_skill.persistence.db_services.skill_management.service import GraphQueryService
    from mindmemos_skill.persistence.db_services.skill_management.typing import (
        SkillGraphEdgeRecord,
        SkillGraphNodeRecord,
        SkillGraphRecord,
        SkillGraphVersionRecord,
    )
    from mindmemos_skill.persistence.db_services.skills.typing import SkillRecord, SkillVersionRecord

    db = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'graphs.db'}"))
    try:
        store = create_persistence(db)
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
                content="content",
                description="Instructions",
                standard_format=False,
                skill_hash=skill_hash("content", {}),
                content_hash=content_hash("content"),
            )
            db.insert(SKILL_VERSIONS_TABLE, [record.model_dump()])
            db.insert(
                SKILLS_TABLE,
                [
                    SkillRecord(
                        project_id="project-a",
                        skill_id=f"skill-{i}",
                        name=f"Skill {i}",
                        active_version_id=f"version-{i}",
                    ).model_dump()
                ],
            )
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
        with pytest.raises(GraphNotFoundError):
            other.snapshot("g", snapshot)
        with pytest.raises(ValueError, match="Skill version not found"):
            other.version("version-1")
    finally:
        db.close()


def test_static_entrypoint(client):
    http, _ = client
    response = http.get("/")
    assert response.status_code == 200
    assert "/assets/" in response.text


def test_cli_starts_without_a_local_skill_database(tmp_path, monkeypatch):
    from mindmemos_ui.cli import main

    monkeypatch.setenv("MINDMEMOS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("sys.argv", ["mindmemos-ui"])
    calls = []

    async def serve(server):
        with TestClient(server.config.app) as http:
            assert http.get("/api/session").json()["graphs_connected"] is False
            assert http.get("/api/skills").status_code == 503
        calls.append((server.config.host, server.config.port))

    monkeypatch.setattr("mindmemos_ui.cli.uvicorn.Server.serve", serve)
    main()
    assert calls == [("127.0.0.1", 8765)]
    assert not (tmp_path / "home" / "data").exists()
