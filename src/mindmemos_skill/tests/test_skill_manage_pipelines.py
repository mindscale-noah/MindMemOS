"""Graph interface, topology, and cleanup contracts before algorithm integration."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.typing import (
    AddGraphMembersRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
)
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.db_services.graphs.typing import GraphNodeType, SkillGraphEdgeRecord
from mindmemos_skill.persistence.db_services.graphs.writer import SkillGraphStore
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import ValidationError

from mindmemos_skill import create_service

from .service_config import deployment_config
from .skill_sources import register_detail, structured


def service_config(path):
    """Configure graph queries without any concrete algorithm or model settings."""
    return deployment_config(
        {
            "home": path.parent / "home",
            "persistence": {"database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{path}"}}},
            "repository_pipeline": {"class_name": "repository"},
            "manage_pipeline": {"class_name": "graph_manage"},
        },
        "project",
    )


async def seed_topology(service):
    """Publish a real Skill/Scene fixture without calling a graph algorithm."""
    source = Skill(SkillContent(content="---\nname: csv\ndescription: CSV processing\n---\nRead rows"))
    detail = await register_detail(
        service,
        RegisterRequest(
            project_id="project",
            skill=structured(source.snapshot()),
        ),
    )
    version = detail.version
    with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as persistence:
        store = SkillGraphStore(
            persistence.database,
            graph_id="skg-fixture",
            alias="fixture",
            actor_id="test",
            version_ids={version.skill_id: version.version_id},
        )
        await store.validate_skills({version.skill_id: source})
        snapshot_id = "skgv-" + uuid4().hex
        skill_node = store.bind_node(snapshot_id, version.skill_id, {"node_id": "skill-node", "name": "csv"})
        scene_node = store.bind_node(
            snapshot_id,
            version.skill_id,
            {
                "node_id": "scene-node",
                "node_type": GraphNodeType.SCENE,
                "owner_node_id": "skill-node",
                "name": "Read CSV",
                "description": "Load the source rows",
            },
        )
        edge = SkillGraphEdgeRecord(
            project_id="project",
            graph_id="skg-fixture",
            graph_version_id=snapshot_id,
            edge_id=uuid4(),
            source_node_id="scene-node",
            target_node_id="skill-node",
            edge_type="containment",
            metadata={"condition": "File exists", "evidence": "Source instructions"},
        )
        await store.publish(
            snapshot_id, request_hash="a" * 64, nodes=[skill_node, scene_node], edges=[edge], metadata={}
        )
    return snapshot_id


async def test_graph_queries_include_scenes_and_preserve_history(tmp_path):
    async with await create_service(service_config(tmp_path / "graphs.db")) as service:
        assert await service.list_graph() == []
        snapshot_id = await seed_topology(service)
        topology = await service.get_graph_snapshot("fixture", snapshot_id)
        preview = await service.preview_graph(snapshot_id)
        assert preview == topology
        assert {node["node_type"] for node in topology["nodes"]} == {GraphNodeType.SKILL, GraphNodeType.SCENE}
        scene = next(node for node in topology["nodes"] if node["node_type"] == GraphNodeType.SCENE)
        assert scene["owner_node_id"] == "skill-node"
        assert scene["skill_version_id"] == topology["nodes"][0]["skill_version_id"]
        assert topology["edges"][0]["condition"] == "File exists"
        assert "embedding" not in str(topology)
        assert len(await service.list_graph_snapshots("fixture")) == 1
        with pytest.raises(ValueError, match="default project"):
            await service.preview_graph("fixture", "foreign")
        with TestClient(create_app(service)) as client:
            response = client.post("/v1/skills/graphs/preview", json={"graph_ref": "fixture"})
            assert response.status_code == 200, response.text
            assert response.json()["data"]["graph_version_id"] == snapshot_id
        # The HTTP app owns and closes the service.


async def test_graph_algorithm_registered_and_missing_models_reported(tmp_path):
    async with await create_service(service_config(tmp_path / "unavailable.db")) as service:
        assert "skill_manage" in Algorithm.registered_types()
        with pytest.raises(ValueError, match="algorithm settings"):
            await service.organize_skills(OrganizeRequest(project_id="project", skill_refs=[]))
        assert await service.list_graph() == []


@pytest.mark.parametrize("updates", [{"skill_refs": []}, {"skill_refs": ["s", "s"]}, {"base_revision": 1}])
def test_member_requests_reject_invalid_batches_and_old_protocol(updates):
    with pytest.raises(ValidationError):
        AddGraphMembersRequest.model_validate(
            {"project_id": "project", "graph_ref": "g", "skill_refs": ["s"], **updates}
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"include_scenes": True},
        {"strategy": "vector", "include_scenes": "false"},
        {"strategy": "vector", "include_scenes": 1},
        {"top_k": True},
        {"top_k": "5"},
        {"top_k": 1.5},
        {"top_k": 0},
        {"top_k": -1},
        {"diagnostics": True},
        {"task": " "},
        {"strategy": "vector", "task": " "},
    ],
)
def test_recommendation_request_validates_strategy(updates):
    with pytest.raises(ValidationError):
        RecommendRequest.model_validate({"project_id": "project", "graph_id": "g", "task": "rows", **updates})


async def test_http_member_routes_forward_scope_and_reject_legacy_fields(tmp_path, monkeypatch):
    from mindmemos_skill.api.typing import OrganizeResult

    service = await create_service(service_config(tmp_path / "http.db"))
    captured = []

    async def add(request):
        captured.append(request)
        return OrganizeResult(graph_id="skg-test", graph_version_id="skgv-" + uuid4().hex, skill_version_ids=["v1"])

    monkeypatch.setattr(service._manage_pipeline, "add_graph_members", add)
    with TestClient(create_app(service)) as client:
        body = {"graph_ref": "g", "skill_refs": ["s"], "force_extract_scene": True}
        response = client.post("/v1/skills/graphs/members/add", json=body)
        assert response.status_code == 200, response.text
        assert captured[0].project_id == "project" and captured[0].force_extract_scene
        for extra in ({"project_id": "other"}, {"request_id": "old"}, {"base_revision": 1}):
            assert client.post("/v1/skills/graphs/members/add", json=body | extra).status_code == 422
        response = client.post("/v1/skills/graphs/members/remove", json={"graph_ref": "g", "skill_refs": ["s"]})
        assert response.status_code == 404, response.text


async def test_vector_options_and_core_scores_cross_http_boundary(tmp_path, monkeypatch):
    from mindmemos_skill.api.typing import RecommendResult, VectorRecommendationScore

    service = await create_service(service_config(tmp_path / "recommend.db"))
    snapshot_id = await seed_topology(service)
    topology = await service.get_graph_snapshot("fixture", snapshot_id)
    version = await service.get_skill_version(topology["nodes"][0]["skill_version_id"])
    captured = []

    async def recommend(request):
        captured.append(request)
        return RecommendResult(
            skills=[version],
            graph_version_id=request.graph_version_id,
            strategy="vector",
            score_type="cosine_similarity",
            include_scenes=request.include_scenes,
            scores=[
                VectorRecommendationScore(
                    skill_version_id=version.version_id,
                    similarity=0.75,
                    node_id="scene-node",
                    node_type=GraphNodeType.SCENE,
                )
            ],
        )

    monkeypatch.setattr(service._manage_pipeline, "recommend", recommend)
    with TestClient(create_app(service)) as client:
        body = {"graph_ref": "fixture", "query": "Read rows", "strategy": "vector", "include_scenes": True}
        response = client.post("/v1/skills/algos/recommend", json=body)
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        assert data["graph_version_id"] == snapshot_id
        assert data["scores"][0]["similarity"] == 0.75
        assert data["skills"][0]["version_id"] == data["scores"][0]["skill_version_id"]
        assert captured[0].task == "Read rows" and captured[0].include_scenes
        assert captured[0].graph_version_id == snapshot_id
        assert client.post("/v1/skills/algos/recommend", json=body | {"strategy": "graph"}).status_code == 422
        assert client.post("/v1/skills/algos/recommend", json=body | {"diagnostics": True}).status_code == 422
