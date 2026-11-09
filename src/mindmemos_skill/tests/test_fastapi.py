"""Verify the public Skill HTTP entry and explicit backend capability boundaries."""

import base64
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import httpx
import jwt
import pytest
import pytest_asyncio
import yaml
from fastapi.testclient import TestClient
from mindmemos_skill.algos.analyze.typing import SkillAnalysisResult
from mindmemos_skill.api.auth import GatewayJwtResolver, SkillHttpIdentity
from mindmemos_skill.api.constants import SKILL_API_HOST_ENV, SKILL_CONFIG_PATH_ENV
from mindmemos_skill.api.entries.fastapi import build_http_service, create_app
from mindmemos_skill.common.time_utils import utcnow
from mindmemos_skill.config.config import FastAPIConfig, HttpAuthConfig
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.constants import (
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
    TRAJECTORIES_TABLE,
    TRAJECTORY_VERSIONS_TABLE,
)
from mindmemos_skill.pipelines import BuiltinAnalyzePipeline
from mindmemos_skill.pipelines.builtin_pipelines.analyze.config import AnalyzePipelineConfig
from pydantic import BaseModel

from mindmemos_skill import MindMemOSSkill, create_service

from .service_config import deployment_config, repository_config

GATEWAY_SECRET = "test-gateway-secret-with-at-least-64-bytes-for-all-test-algorithms"


class _OfflineJudgeConfig(BaseModel):
    """Minimal analyzer configuration supplying stable version metadata."""

    algorithm_version: str = "1"
    prompt_version: str = "1"


class _OfflineJudge:
    """Deterministic analyzer that records offline HTTP invocations."""

    def __init__(self):
        self.algorithm_name = "offline_judge"
        self.config = _OfflineJudgeConfig()
        self.calls = 0

    async def analyze(self, request):
        """Return a stable completed analysis without external calls."""

        self.calls += 1
        return SkillAnalysisResult(
            summary="Offline evaluation",
            metrics={"score_mean": 4.0, "success": True},
        )


def install_offline_analyze(service):
    """Attach an offline Analyze pipeline to an assembled test service."""

    judge = _OfflineJudge()
    analyze = BuiltinAnalyzePipeline(
        AnalyzePipelineConfig(),
        persistence=service.repository.persistence,
    )
    analyze.skill_judge = judge
    analyze.trajectory_judge = judge
    service._analyze_pipeline = analyze
    return analyze, judge


def gateway_token(
    *,
    project_id: str = "project-a",
    scopes: list[str] | None = None,
    secret: str = GATEWAY_SECRET,
    algorithm: str = "HS256",
    **overrides,
) -> str:
    """Sign a Gateway JWT for the independent Skill data plane."""
    now = int(time.time())
    claims = {
        "iss": "mindmemos-gateway",
        "aud": "skill-data-plane",
        "sub": "account-a",
        "account_id": "account-a",
        "project_id": project_id,
        "api_key_uuid": "key-a",
        "memory_algorithm": "schema",
        "scopes": scopes if scopes is not None else ["skill:read"],
        "iat": now,
        "exp": now + 60,
    }
    claims.update(overrides)
    return jwt.encode(claims, secret, algorithm=algorithm)


@pytest_asyncio.fixture
async def service(tmp_path):
    """Build one local Skill service with an isolated persistent database."""
    return await create_service(repository_config(tmp_path / "home", "project"))


@pytest_asyncio.fixture
async def management_service(tmp_path, monkeypatch):
    """Build graph HTTP routes using the separate offline demo backend."""
    from mindmemos_skill.demos.graph import DemoGraphPipeline

    demo_config = DemoGraphPipeline.config_type(delay_seconds=0)

    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'manage-http.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
            "manage_pipeline": {"class_name": "demo_graph", "config": demo_config.model_dump()},
        },
        "project",
    )
    return await create_service(config)


def test_public_route_discovery_and_no_legacy_export(service):
    """Advertise target routes without publishing old or training routes."""
    assert isinstance(service, MindMemOSSkill)
    app = create_app(service)
    assert app.state.service is service
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/docs").status_code == 200
        paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/skills/register" in paths
    assert "/v1/skills/version-resolve" in paths
    assert "/v1/skills/skill-resolve" in paths
    assert "/v1/skills/trajectories/report" in paths
    assert "/v1/skills/algos/organize" in paths
    assert "/v1/skills/algos/recommend" in paths
    assert "/v1/skills/algos/analyze/skill" in paths
    assert "/v1/skills/algos/analyze/trajectory" in paths
    assert "/v1/skills/algos/evaluations/query" in paths
    assert "/v1/skills/algos/evolve" in paths
    assert "/v1/skills/tasks/show" in paths
    assert "201" in paths["/v1/skills/register"]["post"]["responses"]
    for route in ("evolve", "train"):
        assert "202" in paths[f"/v1/skills/algos/{route}"]["post"]["responses"]
    assert "200" in paths["/v1/skills/algos/organize"]["post"]["responses"]
    assert "200" in paths["/v1/skills/algos/recommend"]["post"]["responses"]
    assert not any(path.startswith("/skills/") for path in paths)
    assert "/v1/skills/algos/train" in paths
    assert "/v1/skills/tasks/progress" in paths
    assert not any("export" in path for path in paths)


def test_skill_http_preserves_uploaded_resources_and_exposes_graph_reads(service):
    """The SDK contract retains binary resources without changing the project scope."""
    with TestClient(create_app(service)) as client:
        register = client.post(
            "/v1/skills/register",
            json={
                "alias": "demo",
                "skill": {
                    "content": "---\nname: demo\ndescription: Example\n---\nOriginal",
                    "name": "demo",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                    "resources": {"assets/data.bin": base64.b64encode(b"\x00source").decode("ascii")},
                },
            },
        )
        assert register.status_code == 201, register.text
        skill_id = register.json()["data"]["skill_id"]
        initial = register.json()["data"]["skill_version_id"]
        shown = client.post("/v1/skills/show", json={"skill_id": skill_id})
        assert shown.json()["data"]["version"]["resources"]["assets/data.bin"] == base64.b64encode(
            b"\x00source"
        ).decode("ascii")

        committed = client.post(
            "/v1/skills/commit",
            json={
                "skill_id": skill_id,
                "active": True,
                "base_version_id": initial,
                "skill": {
                    "content": "---\nname: demo\ndescription: Example\n---\nUpdated",
                    "resources": {"assets/data.bin": base64.b64encode(b"\x00updated").decode("ascii")},
                    "commit_message": "Update",
                    "version_bump": "patch",
                },
            },
        )
        assert committed.status_code == 201, committed.text
        assert committed.json()["data"]["version_label"] == "v1.0.1"
        active_version_id = committed.json()["data"]["skill_version_id"]
        assert skill_id.startswith("sk-")
        assert initial.startswith("skv-")
        for reference in ("demo", skill_id, skill_id[:8]):
            resolved = client.post("/v1/skills/skill-resolve", json={"skill_ref": reference})
            assert resolved.status_code == 200, resolved.text
            assert resolved.json()["data"]["skill_version_id"] == active_version_id
        historical = client.post("/v1/skills/skill-resolve", json={"skill_ref": initial[:9]})
        assert historical.status_code == 200, historical.text
        assert historical.json()["data"]["skill_version_id"] == initial
        assert historical.json()["data"]["is_active"] is False
        history = client.post("/v1/skills/history", json={"skill_id": skill_id}).json()["data"]["items"]
        assert len(history) == 2
        assert all("implementation" in row and "status" in row and "origin" in row for row in history)
        assert client.post("/v1/skills/graphs/list", json={}).json()["data"]["items"] == []
        assert client.post("/v1/skills/graphs/snapshots", json={"graph_id": "missing"}).json()["data"]["items"] == []


def test_unavailable_backends_return_stable_errors(service):
    """Keep missing worker-backed operations explicit instead of returning success."""
    with TestClient(create_app(service)) as client:
        requests = {
            "/v1/skills/graph-resolve": {"graph_ref": "g"},
            "/v1/skills/algos/organize": {"alias": "g", "commit_message": "Organize", "skill_refs": []},
            "/v1/skills/algos/recommend": {
                "graph_ref": "g",
                "graph_version_id": "00000000-0000-0000-0000-000000000001",
                "query": "hello",
            },
            "/v1/skills/algos/evolve": {"skill_version_id": "v"},
            "/v1/skills/algos/analyze/skill": {"skill_version_id": "v"},
            "/v1/skills/algos/analyze/trajectory": {"trajectory_id": "t"},
            "/v1/skills/algos/evaluations/query": {"skill_version_ids": ["v"]},
            "/v1/skills/tasks/show": {"task_id": "t"},
        }
        for path, body in requests.items():
            response = client.post(path, json=body)
            assert response.status_code in {400, 404, 422, 501}, response.text
            assert response.json()["error"]["code"] in {
                "trajectory_api_unavailable",
                "graph_resolution_unavailable",
                "graph_not_found",
                "pipeline_unavailable",
                "invalid_request",
                "task_api_unavailable",
            }
        assert client.post("/v1/skills/register", json={}).status_code == 422
        assert client.post("/v1/skills/list", json={"project_id": "project"}).status_code == 422
        assert client.post("/v1/skills/list", json={"offset": 0}).status_code == 422
        assert client.post("/skills/register", json={}).status_code == 404


async def test_manage_http_routes_publish_and_recommend_pinned_versions(management_service):
    """Exercise graph HTTP contracts with the separate offline demo pipeline."""

    def resolver(token):
        scopes = {"skill:write"} if token == "admin" else {"skill:read"}
        return SkillHttpIdentity(project_id="project", scopes=scopes, actor_id="account-1")

    headers = {"Authorization": "Bearer admin"}
    with TestClient(create_app(management_service, identity_from_bearer=resolver)) as client:
        versions = []
        for name, description in (("csv", "Parse CSV tables"), ("images", "Draw pictures")):
            response = client.post(
                "/v1/skills/register",
                headers=headers,
                json={
                    "alias": name,
                    "skill": {
                        "content": f"---\nname: {name}\ndescription: {description}\n---\n{description}",
                        "name": name,
                        "commit_version": "v1.0.0",
                        "commit_message": "Initial version",
                    },
                },
            )
            assert response.status_code == 201, response.text
            versions.append(response.json()["data"]["skill_version_id"])

        body = {"alias": "collection", "commit_message": "Initial organization", "skill_refs": versions}
        organized = client.post("/v1/skills/algos/organize", headers=headers, json=body)
        assert organized.status_code == 200, organized.text
        snapshot = organized.json()["data"]["graph_version_id"]
        graph_id = organized.json()["data"]["graph_id"]
        assert graph_id.startswith("skg-")
        assert snapshot.startswith("skgv-")
        graph_current = client.post("/v1/skills/graph-resolve", headers=headers, json={"graph_ref": "collection"})
        assert graph_current.status_code == 200, graph_current.text
        assert graph_current.json()["data"]["graph_id"] == graph_id
        assert graph_current.json()["data"]["graph_version_id"] == snapshot
        graph_prefix = client.post("/v1/skills/graph-resolve", headers=headers, json={"graph_ref": graph_id[:8]})
        assert graph_prefix.json()["data"]["graph_id"] == graph_id
        pinned = client.post("/v1/skills/graph-resolve", headers=headers, json={"graph_ref": snapshot[:9]})
        assert pinned.json()["data"]["graph_version_id"] == snapshot
        assert organized.json()["data"]["skill_version_ids"] == versions
        graphs = client.post("/v1/skills/graphs/list", headers=headers, json={})
        assert graphs.status_code == 200, graphs.text
        assert any(
            row["graph_id"] == graph_id and row["alias"] == "collection" for row in graphs.json()["data"]["items"]
        )
        snapshots = client.post("/v1/skills/graphs/snapshots", headers=headers, json={"graph_id": "collection"})
        assert snapshots.status_code == 200, snapshots.text
        assert any(row["graph_version_id"] == snapshot for row in snapshots.json()["data"]["items"])
        snapshot_detail = client.post(
            "/v1/skills/graphs/snapshot",
            headers=headers,
            json={"graph_id": "collection", "graph_version_id": snapshot},
        )
        assert snapshot_detail.status_code == 200, snapshot_detail.text
        assert {node["skill_version_id"] for node in snapshot_detail.json()["data"]["nodes"]} == set(versions)
        with management_service.repository.persistence.session(Scope(fields={"project_id": "project"})) as scoped:
            snapshots = await scoped.database.select(SKILL_GRAPH_VERSIONS_TABLE, filters={"graph_id": graph_id})
        body = {"graph_id": graph_id, "commit_message": "Initial organization", "skill_refs": versions}
        assert snapshots[0]["actor_id"] == "account-1"
        assert snapshots[0]["message"] == "Initial organization"
        assert (
            client.post("/v1/skills/algos/organize", headers=headers, json=body).json()["data"]["graph_version_id"]
            == snapshot
        )
        query = {"graph_ref": "collection", "graph_version_id": snapshot, "query": "Parse CSV records", "top_k": 1}
        recommended = client.post("/v1/skills/algos/recommend", headers=headers, json=query)
        assert recommended.status_code == 200, recommended.text
        assert [item["version_id"] for item in recommended.json()["data"]["skills"]] == [versions[0]]

        newer = client.post(
            "/v1/skills/algos/organize",
            headers=headers,
            json={"graph_id": graph_id, "commit_message": "Image-only organization", "skill_refs": [versions[1]]},
        )
        assert newer.status_code == 200, newer.text
        assert newer.json()["data"]["graph_version_id"] != snapshot
        old_graph_version = client.post("/v1/skills/graph-resolve", headers=headers, json={"graph_ref": snapshot})
        assert old_graph_version.json()["data"]["graph_version_id"] == snapshot
        assert old_graph_version.json()["data"]["is_current"] is False
        assert [
            item["version_id"]
            for item in client.post("/v1/skills/algos/recommend", headers=headers, json=query).json()["data"]["skills"]
        ] == [versions[0]]
        latest = client.post(
            "/v1/skills/algos/recommend",
            headers=headers,
            json={"graph_ref": "collection", "query": "Parse CSV records", "top_k": 1},
        )
        assert latest.status_code == 200, latest.text
        assert [item["version_id"] for item in latest.json()["data"]["skills"]] == [versions[1]]

        reader = {"Authorization": "Bearer reader"}
        assert client.post("/v1/skills/algos/organize", headers=reader, json=body).status_code == 403

        assert client.post("/v1/skills/algos/recommend", headers=reader, json=query).status_code == 403
        assert client.post("/v1/skills/graphs/list", headers=reader, json={}).status_code == 200
        assert (
            client.post("/v1/skills/algos/organize", headers=headers, json=body | {"project_id": "other"}).status_code
            == 422
        )
        assert (
            client.post("/v1/skills/algos/recommend", headers=headers, json=query | {"project_id": "other"}).status_code
            == 422
        )


async def test_skill_rollout_mode_keeps_legacy_id_generation(tmp_path):
    """The compatibility phase shares deterministic ID formats with old writers."""
    config = deployment_config(
        {
            "home": tmp_path / "home",
            "persistence": {
                "config": {"prefixed_ids": False},
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'legacy-ids.db'}"}},
            },
            "repository_pipeline": {"class_name": "repository"},
        },
        "project",
    )
    service = await create_service(config)
    with TestClient(create_app(service)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "legacy",
                "skill": {
                    "content": "---\nname: legacy\ndescription: Legacy\n---\nLegacy",
                    "name": "legacy",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        assert registered.status_code == 201, registered.text
        assert registered.json()["data"]["skill_id"].startswith("sk_")
        assert registered.json()["data"]["skill_version_id"].startswith("v_")
        resolved = client.post("/v1/skills/skill-resolve", json={"skill_ref": "legacy"})
        assert resolved.json()["data"]["skill_version_id"] == registered.json()["data"]["skill_version_id"]


async def test_report_trajectory_uses_shared_record_and_preserves_unknown_start(service):
    """Convert sparse Harness messages into a project-owned physical attempt."""
    with TestClient(create_app(service)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "reported-skill",
                "skill": {
                    "content": "---\nname: reported-skill\ndescription: Example\n---\nBody",
                    "name": "reported-skill",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        assert registered.status_code == 201, registered.text
        detail = registered.json()["data"]
        version_id = detail["skill_version_id"]
        report = client.post(
            "/v1/skills/trajectories/report",
            json={
                "user_id": "user-1",
                "agent_id": "agent-1",
                "session_id": "session-1",
                "skill_version_ids": [version_id],
                "messages": [
                    {"role": "system", "content": "Be concise"},
                    {"role": "user", "content": "Summarize the file"},
                    {"role": "assistant", "content": "Done"},
                ],
            },
        )
        assert report.status_code == 201, report.text
        response = report.json()["data"]
        assert response["skill_version_ids"] == [version_id]
        with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
            record = await store.trajectories.get(response["trajectory_id"])
        assert record is not None
        assert record.source == "reported"
        assert record.task_id != record.trajectory_id
        assert record.rollout_id == record.trajectory_id
        assert record.task_system_prompt == "Be concise"
        assert record.task_instruction == "Summarize the file"
        assert record.env_ref == "unknown"
        assert record.started_at is None
        assert record.received_at == record.created_at
        assert record.bound_versions == {version_id: detail["skill_id"]}
        assert record.user_id == "user-1" and record.agent_id == "agent-1"
        assert record.session_id == "session-1"
        assert record.n_turn == 1
        assert record.trajectory[0]["role"] == "system"
        assert record.skill_bindings[0]["version_id"] == version_id
        assert record.skill_bindings[0]["name"] == "reported-skill"
        assert response["stored_at"] == record.received_at.isoformat()
        protected = client.post("/v1/skills/delete", json={"skill_id": detail["skill_id"]})
        assert protected.status_code == 200
        with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
            assert (await store.trajectories.get(response["trajectory_id"])) is not None
        missing = client.post(
            "/v1/skills/trajectories/report",
            json={
                "user_id": "user-1",
                "agent_id": "agent-1",
                "skill_version_ids": ["missing"],
                "messages": [{"role": "user", "content": "Question"}],
            },
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "skill_version_not_found"


def test_trajectory_report_accepts_only_the_version_list(service):
    """An empty list creates a report without attributing it to a Skill version."""
    with TestClient(create_app(service)) as client:
        path = "/v1/skills/trajectories/report"
        actor = {"user_id": "user-1", "agent_id": "agent-1", "messages": [{"role": "user", "content": "Task"}]}
        assert client.post(path, json={**actor, "skill_version_id": "old"}).status_code == 422
        assert client.post(path, json={**actor, "skill_version_ids": None}).status_code == 422
        response = client.post(path, json={**actor, "skill_version_ids": []})
        assert response.status_code == 201, response.text
        report = response.json()["data"]
        assert report["skill_version_ids"] == []
        shown = client.post("/v1/skills/trajectories/show", json={"trajectory_id": report["trajectory_id"]})
        assert shown.json()["data"]["skill_bindings"] == []


async def test_report_without_user_message_keeps_task_instruction_empty(service):
    """Do not invent a user instruction when a report only has Agent output."""
    with TestClient(create_app(service)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "output-only",
                "skill": {
                    "content": "---\nname: output-only\ndescription: Example\n---\nBody",
                    "name": "output-only",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        version_id = registered.json()["data"]["skill_version_id"]
        report = client.post(
            "/v1/skills/trajectories/report",
            json={
                "user_id": "user-1",
                "agent_id": "agent-1",
                "skill_version_ids": [version_id],
                "messages": [{"role": "assistant", "content": "Done"}],
            },
        )
        assert report.status_code == 201, report.text
        with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
            record = await store.trajectories.get(report.json()["data"]["trajectory_id"])
        assert record.task_instruction == ""
        assert record.task_system_prompt is None


async def test_report_show_list_cursor_and_same_execution_actor_filter(service):
    """Read reported attempts without cursor drift or cross-report actor matches."""
    with TestClient(create_app(service)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "tracked",
                "skill": {
                    "content": "---\nname: tracked\ndescription: Example\n---\nBody",
                    "name": "tracked",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        skill_id = registered.json()["data"]["skill_id"]
        version_id = registered.json()["data"]["skill_version_id"]

        def report(user_id, agent_id):
            response = client.post(
                "/v1/skills/trajectories/report",
                json={
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "skill_version_ids": [version_id],
                    "messages": [{"role": "user", "content": user_id}],
                },
            )
            assert response.status_code == 201, response.text
            return response.json()["data"]["trajectory_id"]

        older = report("user-a", "agent-a")
        newer = report("user-b", "agent-b")
        with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
            (await store.trajectories.mark_consumed(version_id, [older], "evolution-a"))
            (await store.trajectories.mark_consumed(version_id, [older], "evolution-b"))
        counts = {"total": 2, "consumed": 1, "unconsumed": 1}
        detail = client.post("/v1/skills/show", json={"skill_id": skill_id}).json()["data"]
        assert "trajectory_counts" not in detail["version"]
        history = client.post("/v1/skills/history", json={"skill_id": skill_id}).json()["data"]
        assert "trajectory_counts" not in history["items"][0]
        statistics = client.post(
            "/v1/skills/trajectories/statistics", json={"skill_id": skill_id, "skill_version_id": version_id}
        )
        assert statistics.status_code == 200, statistics.text
        aggregate = statistics.json()["data"]
        assert aggregate["totals"] == counts
        assert len(aggregate["days"]) == 30
        assert sum(day["total"] for day in aggregate["days"]) == 2
        assert aggregate["last_evolved_at"] is None
        assert "trajectory_id" not in statistics.text
        assert (
            client.post(
                "/v1/skills/trajectories/statistics", json={"skill_id": skill_id, "skill_version_id": "missing"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/v1/skills/trajectories/statistics",
                json={"skill_id": skill_id, "skill_version_id": version_id, "days": 90},
            ).status_code
            == 422
        )
        shown = client.post("/v1/skills/trajectories/show", json={"trajectory_id": older})
        assert shown.status_code == 200
        assert shown.json()["data"]["trajectory"][0]["content"] == "user-a"
        assert shown.json()["data"]["skill_version_ids"] == [version_id]
        assert client.post("/v1/skills/trajectories/show", json={"trajectory_id": "missing"}).status_code == 404

        first = client.post("/v1/skills/trajectories/list", json={"skill_id": skill_id, "limit": 1})
        assert first.status_code == 200, first.text
        first_page = first.json()["data"]
        assert [item["trajectory_id"] for item in first_page["items"]] == [newer]
        assert first_page["next_cursor"]
        late = report("user-c", "agent-c")
        second = client.post(
            "/v1/skills/trajectories/list",
            json={"skill_id": skill_id, "limit": 1, "cursor": first_page["next_cursor"]},
        )
        assert second.status_code == 200, second.text
        assert [item["trajectory_id"] for item in second.json()["data"]["items"]] == [older]
        assert second.json()["data"]["next_cursor"] is None
        assert late not in {newer, older}
        changed = client.post(
            "/v1/skills/trajectories/list",
            json={"skill_id": skill_id, "limit": 1, "cursor": first_page["next_cursor"], "status": "failed"},
        )
        assert changed.status_code == 422
        assert changed.json()["error"]["code"] == "invalid_cursor"

        matched = client.post("/v1/skills/list", json={"filters": {"user_id": "user-a", "agent_id": "agent-a"}})
        assert matched.status_code == 200, matched.text
        assert [item["skill_id"] for item in matched.json()["data"]["items"]] == [skill_id]
        mismatched = client.post("/v1/skills/list", json={"filters": {"user_id": "user-a", "agent_id": "agent-b"}})
        assert mismatched.status_code == 200, mismatched.text
        assert mismatched.json()["data"]["items"] == []


async def test_history_and_trajectory_pages_select_only_summary_fields(service, monkeypatch):
    """Keep large version and attempt content out of paginated database reads."""
    with TestClient(create_app(service)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "projected",
                "skill": {
                    "content": "---\nname: projected\ndescription: Example\n---\nBody",
                    "name": "projected",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        assert registered.status_code == 201, registered.text
        skill_id = registered.json()["data"]["skill_id"]
        version_id = registered.json()["data"]["skill_version_id"]
        for user_id in ("first", "second"):
            reported = client.post(
                "/v1/skills/trajectories/report",
                json={
                    "user_id": user_id,
                    "agent_id": "agent",
                    "skill_version_ids": [version_id],
                    "messages": [{"role": "user", "content": user_id}],
                },
            )
            assert reported.status_code == 201, reported.text

        database = service.backends["database"]
        original_select = database.select
        selected_fields = []

        def capture_select(table, **kwargs):
            selected_fields.append((table, kwargs.get("fields")))
            return original_select(table, **kwargs)

        monkeypatch.setattr(database, "select", capture_select)
        history = client.post("/v1/skills/history", json={"skill_id": skill_id, "limit": 1})
        assert history.status_code == 200, history.text
        assert len(history.json()["data"]["items"]) == 1
        first = client.post(
            "/v1/skills/trajectories/list",
            json={"skill_id": skill_id, "skill_version_id": version_id, "limit": 1},
        )
        assert first.status_code == 200, first.text
        cursor = first.json()["data"]["next_cursor"]
        assert cursor is not None
        second = client.post(
            "/v1/skills/trajectories/list",
            json={"skill_id": skill_id, "skill_version_id": version_id, "limit": 1, "cursor": cursor},
        )
        assert second.status_code == 200, second.text
        assert len(second.json()["data"]["items"]) == 1

        version_fields = [fields for table, fields in selected_fields if table == SKILL_VERSIONS_TABLE]
        trajectory_fields = [fields for table, fields in selected_fields if table == TRAJECTORY_VERSIONS_TABLE]
        assert {frozenset(fields) for fields in version_fields} == {
            frozenset(
                {
                    "version_id",
                    "version_label",
                    "commit_message",
                    "parent_version_ids",
                    "content_hash",
                    "skill_hash",
                    "implementation",
                    "status",
                    "origin",
                    "created_at",
                }
            ),
            frozenset({"skill_id"}),
        }
        assert len(trajectory_fields) == 3
        assert all(
            set(fields)
            == {
                "trajectory_id",
                "task_id",
                "skill_id",
                "skill_version_id",
                "user_id",
                "agent_id",
                "status",
                "started_at",
                "finished_at",
                "received_at",
            }
            for fields in trajectory_fields
        )

        with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
            skill = await store.skills.resolve_family(skill_id)
            custom_history = await store.skills.history(skill, limit=1, offset=0, fields=["version_id", "content"])
            custom_attempts = await store.trajectories.list_reported(
                skill_id,
                skill_version_id=version_id,
                status=None,
                received_at_from=None,
                received_at_to=None,
                snapshot_at=utcnow(),
                after_received_at=None,
                after_trajectory_id=None,
                limit=1,
                fields=["trajectory_id", "trajectory"],
            )
        assert set(custom_history[0]) == {"version_id", "content"}
        assert "Body" in custom_history[0]["content"]
        assert set(custom_attempts[0]) == {"trajectory_id", "trajectory"}
        assert custom_attempts[0]["trajectory"][0]["role"] == "user"
        assert selected_fields[-3:] == [
            (SKILL_VERSIONS_TABLE, ["version_id", "content"]),
            (TRAJECTORY_VERSIONS_TABLE, ["trajectory_id"]),
            (TRAJECTORIES_TABLE, ["trajectory_id", "trajectory"]),
        ]


def test_bearer_project_scope_and_write_permission(service):
    """Resolve project and permission from a trusted Bearer adapter."""

    def resolver(token):
        return SkillHttpIdentity(project_id="project", scopes={"skill:read"} if token == "reader" else {"skill:write"})

    with TestClient(create_app(service, identity_from_bearer=resolver)) as client:
        assert client.post("/v1/skills/list", json={}).status_code == 401
        assert client.post("/v1/skills/list", json={}, headers={"Authorization": "Bearer reader"}).status_code == 200
        assert client.post("/v1/skills/list", json={}, headers={"Authorization": "Bearer writer"}).status_code == 200
        denied = client.post(
            "/v1/skills/delete", json={"skill_id": "missing"}, headers={"Authorization": "Bearer reader"}
        )
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "auth.insufficient_scope"
        assert (
            client.post(
                "/v1/skills/list", json={"project_id": "different"}, headers={"Authorization": "Bearer reader"}
            ).status_code
            == 422
        )


def test_local_key_file_authenticates_and_rejects_other_keys(authenticated_service, tmp_path):
    """Bind a local deployment to the key file's project on each request."""
    key_file = tmp_path / "skill_keys.yaml"
    key_file.write_text("project_id: project\napi_key: local-secret\n", encoding="utf-8")
    config = FastAPIConfig(auth=HttpAuthConfig(mode="local", api_key_file=str(key_file)))

    with TestClient(create_app(authenticated_service, api_config=config)) as client:
        assert client.post("/v1/skills/list", json={}).status_code == 401
        denied = client.post("/v1/skills/list", json={}, headers={"Authorization": "Bearer wrong"})
        assert denied.status_code == 401
        assert denied.json()["error"]["code"] == "auth.invalid_api_key"
        accepted = client.post("/v1/skills/list", json={}, headers={"Authorization": "Bearer local-secret"})
        assert accepted.status_code == 200
        key_file.write_text("project_id: other\napi_key: local-secret\n", encoding="utf-8")
        assert (
            client.post("/v1/skills/list", json={}, headers={"Authorization": "Bearer local-secret"}).status_code == 200
        )


@pytest.mark.parametrize(
    "scopes,enabled,expected", [(["skill:read"], True, 200), ([], True, 403), (["skill:write"], False, 401)]
)
def test_local_key_file_enforces_explicit_scopes_and_enabled(
    authenticated_service, tmp_path, scopes, enabled, expected
):
    """Do not elevate configured local credentials to write permission."""
    import yaml

    key_file = tmp_path / "skill_keys.yaml"
    key_file.write_text(
        yaml.safe_dump({"project_id": "project", "api_key": "local-secret", "scopes": scopes, "enabled": enabled})
    )
    config = FastAPIConfig(auth=HttpAuthConfig(mode="local", api_key_file=str(key_file)))
    headers = {"Authorization": "Bearer local-secret"}
    with TestClient(create_app(authenticated_service, api_config=config)) as client:
        assert client.post("/v1/skills/list", json={}, headers=headers).status_code == expected
        if enabled:
            assert client.post("/v1/skills/delete", json={"skill_id": "missing"}, headers=headers).status_code == 403


def test_auth_config_requires_gateway_secret():
    """Reject missing verification keys and the obsolete identity lookup mode."""
    with pytest.raises(ValueError, match="gateway_jwt_secret"):
        HttpAuthConfig(mode="gateway_jwt")
    with pytest.raises(ValueError, match="gateway_jwt_secret"):
        HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret="")
    with pytest.raises(ValueError):
        HttpAuthConfig(mode="cloud", cloud_base_url="https://account.example")
    assert HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret="secret").mode == "gateway_jwt"


def test_gateway_jwt_resolver_verifies_gateway_claims():
    """Accept only signed, unexpired JWTs for the configured issuer and audience."""
    resolver = GatewayJwtResolver(secret=GATEWAY_SECRET, issuer="mindmemos-gateway", audience="skill-data-plane")
    identity = resolver(gateway_token(project_id="cloud-project", scopes=["skill:read"], request_id="gateway-request"))
    assert identity.project_id == "cloud-project"
    assert identity.request_id == "gateway-request"
    assert identity.scopes == {"skill:read"}

    invalid = [
        gateway_token(secret="another-gateway-secret-with-32-bytes"),
        gateway_token(exp=int(time.time()) - 1),
        gateway_token(iss="other-gateway"),
        gateway_token(aud="memory-data-plane"),
        gateway_token(project_id=""),
        gateway_token(scopes="skill:read"),
        gateway_token(exp="tomorrow"),
        gateway_token(algorithm="HS384"),
        jwt.encode({"iss": "mindmemos-gateway", "aud": "skill-data-plane"}, GATEWAY_SECRET, algorithm="HS256"),
    ]
    for token in invalid:
        with pytest.raises(Exception) as exc_info:
            resolver(token)
        assert exc_info.value.status_code == 401


@pytest_asyncio.fixture
async def authenticated_service(tmp_path):
    """Assemble an unbound store whose project comes only from the request identity."""
    config = repository_config(tmp_path / "home").service
    config.persistence.database.config["url"] = f"sqlite+pysqlite:///{tmp_path / 'auth.db'}"

    def authorize(principal, project_id):
        if principal.project_id != project_id:
            raise PermissionError("Project mismatch")
        return Scope(fields={"project_id": project_id})

    return await create_service(config, project_authorizer=authorize)


def test_local_key_file_selects_project_on_unbound_service(authenticated_service, tmp_path):
    """A local key file selects a project without a service-level local binding."""
    key_file = tmp_path / "skill_keys.yaml"
    key_file.write_text("project_id: project-a\napi_key: local-secret\n", encoding="utf-8")
    config = FastAPIConfig(auth=HttpAuthConfig(mode="local", api_key_file=str(key_file)))
    headers = {"Authorization": "Bearer local-secret"}
    with TestClient(create_app(authenticated_service, api_config=config)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "local-skill",
                "skill": {
                    "content": "---\nname: local-skill\ndescription: Local\n---\nBody",
                    "name": "local-skill",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
            headers=headers,
        )
        assert registered.status_code == 201, registered.text
        key_file.write_text("project_id: project-b\napi_key: local-secret\n", encoding="utf-8")
        listed = client.post("/v1/skills/list", json={}, headers=headers)
        assert listed.status_code == 200
        assert listed.json()["data"]["items"] == []


def test_gateway_jwt_controls_project_and_skill_scopes(authenticated_service):
    """Verified Gateway claims select the project and permitted Skill operations."""
    config = FastAPIConfig(auth=HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret=GATEWAY_SECRET))
    writer_a = gateway_token(project_id="project-a", scopes=["skill:write"])
    reader_a = gateway_token(project_id="project-a", scopes=["skill:read"])
    reader_b = gateway_token(project_id="project-b", scopes=["skill:read"])
    with TestClient(create_app(authenticated_service, api_config=config)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "private-skill",
                "skill": {
                    "content": "---\nname: private-skill\ndescription: Private\n---\nBody",
                    "name": "private-skill",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
            headers={"Authorization": f"Bearer {writer_a}"},
        )
        assert registered.status_code == 201, registered.text
        skill = registered.json()["data"]
        for token, expected_count in ((writer_a, 1), (reader_a, 1), (reader_b, 0)):
            headers = {"Authorization": f"Bearer {token}"}
            listed = client.post("/v1/skills/list", json={}, headers=headers)
            assert listed.status_code == 200
            assert len(listed.json()["data"]["items"]) == expected_count
            if token == reader_a:
                shown = client.post("/v1/skills/show", json={"skill_id": skill["skill_id"]}, headers=headers)
                assert shown.status_code == 200
                compared = client.post(
                    "/v1/skills/diff",
                    json={
                        "skill_id": skill["skill_id"],
                        "to_skill_version_id": skill["skill_version_id"],
                    },
                    headers=headers,
                )
                assert compared.status_code == 200
            if token != writer_a:
                denied = client.post("/v1/skills/delete", json={"skill_id": "missing"}, headers=headers)
                assert denied.status_code == 403
                assert denied.json()["error"]["code"] == "auth.insufficient_scope"
        assert client.post("/v1/skills/list", json={}).status_code == 401
        memory_only = client.post(
            "/v1/skills/list",
            json={},
            headers={"Authorization": f"Bearer {gateway_token(scopes=['memory:read'])}"},
        )
        assert memory_only.status_code == 403
        assert memory_only.json()["error"]["code"] == "auth.insufficient_scope"
        for obsolete_scope in ("skill:*", "skill:manage", "skill:evolve", "skill:train"):
            obsolete_headers = {"Authorization": f"Bearer {gateway_token(scopes=[obsolete_scope])}"}
            for path, body in (
                ("/v1/skills/list", {}),
                ("/v1/skills/delete", {"skill_id": "missing"}),
            ):
                obsolete = client.post(path, json=body, headers=obsolete_headers)
                assert obsolete.status_code == 403
        expired = client.post(
            "/v1/skills/list",
            json={},
            headers={"Authorization": f"Bearer {gateway_token(exp=int(time.time()) - 1)}"},
        )
        assert expired.status_code == 401
        assert expired.json()["error"]["code"] == "auth.internal_token_expired"
        invalid = client.post(
            "/v1/skills/list",
            json={},
            headers={"Authorization": f"Bearer {gateway_token(secret='another-gateway-secret-with-32-bytes')}"},
        )
        assert invalid.status_code == 401
        assert invalid.json()["error"]["code"] == "auth.invalid_internal_token"


def test_list_cursor_keeps_scroll_stable_across_insert_and_rename(service):
    """Continue a filtered page without offset drift or newly inserted rows."""

    def register(client, name):
        response = client.post(
            "/v1/skills/register",
            json={
                "alias": name,
                "skill": {
                    "content": f"---\nname: {name}\ndescription: Example\n---\nBody",
                    "name": name,
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        assert response.status_code == 201, response.text
        return response.json()["data"]["skill_id"]

    with TestClient(create_app(service)) as client:
        original = {register(client, f"item-{index}") for index in range(5)}
        first = client.post("/v1/skills/list", json={"filters": {"alias": "item-*"}, "limit": 2})
        assert first.status_code == 200, first.text
        page = first.json()["data"]
        assert page["has_more"] is True
        assert page["next_cursor"]
        assert "offset" not in page
        seen = [item["skill_id"] for item in page["items"]]

        added_later = register(client, "item-later")
        renamed = client.post("/v1/skills/update-info", json={"skill_id": seen[0], "name": "renamed"})
        assert renamed.status_code == 200
        cursor = page["next_cursor"]
        changed_filter = client.post(
            "/v1/skills/list", json={"cursor": cursor, "filters": {"alias": "other-*"}, "limit": 2}
        )
        assert changed_filter.status_code == 422
        assert changed_filter.json()["error"]["code"] == "invalid_cursor"
        assert client.post("/v1/skills/list", json={"cursor": "broken", "limit": 2}).status_code == 422

        while cursor is not None:
            next_page = client.post("/v1/skills/list", json={"cursor": cursor, "limit": 2})
            assert next_page.status_code == 200, next_page.text
            data = next_page.json()["data"]
            seen.extend(item["skill_id"] for item in data["items"])
            cursor = data["next_cursor"]
        assert len(seen) == len(set(seen))
        assert set(seen) == original
        assert added_later not in seen


async def test_list_cursor_breaks_creation_time_ties_with_skill_id(service):
    """Read every family once when imported records share a creation time."""
    with TestClient(create_app(service)) as client:
        ids = []
        for index in range(3):
            name = f"tie-{index}"
            response = client.post(
                "/v1/skills/register",
                json={
                    "alias": name,
                    "skill": {
                        "content": f"---\nname: {name}\ndescription: Example\n---\nBody",
                        "name": name,
                        "commit_version": "v1.0.0",
                        "commit_message": "Initial version",
                    },
                },
            )
            assert response.status_code == 201
            ids.append(response.json()["data"]["skill_id"])
        with service.repository.persistence.session(Scope(fields={"project_id": "project"})) as store:
            rows = await store.database.select(SKILLS_TABLE, order_by=["created_at"], limit=3)
            for skill_id in ids:
                (
                    await store.database.update(
                        SKILLS_TABLE, {"created_at": rows[0]["created_at"]}, filters={"skill_id": skill_id}
                    )
                )

        seen = []
        cursor = None
        while True:
            body = {"limit": 1}
            if cursor is not None:
                body["cursor"] = cursor
            response = client.post("/v1/skills/list", json=body)
            assert response.status_code == 200, response.text
            page = response.json()["data"]
            seen.extend(item["skill_id"] for item in page["items"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert seen == sorted(ids, reverse=True)


def test_http_policy_yaml_remains_strict():
    """Validate the deployment policy without accepting request-level pipeline names."""
    path = Path(__file__).resolve().parents[3] / "config/mindmemos_skill/dev.example.yaml"
    policy = FastAPIConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))["fastapi"])
    assert policy.evolution_pipelines == {"offline"}
    assert policy.default_evolution_pipeline == "offline"


def test_http_default_must_be_an_allowed_evolution_pipeline():
    """Reject deployment defaults that would bypass the allowed pipeline policy."""
    with pytest.raises(ValueError, match="default_evolution_pipeline"):
        FastAPIConfig(
            auth=HttpAuthConfig(mode="none", default_project_id="project"),
            evolution_pipelines={"custom"},
            default_evolution_pipeline="offline",
        )


async def test_custom_service_reports_unavailable_evolution_through_common_contract():
    """Use declared optional methods instead of requiring algorithm-specific Service methods."""
    service = Mock(spec=MindMemOSSkill)
    service.submit_evolution.side_effect = NotImplementedError("Evolution is unavailable")
    policy = FastAPIConfig(
        auth=HttpAuthConfig(mode="none", default_project_id="project"),
        evolution_pipelines={"offline"},
        default_evolution_pipeline="offline",
    )
    app = create_app(
        service,
        api_config=policy,
        identity_from_bearer=lambda credential: SkillHttpIdentity(project_id="project", scopes={"skill:write"}),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://127.0.0.1", headers={"authorization": "Bearer test"}
    ) as client:
        for body in ({"skill_version_id": "skv-a"}, {"pipeline": "offline", "skill_version_ids": ["skv-a", "skv-b"]}):
            response = await client.post("/v1/skills/algos/evolve", json=body)
            assert response.status_code == 422
            assert service.submit_evolution.call_args.kwargs["pipeline"] == "offline"


def test_local_import_does_not_load_http_dependencies():
    """Keep FastAPI and Uvicorn optional until create_app is called."""
    source = Path(__file__).resolve().parents[3] / "src/mindmemos_skill"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, asyncio; from mindmemos_skill import ServiceConfig, create_service; "
            "service = asyncio.run(create_service(ServiceConfig(home='.'))); "
            "assert service.backends == {}; "
            "assert 'fastapi' not in sys.modules; assert 'uvicorn' not in sys.modules; assert 'mindmemos_sdk' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": str(source)},
        check=True,
    )


async def test_local_entry_rejects_public_bind_without_authentication(monkeypatch):
    """Reject unauthenticated deployment on a public configured address."""
    path = Path(__file__).resolve().parents[3] / "config/mindmemos_skill/dev.example.yaml"
    monkeypatch.setenv(SKILL_CONFIG_PATH_ENV, str(path))
    monkeypatch.setenv(SKILL_API_HOST_ENV, "0.0.0.0")
    with pytest.raises(ValueError, match="loopback address"):
        (await build_http_service())


def test_http_policy_cannot_override_project_binding(service):
    """Reject conflicting default projects and authentication over a none-mode service."""
    with pytest.raises(ValueError, match="default project"):
        create_app(service, api_config=FastAPIConfig(auth=HttpAuthConfig(mode="none", default_project_id="other")))
    with pytest.raises(ValueError, match="without a default project"):
        create_app(
            service, api_config=FastAPIConfig(auth=HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret="secret"))
        )


@pytest.mark.parametrize("mode", ["local", "gateway_jwt"])
async def test_complete_yaml_installs_authentication_without_host_overrides(tmp_path, mode):
    """The complete YAML supplies credentials, project authorization and HTTP policy."""
    config = repository_config(tmp_path / "home").model_dump(mode="json")
    if mode == "local":
        (tmp_path / "key.yaml").write_text("project_id: project-a\napi_key: local-secret\n")
        config["fastapi"]["auth"] = {"mode": "local", "api_key_file": "key.yaml"}
        credential = "local-secret"
    else:
        config["fastapi"]["auth"] = {"mode": "gateway_jwt", "gateway_jwt_secret": GATEWAY_SECRET}
        credential = gateway_token()
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(config))
    service = await create_service(path)
    assert service.default_project_id is None
    with TestClient(create_app(service)) as client:
        assert client.post("/v1/skills/list", json={}).status_code == 401
        assert client.post("/v1/skills/list", json={}, headers={"Authorization": "Bearer invalid"}).status_code == 401
        accepted = client.post("/v1/skills/list", json={}, headers={"Authorization": f"Bearer {credential}"})
        assert accepted.status_code == 200


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/skills/algos/analyze/skill", {"skill_version_id": "v", "unexpected": 1}),
        ("/v1/skills/algos/analyze/trajectory", {"trajectory_id": "t", "unexpected": 1}),
        ("/v1/skills/algos/evaluations/query", {"skill_version_ids": ["v"], "unexpected": 1}),
    ],
)
def test_analyze_routes_reject_unknown_fields(service, path, body):
    """Keep the new Analyze and Evaluation contracts closed to unknown fields."""
    with TestClient(create_app(service)) as client:
        response = client.post(path, json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


async def test_analyze_http_runs_offline_and_queries_both_judge_types(service):
    """Exercise the Analyze and Evaluation routes with an injected offline judge."""
    _, judge = install_offline_analyze(service)
    with TestClient(create_app(service)) as client:
        registered = client.post(
            "/v1/skills/register",
            json={
                "alias": "analyzed",
                "skill": {
                    "content": "---\nname: analyzed\ndescription: Example\n---\nBody",
                    "name": "analyzed",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        assert registered.status_code == 201, registered.text
        version_id = registered.json()["data"]["skill_version_id"]

        static = client.post("/v1/skills/algos/analyze/skill", json={"skill_version_id": version_id})
        assert static.status_code == 200, static.text
        assert static.json()["data"]["status"] == "completed"
        assert static.json()["data"]["judge_type"] == "static_skill"

        report = client.post(
            "/v1/skills/trajectories/report",
            json={
                "user_id": "user-1",
                "agent_id": "agent-1",
                "skill_version_ids": [version_id],
                "messages": [{"role": "user", "content": "Summarize the file"}],
            },
        )
        assert report.status_code == 201, report.text
        trajectory_id = report.json()["data"]["trajectory_id"]

        analyzed = client.post("/v1/skills/algos/analyze/trajectory", json={"trajectory_id": trajectory_id})
        assert analyzed.status_code == 200, analyzed.text
        assert analyzed.json()["data"]["judge_type"] == "trajectory"
        assert analyzed.json()["data"]["status"] == "completed"

        cached = client.post("/v1/skills/algos/analyze/skill", json={"skill_version_id": version_id})
        assert cached.status_code == 200, cached.text
        assert cached.json()["data"]["evaluation_id"] == static.json()["data"]["evaluation_id"]

        queried = client.post("/v1/skills/algos/evaluations/query", json={"skill_version_ids": [version_id]})
        assert queried.status_code == 200, queried.text
        judge_types = {item["judge_type"] for item in queried.json()["data"]["items"]}
    assert judge_types == {"static_skill", "trajectory"}
    assert judge.calls == 2


def test_gateway_jwt_isolates_analyze_results_by_project(authenticated_service):
    """Keep Analyze writes and Evaluation reads within the credential's project."""
    install_offline_analyze(authenticated_service)
    config = FastAPIConfig(auth=HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret=GATEWAY_SECRET))
    writer_a = gateway_token(project_id="project-a", scopes=["skill:write", "skill:read"])
    reader_b = gateway_token(project_id="project-b", scopes=["skill:read"])
    with TestClient(create_app(authenticated_service, api_config=config)) as client:
        registered = client.post(
            "/v1/skills/register",
            headers={"Authorization": f"Bearer {writer_a}"},
            json={
                "alias": "scoped",
                "skill": {
                    "content": "---\nname: scoped\ndescription: Scoped\n---\nBody",
                    "name": "scoped",
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        assert registered.status_code == 201, registered.text
        version_id = registered.json()["data"]["skill_version_id"]
        analyzed = client.post(
            "/v1/skills/algos/analyze/skill",
            headers={"Authorization": f"Bearer {writer_a}"},
            json={"skill_version_id": version_id},
        )
        assert analyzed.status_code == 200, analyzed.text
        mine = client.post(
            "/v1/skills/algos/evaluations/query",
            headers={"Authorization": f"Bearer {writer_a}"},
            json={"skill_version_ids": [version_id]},
        )
        assert mine.status_code == 200, mine.text
        assert len(mine.json()["data"]["items"]) == 1
        other = client.post(
            "/v1/skills/algos/evaluations/query",
            headers={"Authorization": f"Bearer {reader_b}"},
            json={"skill_version_ids": [version_id]},
        )
        assert other.status_code == 200, other.text
        assert other.json()["data"]["items"] == []


def _publish_recommend_graph(client, headers) -> tuple[list[str], str]:
    """Register two versioned Skills and organize them into one graph snapshot."""
    versions: list[str] = []
    for name, description in (("csv", "Parse CSV tables"), ("images", "Draw pictures")):
        registered = client.post(
            "/v1/skills/register",
            headers=headers,
            json={
                "alias": name,
                "skill": {
                    "content": f"---\nname: {name}\ndescription: {description}\n---\n{description}",
                    "name": name,
                    "commit_version": "v1.0.0",
                    "commit_message": "Initial version",
                },
            },
        )
        assert registered.status_code == 201, registered.text
        versions.append(registered.json()["data"]["skill_version_id"])
    organized = client.post(
        "/v1/skills/algos/organize",
        headers=headers,
        json={"alias": "collection", "commit_message": "Initial organization", "skill_refs": versions},
    )
    assert organized.status_code == 200, organized.text
    return versions, organized.json()["data"]["graph_version_id"]


async def test_recommend_attaches_evaluations_without_changing_ranking(management_service):
    """include_evaluation appends stored rows while keeping order and scores stable."""
    _, judge = install_offline_analyze(management_service)

    def resolver(token):
        return SkillHttpIdentity(project_id="project", scopes={"skill:write"}, actor_id="account-1")

    headers = {"Authorization": "Bearer admin"}
    with TestClient(create_app(management_service, identity_from_bearer=resolver)) as client:
        versions, snapshot = _publish_recommend_graph(client, headers)
        for version_id in versions:
            analyzed = client.post(
                "/v1/skills/algos/analyze/skill", headers=headers, json={"skill_version_id": version_id}
            )
            assert analyzed.status_code == 200, analyzed.text

        query = {"graph_ref": "collection", "graph_version_id": snapshot, "query": "Parse CSV records", "top_k": 2}
        plain = client.post("/v1/skills/algos/recommend", headers=headers, json=query)
        assert plain.status_code == 200, plain.text
        assert "evaluations" not in plain.json()["data"]
        plain_data = plain.json()["data"]

        enriched = client.post("/v1/skills/algos/recommend", headers=headers, json=query | {"include_evaluation": True})
        assert enriched.status_code == 200, enriched.text
        enriched_data = enriched.json()["data"]
        assert [item["version_id"] for item in enriched_data["skills"]] == [
            item["version_id"] for item in plain_data["skills"]
        ]
        assert enriched_data["scores"] == plain_data["scores"]
        assert {item["skill_version_id"] for item in enriched_data["evaluations"]} == set(versions)
        assert all(item["status"] == "completed" for item in enriched_data["evaluations"])
    assert judge.calls == 2


async def test_recommend_keeps_results_when_evaluation_query_fails(management_service, monkeypatch):
    """A stored-evaluation read failure degrades to an empty list without failing Recommend."""
    _, judge = install_offline_analyze(management_service)

    def resolver(token):
        return SkillHttpIdentity(project_id="project", scopes={"skill:write"}, actor_id="account-1")

    headers = {"Authorization": "Bearer admin"}
    with TestClient(create_app(management_service, identity_from_bearer=resolver)) as client:
        versions, snapshot = _publish_recommend_graph(client, headers)
        for version_id in versions:
            analyzed = client.post(
                "/v1/skills/algos/analyze/skill", headers=headers, json={"skill_version_id": version_id}
            )
            assert analyzed.status_code == 200, analyzed.text

        async def broken_query(request):
            raise RuntimeError("evaluation storage offline")

        monkeypatch.setattr(management_service._analyze_pipeline, "query", broken_query)
        response = client.post(
            "/v1/skills/algos/recommend",
            headers=headers,
            json={
                "graph_ref": "collection",
                "graph_version_id": snapshot,
                "query": "Parse CSV records",
                "top_k": 2,
                "include_evaluation": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["data"]["evaluations"] == []
    assert judge.calls == 2
