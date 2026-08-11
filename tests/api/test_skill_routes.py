"""HTTP acceptance tests for the unified relational Skill protocol."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from mindmemos.api.deps import get_request_context
from mindmemos.api.schemas import AddRequest, AuthContext
from mindmemos.api.services import get_skill_service
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.api.services.skill_service import SkillService
from mindmemos.api.skill_routes import router
from mindmemos.api.skill_schemas import SkillEvolveRequest
from mindmemos.errors import ApiError
from mindmemos.infra.db import SkillRelationalRepository, build_cloud_skill_tables
from mindmemos_skill.contracts import (
    SkillBundle,
    SkillTrajectoryReportRequest,
    SkillTrajectoryReportResult,
    SkillTrajectoryReportResultItem,
    SkillVersionCore,
    compute_trajectory_hash,
)
from mindmemos_skill.infra.database import DatabaseConfig, bootstrap_database


class _Evolver:
    async def evolve(self, **_kwargs):
        return [SkillBundle.from_files({"SKILL.md": "improved"})]


class _Producer:
    def __init__(self) -> None:
        self.messages = []

    async def send(self, topic, value, **kwargs):
        self.messages.append((topic, value, kwargs))


class _TrajectoryIngestService:
    def __init__(self) -> None:
        self.request = None

    async def report_trajectories(self, _auth, request):
        self.request = request
        return SkillTrajectoryReportResult(
            items=[
                SkillTrajectoryReportResultItem(
                    trajectory_id=request.items[0].trajectory.trajectory_id,
                    status="stored",
                )
            ]
        )


@pytest_asyncio.fixture
async def client():
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": ":memory:"}),
        build_cloud_skill_tables(),
    )
    service = SkillService(repository=SkillRelationalRepository(database), evolver=_Evolver())
    app = FastAPI()

    @app.exception_handler(ApiError)
    async def _handle_api_error(_request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    app.include_router(router)
    app.dependency_overrides[get_skill_service] = lambda: service
    app.dependency_overrides[get_request_context] = lambda: AuthContext(
        request_id="req-1",
        account_id="acct",
        project_id="proj",
        api_key_uuid="key",
        memory_algorithm="schema",
        scopes=["skills:read", "skills:write", "skills:trajectory:read", "skills:trajectory:write", "skills:evolve"],
    )
    try:
        yield TestClient(app)
    finally:
        await database.close()


def _register_payload(
    *,
    operation_id: str,
    version_id: str,
    text: str,
    cloud_skill_id: str | None = None,
    parents: list[str] | None = None,
    label: str = "1.0.0",
    origin: str = "local",
):
    now = datetime(2026, 8, 7, tzinfo=UTC)
    bundle = SkillBundle.from_files({"SKILL.md": text})
    return {
        "operation_id": operation_id,
        "version": {
            "version_id": version_id,
            "cloud_skill_id": cloud_skill_id,
            "parent_version_ids": parents or [],
            "name": "demo",
            "content_hash": bundle.content_hash,
            "version_label": label,
            "status": "draft",
            "version_revision": 0,
            "origin": origin,
            "metadata": {},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        },
        "bundle": bundle.model_dump(mode="json"),
    }


def _trajectory_payload(*, cloud_skill_id: str, version_id: str, content_hash: str):
    now = datetime(2026, 8, 7, tzinfo=UTC)
    payload = {
        "trajectory_id": "trajectory-1",
        "task_id": "task-1",
        "rollout_id": "rollout-1",
        "attempt_no": 0,
        "rollout_type": "inference",
        "task_instruction": "do it",
        "status": "succeeded",
        "trajectory": [],
        "skill_bindings": [
            {
                "name": "demo",
                "cloud_skill_id": cloud_skill_id,
                "version_id": version_id,
                "content_hash": content_hash,
                "usage": "injected",
            }
        ],
        "started_at": now.isoformat(),
        "finished_at": (now + timedelta(seconds=1)).isoformat(),
        "source": "skill_runtime",
        "created_at": (now + timedelta(seconds=1)).isoformat(),
    }
    payload["trajectory_hash"] = compute_trajectory_hash(payload)
    return payload


def test_trajectory_hash_keeps_unknown_env_ref_backward_compatible() -> None:
    payload = _trajectory_payload(
        cloud_skill_id="cloud-skill",
        version_id="version-1",
        content_hash="a" * 64,
    )
    payload.pop("trajectory_hash")

    legacy_hash = compute_trajectory_hash(payload)

    assert compute_trajectory_hash({**payload, "env_ref": "unknown"}) == legacy_hash
    assert compute_trajectory_hash({**payload, "env_ref": "alfworld"}) != legacy_hash


def test_version_dag_idempotency_and_status(client):
    root_payload = _register_payload(operation_id="push-root", version_id="v1", text="root")
    root = client.post("/v1/skills/register", json=root_payload)
    replay = client.post("/v1/skills/register", json=root_payload)
    assert root.status_code == replay.status_code == 200
    assert root.json()["data"] == replay.json()["data"]
    root_version = root.json()["data"]["version"]
    family = root_version["cloud_skill_id"]

    conflict = client.post(
        "/v1/skills/register",
        json=_register_payload(operation_id="push-root", version_id="different", text="different"),
    )
    assert conflict.status_code == 409

    client.post(
        "/v1/skills/register",
        json=_register_payload(
            operation_id="push-left",
            version_id="v2",
            text="left",
            cloud_skill_id=family,
            parents=["v1"],
            label="1.0.1",
        ),
    )
    right = client.post(
        "/v1/skills/register",
        json=_register_payload(
            operation_id="push-right",
            version_id="v3",
            text="right",
            cloud_skill_id=family,
            parents=["v1"],
            label="1.0.2",
        ),
    ).json()["data"]["version"]
    status = client.post("/v1/skills/versions/v3/status", json={"status": "published", "expected_revision": 0})
    assert status.status_code == 200
    assert status.json()["data"]["version"]["version_revision"] == 1
    listing = client.get("/v1/skills").json()["data"]["skills"]
    assert listing[0]["latest_version"]["version_id"] == right["version_id"]
    assert "published_head_id" not in listing[0]


def test_openapi_exposes_distinct_trajectory_post_routes(client):
    paths = client.app.openapi()["paths"]
    assert "/v1/skills/{cloud_skill_id}/merge" not in paths
    assert "/v1/skills/trajectories" not in paths
    assert set(paths["/v1/skills/trajectory/report"]) == {"post"}
    assert set(paths["/v1/skills/trajectory/list"]) == {"post"}
    assert set(paths["/v1/skills/trajectories/{trajectory_id}"]) == {"get"}


def test_trajectory_sync_async_pull_and_evolve(client):
    root_payload = _register_payload(operation_id="push-root", version_id="v1", text="root")
    version = client.post("/v1/skills/register", json=root_payload).json()["data"]["version"]
    trajectory = _trajectory_payload(
        cloud_skill_id=version["cloud_skill_id"],
        version_id=version["version_id"],
        content_hash=version["content_hash"],
    )
    stored_trajectory_hash = trajectory["trajectory_hash"]
    report = {"operation_id": "report-1", "mode": "sync", "items": [{"trajectory": trajectory}]}
    first = client.post("/v1/skills/trajectory/report", json=report)
    replay = client.post("/v1/skills/trajectory/report", json=report)
    assert first.json()["data"]["items"][0]["status"] == "stored"
    assert replay.json()["data"] == first.json()["data"]

    trajectory["trajectory_id"] = "trajectory-2"
    trajectory["rollout_id"] = "rollout-2"
    trajectory["trajectory_hash"] = compute_trajectory_hash(trajectory)
    queued = client.post(
        "/v1/skills/trajectory/report",
        json={"operation_id": "report-2", "mode": "async", "items": [{"trajectory": trajectory}]},
    )
    assert queued.json()["data"]["items"][0]["status"] == "queued"

    pulled = client.post(
        "/v1/skills/trajectory/list",
        json={"cloud_skill_id": version["cloud_skill_id"], "version_id": "v1"},
    )
    assert [item["trajectory_id"] for item in pulled.json()["data"]["items"]] == ["trajectory-1"]
    projected = client.post(
        "/v1/skills/trajectory/list",
        json={"cloud_skill_id": version["cloud_skill_id"], "include_events": False},
    ).json()["data"]["items"][0]
    assert projected["trajectory"] == []
    assert projected["trajectory_hash"] == stored_trajectory_hash

    evolved = client.post(
        "/v1/skills/evolve",
        json={
            "operation_id": "evolve-1",
            "cloud_skill_id": version["cloud_skill_id"],
            "base_version_id": "v1",
            "algorithm": "fake",
            "mode": "sync",
        },
    )
    assert evolved.status_code == 200
    assert evolved.json()["data"]["status"] == "succeeded"
    candidate_id = evolved.json()["data"]["selected_version_id"]
    versions = client.get(f"/v1/skills/{version['cloud_skill_id']}/versions").json()["data"]["versions"]
    candidate = next(item for item in versions if item["version_id"] == candidate_id)
    assert candidate["metadata"]["evolution"]["base_version_id"] == "v1"
    assert candidate["metadata"]["evolution"]["evidence"][0]["trajectory_id"] == "trajectory-1"


@pytest.mark.asyncio
async def test_async_evolution_is_dispatched_and_resumes_from_frozen_operation():
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": ":memory:"}),
        build_cloud_skill_tables(),
    )
    repository = SkillRelationalRepository(database)
    producer = _Producer()
    service = SkillService(repository=repository, evolver=_Evolver(), producer=producer)
    auth = AuthContext(
        request_id="request",
        account_id="account",
        project_id="project",
        api_key_uuid="key",
        memory_algorithm="schema",
    )
    now = datetime(2026, 8, 7, tzinfo=UTC)
    bundle = SkillBundle.from_files({"SKILL.md": "root"})
    version = SkillVersionCore(
        version_id="v1",
        cloud_skill_id="family",
        name="demo",
        content_hash=bundle.content_hash,
        version_label="1.0.0",
        status="draft",
        origin="local",
        created_at=now,
        updated_at=now,
    )
    try:
        await repository.create_version(
            project_id="project",
            operation_id="push",
            version=version,
            bundle=bundle,
        )
        trajectory = _trajectory_payload(
            cloud_skill_id="family",
            version_id="v1",
            content_hash=bundle.content_hash,
        )
        await repository.ingest_trajectories(
            project_id="project",
            request=SkillTrajectoryReportRequest.model_validate(
                {"operation_id": "report", "items": [{"trajectory": trajectory}]}
            ),
        )
        async_trajectory = {**trajectory, "trajectory_id": "trajectory-async", "rollout_id": "rollout-async"}
        async_trajectory["trajectory_hash"] = compute_trajectory_hash(async_trajectory)
        queued_trajectory = await service.report_trajectories(
            auth,
            SkillTrajectoryReportRequest.model_validate(
                {
                    "operation_id": "report-async",
                    "mode": "async",
                    "items": [{"trajectory": async_trajectory}],
                }
            ),
        )
        resumed_trajectory = await repository.resume_trajectory_ingest(
            project_id="project",
            operation_id="report-async",
        )
        queued = await service.evolve(
            auth,
            SkillEvolveRequest(
                operation_id="evolve-async",
                cloud_skill_id="family",
                base_version_id="v1",
                algorithm="fake",
                mode="async",
            ),
        )
        resumed = await service.resume_evolution(project_id="project", operation_id="evolve-async")
        replay = await service.resume_evolution(project_id="project", operation_id="evolve-async")

        assert queued.status == "queued"
        assert queued_trajectory.items[0].status == "queued"
        assert resumed_trajectory.items[0].status == "stored"
        assert [message[0] for message in producer.messages] == ["skill.trajectory.ingest", "skill.evolve"]
        assert resumed.status == replay.status == "succeeded"
        assert resumed.selected_version_id == replay.selected_version_id
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_memory_add_assigns_add_record_reference_before_canonical_ingest():
    trajectory = _trajectory_payload(
        cloud_skill_id="family",
        version_id="v1",
        content_hash="0" * 64,
    )
    trajectory["source"] = "memory_add"
    trajectory["trajectory_hash"] = compute_trajectory_hash(trajectory)
    ingest = _TrajectoryIngestService()
    service = MemoryService(add_pipeline=object(), skill_store=None, skill_service=ingest)
    auth = AuthContext(
        request_id="request",
        account_id="account",
        project_id="project",
        api_key_uuid="key",
        memory_algorithm="schema",
        scopes=["skills:trajectory:write"],
    )
    request = AddRequest.model_validate(
        {
            "user_id": "user",
            "messages": [{"role": "user", "content": "remember", "timestamp": 1786090000000}],
            "skill_trajectory": trajectory,
            "skill_trajectory_delivery": "required",
        }
    )

    result = await service._ingest_skill_trajectory(auth, "add-record-1", request)

    upload = ingest.request.items[0].trajectory
    assert upload.source_add_record_id == "add-record-1"
    assert result == {
        "trajectory_id": upload.trajectory_id,
        "trajectory_hash": upload.trajectory_hash,
        "status": "stored",
    }
