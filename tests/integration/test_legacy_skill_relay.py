"""Exercise old Skill HTTP writes and Add lifecycle delivery against the new repository."""

import asyncio
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import httpx
import jwt
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mindmemos.api.deps import get_request_context
from mindmemos.api.schemas import AuthContext
from mindmemos.api.services import get_skill_service
from mindmemos.api.services.skill_service import SkillService as MemorySkillService
from mindmemos.api.skill_routes import router
from mindmemos.config import QdrantConfig
from mindmemos.errors import ApiError
from mindmemos.infra.db import SkillVersionRepository
from mindmemos.infra.db.qdrant import QdrantStore
from mindmemos.pipelines.memory_db.add_record_store import AddRecordStore
from mindmemos.pipelines.memory_db.operation_records import MemoryOperationRecorder
from mindmemos.pipelines.skill import SkillVersionStore
from mindmemos.typing.memory import DialogueMessage, MemoryRequestContext
from mindmemos.typing.service import AddPipelineInput, AddPipelineSyncResult
from mindmemos.typing.skill import SkillBinding
from mindmemos.workers import legacy_skill_trajectory as relay
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.service import SkillService
from mindmemos_skill.config.config import FastAPIConfig, HttpAuthConfig
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence.constants import SKILL_VERSIONS_TABLE, SKILLS_TABLE, TRAJECTORIES_TABLE
from mindmemos_skill.persistence.factory import create_persistence
from mindmemos_skill.persistence.observability import AlgorithmAuditPersistence
from mindmemos_skill.pipelines.builtin_pipelines.repository.config import RepositoryPipelineConfig
from mindmemos_skill.pipelines.builtin_pipelines.repository.repository import RepositoryPipeline
from qdrant_client import AsyncQdrantClient

PROJECT = "legacy-http-project"
MEMORY_SECRET = "memory-gateway-signing-key-for-test-only"
SKILL_SECRET = "skill-gateway-signing-key-for-test-only"
CONTENT = "---\nname: demo\ndescription: Example\n---\nOriginal"


@pytest.fixture
async def relay_apps(tmp_path, monkeypatch):
    """Connect real old and new stores through authenticated in-process HTTP."""
    qclient = AsyncQdrantClient(":memory:")
    config = QdrantConfig(vector_size=2)
    qdrant = QdrantStore(config, client=qclient)
    await qdrant.ensure_schema()
    repository = SkillVersionRepository(config, engine=qdrant.engine)
    source = SkillVersionStore(skill_repo=repository, add_record_repo=qdrant.add_record)
    clients = SimpleNamespace(skill=repository, qdrant=qdrant)
    monkeypatch.setattr(relay, "get_database_clients", lambda: clients)
    queue = asyncio.Queue()
    monkeypatch.setattr(relay, "_queue", queue)

    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'target.db'}"))
    target = create_persistence(database)
    await target.initialize_storage()
    service = SkillService(
        backends={},
        observability=AlgorithmAuditPersistence(None),
        persistence=target,
        repository=RepositoryPipeline(RepositoryPipelineConfig(), persistence=target),
        project_authorizer=lambda principal, project: Scope(fields={"project_id": principal.project_id}),
    )
    target_app = create_app(
        service,
        api_config=FastAPIConfig(auth=HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret=SKILL_SECRET)),
    )
    source_app = FastAPI()
    source_app.include_router(router)
    source_app.dependency_overrides[get_skill_service] = lambda: MemorySkillService(store=source)
    source_app.dependency_overrides[get_request_context] = lambda: AuthContext(
        request_id="request",
        account_id="account",
        project_id=PROJECT,
        api_key_uuid="key",
        memory_algorithm="vanilla",
        scopes=["memory:read", "memory:write"],
    )

    @source_app.exception_handler(ApiError)
    async def handle_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"code": exc.code})

    target_transport = httpx.ASGITransport(app=target_app)

    async def forward(request):
        claims = jwt.decode(
            request.headers["Authorization"].removeprefix("Bearer "),
            MEMORY_SECRET,
            algorithms=["HS256"],
            issuer="mindmemos-memory",
            audience="platform-legacy-skill",
        )
        token = jwt.encode(
            {
                "iss": "mindmemos-gateway",
                "aud": "skill-data-plane",
                "sub": "mindmemos-memory",
                "project_id": claims["project_id"],
                "scopes": claims["scopes"],
                "exp": int(time.time()) + 60,
            },
            SKILL_SECRET,
            algorithm="HS256",
        )
        request.headers["Authorization"] = f"Bearer {token}"
        return await target_transport.handle_async_request(request)

    recorder = MemoryOperationRecorder(add_record_store=AddRecordStore(clients=clients))
    try:
        async with (
            httpx.AsyncClient(transport=httpx.ASGITransport(app=source_app), base_url="http://memory") as old,
            httpx.AsyncClient(transport=httpx.MockTransport(forward), base_url="http://gateway") as outbound,
        ):
            yield old, outbound, target, queue, recorder
    finally:
        await service.close()
        await qclient.close()


async def flush(queue, outbound):
    """Deliver pending operations using the production worker's dispatch and retry path."""
    while not queue.empty():
        project, record, kind = queue.get_nowait()
        try:
            assert await relay._deliver(outbound, project, record, MEMORY_SECRET, kind=kind)
        finally:
            queue.task_done()


async def register(old, outbound, queue, *, parent=None):
    """Register over the old HTTP protocol and complete its independent version delivery."""
    body = {"name": "demo", "content": CONTENT + (" changed" if parent else "")}
    if parent:
        body["parent_version_id"] = parent
    response = await old.post("/v1/skills/register", json=body)
    assert response.status_code == 200
    assert queue.qsize() == 1
    await flush(queue, outbound)
    return response.json()["data"]


@pytest.mark.asyncio
async def test_register_child_reads_and_delete_relay_over_http(relay_apps):
    """New versions arrive without trajectories; reads are passive and deletion hides the copy."""
    old, outbound, target, queue, _ = relay_apps
    root = await register(old, outbound, queue)
    child = await register(old, outbound, queue, parent=root["version_id"])
    family_id = root["cloud_skill_id"]
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        imported_root = await session.skills.get(root["version_id"])
        imported_child = await session.skills.get(child["version_id"])
        assert imported_child.parent_version_ids == [imported_root.version_id]
        assert await session.database.count(SKILL_VERSIONS_TABLE) == 2
        assert await session.database.count(TRAJECTORIES_TABLE) == 0

    for method, path, body in (
        ("GET", "/v1/skills", None),
        ("POST", f"/v1/skills/{family_id}/get", None),
        ("GET", f"/v1/skills/{family_id}/versions", None),
        ("GET", f"/v1/skills/{family_id}/versions/{child['version_id']}/content", None),
        ("POST", "/v1/skills/sync", [{"cloud_skill_id": family_id, "local_version_id": child["version_id"]}]),
    ):
        response = await old.request(method, path, json=body)
        assert response.status_code == 200, response.text
    assert queue.empty()

    assert (await old.post(f"/v1/skills/{family_id}/delete")).status_code == 200
    await flush(queue, outbound)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        family = await session.database.get(SKILLS_TABLE, {"project_id": PROJECT, "skill_id": imported_root.skill_id})
        assert family["is_deleted"] is True
        assert await session.database.count(SKILL_VERSIONS_TABLE) == 2
    assert (await old.post(f"/v1/skills/{family_id}/get")).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", "partial"])
async def test_add_lifecycle_updates_existing_imported_trajectory(relay_apps, outcome):
    """An already delivered input receives later completion or failure details."""
    old, outbound, target, queue, recorder = relay_apps
    root = await register(old, outbound, queue)
    trace_id = str(uuid4())
    context = MemoryRequestContext(request_id="request", project_id=PROJECT, account_id="account", api_key_uuid="key")
    await recorder.record_add_input(
        AddPipelineInput(messages=[DialogueMessage(role="user", content="Remember", timestamp=1770000000000)]),
        ctx=context,
        request_submitted_at=datetime.now(UTC),
        add_record_id=trace_id,
        status="queued",
        skill_bindings=[SkillBinding(name="demo", version_id=root["version_id"], content_hash=root["content_hash"])],
    )
    await flush(queue, outbound)
    await recorder.mark_add_processing(context, trace_id)
    await flush(queue, outbound)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert (await session.trajectories.get(trace_id)).metadata["legacy-memory"]["add_status"] == "processing"

    expected_error = None
    if outcome == "completed":
        await recorder.mark_add_completed(context, trace_id, AddPipelineSyncResult(status="ok", memories=[]))
    elif outcome == "failed":
        expected_error = "Worker failed"
        await recorder.mark_add_failed(context, trace_id, expected_error)
    elif outcome == "cancelled":
        expected_error = "Cancelled by caller"
        await recorder.mark_add_cancelled(context, trace_id, expected_error)
    else:
        await recorder.append_add_output(context, trace_id, [])
    await flush(queue, outbound)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        evidence = await session.trajectories.get(trace_id)
        metadata = evidence.metadata["legacy-memory"]
        assert (
            metadata["add_status"]
            == {"completed": "ok", "failed": "error", "cancelled": "cancelled", "partial": "ok"}[outcome]
        )
        assert metadata["memory_completed_at"] is not None
        assert metadata["memory_error"] == expected_error
        assert await session.database.count(TRAJECTORIES_TABLE) == 1

    assert (await old.post(f"/v1/skills/{root['cloud_skill_id']}/delete")).status_code == 200
    await flush(queue, outbound)
    with target.session(Scope(fields={"project_id": PROJECT})) as session:
        assert await session.database.count(TRAJECTORIES_TABLE) == 1
        assert await session.database.count(SKILL_VERSIONS_TABLE) == 1
