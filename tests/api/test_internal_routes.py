from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from mindmemos.api.internal_routes import router
from mindmemos.config import init_config, reset_config
from mindmemos.errors import ApiError, BadRequestError
from mindmemos.infra.db import QdrantRecord
from mindmemos.infra.db.read_cache import ProjectReadCache
from mindmemos.pipelines.memory_db import MemoryDbReader


def write_config(tmp_path, *, secret: str) -> None:
    (tmp_path / "dev.yaml").write_text(
        f"""
auth:
  mode: gateway_jwt
  gateway_jwt_secret: {secret}
""",
        encoding="utf-8",
    )


def make_app(monkeypatch, clients=None) -> FastAPI:
    app = FastAPI()
    fake_clients = clients or FakeClients()

    @app.exception_handler(ApiError)
    async def _handle_api_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    app.include_router(router)
    monkeypatch.setattr("mindmemos.api.internal_routes.MemoryDbReader", lambda: FakeMemoryDbReader(fake_clients))
    return app


def make_app_with_provider_service(monkeypatch, service) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(ApiError)
    async def _handle_api_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    app.include_router(router)
    monkeypatch.setattr("mindmemos.api.internal_routes.get_provider_binding_service", lambda: service)
    return app


def make_app_with_vector_repair_service(monkeypatch, service) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(ApiError)
    async def _handle_api_error(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    app.include_router(router)
    monkeypatch.setattr("mindmemos.api.internal_routes.get_vector_repair_service", lambda: service)
    return app


def test_internal_memory_list_uses_gateway_token_project_scope(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch)).get(
            "/internal/v1/projects/proj_001/memories",
            headers={"authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        body = response.json()
        # request_id is generated per request, no longer carried in the gateway token.
        assert uuid.UUID(body["request_id"])
        assert body["data"]["items"][0]["memory_id"] == "mem_001"
    finally:
        reset_config()


def test_internal_memory_detail_missing_uses_standard_memory_not_found_envelope(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read"],
    )
    qdrant = FakeQdrant()

    async def missing_memory(*args, **kwargs):
        return None

    qdrant.get_memory = missing_memory
    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch, clients=FakeClients(qdrant=qdrant))).get(
            "/internal/v1/projects/proj_001/memories/missing",
            headers={"authorization": f"Bearer {token}"},
        )

        assert response.status_code == 404
        assert response.json() == {
            "code": "memory.not_found",
            "message": "memory not found: missing",
            "data": None,
        }
    finally:
        reset_config()


def test_internal_memory_list_pushes_query_to_qdrant_filter_before_pagination(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read"],
    )
    qdrant = FakeQdrant(
        unfiltered_records=[
            QdrantRecord(
                point_id="mem_unmatched",
                payload={"project_id": "proj_001", "content": "first page without the target"},
            )
        ],
        filtered_records=[
            QdrantRecord(
                point_id="mem_target",
                payload={"project_id": "proj_001", "content": "target memory"},
            )
        ],
    )

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch, clients=FakeClients(qdrant=qdrant))).get(
            "/internal/v1/projects/proj_001/memories",
            headers={"authorization": f"Bearer {token}"},
            params={"q": " target ", "limit": 1},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["data"]["items"][0]["memory_id"] == "mem_target"
        assert qdrant.last_limit == 1
        assert _content_match_text(qdrant.last_filter) == "target"
    finally:
        reset_config()


def test_internal_memory_list_rejects_other_project(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read"],
    )

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch)).get(
            "/internal/v1/projects/proj_002/memories",
            headers={"authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403
        assert response.json()["code"] == "auth.project_scope_mismatch"
    finally:
        reset_config()


def test_internal_memory_list_rejects_malformed_gateway_token_as_401(tmp_path, monkeypatch) -> None:
    write_config(tmp_path, secret="test-internal-secret")

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch)).get(
            "/internal/v1/projects/proj_001/memories",
            headers={"authorization": "Bearer a.b.c"},
        )

        assert response.status_code == 401
        assert response.json()["code"] == "auth.invalid_internal_token"
    finally:
        reset_config()


def test_internal_memory_insights_are_project_scoped_and_evidence_backed(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )

    qdrant = FakeQdrant()
    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch, clients=FakeClients(qdrant=qdrant))).get(
            "/internal/v1/projects/proj_001/memory-insights",
            headers={"authorization": f"Bearer {token}"},
            params={"range_days": 30},
        )

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["counts"] == {"active": 29, "archived": 5, "vector_pending": 3, "total": 37}
        assert data["actions"]["created"] == 3
        assert data["actions"]["reinforced"] == 2
        assert data["availability"]["scanned_all"] is True
        assert data["range_days"] == 30
        assert data["as_of"].endswith("Z")
        assert qdrant.last_payload_fields is not None
        assert "content" not in qdrant.last_payload_fields
        assert {"status", "created_at", "update_at", "status_changed_at", "reinforcement_count"}.issubset(
            set(qdrant.last_payload_fields)
        )
        assert qdrant.last_filter is not None
        assert qdrant.last_filter.should
    finally:
        reset_config()


def test_internal_memory_graph_rejects_cross_project_and_enforces_bounds(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        client = TestClient(make_app(monkeypatch))
        denied = client.get(
            "/internal/v1/projects/proj_002/memories/mem_001/graph",
            headers={"authorization": f"Bearer {token}"},
        )
        too_large = client.get(
            "/internal/v1/projects/proj_001/memories/mem_001/graph",
            params={"limit": 201},
            headers={"authorization": f"Bearer {token}"},
        )

        assert denied.status_code == 403
        assert too_large.status_code == 422
    finally:
        reset_config()


def test_internal_node_graph_is_additive_owned_and_bounded(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        client = TestClient(make_app(monkeypatch))
        response = client.get(
            "/internal/v1/projects/proj_001/graph/neighbors",
            params={"node_type": "entity", "node_id": "entity-1", "limit": 500},
            headers={"authorization": f"Bearer {token}"},
        )
        denied = client.get(
            "/internal/v1/projects/proj_002/graph/neighbors",
            params={"node_type": "entity", "node_id": "entity-1"},
            headers={"authorization": f"Bearer {token}"},
        )
        too_large = client.get(
            "/internal/v1/projects/proj_001/graph/neighbors",
            params={"node_type": "entity", "node_id": "entity-1", "limit": 2001},
            headers={"authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["data"]["nodes"][0]["node_type"] == "entity"
        assert denied.status_code == 403
        assert too_large.status_code == 422
    finally:
        reset_config()


def test_internal_memory_insight_scan_is_bounded(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch)).get(
            "/internal/v1/projects/proj_001/memory-insights",
            params={"scan_limit": 10001},
            headers={"authorization": f"Bearer {token}"},
        )

        assert response.status_code == 422
    finally:
        reset_config()


def test_internal_insights_cache_and_force_refresh_preserve_project_authorization(tmp_path, monkeypatch) -> None:
    write_config(tmp_path, secret="cache-test-secret")
    token = make_gateway_token(
        secret="cache-test-secret",
        account_id="account",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )
    qdrant = FakeQdrant()
    try:
        init_config(config_path=tmp_path / "dev.yaml")
        with TestClient(make_app(monkeypatch, clients=FakeClients(qdrant=qdrant))) as client:
            headers = {"authorization": f"Bearer {token}"}
            path = "/internal/v1/projects/proj_001/memory-insights"
            first = client.get(path, headers=headers)
            second = client.get(path, headers=headers)
            assert first.status_code == second.status_code == 200
            assert first.json()["data"]["as_of"] == second.json()["data"]["as_of"]
            assert qdrant.count_calls == 4
            assert client.get(path, headers=headers, params={"refresh": True}).status_code == 200
            assert qdrant.count_calls == 8
            assert client.get(path, headers=headers, params={"range_days": 7}).status_code == 200
            assert qdrant.count_calls == 12
            assert client.get(path.replace("proj_001", "other"), headers=headers).status_code == 403
            assert qdrant.count_calls == 12
            assert "metadata" not in qdrant.last_payload_fields
    finally:
        reset_config()


def test_internal_lifecycle_does_not_serialize_original_add_input(tmp_path, monkeypatch) -> None:
    write_config(tmp_path, secret="projection-test-secret")
    token = make_gateway_token(
        secret="projection-test-secret",
        account_id="account",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )
    qdrant = FakeQdrant()
    qdrant.provenance = {
        "add_record_id": "add-1",
        "request_id": "request-1",
        "source_type": "chat",
        "messages": ["PRIVATE_ORIGINAL_INPUT"],
    }
    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch, clients=FakeClients(qdrant=qdrant))).get(
            "/internal/v1/projects/proj_001/memories/memory-1/lifecycle",
            headers={"authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        assert "PRIVATE_ORIGINAL_INPUT" not in response.text
        assert set(response.json()["data"]["provenance"]) == {
            "available",
            "add_record_id",
            "request_id",
            "source_type",
            "submitted_at",
        }
    finally:
        reset_config()


def test_internal_inventory_projects_legacy_fields_without_changing_pagination(tmp_path, monkeypatch) -> None:
    write_config(tmp_path, secret="inventory-contract-secret")
    token = make_gateway_token(
        secret="inventory-contract-secret",
        account_id="account",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read", "memory:visualize"],
    )
    qdrant = FakeQdrant(
        unfiltered_records=[
            QdrantRecord(
                point_id="memory-1",
                payload={
                    "project_id": "proj_001",
                    "content": "memory",
                    "mem_extract_type": "vanilla",
                    "update_at": "2026-10-31T00:00:00Z",
                    "metadata": {"input": "PRIVATE_METADATA"},
                },
            )
        ]
    )
    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app(monkeypatch, clients=FakeClients(qdrant=qdrant))).get(
            "/internal/v1/projects/proj_001/memories?limit=1",
            headers={"authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        item = data["items"][0]
        assert item["algorithm"] == "vanilla" and item["updated_at"] == "2026-10-31T00:00:00Z"
        assert item["memory_id"] == "memory-1"
        assert "mem_extract_type" not in item and "update_at" not in item and "id" not in item
        assert "PRIVATE_METADATA" not in response.text
        assert "next_cursor" in data
    finally:
        reset_config()


def test_internal_vector_repair_is_project_scoped_and_bounded(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:write"],
    )
    service = FakeVectorRepairService()

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app_with_vector_repair_service(monkeypatch, service)).post(
            "/internal/v1/projects/proj_001/vector-repairs",
            headers={"authorization": f"Bearer {token}"},
            json={"limit": 3, "memory_ids": ["mem-1"], "force": True},
        )

        assert response.status_code == 200
        assert response.json()["data"]["repaired"] == 1
        assert service.repair_calls[0][0].project_id == "proj_001"
        assert service.repair_calls[0][1].limit == 3
        assert service.repair_calls[0][1].memory_ids == ["mem-1"]
    finally:
        reset_config()


def test_internal_vector_repair_rejects_other_project(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:write"],
    )

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app_with_vector_repair_service(monkeypatch, FakeVectorRepairService())).post(
            "/internal/v1/projects/proj_002/vector-repairs",
            headers={"authorization": f"Bearer {token}"},
            json={"limit": 1},
        )

        assert response.status_code == 403
        assert response.json()["code"] == "auth.project_scope_mismatch"
    finally:
        reset_config()


def test_internal_vector_repair_status_requires_read_scope(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["memory:read"],
    )
    service = FakeVectorRepairService()

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app_with_vector_repair_service(monkeypatch, service)).get(
            "/internal/v1/projects/proj_001/vector-repairs/status",
            headers={"authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["data"] == {"pending": 2, "due": 1, "exhausted": 0}
    finally:
        reset_config()


def test_internal_provider_binding_create_uses_gateway_project_scope(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["provider:write"],
    )
    service = FakeProviderBindingService()

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app_with_provider_service(monkeypatch, service)).post(
            "/internal/v1/projects/proj_001/provider-bindings",
            headers={"authorization": f"Bearer {token}"},
            json={
                "scope": {"user_id": "user-1"},
                "routers": {
                    "embed_model_router": {
                        "endpoints": [
                            {
                                "model": "openai/text-embedding-3-large",
                                "api_key": "secret-key",
                                "api_base": "https://embed.example/v1",
                                "dimensions": 1024,
                            }
                        ]
                    }
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["data"]["binding_id"] == "binding-1"
        assert service.create_calls[0]["project_id"] == "proj_001"
        assert service.create_calls[0]["scope"] == {"user_id": "user-1"}
    finally:
        reset_config()


def test_internal_provider_binding_patch_returns_immutable_error(tmp_path, monkeypatch) -> None:
    secret = "test-internal-secret"
    write_config(tmp_path, secret=secret)
    token = make_gateway_token(
        secret=secret,
        account_id="acct_001",
        project_id="proj_001",
        api_key_uuid="console",
        scopes=["provider:write"],
    )
    service = FakeProviderBindingService(raise_on_patch=True)

    try:
        init_config(config_path=tmp_path / "dev.yaml")
        response = TestClient(make_app_with_provider_service(monkeypatch, service)).patch(
            "/internal/v1/projects/proj_001/provider-bindings/binding-1",
            headers={"authorization": f"Bearer {token}"},
            json={"routers": {"embed_model_router": {"endpoints": [{"model": "changed"}]}}},
        )

        assert response.status_code == 400
        assert response.json()["code"] == "provider_binding.immutable_embedding_config"
    finally:
        reset_config()


class FakeClients:
    def __init__(self, qdrant=None) -> None:
        self.qdrant = qdrant or FakeQdrant()


class FakeMemoryDbReader(MemoryDbReader):
    def __init__(self, clients: FakeClients) -> None:
        super().__init__(clients=clients)
        self.clients = clients

    async def list_memory_records(self, ctx, *, filters=None, limit=50, cursor=None, payload_fields=None):
        self.last_payload_fields = payload_fields
        return await self.clients.qdrant.scroll_memories(
            ctx.project_id,
            filter_=filters,
            limit=limit,
            cursor=cursor,
            with_vectors=False,
            payload_fields=payload_fields,
        )

    async def count_memory_records(self, ctx, *, filters=None):
        return await self.clients.qdrant.count_memories(ctx.project_id, filter_=filters)

    async def get_memory_record(self, ctx, memory_id):
        return await self.clients.qdrant.get_memory(ctx.project_id, memory_id, with_vectors=False)

    async def get_memory_lineage(self, ctx, memory_ids):
        return {memory_id: [] for memory_id in memory_ids}

    async def get_add_records_by_ids(self, ctx, add_record_ids):
        return []

    async def list_add_records(self, ctx, *, filters=None, limit=50, cursor=None):
        return [], None

    async def get_memory_graph(self, ctx, memory_id, *, limit=100):
        from mindmemos.pipelines.memory_db import LifecycleAvailability, MemoryGraph, MemoryGraphNode

        return MemoryGraph(
            nodes=[MemoryGraphNode(id=memory_id, node_type="memory", label="hello memory")],
            availability=LifecycleAvailability(graph_available=True),
        )

    async def get_node_graph(self, ctx, node_id, *, node_type, limit=100, cursor=None):
        from mindmemos.pipelines.memory_db import LifecycleAvailability, MemoryGraph, MemoryGraphNode

        return MemoryGraph(
            nodes=[MemoryGraphNode(id=node_id, node_type=node_type, label=f"{node_type} label")],
            availability=LifecycleAvailability(graph_available=True),
        )


class FakeQdrant:
    def __init__(self, *, unfiltered_records=None, filtered_records=None) -> None:
        self.memory = SimpleNamespace(read_cache=ProjectReadCache())
        self.provenance = None
        self.count_calls = 0
        self.unfiltered_records = unfiltered_records
        self.filtered_records = filtered_records
        self.last_filter = None
        self.last_limit = None
        self.last_payload_fields = None

    async def get_memory_provenance(self, project_id, memory_id, *, add_record_id=None):
        return self.provenance

    async def scroll_memories(
        self, project_id, *, filter_=None, limit=50, cursor=None, with_vectors=False, payload_fields=None
    ):
        self.last_filter = filter_
        self.last_limit = limit
        self.last_payload_fields = payload_fields
        if filter_ is not None and self.filtered_records is not None:
            return self.filtered_records, None
        if filter_ is None and self.unfiltered_records is not None:
            return self.unfiltered_records, None
        return [
            QdrantRecord(
                point_id="mem_001",
                payload={
                    "project_id": project_id,
                    "content": "hello memory",
                    "status": "active",
                    "created_at": "2026-08-01T08:00:00Z",
                    "reinforcement_count": 2,
                },
            ),
            QdrantRecord(
                point_id="mem_002",
                payload={
                    "project_id": project_id,
                    "content": "archived memory",
                    "status": "archived",
                    "created_at": "2026-08-02T08:00:00Z",
                },
            ),
            QdrantRecord(
                point_id="mem_003",
                payload={
                    "project_id": project_id,
                    "content": "pending memory",
                    "status": "vector_pending",
                    "created_at": "2026-08-03T08:00:00Z",
                },
            ),
        ], None

    async def count_memories(self, project_id, *, filter_=None):
        self.count_calls += 1
        del project_id
        if filter_ is None:
            return 37
        for condition in filter_.must or []:
            if getattr(condition, "key", None) == "status":
                return {"active": 29, "archived": 5, "vector_pending": 3}.get(condition.match.value, 0)
        return 0

    async def get_memory(self, project_id, memory_id, *, with_vectors=False):
        return QdrantRecord(point_id=memory_id, payload={"project_id": project_id, "content": "hello memory"})


class FakeProviderBindingService:
    def __init__(self, *, raise_on_patch: bool = False) -> None:
        self.raise_on_patch = raise_on_patch
        self.create_calls = []

    async def create_binding(self, *, project_id, scope, routers, request_id):
        self.create_calls.append(
            {
                "project_id": project_id,
                "scope": scope,
                "routers": routers,
                "request_id": request_id,
            }
        )
        return {"binding_id": "binding-1", "project_id": project_id, "scope": scope, "routers": routers}

    async def patch_binding(self, *, project_id, binding_id, routers, request_id):
        if self.raise_on_patch:
            raise BadRequestError(
                "cannot update embedding model identity fields for an existing provider binding: "
                "embed_model_router.endpoints[0].model",
                code="provider_binding.immutable_embedding_config",
            )
        return {"binding_id": binding_id, "project_id": project_id, "scope": {}, "routers": routers}

    async def list_bindings(self, *, project_id):
        return []


class FakeVectorRepairService:
    def __init__(self) -> None:
        self.repair_calls = []

    async def repair(self, ctx, request):
        from mindmemos.pipelines.memory_db.vector_repair import VectorRepairResult

        self.repair_calls.append((ctx, request))
        return VectorRepairResult(selected=1, repaired=1)

    async def status(self, ctx):
        from mindmemos.pipelines.memory_db.vector_repair import VectorRepairStatus

        return VectorRepairStatus(pending=2, due=1, exhausted=0)


def _content_match_text(filter_) -> str | None:
    if filter_ is None:
        return None
    for condition in filter_.must or []:
        if getattr(condition, "key", None) == "content":
            match = getattr(condition, "match", None)
            return getattr(match, "text", None)
    return None


def make_gateway_token(
    *,
    secret: str,
    account_id: str,
    project_id: str,
    api_key_uuid: str,
    scopes: list[str],
) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": "mindmemos-gateway",
        "aud": "memory-data-plane",
        "sub": account_id,
        "account_id": account_id,
        "project_id": project_id,
        "api_key_uuid": api_key_uuid,
        "memory_algorithm": "schema",
        "scopes": scopes,
        "iat": now,
        "exp": now + 60,
    }
    signing_input = f"{_json_b64(header)}.{_json_b64(payload)}"
    sig = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


def _json_b64(value: dict) -> str:
    return _b64url(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
