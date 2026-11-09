from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from mindmemos.api.internal_routes import router
from mindmemos.config import init_config, reset_config
from mindmemos.errors import ApiError, BadRequestError
from mindmemos.infra.db import QdrantRecord


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


class FakeMemoryDbReader:
    def __init__(self, clients: FakeClients) -> None:
        self.clients = clients

    async def list_memory_records(self, ctx, *, filters=None, limit=50, cursor=None):
        return await self.clients.qdrant.scroll_memories(
            ctx.project_id,
            filter_=filters,
            limit=limit,
            cursor=cursor,
            with_vectors=False,
        )

    async def get_memory_record(self, ctx, memory_id):
        return await self.clients.qdrant.get_memory(ctx.project_id, memory_id, with_vectors=False)


class FakeQdrant:
    def __init__(self, *, unfiltered_records=None, filtered_records=None) -> None:
        self.unfiltered_records = unfiltered_records
        self.filtered_records = filtered_records
        self.last_filter = None
        self.last_limit = None

    async def scroll_memories(self, project_id, *, filter_=None, limit=50, cursor=None, with_vectors=False):
        self.last_filter = filter_
        self.last_limit = limit
        if filter_ is not None and self.filtered_records is not None:
            return self.filtered_records, None
        if filter_ is None and self.unfiltered_records is not None:
            return self.unfiltered_records, None
        return [
            QdrantRecord(
                point_id="mem_001",
                payload={"project_id": project_id, "content": "hello memory"},
            )
        ], None

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
