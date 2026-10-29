"""HTTP contract for GET /v1/memory/images/{blob_id} (raw bytes, owner-verified)."""

from __future__ import annotations

import hashlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mindmemos.api.app import register_exception_handlers
from mindmemos.api.deps import get_request_context
from mindmemos.api.routes import router
from mindmemos.api.schemas import AuthContext
from mindmemos.api.services import get_memory_service
from mindmemos.api.services import memory_service as memory_service_module
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.errors import ResourceNotFoundError
from mindmemos.infra.blob import LocalBlobStore
from mindmemos.pipelines.memory_db import ImageMemoryStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
BLOB_ID = hashlib.sha256(PNG_BYTES).hexdigest()
OWNER = "user-1"


@pytest.fixture
def auth_context() -> AuthContext:
    return AuthContext(
        request_id="req-image-route",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        memory_algorithm="vanilla",
        scopes=["memory:read", "memory:write"],
    )


def _app(auth_context: AuthContext, service: MemoryService) -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_request_context] = lambda: auth_context
    app.dependency_overrides[get_memory_service] = lambda: service
    return app


def _client(auth_context: AuthContext, service: MemoryService) -> TestClient:
    return TestClient(_app(auth_context, service))


def _service(monkeypatch: pytest.MonkeyPatch, get_image) -> MemoryService:
    service = MemoryService(skill_store=object())
    monkeypatch.setattr(service, "get_image", get_image)
    return service


def test_get_image_returns_bytes_with_cache_headers(monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        return PNG_BYTES, "image/png"

    response = _client(auth_context, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params={"user_id": OWNER}
    )

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, max-age=86400"


def test_get_image_missing_returns_standard_envelope(
    monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext
) -> None:
    async def _get_image(_auth, blob_id, _user_id):
        raise ResourceNotFoundError(f"image {blob_id} not found", code="image.not_found")

    response = _client(auth_context, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params={"user_id": OWNER}
    )

    assert response.status_code == 404
    assert response.json() == {
        "code": "image.not_found",
        "message": f"image {BLOB_ID} not found",
        "data": None,
    }


def test_get_image_requires_user_id_query_param(monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        raise AssertionError("get_image must not run without user_id")

    response = _client(auth_context, _service(monkeypatch, _get_image)).get(f"/v1/memory/images/{BLOB_ID}")

    assert response.status_code == 422


def test_get_image_requires_memory_read_scope(monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        raise AssertionError("get_image must not run without memory:read")

    auth_context.scopes = ["memory:write"]
    response = _client(auth_context, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params={"user_id": OWNER}
    )

    assert response.status_code == 403
    assert response.json()["code"] == "auth.insufficient_scope"


@pytest.mark.asyncio
async def test_get_image_serves_blob_from_the_real_service_and_local_store(
    monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext, tmp_path
) -> None:
    store = LocalBlobStore(tmp_path)
    await store.put(auth_context.project_id, PNG_BYTES, mime_type="image/png")
    monkeypatch.setattr(memory_service_module, "get_blob_store", lambda cfg=None: store)

    async def _owned(_self, _ctx, _blob_id, user_id):
        return user_id == OWNER

    monkeypatch.setattr(ImageMemoryStore, "blob_owned_by", _owned)
    service = MemoryService(skill_store=object())

    response = _client(auth_context, service).get(f"/v1/memory/images/{BLOB_ID}", params={"user_id": OWNER})
    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"] == "image/png"

    # Another user's blob id and a missing blob raise the same 404 (no existence leak).
    other_user = _client(auth_context, service).get(f"/v1/memory/images/{BLOB_ID}", params={"user_id": "user-2"})
    assert other_user.status_code == 404
    assert other_user.json()["code"] == "image.not_found"

    other_project = auth_context.model_copy(update={"project_id": "project-2"})
    cross_project = _client(other_project, service)
    assert cross_project.get(f"/v1/memory/images/{BLOB_ID}", params={"user_id": OWNER}).status_code == 404
