"""HTTP contract for GET /v1/memory/images/{blob_id} (raw bytes, owner-verified)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mindmemos.api.app import register_exception_handlers
from mindmemos.api.deps import get_optional_request_context, get_request_context
from mindmemos.api.routes import router
from mindmemos.api.schemas import AuthContext
from mindmemos.api.services import get_memory_service
from mindmemos.api.services import memory_service as memory_service_module
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.config import init_config, reset_config
from mindmemos.errors import ResourceNotFoundError
from mindmemos.infra.blob.signed_url import image_url_signature
from mindmemos.pipelines.memory_db import ImageMemoryStore

from tests.blob_fakes import MemoryBlobStore

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
BLOB_ID = hashlib.sha256(PNG_BYTES).hexdigest()
OWNER = "user-1"
PROJECT = "project-1"
SIGNING_SECRET = "test-secret"


@pytest.fixture
def auth_context() -> AuthContext:
    return AuthContext(
        request_id="req-image-route",
        account_id="account-1",
        project_id=PROJECT,
        api_key_uuid="key-1",
        memory_algorithm="vanilla",
        scopes=["memory:read", "memory:write"],
    )


@pytest.fixture
def config(tmp_path: Path) -> None:
    """Minimal valid config: the signed-URL branch derives its secret from gateway_jwt_secret."""
    path = tmp_path / "dev.yaml"
    path.write_text(f"auth:\n  mode: gateway_jwt\n  gateway_jwt_secret: {SIGNING_SECRET}\n", encoding="utf-8")
    init_config(config_path=path)
    yield
    reset_config()


def _app(auth_context: AuthContext | None, service: MemoryService) -> FastAPI:
    """Bearer override unless ``auth_context`` is None (signed-URL mode exercises the real dependency)."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    if auth_context is not None:
        app.dependency_overrides[get_optional_request_context] = lambda: auth_context
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
async def test_get_image_serves_blob_from_the_real_service_and_memory_store(
    monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext
) -> None:
    store = MemoryBlobStore()
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


def _signed_params(user_id: str = OWNER, *, expires_offset: int = 3600, project_id: str = PROJECT) -> dict:
    expires = int(time.time()) + expires_offset
    signature = image_url_signature(project_id, BLOB_ID, user_id, expires, SIGNING_SECRET)
    return {
        "user_id": user_id,
        "project_id": project_id,
        "expires": expires,
        "signature": signature,
    }


def test_signed_url_serves_bytes_without_any_authorization(monkeypatch: pytest.MonkeyPatch, config: None) -> None:
    async def _get_image(auth, _blob_id, _user_id):
        assert auth.project_id == PROJECT  # synthesized context carries the signed project
        return PNG_BYTES, "image/png"

    response = _client(None, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params=_signed_params()
    )

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"] == "image/png"


def test_signed_url_expired_is_rejected(monkeypatch: pytest.MonkeyPatch, config: None) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        raise AssertionError("get_image must not run on an expired link")

    response = _client(None, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params=_signed_params(expires_offset=-10)
    )

    assert response.status_code == 401
    assert response.json()["code"] == "image.signature_invalid"


def test_signed_url_tampered_signature_is_rejected(monkeypatch: pytest.MonkeyPatch, config: None) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        raise AssertionError("get_image must not run on a tampered link")

    params = _signed_params()
    params["signature"] = "0" * 64
    response = _client(None, _service(monkeypatch, _get_image)).get(f"/v1/memory/images/{BLOB_ID}", params=params)

    assert response.status_code == 401
    assert response.json()["code"] == "image.signature_invalid"


def test_signed_url_non_ascii_signature_is_rejected_not_a_500(
    monkeypatch: pytest.MonkeyPatch, config: None
) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        raise AssertionError("get_image must not run on a malformed link")

    params = _signed_params()
    params["signature"] = "签名" * 20  # non-ASCII would crash compare_digest without the hex pre-check
    response = _client(None, _service(monkeypatch, _get_image)).get(f"/v1/memory/images/{BLOB_ID}", params=params)

    assert response.status_code == 401
    assert response.json()["code"] == "image.signature_invalid"


def test_signed_url_with_no_secret_configured_rejects_even_a_forged_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty signing secret must fail closed: an HMAC over b\"\" is publicly computable."""

    async def _get_image(_auth, _blob_id, _user_id):
        raise AssertionError("get_image must not run when no secret is configured")

    path = tmp_path / "dev.yaml"
    path.write_text("auth:\n  mode: api_key\n", encoding="utf-8")  # no gateway_jwt_secret, no url_signing_secret
    init_config(config_path=path)
    try:
        expires = int(time.time()) + 3600
        forged = image_url_signature(PROJECT, BLOB_ID, OWNER, expires, "")  # what an attacker computes
        response = _client(None, _service(monkeypatch, _get_image)).get(
            f"/v1/memory/images/{BLOB_ID}",
            params={"user_id": OWNER, "project_id": PROJECT, "expires": expires, "signature": forged},
        )
        assert response.status_code == 401
        assert response.json()["code"] == "image.signature_invalid"
    finally:
        reset_config()


def test_signed_url_cache_control_is_capped_at_the_remaining_lifetime(
    monkeypatch: pytest.MonkeyPatch, config: None
) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        return PNG_BYTES, "image/png"

    response = _client(None, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params=_signed_params(expires_offset=60)
    )

    assert response.status_code == 200
    max_age = int(response.headers["cache-control"].rpartition("=")[2])
    assert 0 <= max_age <= 60  # the browser cache must not outlive the signature


def test_signed_url_without_signature_params_requires_bearer(monkeypatch: pytest.MonkeyPatch, config: None) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        raise AssertionError("get_image must not run without either credential")

    response = _client(None, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params={"user_id": OWNER}
    )

    assert response.status_code == 401
    assert response.json()["code"] == "auth.missing_authorization"


def test_signed_url_with_wrong_owner_still_hits_the_ownership_check(
    monkeypatch: pytest.MonkeyPatch, config: None
) -> None:
    """A correctly signed link naming another owner authenticates, then 404s on ownership."""

    async def _get_image(_auth, _blob_id, user_id):
        raise ResourceNotFoundError(f"image {_blob_id} not found", code="image.not_found")

    # signature is minted FOR user-2 (valid HMAC), so auth passes; user-2 owns nothing here
    response = _client(None, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params=_signed_params(user_id="user-2")
    )

    assert response.status_code == 404
    assert response.json()["code"] == "image.not_found"


def test_bearer_wins_over_a_broken_signature(monkeypatch: pytest.MonkeyPatch, auth_context: AuthContext) -> None:
    async def _get_image(_auth, _blob_id, _user_id):
        return PNG_BYTES, "image/png"

    params = _signed_params()
    params["signature"] = "garbage"
    response = _client(auth_context, _service(monkeypatch, _get_image)).get(
        f"/v1/memory/images/{BLOB_ID}", params=params
    )

    assert response.status_code == 200
    assert response.content == PNG_BYTES
