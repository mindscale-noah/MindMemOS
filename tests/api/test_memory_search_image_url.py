"""HTTP contract: POST /v1/memory/search decorates image hits with metadata.image_url."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mindmemos.api.app import register_exception_handlers
from mindmemos.api.deps import get_request_context
from mindmemos.api.routes import router
from mindmemos.api.schemas import AuthContext, SearchRequest
from mindmemos.api.services import get_memory_service
from mindmemos.api.services import memory_service as memory_service_module
from mindmemos.api.services.memory_service import MemoryService
from mindmemos.config import init_config, reset_config
from mindmemos.infra.blob import LocalBlobStore
from mindmemos.typing import MemorySearchItem, SearchPipelineResult

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"
BLOB_ID = hashlib.sha256(PNG_BYTES).hexdigest()
OTHER_BLOB_ID = hashlib.sha256(b"another-image").hexdigest()
OWNER = "user-1"


@pytest.fixture
def auth_context() -> AuthContext:
    return AuthContext(
        request_id="req-search-image-url",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        memory_algorithm="vanilla",
        scopes=["memory:read", "memory:write"],
    )


@pytest.fixture
def config(tmp_path: Path) -> None:
    """Minimal valid config: search decoration reads blob_store settings from it."""
    path = tmp_path / "dev.yaml"
    path.write_text("auth:\n  mode: gateway_jwt\n  gateway_jwt_secret: test-secret\n", encoding="utf-8")
    init_config(config_path=path)
    yield
    reset_config()


@pytest.fixture
def config_expiry_7200(tmp_path: Path) -> None:
    path = tmp_path / "dev.yaml"
    path.write_text(
        "auth:\n  mode: gateway_jwt\n  gateway_jwt_secret: test-secret\n"
        "blob_store:\n  s3_presign_expiry_seconds: 7200\n",
        encoding="utf-8",
    )
    init_config(config_path=path)
    yield
    reset_config()


class _NoopRecorder:
    async def record_search(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeSearchPipeline:
    def __init__(self, result: SearchPipelineResult) -> None:
        self._result = result

    async def search(self, payload, ctx) -> SearchPipelineResult:
        return self._result


class _PresigningStore(LocalBlobStore):
    """Local store that pretends to be an S3 backend able to sign."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.presigned: list[tuple[str, str, int]] = []

    def presigned_get(self, namespace: str, blob_id: str, *, expires_seconds: int = 3600) -> str | None:
        self.presigned.append((namespace, blob_id, expires_seconds))
        return f"https://minio.example.com/blobs/{namespace}/{blob_id}?X-Amz-Signature=fake&expires={expires_seconds}"


def _image_result(*, owner: str = OWNER) -> SearchPipelineResult:
    return SearchPipelineResult(
        status="ok",
        memories=[
            MemorySearchItem(
                id="mem-1",
                memory="a text fact",
                last_update_at="2026-10-28 00:00:00",
                metadata={"modality": "text"},
            ),
            MemorySearchItem(
                id="img-1",
                memory="[image] a screenshot",
                memory_type="image",
                last_update_at="2026-10-28 00:00:00",
                metadata={"modality": "image", "blob_id": BLOB_ID, "user_id": owner, "image_path": f"/v1/memory/images/{BLOB_ID}"},
            ),
            MemorySearchItem(  # bare image hit: blob known, owner unknown
                id=BLOB_ID,
                memory="[image]",
                memory_type="image",
                last_update_at="2026-10-28 00:00:00",
                metadata={"modality": "image", "blob_id": OTHER_BLOB_ID, "image_path": f"/v1/memory/images/{OTHER_BLOB_ID}"},
            ),
            MemorySearchItem(  # malformed blob id: never decorated
                id="img-bad",
                memory="[image] broken",
                memory_type="image",
                last_update_at="2026-10-28 00:00:00",
                metadata={"modality": "image", "blob_id": "not-an-id", "user_id": owner},
            ),
        ],
    )


def _client(auth_context: AuthContext, service: MemoryService) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(router)
    app.dependency_overrides[get_request_context] = lambda: auth_context
    app.dependency_overrides[get_memory_service] = lambda: service
    return TestClient(app)


def _service(monkeypatch: pytest.MonkeyPatch, store: Any, *, owner: str = OWNER) -> MemoryService:
    monkeypatch.setattr(memory_service_module, "get_blob_store", lambda cfg=None: store)
    return MemoryService(
        skill_store=object(),
        search_pipeline=_FakeSearchPipeline(_image_result(owner=owner)),
        operation_recorder=_NoopRecorder(),
    )


def _search(client: TestClient) -> list[dict]:
    response = client.post("/v1/memory/search", json={"query": "screenshot", "user_id": OWNER})
    assert response.status_code == 200
    return response.json()["data"]["memories"]


def test_local_backend_falls_back_to_the_relative_proxy_route(tmp_path, monkeypatch, auth_context, config) -> None:
    client = _client(auth_context, _service(monkeypatch, LocalBlobStore(tmp_path)))

    memories = _search(client)
    by_id = {item["id"]: item for item in memories}

    assert "image_url" not in by_id["mem-1"]["metadata"]  # text hit untouched
    assert by_id["img-1"]["metadata"]["image_url"] == f"/v1/memory/images/{BLOB_ID}?user_id={OWNER}"
    # bare hit has no owner to name on the proxy route -> stays undecorated
    assert "image_url" not in by_id[BLOB_ID]["metadata"]
    assert "image_url" not in by_id["img-bad"]["metadata"]


def test_proxy_fallback_quotes_special_owner_ids(tmp_path, monkeypatch, auth_context, config) -> None:
    """An owner like ``a&b=张三`` must stay one query value, not corrupt the URL."""
    client = _client(auth_context, _service(monkeypatch, LocalBlobStore(tmp_path), owner="a&b=张三"))

    memories = _search(client)

    assert memories[1]["metadata"]["image_url"] == (
        f"/v1/memory/images/{BLOB_ID}?user_id=a%26b%3D%E5%BC%A0%E4%B8%89"
    )


def test_presigning_backend_decorates_every_valid_image_hit(tmp_path, monkeypatch, auth_context, config) -> None:
    store = _PresigningStore(tmp_path)
    client = _client(auth_context, _service(monkeypatch, store))

    memories = _search(client)
    by_id = {item["id"]: item for item in memories}

    assert by_id["img-1"]["metadata"]["image_url"].startswith("https://minio.example.com/blobs/")
    assert by_id[BLOB_ID]["metadata"]["image_url"].startswith("https://minio.example.com/blobs/")  # bare hit too
    assert "image_url" not in by_id["mem-1"]["metadata"]
    # signed per hit, in the project namespace, with the configured expiry
    assert store.presigned == [
        ("project-1", BLOB_ID, 3600),
        ("project-1", OTHER_BLOB_ID, 3600),
    ]


def test_internal_caller_gets_the_relative_proxy_path(tmp_path, monkeypatch, auth_context, config) -> None:
    """Internal callers (no HTTP request) get the same inert relative path, never a bogus absolute origin."""
    service = _service(monkeypatch, LocalBlobStore(tmp_path))

    result = asyncio.run(service.search(auth_context, SearchRequest(query="q")))

    by_id = {item.id: item for item in result.memories}
    assert by_id["img-1"].metadata["image_url"] == f"/v1/memory/images/{BLOB_ID}?user_id={OWNER}"
    assert "image_url" not in by_id[BLOB_ID].metadata  # bare hit stays undecorated


def test_expiry_flows_from_config_into_the_signed_urls(tmp_path, monkeypatch, auth_context, config_expiry_7200) -> None:
    """s3_presign_expiry_seconds is plumbed through, not just the 3600 default."""
    store = _PresigningStore(tmp_path)
    client = _client(auth_context, _service(monkeypatch, store))

    _search(client)

    assert store.presigned == [
        ("project-1", BLOB_ID, 7200),
        ("project-1", OTHER_BLOB_ID, 7200),
    ]


def test_decoration_failure_never_breaks_the_search(tmp_path, monkeypatch, auth_context, config) -> None:
    """A broken blob store (e.g. backend s3 without the extra) degrades to no image_url."""

    class _ExplodingStore(LocalBlobStore):
        def presigned_get(self, *args, **kwargs):
            raise ImportError("mindmemos[s3] extra missing")

    client = _client(auth_context, _service(monkeypatch, _ExplodingStore(tmp_path)))

    memories = _search(client)  # 200, not 500

    assert len(memories) == 4 and all("image_url" not in item["metadata"] for item in memories)
