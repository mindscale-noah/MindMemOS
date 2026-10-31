"""HTTP contract: POST /v1/memory/search decorates image hits with metadata.image_url."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

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
from mindmemos.infra.blob.signed_url import image_url_signature
from mindmemos.typing import MemorySearchItem, SearchPipelineResult

from tests.blob_fakes import MemoryBlobStore, MirroredBlobStore, PresigningBlobStore

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


@pytest.fixture
def config_proxy_base(tmp_path: Path) -> None:
    path = tmp_path / "dev.yaml"
    path.write_text(
        "auth:\n  mode: gateway_jwt\n  gateway_jwt_secret: test-secret\n"
        "blob_store:\n  proxy_public_base_url: https://memory.example.com\n",
        encoding="utf-8",
    )
    init_config(config_path=path)
    yield
    reset_config()


@pytest.fixture
def config_proxy_expiry_720(tmp_path: Path) -> None:
    path = tmp_path / "dev.yaml"
    path.write_text(
        "auth:\n  mode: gateway_jwt\n  gateway_jwt_secret: test-secret\nblob_store:\n  proxy_url_expiry_seconds: 720\n",
        encoding="utf-8",
    )
    init_config(config_path=path)
    yield
    reset_config()


@pytest.fixture
def config_dedicated_signing_secret(tmp_path: Path) -> None:
    """Both secrets set, distinct: the dedicated one must win on mint and on verify."""
    path = tmp_path / "dev.yaml"
    path.write_text(
        "auth:\n  mode: gateway_jwt\n  gateway_jwt_secret: test-jwt-secret\n"
        "blob_store:\n  url_signing_secret: dedicated-signing-secret\n",
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
                metadata={"modality": "image", "blob_id": BLOB_ID, "user_id": owner},
            ),
            MemorySearchItem(  # bare image hit: blob known, owner unknown
                id=BLOB_ID,
                memory="[image]",
                memory_type="image",
                last_update_at="2026-10-28 00:00:00",
                metadata={"modality": "image", "blob_id": OTHER_BLOB_ID},
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


def test_presign_failure_falls_back_to_the_signed_proxy_url(monkeypatch, auth_context, config) -> None:
    client = _client(auth_context, _service(monkeypatch, MemoryBlobStore()))

    memories = _search(client)
    by_id = {item["id"]: item for item in memories}

    assert "image_url" not in by_id["mem-1"]["metadata"]  # text hit untouched
    url = by_id["img-1"]["metadata"]["image_url"]
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.path == f"/v1/memory/images/{BLOB_ID}"  # relative: no trusted public base configured
    assert query["user_id"] == [OWNER]
    assert query["project_id"] == ["project-1"]
    assert int(query["expires"][0]) > int(time.time())
    assert query["signature"] == [
        image_url_signature("project-1", BLOB_ID, OWNER, int(query["expires"][0]), "test-secret")
    ]
    # bare hit has no owner to name on the proxy route -> stays undecorated
    assert "image_url" not in by_id[BLOB_ID]["metadata"]
    assert "image_url" not in by_id["img-bad"]["metadata"]


def test_dedicated_signing_secret_round_trips_the_image_route(
    monkeypatch, auth_context, config_dedicated_signing_secret
) -> None:
    """url_signing_secret must win over gateway_jwt_secret on BOTH sides of the link.

    Search mints the signature and the image route verifies it; every other fixture leaves
    url_signing_secret unset, so a swapped argument or wrong-secret lookup on either side
    would go unnoticed. Here the two secrets differ, the mint must sign with the dedicated
    one, and the minted link must then pass the route's own verification with no API key.
    """
    service = _service(monkeypatch, MemoryBlobStore())

    async def _get_image(_auth, _blob_id, _user_id):
        return PNG_BYTES, "image/png"

    monkeypatch.setattr(service, "get_image", _get_image)
    client = _client(auth_context, service)  # get_optional_request_context stays real: verify runs

    memories = _search(client)
    parsed = urlparse(memories[1]["metadata"]["image_url"])
    query = parse_qs(parsed.query)
    expires = int(query["expires"][0])
    assert query["signature"] == [image_url_signature("project-1", BLOB_ID, OWNER, expires, "dedicated-signing-secret")]
    assert query["signature"] != [image_url_signature("project-1", BLOB_ID, OWNER, expires, "test-jwt-secret")]

    # round trip: the same server verifies its own mint with the dedicated secret, keyless
    response = client.get(f"{parsed.path}?{parsed.query}")
    assert response.status_code == 200
    assert response.content == PNG_BYTES


def test_no_signing_secret_skips_the_signed_tier_instead_of_minting_with_an_empty_key(
    monkeypatch, auth_context, tmp_path
) -> None:
    """Presign fails and no secret is configured: the link is absent, never forgeable-empty."""

    path = tmp_path / "dev.yaml"
    path.write_text("auth:\n  mode: api_key\n", encoding="utf-8")  # no gateway_jwt_secret / url_signing_secret
    init_config(config_path=path)
    try:
        client = _client(auth_context, _service(monkeypatch, MemoryBlobStore()))
        memories = _search(client)
    finally:
        reset_config()

    by_id = {item["id"]: item for item in memories}
    assert "image_url" not in by_id["mem-1"]["metadata"]
    assert "image_url" not in by_id["img-1"]["metadata"]  # tier skipped, not minted with secret=""


def test_signed_proxy_url_is_absolute_under_proxy_public_base_url(monkeypatch, auth_context, config_proxy_base) -> None:
    client = _client(auth_context, _service(monkeypatch, MemoryBlobStore()))

    memories = _search(client)

    url = memories[1]["metadata"]["image_url"]
    assert url.startswith("https://memory.example.com/v1/memory/images/")
    parsed = urlparse(url)
    assert parsed.scheme == "https" and parsed.netloc == "memory.example.com"
    query = parse_qs(parsed.query)
    assert query["signature"] == [
        image_url_signature("project-1", BLOB_ID, OWNER, int(query["expires"][0]), "test-secret")
    ]


def test_signed_proxy_url_honours_proxy_url_expiry_seconds(monkeypatch, auth_context, config_proxy_expiry_720) -> None:
    client = _client(auth_context, _service(monkeypatch, MemoryBlobStore()))

    memories = _search(client)

    query = parse_qs(urlparse(memories[1]["metadata"]["image_url"]).query)
    remaining = int(query["expires"][0]) - int(time.time())
    assert 700 < remaining <= 720  # proxy knob wins over the 3600 s3 default


def test_proxy_fallback_quotes_special_owner_ids(monkeypatch, auth_context, config) -> None:
    """An owner like ``a&b=张三`` must stay one query value, not corrupt the URL."""
    client = _client(auth_context, _service(monkeypatch, MemoryBlobStore(), owner="a&b=张三"))

    memories = _search(client)

    parsed = urlparse(memories[1]["metadata"]["image_url"])
    query = parse_qs(parsed.query)
    assert parsed.path == f"/v1/memory/images/{BLOB_ID}"
    assert query["user_id"] == ["a&b=张三"]  # parse_qs decodes exactly one value back
    assert query["signature"] == [
        image_url_signature("project-1", BLOB_ID, "a&b=张三", int(query["expires"][0]), "test-secret")
    ]


def test_presigning_backend_decorates_every_valid_image_hit(monkeypatch, auth_context, config) -> None:
    store = PresigningBlobStore()
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


def test_public_mirror_url_wins_over_presigning(monkeypatch, auth_context, config) -> None:
    """A configured s3_public_base_url serves permanent URLs; no signing round per hit."""
    store = MirroredBlobStore()
    client = _client(auth_context, _service(monkeypatch, store))

    memories = _search(client)
    by_id = {item["id"]: item for item in memories}

    assert by_id["img-1"]["metadata"]["image_url"] == f"https://cdn.example.com/mirror/project-1/{BLOB_ID}"
    assert by_id[BLOB_ID]["metadata"]["image_url"] == f"https://cdn.example.com/mirror/project-1/{OTHER_BLOB_ID}"
    assert store.presigned == []  # the mirror makes presigning unnecessary


def test_internal_caller_gets_the_signed_proxy_path(monkeypatch, auth_context, config) -> None:
    """Internal callers (no HTTP request) get the same relative signed path, never a bogus absolute origin."""
    service = _service(monkeypatch, MemoryBlobStore())

    result = asyncio.run(service.search(auth_context, SearchRequest(query="q")))

    by_id = {item.id: item for item in result.memories}
    url = by_id["img-1"].metadata["image_url"]
    parsed = urlparse(url)
    assert parsed.path == f"/v1/memory/images/{BLOB_ID}"
    query = parse_qs(parsed.query)
    assert query["user_id"] == [OWNER]
    assert query["signature"] == [
        image_url_signature("project-1", BLOB_ID, OWNER, int(query["expires"][0]), "test-secret")
    ]
    assert "image_url" not in by_id[BLOB_ID].metadata  # bare hit stays undecorated


def test_expiry_flows_from_config_into_the_signed_urls(monkeypatch, auth_context, config_expiry_7200) -> None:
    """s3_presign_expiry_seconds is plumbed through, not just the 3600 default."""
    store = PresigningBlobStore()
    client = _client(auth_context, _service(monkeypatch, store))

    _search(client)

    assert store.presigned == [
        ("project-1", BLOB_ID, 7200),
        ("project-1", OTHER_BLOB_ID, 7200),
    ]


def test_decoration_failure_never_breaks_the_search(monkeypatch, auth_context, config) -> None:
    """A broken blob store (e.g. the minio client missing its extra) degrades to no image_url."""

    class _ExplodingStore(MemoryBlobStore):
        def presigned_get(self, *args, **kwargs):
            raise ImportError("mindmemos[s3] extra missing")

    client = _client(auth_context, _service(monkeypatch, _ExplodingStore()))

    memories = _search(client)  # 200, not 500

    assert len(memories) == 4 and all("image_url" not in item["metadata"] for item in memories)
