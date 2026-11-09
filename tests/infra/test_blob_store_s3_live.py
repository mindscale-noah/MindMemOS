"""Live round-trip against a real S3 endpoint (MinIO and friends).

Opt-in — skipped unless ``MINDMEMOS_TEST_S3_ENDPOINT`` is set:

    MINDMEMOS_TEST_S3_ENDPOINT=http://localhost:9100 uv run pytest tests/infra/test_blob_store_s3_live.py -q

Credentials default to the compose dev ones (minioadmin/minioadmin, see ``make dev-blobs``).
Each run gets its own bucket and removes it afterwards; set ``MINDMEMOS_TEST_S3_BUCKET`` to
reuse an existing bucket instead (it is left in place, only the test namespaces are cleared).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from mindmemos.infra.blob.s3 import S3BlobStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("MINDMEMOS_TEST_S3_ENDPOINT"),
    reason="set MINDMEMOS_TEST_S3_ENDPOINT (e.g. http://localhost:9100) to run live S3 tests",
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + os.urandom(64)


@pytest.fixture()
def store():
    bucket = os.environ.get("MINDMEMOS_TEST_S3_BUCKET") or f"mindmemos-blobs-test-{uuid.uuid4().hex[:8]}"
    store = S3BlobStore(
        endpoint=os.environ["MINDMEMOS_TEST_S3_ENDPOINT"],
        bucket=bucket,
        access_key=os.environ.get("MINDMEMOS_TEST_S3_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("MINDMEMOS_TEST_S3_SECRET_KEY", "minioadmin"),
    )
    yield store
    for namespace in ("live-a", "live-b"):  # best-effort cleanup, never masks the test outcome
        try:
            asyncio.run(store.delete_namespace(namespace))
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"live cleanup failed for namespace {namespace}: {exc}")
    if bucket == store._bucket and "mindmemos-blobs-test-" in bucket:
        try:
            store._client.remove_bucket(store._bucket)
        except Exception:  # noqa: BLE001 - leftover objects keep the bucket; the next run uses a new one
            pass


@pytest.mark.asyncio
async def test_bucket_is_created_lazily_and_round_trips(store) -> None:
    blob = await store.put("live-a", PNG_BYTES, mime_type="image/png")  # first use creates the bucket
    assert blob.size_bytes == len(PNG_BYTES)
    assert (await store.put("live-a", PNG_BYTES, mime_type="image/png")).blob_id == blob.blob_id

    stat = await store.stat("live-a", blob.blob_id)
    assert stat is not None and stat.mime_type == "image/png"

    fetched = await store.get("live-a", blob.blob_id)
    assert fetched is not None
    stored, data = fetched
    assert data == PNG_BYTES and stored.blob_id == blob.blob_id

    assert await store.stat("live-b", blob.blob_id) is None  # namespaces are isolated
    assert await store.delete("live-a", blob.blob_id) is True
    assert await store.delete("live-a", blob.blob_id) is False  # idempotent
    assert await store.get("live-a", blob.blob_id) is None


@pytest.mark.asyncio
async def test_delete_namespace_clears_only_that_namespace(store) -> None:
    a = await store.put("live-a", b"one", mime_type="image/png")
    b = await store.put("live-b", b"two", mime_type="image/png")

    await store.delete_namespace("live-a")
    await store.delete_namespace("live-a")  # idempotent

    assert await store.get("live-a", a.blob_id) is None
    assert await store.get("live-b", b.blob_id) is not None


@pytest.mark.asyncio
async def test_presigned_url_downloads_the_blob_without_credentials(store) -> None:
    """What search hands clients: a bare GET on the signed URL returns the exact bytes."""
    import urllib.request

    blob = await store.put("live-a", PNG_BYTES, mime_type="image/png")
    url = store.presigned_get("live-a", blob.blob_id, expires_seconds=300)

    assert url is not None and "X-Amz-Signature" in url
    # No auth headers, just the link. The opener ignores HTTP(S)_PROXY so a stray
    # local proxy cannot hijack the localhost fetch.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as response:
        assert response.status == 200
        assert response.read() == PNG_BYTES

    assert await store.get("live-b", blob.blob_id) is None  # the signature covers only this object
