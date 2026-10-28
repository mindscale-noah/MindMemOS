"""Unit tests for the S3 blob store against an injected fake minio client."""

from __future__ import annotations

import hashlib
from datetime import timedelta
from types import SimpleNamespace

import pytest
from mindmemos.infra.blob import is_valid_blob_id
from mindmemos.infra.blob.s3 import S3BlobStore, namespace_prefix
from minio.error import S3Error

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"


def _no_such_key() -> S3Error:
    return S3Error(None, "NoSuchKey", "The specified key does not exist.", "resource", "request-id", "host-id")


class _FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.closed = False
        self.released = False

    def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        self.closed = True

    def release_conn(self) -> None:
        self.released = True


class _FakeMinio:
    """In-memory stand-in for minio.Minio: dict storage plus call counters."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.buckets: set[str] = set()
        self.delete_batches: list[list[str]] = []
        self.last_response: _FakeResponse | None = None
        self.exists_answers: list[bool] | None = None
        self.race_on_make = False
        self.presigned_args: list[tuple[str, str, object]] = []
        self.presign_error: Exception | None = None
        self.calls = {"put_object": 0, "bucket_exists": 0, "make_bucket": 0}

    def bucket_exists(self, bucket: str) -> bool:
        self.calls["bucket_exists"] += 1
        if self.exists_answers is not None:
            return self.exists_answers.pop(0)
        return bucket in self.buckets

    def make_bucket(self, bucket: str, location: str | None = None) -> None:
        self.calls["make_bucket"] += 1
        already = bucket in self.buckets
        self.buckets.add(bucket)
        if already or self.race_on_make:  # race_on_make: someone else created it between check and create
            raise S3Error(None, "BucketAlreadyOwnedByYou", "own bucket", "resource", "request-id", "host-id")

    def put_object(
        self, bucket: str, name: str, data, length: int, content_type: str = "application/octet-stream"
    ) -> None:
        payload = data.read()
        assert len(payload) == length
        self.calls["put_object"] += 1
        self.objects[(bucket, name)] = (payload, content_type)

    def stat_object(self, bucket: str, name: str):
        entry = self.objects.get((bucket, name))
        if entry is None:
            raise _no_such_key()
        return SimpleNamespace(size=len(entry[0]), content_type=entry[1], is_dir=False)

    def get_object(self, bucket: str, name: str) -> _FakeResponse:
        entry = self.objects[(bucket, name)]
        self.last_response = _FakeResponse(entry[0])
        return self.last_response

    def remove_object(self, bucket: str, name: str) -> None:
        self.objects.pop((bucket, name), None)

    def remove_objects(self, bucket: str, delete_object_list):
        batch = [obj.name for obj in delete_object_list]
        self.delete_batches.append(batch)
        for name in batch:
            self.objects.pop((bucket, name), None)
        return iter(())  # no errors

    def list_objects(self, bucket: str, prefix: str | None = None, recursive: bool = False):
        for b, name in sorted(self.objects):
            if b == bucket and name.startswith(prefix or ""):
                yield SimpleNamespace(object_name=name, is_dir=False)

    def presigned_get_object(self, bucket, name, expires=None, **kwargs):
        self.calls["presigned_get_object"] = self.calls.get("presigned_get_object", 0) + 1
        self.presigned_args.append((bucket, name, expires))
        if self.presign_error is not None:
            raise self.presign_error
        return f"http://localhost:9100/{bucket}/{name}?X-Amz-Signature=fake"


def _store(client: _FakeMinio | None = None, **kwargs) -> tuple[S3BlobStore, _FakeMinio]:
    client = client or _FakeMinio()
    store = S3BlobStore(
        endpoint="http://localhost:9100",
        bucket="blobs",
        access_key="a",
        secret_key="s",
        client=client,
        **kwargs,
    )
    return store, client


@pytest.mark.asyncio
async def test_put_is_content_addressed_and_idempotent() -> None:
    store, client = _store()
    first = await store.put("proj-a", PNG_BYTES, mime_type="image/png")
    second = await store.put("proj-a", PNG_BYTES, mime_type="image/png")

    assert first == second
    assert first.blob_id == hashlib.sha256(PNG_BYTES).hexdigest()
    # no stat-first dedup: the second put overwrites the identical content-addressed object
    # (idempotent) instead of costing every first store an extra round trip
    assert client.calls["put_object"] == 2
    prefix = namespace_prefix("proj-a")
    assert ("blobs", f"{prefix}/{first.blob_id[:2]}/{first.blob_id}") in client.objects
    blob, data = await store.get("proj-a", first.blob_id)
    assert data == PNG_BYTES and blob.mime_type == "image/png" and blob.size_bytes == len(PNG_BYTES)


@pytest.mark.asyncio
async def test_namespaces_are_isolated_and_ids_are_validated() -> None:
    store, client = _store()
    blob = await store.put("proj-a", PNG_BYTES, mime_type="image/png")

    assert await store.get("proj-b", blob.blob_id) is None
    assert await store.get("proj-a", "../../etc/passwd") is None
    assert await store.stat("proj-a", "0" * 63) is None
    assert await store.delete("proj-a", "not-an-id") is False
    assert not is_valid_blob_id("../evil")

    evil = await store.put("../evil/..", b"still safe", mime_type="image/png")  # namespace is sanitized
    key = f"{namespace_prefix('../evil/..')}/{evil.blob_id[:2]}/{evil.blob_id}"
    assert all("/.." not in name for (_, name) in client.objects)
    assert ("blobs", key) in client.objects


def test_public_url_reflects_the_configured_mirror() -> None:
    private, _ = _store()
    blob_id = hashlib.sha256(PNG_BYTES).hexdigest()
    assert private.public_url("proj-a", blob_id) is None

    public, _ = _store(public_base_url="https://cdn.example.com/minio")
    expected = f"https://cdn.example.com/minio/blobs/{namespace_prefix('proj-a')}/{blob_id[:2]}/{blob_id}"
    assert public.public_url("proj-a", blob_id) == expected
    assert public.public_url("proj-a", "bad-id") is None


def test_presigned_get_signs_the_object_key_with_the_requested_expiry() -> None:
    store, client = _store()
    blob_id = hashlib.sha256(PNG_BYTES).hexdigest()

    url = store.presigned_get("proj-a", blob_id)

    assert url == f"http://localhost:9100/blobs/{namespace_prefix('proj-a')}/{blob_id[:2]}/{blob_id}?X-Amz-Signature=fake"
    assert client.presigned_args == [("blobs", f"{namespace_prefix('proj-a')}/{blob_id[:2]}/{blob_id}", timedelta(seconds=3600))]

    store.presigned_get("proj-a", blob_id, expires_seconds=120)
    assert client.presigned_args[-1][2] == timedelta(seconds=120)


def test_presigned_get_rejects_bad_ids_and_never_raises() -> None:
    store, client = _store()

    assert store.presigned_get("proj-a", "not-an-id") is None
    assert store.presigned_get("proj-a", "not-an-id", expires_seconds=60) is None
    assert client.calls.get("presigned_get_object", 0) == 0  # invalid ids never reach the client

    client.presign_error = S3Error(None, "InternalError", "boom", "resource", "request-id", "host-id")
    blob_id = hashlib.sha256(PNG_BYTES).hexdigest()
    assert store.presigned_get("proj-a", blob_id) is None  # signing failure degrades to None


def test_presigned_get_uses_the_presign_client_when_configured() -> None:
    main = _FakeMinio()
    presign = _FakeMinio()
    store, _ = _store(client=main, presign_endpoint="https://blobs.example.com", presign_client=presign)
    blob_id = hashlib.sha256(PNG_BYTES).hexdigest()

    url = store.presigned_get("proj-a", blob_id)

    assert "X-Amz-Signature=fake" in url
    assert presign.calls.get("presigned_get_object", 0) == 1
    assert main.calls.get("presigned_get_object", 0) == 0  # the internal client never signs

    # without an injected presign client the main client signs against the internal endpoint
    default_store, default_client = _store()
    default_store.presigned_get("proj-a", blob_id)
    assert default_client.calls.get("presigned_get_object", 0) == 1


def test_presigned_get_without_any_injection_signs_with_the_main_client() -> None:
    """The default production path: no injected client, no presign endpoint.

    Signing is local HMAC, so a real Minio client works offline here. Guards against
    lazily building a hostless presign client from the (None) ctor argument instead
    of the constructed main client — the bug the live test caught (`https:///bucket/...`).
    """

    store = S3BlobStore(endpoint="http://localhost:9100", bucket="blobs", access_key="a", secret_key="s")
    blob_id = hashlib.sha256(PNG_BYTES).hexdigest()

    url = store.presigned_get("proj-a", blob_id)

    assert url is not None
    assert url.startswith("http://localhost:9100/blobs/")


def test_presigned_get_lazily_builds_the_client_from_the_presign_endpoint() -> None:
    """What get_blob_store() builds when MINDMEMOS_S3_PRESIGN_ENDPOINT is set: no injected
    clients at all — the lazy branch must produce URLs carrying the external endpoint's host."""

    store = S3BlobStore(
        endpoint="http://localhost:9100",
        bucket="blobs",
        access_key="a",
        secret_key="s",
        presign_endpoint="https://blobs.example.com",
    )
    blob_id = hashlib.sha256(PNG_BYTES).hexdigest()

    url = store.presigned_get("proj-a", blob_id)

    assert url is not None
    assert url.startswith("https://blobs.example.com/blobs/")
    assert store._presign_client is not store._client  # a separate client was built


def test_presigned_get_survives_an_unbuildable_presign_endpoint() -> None:
    """An endpoint the Minio constructor rejects (bad port) must degrade to None, not raise —
    the never-raise contract every search response relies on."""

    store = S3BlobStore(
        endpoint="http://localhost:9100",
        bucket="blobs",
        access_key="a",
        secret_key="s",
        presign_endpoint="http://blobs.example.com:90oo",
    )
    blob_id = hashlib.sha256(PNG_BYTES).hexdigest()

    assert store.presigned_get("proj-a", blob_id) is None
    assert store.presigned_get("proj-a", blob_id) is None  # failed build stays disabled, no retry storm


@pytest.mark.asyncio
async def test_delete_one_blob_and_a_whole_namespace() -> None:
    store, client = _store()
    a1 = await store.put("proj-a", b"one", mime_type="image/png")
    a2 = await store.put("proj-a", b"two", mime_type="image/png")
    b1 = await store.put("proj-b", b"three", mime_type="image/png")

    assert await store.delete("proj-a", a1.blob_id) is True
    assert await store.delete("proj-a", a1.blob_id) is False  # idempotent
    assert await store.get("proj-a", a2.blob_id) is not None

    await store.delete_namespace("proj-a")
    await store.delete_namespace("proj-a")  # idempotent, no error
    assert await store.get("proj-a", a2.blob_id) is None
    assert await store.get("proj-b", b1.blob_id) is not None  # other namespace untouched
    assert client.delete_batches  # deletions went through remove_objects


@pytest.mark.asyncio
async def test_get_releases_the_underlying_connection() -> None:
    store, client = _store()
    blob = await store.put("proj-a", PNG_BYTES, mime_type="image/png")

    _, data = await store.get("proj-a", blob.blob_id)

    assert data == PNG_BYTES
    assert client.last_response is not None
    assert client.last_response.closed and client.last_response.released


@pytest.mark.asyncio
async def test_bucket_is_ensured_once() -> None:
    store, client = _store()
    await store.put("proj-a", b"x", mime_type="image/png")
    await store.put("proj-a", b"y", mime_type="image/png")

    assert client.calls["bucket_exists"] == 1
    assert client.calls["make_bucket"] == 1
    assert "blobs" in client.buckets


@pytest.mark.asyncio
async def test_bucket_creation_race_is_swallowed_when_the_bucket_exists() -> None:
    client = _FakeMinio()
    client.race_on_make = True  # another writer creates the bucket between check and create
    client.exists_answers = [False, True]  # missing at the check, present at the re-check
    store, _ = _store(client)

    await store.put("proj-a", b"x", mime_type="image/png")

    assert client.calls["make_bucket"] == 1  # raised BucketAlreadyOwnedByYou, re-checked, swallowed


def test_missing_minio_extra_raises_install_hint(monkeypatch) -> None:
    from mindmemos.infra.blob import s3 as s3_module

    monkeypatch.setattr(s3_module, "Minio", None)
    with pytest.raises(ImportError, match=r"mindmemos\[s3\]"):
        S3BlobStore(endpoint="http://localhost:9100", bucket="b", access_key="a", secret_key="s")
