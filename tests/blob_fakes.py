"""In-memory BlobStore fakes shared by the api/components/pipelines/infra tests.

The blob store is S3-only in production, but tests must not need a live MinIO:
``MemoryBlobStore`` implements the same contract with a dict, and the subclasses
below add the URL shapes the S3 backend can produce (public mirror, presigning).
Import as ``from tests.blob_fakes import ...`` — ``tests/__init__.py`` makes the
package importable across the test directories.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from mindmemos.infra.blob import StoredBlob, blob_id_for


class MemoryBlobStore:
    """Dict-backed stand-in for the S3 store; private (no public or presigned URLs)."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], tuple[StoredBlob, bytes]] = {}

    async def put(self, namespace: str, data: bytes, *, mime_type: str) -> StoredBlob:
        blob = StoredBlob(blob_id=blob_id_for(data), mime_type=mime_type, size_bytes=len(data))
        self._objects[(namespace, blob.blob_id)] = (blob, data)
        return blob

    async def get(self, namespace: str, blob_id: str) -> tuple[StoredBlob, bytes] | None:
        return self._objects.get((namespace, blob_id))

    async def stat(self, namespace: str, blob_id: str) -> StoredBlob | None:
        entry = self._objects.get((namespace, blob_id))
        return entry[0] if entry else None

    async def delete(self, namespace: str, blob_id: str) -> bool:
        return self._objects.pop((namespace, blob_id), None) is not None

    async def delete_namespace(self, namespace: str) -> None:
        for key in [key for key in self._objects if key[0] == namespace]:
            del self._objects[key]

    def public_url(self, namespace: str, blob_id: str) -> str | None:
        return None

    def presigned_get(self, namespace: str, blob_id: str, *, expires_seconds: int = 3600) -> str | None:
        return None


class PresigningBlobStore(MemoryBlobStore):
    """Signing-capable store: records presign calls like the MinIO client path."""

    def __init__(self) -> None:
        super().__init__()
        self.presigned: list[tuple[str, str, int]] = []

    def presigned_get(self, namespace: str, blob_id: str, *, expires_seconds: int = 3600) -> str | None:
        self.presigned.append((namespace, blob_id, expires_seconds))
        return f"https://minio.example.com/blobs/{namespace}/{blob_id}?X-Amz-Signature=fake&expires={expires_seconds}"


class MirroredBlobStore(PresigningBlobStore):
    """Store behind a public mirror (``s3_public_base_url``): signing stays unused."""

    def public_url(self, namespace: str, blob_id: str) -> str | None:
        return f"https://cdn.example.com/mirror/{namespace}/{blob_id}"


class ExplodingPresignBlobStore(MemoryBlobStore):
    """Store whose presigning raises, e.g. the minio client missing its extra."""

    def presigned_get(self, namespace: str, blob_id: str, *, expires_seconds: int = 3600) -> str | None:
        raise ImportError("mindmemos[s3] extra missing")


def deterministic_blob_id(seed: str) -> str:
    """A valid 64-hex blob id derived from a seed (content addressing without real bytes)."""

    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


class FakeResponse:
    """Stand-in for a minio object download stream."""

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


def no_such_key():
    """The S3Error a stat of a missing object raises (minio is installed in the dev env)."""

    from minio.error import S3Error

    return S3Error(None, "NoSuchKey", "The specified key does not exist.", "resource", "request-id", "host-id")


class FakeMinio:
    """In-memory stand-in for minio.Minio: dict storage plus call counters."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], tuple[bytes, str]] = {}
        self.buckets: set[str] = set()
        self.delete_batches: list[list[str]] = []
        self.last_response: FakeResponse | None = None
        self.exists_answers: list[bool] | None = None
        self.race_on_make = False
        self.presigned_args: list[tuple[str, str, object]] = []
        self.presign_error: Exception | None = None
        self.calls: dict[str, int] = {"put_object": 0, "bucket_exists": 0, "make_bucket": 0}

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
            from minio.error import S3Error

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
            raise no_such_key()
        return SimpleNamespace(size=len(entry[0]), content_type=entry[1], is_dir=False)

    def get_object(self, bucket: str, name: str) -> FakeResponse:
        entry = self.objects[(bucket, name)]
        self.last_response = FakeResponse(entry[0])
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
