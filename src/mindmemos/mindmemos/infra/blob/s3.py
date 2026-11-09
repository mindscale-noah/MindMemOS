"""S3-compatible blob store (MinIO, Cloudflare R2, OSS/R2 S3 gateways) — the only backend.

Object layout: ``<namespace-prefix>/<id[:2]>/<id>``; the MIME type rides on the object's
content-type (no sidecar object). The bucket is created lazily on first use so a missing
bucket self-heals in non-compose deployments too.

Requires the ``s3`` extra (``pip install mindmemos[s3]``); the import stays optional so the
package core does not pull the minio client until blob storage is first used.
"""

from __future__ import annotations

import asyncio
import io
import threading
from datetime import timedelta
from typing import Any
from urllib.parse import urlsplit

from ...logging import get_logger
from .base import StoredBlob, blob_id_for, is_valid_blob_id, namespace_prefix

try:
    from minio import Minio
    from minio.deleteobjects import DeleteObject
    from minio.error import S3Error
except ImportError:  # pragma: no cover - exercised only without the s3 extra
    Minio = None  # type: ignore[assignment]
    DeleteObject = None  # type: ignore[assignment]
    S3Error = Exception  # type: ignore[assignment,misc]

logger = get_logger(__name__)

_MISSING_OBJECT_CODES = {"NoSuchKey", "NoSuchObject"}
_DELETE_BATCH_SIZE = 1000


class S3BlobStore:
    """Blob store backed by any S3-compatible endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        public_base_url: str | None = None,
        presign_endpoint: str | None = None,
        client: Any | None = None,
        presign_client: Any | None = None,
    ) -> None:
        host, secure = _parse_endpoint(endpoint)
        if client is None and Minio is None:
            raise ImportError("the S3 blob store requires the 's3' extra (pip install mindmemos[s3])")
        self._client = client or Minio(
            host, access_key=access_key, secret_key=secret_key, secure=secure, region=region or None
        )
        self._bucket = bucket
        self._region = region
        self._public_base_url = public_base_url
        self._access_key = access_key
        self._secret_key = secret_key
        # Presigned URLs handed to clients must point at an endpoint they can reach; when the
        # server talks to MinIO over an internal address, sign against this external one instead.
        self._presign_endpoint = presign_endpoint
        # Without a presign endpoint the main client signs; self._client, not the ctor
        # argument, which is None when no client was injected (the default path).
        self._presign_client: Any | None = presign_client or (None if presign_endpoint else self._client)
        self._presign_client_lock = threading.Lock()
        self._bucket_ready = False
        self._bucket_lock = threading.Lock()

    async def put(self, namespace: str, data: bytes, *, mime_type: str) -> StoredBlob:
        return await asyncio.to_thread(self._put, namespace, data, mime_type)

    async def get(self, namespace: str, blob_id: str) -> tuple[StoredBlob, bytes] | None:
        return await asyncio.to_thread(self._get, namespace, blob_id)

    async def stat(self, namespace: str, blob_id: str) -> StoredBlob | None:
        return await asyncio.to_thread(self._stat, namespace, blob_id)

    async def delete(self, namespace: str, blob_id: str) -> bool:
        return await asyncio.to_thread(self._delete, namespace, blob_id)

    async def delete_namespace(self, namespace: str) -> None:
        await asyncio.to_thread(self._delete_namespace, namespace)

    async def count_namespace(self, namespace: str) -> int:
        return await asyncio.to_thread(self._count_namespace, namespace)

    def public_url(self, namespace: str, blob_id: str) -> str | None:
        if not self._public_base_url or not is_valid_blob_id(blob_id):
            return None
        return f"{self._public_base_url.rstrip('/')}/{self._bucket}/{self._key(namespace, blob_id)}"

    def presigned_get(self, namespace: str, blob_id: str, *, expires_seconds: int = 3600) -> str | None:
        """Short-lived signed GET URL (local HMAC, no store I/O); None on any failure.

        Search decorates its results with these URLs; a signing failure must degrade to the
        proxy route, never break the search — so the lazy client build is guarded here too
        (the Minio constructor rejects some endpoints validation cannot, e.g. bad ports).
        """

        if not is_valid_blob_id(blob_id):
            return None
        try:
            client = self._presign_client_for()
            if client is None:
                return None
            return client.presigned_get_object(
                self._bucket,
                self._key(namespace, blob_id),
                expires=timedelta(seconds=max(1, int(expires_seconds))),
            )
        except Exception as exc:  # noqa: BLE001 - presign failure downgrades to the proxy route
            logger.warning("blob_s3_presign_failed", blob_id=blob_id, error=str(exc))
            return None

    def _presign_client_for(self) -> Any | None:
        # False marks a build that already failed: stop retrying (and re-warning) per search.
        if self._presign_client is not None:
            return self._presign_client or None
        with self._presign_client_lock:
            if self._presign_client is not None:
                return self._presign_client or None
            if Minio is None:  # injected client path without the s3 extra
                return None
            try:
                host, secure = _parse_endpoint(self._presign_endpoint or "")
                client: Any = Minio(
                    host,
                    access_key=self._access_key,
                    secret_key=self._secret_key,
                    secure=secure,
                    region=self._region or None,  # explicit region keeps presigning offline
                )
            except Exception as exc:  # noqa: BLE001 - unusable presign endpoint disables presigning
                logger.warning("blob_s3_presign_client_build_failed", error=str(exc))
                self._presign_client = False
                return None
            self._presign_client = client
            return client

    def _put(self, namespace: str, data: bytes, mime_type: str) -> StoredBlob:
        # No stat-first dedup: the key is the content hash, so a duplicate put overwrites the
        # identical object (idempotent) instead of costing every first store an extra round
        # trip. The mime type is sniffed from the same bytes and therefore identical too.
        self._ensure_bucket()
        blob_id = blob_id_for(data)
        self._client.put_object(
            self._bucket, self._key(namespace, blob_id), io.BytesIO(data), len(data), content_type=mime_type
        )
        return StoredBlob(blob_id=blob_id, mime_type=mime_type, size_bytes=len(data))

    def _get(self, namespace: str, blob_id: str) -> tuple[StoredBlob, bytes] | None:
        blob = self._stat(namespace, blob_id)
        if blob is None:
            return None
        response = None
        try:
            response = self._client.get_object(self._bucket, self._key(namespace, blob_id))
            return blob, response.read()
        finally:
            if response is not None:
                response.close()
                response.release_conn()

    def _stat(self, namespace: str, blob_id: str) -> StoredBlob | None:
        if not is_valid_blob_id(blob_id):
            return None
        self._ensure_bucket()
        try:
            obj = self._client.stat_object(self._bucket, self._key(namespace, blob_id))
        except S3Error as exc:
            if getattr(exc, "code", None) in _MISSING_OBJECT_CODES:
                return None
            raise
        return StoredBlob(
            blob_id=blob_id,
            mime_type=obj.content_type or "application/octet-stream",
            size_bytes=int(obj.size or 0),
        )

    def _delete(self, namespace: str, blob_id: str) -> bool:
        if self._stat(namespace, blob_id) is None:
            return False  # S3 DELETE is silent for missing objects; report non-existence plainly
        self._client.remove_object(self._bucket, self._key(namespace, blob_id))
        return True

    def _delete_namespace(self, namespace: str) -> None:
        self._ensure_bucket()
        prefix = f"{namespace_prefix(namespace)}/"
        batch: list[Any] = []
        for obj in self._client.list_objects(self._bucket, prefix=prefix, recursive=True):
            if obj.is_dir:
                continue
            batch.append(DeleteObject(obj.object_name))
            if len(batch) >= _DELETE_BATCH_SIZE:
                self._drain(self._client.remove_objects(self._bucket, batch))
                batch.clear()
        if batch:
            self._drain(self._client.remove_objects(self._bucket, batch))

    def _count_namespace(self, namespace: str) -> int:
        """Objects under one namespace's prefix (the eval project reset reports its work with it)."""

        self._ensure_bucket()
        prefix = f"{namespace_prefix(namespace)}/"
        return sum(
            1 for obj in self._client.list_objects(self._bucket, prefix=prefix, recursive=True) if not obj.is_dir
        )

    def _drain(self, errors: Any) -> None:
        # remove_objects is lazy: the deletes only happen as the error iterator is consumed.
        for error in errors:
            if getattr(error, "code", None) == "NoSuchKey":
                continue
            logger.warning("blob_s3_delete_failed", key=getattr(error, "name", None), code=getattr(error, "code", None))

    def _key(self, namespace: str, blob_id: str) -> str:
        if not is_valid_blob_id(blob_id):
            raise ValueError("invalid blob id")
        return f"{namespace_prefix(namespace)}/{blob_id[:2]}/{blob_id}"

    def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        with self._bucket_lock:
            if self._bucket_ready:
                return
            if not self._client.bucket_exists(self._bucket):
                try:
                    self._client.make_bucket(self._bucket, location=self._region or None)
                except S3Error as exc:
                    already_there = getattr(exc, "code", None) in {
                        "BucketAlreadyOwnedByYou",
                        "BucketAlreadyExists",
                    } and self._client.bucket_exists(self._bucket)
                    if not already_there:
                        raise
            self._bucket_ready = True


def _parse_endpoint(endpoint: str) -> tuple[str, bool]:
    parts = urlsplit(endpoint)
    return parts.netloc or parts.path, parts.scheme != "http"  # https (or scheme-less) -> TLS
