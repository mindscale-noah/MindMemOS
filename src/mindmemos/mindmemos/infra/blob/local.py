"""Local-directory blob store.

Layout: ``<root>/<namespace-dir>/<id[:2]>/<id>`` holds the bytes and ``<id>.json`` the descriptor.
Writes go through a temp file and ``os.replace`` so readers never see partial blobs.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

from .base import StoredBlob, blob_id_for, is_valid_blob_id
from .base import namespace_prefix as _namespace_dir


class LocalBlobStore:
    """Blob store backed by a local directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    async def put(self, namespace: str, data: bytes, *, mime_type: str) -> StoredBlob:
        return await asyncio.to_thread(self._put, namespace, data, mime_type)

    async def get(self, namespace: str, blob_id: str) -> tuple[StoredBlob, bytes] | None:
        return await asyncio.to_thread(self._get, namespace, blob_id)

    async def stat(self, namespace: str, blob_id: str) -> StoredBlob | None:
        return await asyncio.to_thread(self._stat, namespace, blob_id)

    async def delete(self, namespace: str, blob_id: str) -> bool:
        return await asyncio.to_thread(self._delete, namespace, blob_id)

    async def delete_namespace(self, namespace: str) -> None:
        await asyncio.to_thread(shutil.rmtree, self._root / _namespace_dir(namespace), True)

    def public_url(self, namespace: str, blob_id: str) -> str | None:
        del namespace, blob_id
        return None

    def presigned_get(self, namespace: str, blob_id: str, *, expires_seconds: int = 3600) -> str | None:
        del namespace, blob_id, expires_seconds
        return None  # no object-storage signing on the local backend; callers use the proxy route

    def _delete(self, namespace: str, blob_id: str) -> bool:
        if not is_valid_blob_id(blob_id):
            return False
        path = self._path(namespace, blob_id)
        existed = path.exists()
        path.unlink(missing_ok=True)
        path.with_suffix(".json").unlink(missing_ok=True)
        return existed

    def _put(self, namespace: str, data: bytes, mime_type: str) -> StoredBlob:
        blob_id = blob_id_for(data)
        blob = StoredBlob(blob_id=blob_id, mime_type=mime_type, size_bytes=len(data))
        path = self._path(namespace, blob_id)
        if path.exists():
            return self._stat(namespace, blob_id) or blob
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, data)
        meta = {"mime_type": mime_type, "size_bytes": len(data)}
        _atomic_write(path.with_suffix(".json"), json.dumps(meta).encode("utf-8"))
        return blob

    def _get(self, namespace: str, blob_id: str) -> tuple[StoredBlob, bytes] | None:
        blob = self._stat(namespace, blob_id)
        if blob is None:
            return None
        return blob, self._path(namespace, blob_id).read_bytes()

    def _stat(self, namespace: str, blob_id: str) -> StoredBlob | None:
        if not is_valid_blob_id(blob_id):
            return None
        path = self._path(namespace, blob_id)
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return StoredBlob(blob_id=blob_id, mime_type=str(meta["mime_type"]), size_bytes=int(meta["size_bytes"]))

    def _path(self, namespace: str, blob_id: str) -> Path:
        if not is_valid_blob_id(blob_id):
            raise ValueError("invalid blob id")
        return self._root / _namespace_dir(namespace) / blob_id[:2] / blob_id


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
