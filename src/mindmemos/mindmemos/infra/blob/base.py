"""Blob store contract for raw image bytes.

Blobs are content addressed: ``blob_id`` is the SHA-256 hex digest of the bytes, so storing the
same image twice keeps one copy. Every blob lives in a namespace (the project id) and is only
readable through that namespace, which keeps projects isolated.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

_BLOB_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class StoredBlob:
    blob_id: str
    mime_type: str
    size_bytes: int


class BlobStore(Protocol):
    """Content-addressed byte storage, scoped by namespace."""

    async def put(self, namespace: str, data: bytes, *, mime_type: str) -> StoredBlob:
        """Store ``data`` (idempotent) and return its descriptor."""
        ...

    async def get(self, namespace: str, blob_id: str) -> tuple[StoredBlob, bytes] | None:
        """Return the blob and its bytes, or None when absent."""
        ...

    async def stat(self, namespace: str, blob_id: str) -> StoredBlob | None:
        """Return the blob descriptor without reading the bytes."""
        ...

    async def delete(self, namespace: str, blob_id: str) -> bool:
        """Delete one blob; return whether it existed."""
        ...

    async def delete_namespace(self, namespace: str) -> None:
        """Delete every blob of a namespace (a project purge)."""
        ...

    def public_url(self, namespace: str, blob_id: str) -> str | None:
        """Return a URL third parties can fetch (object storage); None for private backends."""
        ...

    def presigned_get(self, namespace: str, blob_id: str, *, expires_seconds: int = 3600) -> str | None:
        """Return a short-lived signed download URL, or None when the backend cannot sign one.

        Signing is a local computation (no store I/O) and must never raise; callers fall back
        to the authenticated proxy route when None comes back.
        """
        ...


def blob_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_valid_blob_id(blob_id: str) -> bool:
    return bool(_BLOB_ID_RE.match(blob_id or ""))


def namespace_prefix(namespace: str) -> str:
    """Readable, collision-free, traversal-safe prefix for a namespace (directory or object key)."""

    if not namespace:
        raise ValueError("blob namespace must not be empty")
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()[:12]
    return f"{_UNSAFE.sub('_', namespace)[:64]}-{digest}"
