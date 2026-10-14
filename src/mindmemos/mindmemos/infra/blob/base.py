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

    def public_url(self, namespace: str, blob_id: str) -> str | None:
        """Return a URL third parties can fetch (object storage); None for private backends."""
        ...


def blob_id_for(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_valid_blob_id(blob_id: str) -> bool:
    return bool(_BLOB_ID_RE.match(blob_id or ""))
