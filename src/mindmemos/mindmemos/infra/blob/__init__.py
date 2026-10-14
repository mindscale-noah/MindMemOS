"""Raw image (blob) storage used by multimodal memory algorithms."""

from __future__ import annotations

from pathlib import Path

from ...config import REPO_ROOT, BlobStoreConfig, get_config
from .base import BlobStore, StoredBlob, blob_id_for, is_valid_blob_id
from .local import LocalBlobStore

_STORES: dict[tuple[str, str], BlobStore] = {}


def get_blob_store(cfg: BlobStoreConfig | None = None) -> BlobStore:
    """Return the configured blob store (cached per backend and root)."""

    cfg = cfg or get_config().blob_store
    if cfg.backend != "local":
        raise ValueError(f"unsupported blob_store.backend: {cfg.backend}")
    root = Path(cfg.local_root).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    key = (cfg.backend, str(root))
    store = _STORES.get(key)
    if store is None:
        store = _STORES[key] = LocalBlobStore(root)
    return store


__all__ = [
    "BlobStore",
    "LocalBlobStore",
    "StoredBlob",
    "blob_id_for",
    "get_blob_store",
    "is_valid_blob_id",
]
