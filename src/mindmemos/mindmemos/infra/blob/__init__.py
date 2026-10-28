"""Raw image (blob) storage used by multimodal memory (vanilla, ``algo_config.multimodal``)."""

from __future__ import annotations

from pathlib import Path

from ...config import REPO_ROOT, BlobStoreConfig, get_config
from .base import BlobStore, StoredBlob, blob_id_for, is_valid_blob_id
from .local import LocalBlobStore

_STORES: dict[tuple, BlobStore] = {}


def get_blob_store(cfg: BlobStoreConfig | None = None) -> BlobStore:
    """Return the configured blob store (cached per backend and endpoint)."""

    cfg = cfg or get_config().blob_store
    if cfg.backend == "local":
        root = Path(cfg.local_root).expanduser()
        if not root.is_absolute():
            root = REPO_ROOT / root
        key = (cfg.backend, str(root))
        store = _STORES.get(key)
        if store is None:
            store = _STORES[key] = LocalBlobStore(root)
        return store
    if cfg.backend == "s3":
        from .s3 import S3BlobStore  # lazy: the minio client stays an optional extra

        key = (cfg.backend, cfg.s3_endpoint, cfg.s3_bucket, cfg.s3_presign_endpoint or "")
        store = _STORES.get(key)
        if store is None:
            store = _STORES[key] = S3BlobStore(
                endpoint=cfg.s3_endpoint,
                bucket=cfg.s3_bucket,
                access_key=cfg.s3_access_key or "",
                secret_key=cfg.s3_secret_key or "",
                region=cfg.s3_region,
                public_base_url=cfg.s3_public_base_url,
                presign_endpoint=cfg.s3_presign_endpoint,
            )
        return store
    raise ValueError(f"unsupported blob_store.backend: {cfg.backend}")


__all__ = [
    "BlobStore",
    "LocalBlobStore",
    "StoredBlob",
    "blob_id_for",
    "get_blob_store",
    "is_valid_blob_id",
]
