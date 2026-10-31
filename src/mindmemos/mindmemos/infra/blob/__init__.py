"""Raw image (blob) storage used by multimodal memory (vanilla, ``algo_config.multimodal``)."""

from __future__ import annotations

from ...config import BlobStoreConfig, get_config
from .base import BlobStore, StoredBlob, blob_id_for, is_valid_blob_id

_STORES: dict[tuple, BlobStore] = {}


def get_blob_store(cfg: BlobStoreConfig | None = None) -> BlobStore:
    """Return the configured S3-compatible blob store (cached per endpoint and credentials)."""

    cfg = cfg or get_config().blob_store
    from .s3 import S3BlobStore  # lazy: the minio client stays an optional extra

    key = (
        cfg.s3_endpoint,
        cfg.s3_bucket,
        cfg.s3_presign_endpoint or "",
        # clients sign with their credentials, so a rotated key or a changed region must
        # not keep serving the cached client built from the old ones
        cfg.s3_access_key or "",
        cfg.s3_secret_key or "",
        cfg.s3_region or "",
        cfg.s3_public_base_url or "",
    )
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


__all__ = [
    "BlobStore",
    "StoredBlob",
    "blob_id_for",
    "get_blob_store",
    "is_valid_blob_id",
]
