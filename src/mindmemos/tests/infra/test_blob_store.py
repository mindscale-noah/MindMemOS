import dataclasses

from mindmemos.config import BlobStoreConfig
from mindmemos.infra import blob as blob_module
from mindmemos.infra.blob import get_blob_store
from mindmemos.infra.blob.s3 import S3BlobStore

S3 = BlobStoreConfig(
    s3_endpoint="http://localhost:9100",
    s3_bucket="blobs",
    s3_access_key="a",
    s3_secret_key="s",
)


def test_get_blob_store_builds_and_caches_the_s3_store(monkeypatch) -> None:
    monkeypatch.setattr(blob_module, "_STORES", {})

    first = get_blob_store(S3)
    assert isinstance(first, S3BlobStore)
    assert get_blob_store(S3) is first  # cached per (endpoint, bucket, credentials, ...)
    assert get_blob_store(dataclasses.replace(S3, s3_bucket="other")) is not first
    # the client signs with its credentials, so rotated keys must not keep the cached client
    assert get_blob_store(dataclasses.replace(S3, s3_access_key="a2", s3_secret_key="s2")) is not first
    assert get_blob_store(dataclasses.replace(S3, s3_endpoint="http://localhost:9101")) is not first
