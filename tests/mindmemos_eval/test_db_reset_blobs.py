"""The eval project reset also deletes the project's stored image objects in S3."""

import pytest
from mindmemos.infra.blob.s3 import S3BlobStore, namespace_prefix
from mindmemos_eval.memory.db_reset import (
    ProjectResetError,
    ResetConfig,
    S3BlobSettings,
    _reset_blobs,
    resolve_s3_blob_settings,
)

from tests.blob_fakes import FakeMinio


@pytest.fixture
def fake_minio(monkeypatch: pytest.MonkeyPatch) -> FakeMinio:
    """_reset_blobs builds its own S3BlobStore; route that construction at the fake client."""

    client = FakeMinio()
    real_cls = S3BlobStore

    def factory(**kwargs):
        return real_cls(
            endpoint=kwargs["endpoint"],
            bucket=kwargs["bucket"],
            access_key=kwargs["access_key"],
            secret_key=kwargs["secret_key"],
            region=kwargs["region"],
            client=client,
        )

    monkeypatch.setattr("mindmemos.infra.blob.s3.S3BlobStore", factory)
    return client


async def _seed(client: FakeMinio) -> None:
    store = S3BlobStore(
        endpoint="http://localhost:9100", bucket="blobs", access_key="a", secret_key="s", client=client
    )
    for namespace, payload in (("proj-1", b"a"), ("proj-1", b"b"), ("proj-2", b"a")):
        await store.put(namespace, payload, mime_type="image/png")


def _objects_of(client: FakeMinio, namespace: str) -> list[str]:
    prefix = f"{namespace_prefix(namespace)}/"
    return [name for _bucket, name in client.objects if name.startswith(prefix)]


def _config() -> ResetConfig:
    return ResetConfig(s3_blobs=S3BlobSettings(endpoint="http://localhost:9100", bucket="blobs"))


@pytest.mark.asyncio
async def test_reset_deletes_only_the_project_namespace(fake_minio) -> None:
    await _seed(fake_minio)

    counts = await _reset_blobs(_config(), "proj-1")

    assert counts == {"s3:images": 2}
    assert _objects_of(fake_minio, "proj-1") == []
    assert len(_objects_of(fake_minio, "proj-2")) == 1  # other projects keep their images
    assert await _reset_blobs(_config(), "proj-1") == {"s3:images": 0}  # idempotent


@pytest.mark.asyncio
async def test_reset_without_s3_settings_skips_blob_cleanup(monkeypatch) -> None:
    async def explode(*args, **kwargs):
        raise AssertionError("no store may be built when s3 settings are absent")

    monkeypatch.setattr("mindmemos.infra.blob.s3.S3BlobStore", explode)
    assert await _reset_blobs(ResetConfig(s3_blobs=None), "proj-1") == {}


@pytest.mark.asyncio
async def test_unreachable_s3_is_a_project_reset_error(monkeypatch) -> None:
    def explode(**kwargs):
        raise ConnectionError("minio down")

    monkeypatch.setattr("mindmemos.infra.blob.s3.S3BlobStore", explode)
    with pytest.raises(ProjectResetError) as excinfo:
        await _reset_blobs(_config(), "proj-1")
    assert excinfo.value.store == "s3"


def test_s3_settings_come_from_the_server_config(tmp_path, monkeypatch) -> None:
    for name in ("MINDMEMOS_S3_ENDPOINT", "MINDMEMOS_S3_BUCKET", "MINDMEMOS_S3_ACCESS_KEY",
                 "MINDMEMOS_S3_SECRET_KEY", "MINDMEMOS_S3_REGION"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "dev.yaml"
    config.write_text(
        "blob_store:\n"
        "  s3_endpoint: http://minio.internal:9100\n"
        "  s3_bucket: eval-blobs\n"
        "  s3_access_key: eval-access\n"
        "  s3_secret_key: eval-secret\n"
        "  s3_region: eu-west-1\n",
        encoding="utf-8",
    )
    settings = resolve_s3_blob_settings(config)
    assert (settings.endpoint, settings.bucket) == ("http://minio.internal:9100", "eval-blobs")
    assert (settings.access_key, settings.secret_key, settings.region) == ("eval-access", "eval-secret", "eu-west-1")

    # unreadable or blob-less configs fall back to the env-var dev defaults
    assert resolve_s3_blob_settings(tmp_path / "missing.yaml") == S3BlobSettings()
    empty = tmp_path / "empty.yaml"
    empty.write_text("auth: {}\n", encoding="utf-8")
    assert resolve_s3_blob_settings(empty) == S3BlobSettings()


def test_s3_env_vars_override_the_server_config_like_the_server_does(tmp_path, monkeypatch) -> None:
    """Server precedence is env over yaml (config/app.py _merge_blob_env); the reset must match."""

    monkeypatch.setenv("MINDMEMOS_S3_ENDPOINT", "http://env-minio:9100")
    monkeypatch.setenv("MINDMEMOS_S3_BUCKET", "env-blobs")
    config = tmp_path / "dev.yaml"
    config.write_text(
        "blob_store:\n"
        "  s3_endpoint: http://yaml-minio:9100\n"
        "  s3_bucket: yaml-blobs\n"
        "  s3_access_key: yaml-access\n"
        "  s3_secret_key: yaml-secret\n",
        encoding="utf-8",
    )

    settings = resolve_s3_blob_settings(config)

    assert settings.endpoint == "http://env-minio:9100"
    assert settings.bucket == "env-blobs"
    assert settings.access_key == "yaml-access"  # unset env fields keep the yaml value
    assert settings.secret_key == "yaml-secret"
