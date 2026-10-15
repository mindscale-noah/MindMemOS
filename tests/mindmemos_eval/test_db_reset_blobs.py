"""The eval project reset also deletes the project's stored images (blob files)."""

import pytest
from mindmemos.infra.blob import LocalBlobStore
from mindmemos.infra.blob.local import _namespace_dir
from mindmemos_eval.memory.db_reset import (
    DEFAULT_BLOB_ROOT,
    ResetConfig,
    _reset_blobs,
    blob_namespace_dir,
    resolve_blob_root,
)


@pytest.mark.parametrize("namespace", ["proj-1", "eval/memeye run:42", "../evil/..", "x" * 100])
def test_namespace_dir_matches_the_server_blob_store(namespace: str) -> None:
    assert blob_namespace_dir(namespace) == _namespace_dir(namespace)


@pytest.mark.asyncio
async def test_reset_deletes_only_the_project_blobs(tmp_path) -> None:
    store = LocalBlobStore(tmp_path)
    mine = await store.put("proj-1", b"a", mime_type="image/png")
    await store.put("proj-1", b"b", mime_type="image/png")
    other = await store.put("proj-2", b"a", mime_type="image/png")

    counts = _reset_blobs(ResetConfig(blob_root=str(tmp_path)), "proj-1")

    assert counts == {"blob:images": 2}
    assert await store.get("proj-1", mine.blob_id) is None
    assert await store.get("proj-2", other.blob_id) is not None
    assert _reset_blobs(ResetConfig(blob_root=str(tmp_path)), "proj-1") == {"blob:images": 0}
    assert _reset_blobs(ResetConfig(blob_root=None), "proj-2") == {}


def test_blob_root_comes_from_the_server_config(tmp_path) -> None:
    config = tmp_path / "dev.yaml"
    config.write_text("blob_store:\n  backend: local\n  local_root: /data/blobs\n")
    assert resolve_blob_root(config) == "/data/blobs"
    assert resolve_blob_root(None) == DEFAULT_BLOB_ROOT
    assert resolve_blob_root(tmp_path / "missing.yaml") == DEFAULT_BLOB_ROOT
