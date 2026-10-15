import hashlib

import pytest
from mindmemos.infra.blob import LocalBlobStore


@pytest.mark.asyncio
async def test_put_is_content_addressed_and_idempotent(tmp_path) -> None:
    store = LocalBlobStore(tmp_path)
    first = await store.put("proj-a", b"\x89PNG data", mime_type="image/png")
    second = await store.put("proj-a", b"\x89PNG data", mime_type="image/png")

    assert first == second
    assert first.blob_id == hashlib.sha256(b"\x89PNG data").hexdigest()
    assert len([p for p in tmp_path.rglob("*") if p.is_file()]) == 2  # bytes + descriptor
    blob, data = await store.get("proj-a", first.blob_id)
    assert data == b"\x89PNG data" and blob.mime_type == "image/png"


@pytest.mark.asyncio
async def test_namespaces_are_isolated_and_ids_are_validated(tmp_path) -> None:
    store = LocalBlobStore(tmp_path)
    blob = await store.put("proj-a", b"x", mime_type="image/png")

    assert await store.get("proj-b", blob.blob_id) is None
    assert await store.get("proj-a", "../../etc/passwd") is None
    assert await store.stat("proj-a", "0" * 63) is None
    weird = await store.put("../evil/..", b"y", mime_type="image/png")
    assert all(tmp_path in p.parents for p in tmp_path.rglob("*"))
    assert (await store.get("../evil/..", weird.blob_id))[1] == b"y"


@pytest.mark.asyncio
async def test_delete_one_blob_and_a_whole_namespace(tmp_path) -> None:
    store = LocalBlobStore(tmp_path)
    a = await store.put("proj-a", b"a", mime_type="image/png")
    b = await store.put("proj-a", b"b", mime_type="image/png")
    other = await store.put("proj-b", b"a", mime_type="image/png")

    assert await store.delete("proj-a", a.blob_id) is True
    assert await store.delete("proj-a", a.blob_id) is False
    assert await store.delete("proj-a", "../../etc/passwd") is False
    assert await store.get("proj-a", a.blob_id) is None and await store.get("proj-a", b.blob_id) is not None

    await store.delete_namespace("proj-a")
    await store.delete_namespace("proj-a")  # idempotent
    assert await store.get("proj-a", b.blob_id) is None
    assert await store.get("proj-b", other.blob_id) is not None  # other projects keep theirs
