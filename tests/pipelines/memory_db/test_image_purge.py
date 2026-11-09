"""Deleting a user's or a project's memory takes their image records and unused blobs with it."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.infra.db.models import QdrantRecord
from mindmemos.pipelines.memory_db import ImageMemoryStore
from mindmemos.typing import MemoryRequestContext

from tests.blob_fakes import MemoryBlobStore

NOW = datetime(2024, 1, 5, tzinfo=UTC)
CTX = MemoryRequestContext(request_id="r", account_id="a", project_id="p", api_key_uuid="k")


def _matches(filter_, payload) -> bool:
    """Evaluate the ``must`` match conditions of a Qdrant filter (nested filters included)."""

    for condition in filter_.must or []:
        if hasattr(condition, "must"):
            if not _matches(condition, payload):
                return False
        elif payload.get(condition.key) != condition.match.value:
            return False
    return True


class FakeImageRepository:
    def __init__(self, payloads: list[dict]) -> None:
        self.points = {payload["image_id"]: payload for payload in payloads}

    async def list_all(self, project_id, *, filter_=None):
        return [
            QdrantRecord(point_id=point_id, payload=payload)
            for point_id, payload in self.points.items()
            if payload["project_id"] == project_id and (filter_ is None or _matches(filter_, payload))
        ]

    async def delete_many(self, project_id, image_ids):
        for image_id in image_ids:
            payload = self.points.get(image_id)
            if payload is not None and payload["project_id"] == project_id:
                self.points.pop(image_id)

    async def count(self, project_id, *, filter_=None):
        return len(await self.list_all(project_id, filter_=filter_))


def _payload(image_id: str, user_id: str, blob_id: str, project_id: str = "p") -> dict:
    return {"image_id": image_id, "project_id": project_id, "user_id": user_id, "blob_id": blob_id}


async def _setup():
    blobs = MemoryBlobStore()
    shared = await blobs.put("p", b"shared", mime_type="image/png")
    own = await blobs.put("p", b"own", mime_type="image/png")
    unindexed = await blobs.put("p", b"unindexed", mime_type="image/png")
    elsewhere = await blobs.put("q", b"own", mime_type="image/png")
    repo = FakeImageRepository(
        [
            _payload("i1", "alice", shared.blob_id),
            _payload("i2", "alice", own.blob_id),
            _payload("i3", "bob", shared.blob_id),
            _payload("i4", "alice", own.blob_id, project_id="q"),
        ]
    )
    store = ImageMemoryStore(clients=SimpleNamespace(qdrant=SimpleNamespace(image=repo)))
    return store, repo, blobs, SimpleNamespace(shared=shared, own=own, unindexed=unindexed, elsewhere=elsewhere)


@pytest.mark.asyncio
async def test_user_purge_deletes_their_records_and_the_blobs_nobody_else_uses() -> None:
    store, repo, blobs, b = await _setup()

    assert await store.purge_images(CTX, blobs, user_id="alice") == 2

    assert set(repo.points) == {"i3", "i4"}
    assert await blobs.get("p", b.own.blob_id) is None
    assert await blobs.get("p", b.shared.blob_id) is not None  # bob still has it
    assert await blobs.get("q", b.elsewhere.blob_id) is not None  # another project


@pytest.mark.asyncio
async def test_project_purge_deletes_every_record_and_the_whole_blob_namespace() -> None:
    store, repo, blobs, b = await _setup()

    assert await store.purge_images(CTX, blobs) == 3

    assert set(repo.points) == {"i4"}
    for blob in (b.shared, b.own, b.unindexed):
        assert await blobs.get("p", blob.blob_id) is None
    assert await blobs.get("q", b.elsewhere.blob_id) is not None
