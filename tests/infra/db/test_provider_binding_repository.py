from __future__ import annotations

import uuid

import pytest

from mindmemos.infra.db.collections.provider_binding import ProviderBindingRepository
from mindmemos.infra.db.models import ProviderBindingPoint


class FakeQdrantEngine:
    def __init__(self) -> None:
        self.upsert_point_ids: list[str] = []
        self.retrieve_point_ids: list[str] = []

    def safe_payload(self, payload: dict) -> dict:
        return payload

    async def upsert(self, collection: str, points: list) -> None:
        del collection
        self.upsert_point_ids = [str(point.id) for point in points]

    async def retrieve(self, collection: str, point_ids: list[str], *, with_vectors: bool = False) -> list:
        del collection, with_vectors
        self.retrieve_point_ids = point_ids
        return []

    def first_project_match(self, records: list, project_id: str):
        del records, project_id
        return None


class FakeQdrantConfig:
    provider_binding_collection = "provider_binding_v1"
    project_collection_namespace_enabled = False
    semantic_vector_name = "semantic"
    bm25_vector_name = "bm25"
    vector_size = 3
    distance = "Cosine"


@pytest.mark.asyncio
async def test_provider_binding_repository_uses_uuid_point_id_for_qdrant() -> None:
    engine = FakeQdrantEngine()
    repository = ProviderBindingRepository(engine, FakeQdrantConfig())
    binding_id = "pb_274f82cbed8463d01e86cec4c7a396ba"

    await repository.upsert(
        [
            ProviderBindingPoint(
                binding_id=binding_id,
                payload={"project_id": "project-1", "binding_id": binding_id},
            )
        ]
    )
    await repository.get("project-1", binding_id)

    upsert_point_id = engine.upsert_point_ids[0]
    retrieve_point_id = engine.retrieve_point_ids[0]
    uuid.UUID(upsert_point_id)
    assert upsert_point_id == retrieve_point_id
    assert upsert_point_id != binding_id
