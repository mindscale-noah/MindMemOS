"""Repository for the ``source_item_v1`` collection (single dense vector)."""

from __future__ import annotations

from ..filters import SOURCE_PAYLOAD_INDEX_SCHEMA
from ..models import QdrantRecord, SourcePoint
from .base import CollectionRepository


class SourceRepository(CollectionRepository):
    """Typed adapter for ``source_item_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.source_collection

    async def upsert(self, points: list[SourcePoint]) -> None:
        """Upsert many source points."""

        by_collection: dict[str, list[SourcePoint]] = {}
        for point in points:
            project_id = str(point.payload.get("project_id") or "")
            vector_size = len(point.vector or []) or self._cfg.vector_size
            collection = await self._ensure_project_vector_collection(
                project_id,
                vector_size=vector_size,
                enable_sparse=False,
                payload_indexes=list(SOURCE_PAYLOAD_INDEX_SCHEMA),
            )
            by_collection.setdefault(collection, []).append(point)
        for collection, collection_points in by_collection.items():
            await self._engine.upsert(
                collection,
                [self._dense_point(point.source_id, point.vector, point.payload) for point in collection_points],
            )

    async def get(self, project_id: str, source_id: str, *, with_vectors: bool = False) -> QdrantRecord | None:
        """Retrieve one source by project and id."""

        return await self._get_one_scoped(project_id, source_id, with_vectors=with_vectors)
