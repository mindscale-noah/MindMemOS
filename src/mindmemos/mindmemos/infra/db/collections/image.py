"""Repository for the image collection (multimodal vectors of stored images).

The collection is created on demand: its dimension comes from the multimodal embedding endpoint,
which is optional, so it cannot be bootstrapped with the fixed-dimension text collections.
"""

from __future__ import annotations

from qdrant_client import models as qmodels

from ..filters import IMAGE_PAYLOAD_INDEX_SCHEMA
from ..models import ImagePoint, QdrantCollectionSpec, QdrantRecord, QdrantSearchRecord
from .base import CollectionRepository

IMAGE_VECTOR_NAME = "image"


class ImageRepository(CollectionRepository):
    """Typed adapter for ``image_item_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.image_collection

    def collection_spec(self, vector_size: int) -> QdrantCollectionSpec:
        return QdrantCollectionSpec(
            name=self.collection,
            vector_size=vector_size,
            dense_vector_name=IMAGE_VECTOR_NAME,
            distance="Cosine",
            enable_dense=True,
            enable_sparse=False,
            payload_indexes=list(IMAGE_PAYLOAD_INDEX_SCHEMA),
        )

    async def ensure(self, vector_size: int) -> None:
        """Create the collection with ``vector_size`` dimensions if it does not exist yet."""

        await self._engine.ensure_collection(self.collection_spec(vector_size))

    async def vector_size(self) -> int | None:
        """Dimension of the existing collection, or None when it does not exist."""

        if not await self._engine.collection_exists(self.collection):
            return None
        return await self._engine.dense_vector_size(self.collection, IMAGE_VECTOR_NAME)

    async def upsert(self, points: list[ImagePoint]) -> None:
        if not points:
            return
        await self.ensure(len(points[0].vector))
        await self._engine.upsert(
            self.collection,
            [
                qmodels.PointStruct(
                    id=point.image_id,
                    vector={IMAGE_VECTOR_NAME: point.vector},
                    payload=self._engine.safe_payload(point.payload),
                )
                for point in points
            ],
        )

    async def set_payload(self, image_id: str, payload: dict) -> None:
        """Replace payload fields of an existing point, keeping its vector."""

        await self._engine.set_payload(self.collection, image_id, payload)

    async def get_many(self, project_id: str, image_ids: list[str]) -> list[QdrantRecord]:
        if not image_ids or not await self._engine.collection_exists(self.collection):
            return []
        records = await self._engine.retrieve(self.collection, image_ids)
        return [record for record in records if record.payload.get("project_id") == project_id]

    async def search(
        self,
        project_id: str,
        vector: list[float],
        *,
        filter_: qmodels.Filter | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[QdrantSearchRecord]:
        if not await self._engine.collection_exists(self.collection):
            return []
        return await self._engine.query(
            self.collection,
            source="image",
            query=vector,
            using=IMAGE_VECTOR_NAME,
            query_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            score_threshold=score_threshold,
        )
