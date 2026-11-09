"""Repository for the image collection (stored images of multimodal vanilla memory).

Each point has two named dense vectors: ``image`` (multimodal embedding of the pixels) and, once
the image has been described (enhanced mode), ``text`` (text embedding of its description and
visual facts, same model and size as the text memory collection). The collection is created on
demand: the image dimension comes from the multimodal embedding endpoint, which is optional, so it
cannot be bootstrapped with the fixed-dimension text collections.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import models as qmodels

from ..filters import IMAGE_PAYLOAD_INDEX_SCHEMA
from ..models import ImagePoint, QdrantCollectionSpec, QdrantRecord, QdrantSearchRecord
from .base import CollectionRepository

IMAGE_VECTOR_NAME = "image"
TEXT_VECTOR_NAME = "text"
_SCROLL_PAGE = 256


class ImageRepository(CollectionRepository):
    """Typed adapter for ``image_item_v1``."""

    @property
    def collection(self) -> str:
        return self._cfg.image_collection

    @property
    def text_vector_size(self) -> int:
        """Size of the ``text`` vector: the text embedding dimension of the memory collection."""

        return int(self._cfg.vector_size)

    def collection_spec(self, vector_size: int) -> QdrantCollectionSpec:
        return QdrantCollectionSpec(
            name=self.collection,
            vector_size=vector_size,
            dense_vector_name=IMAGE_VECTOR_NAME,
            distance="Cosine",
            enable_dense=True,
            enable_sparse=False,
            payload_indexes=list(IMAGE_PAYLOAD_INDEX_SCHEMA),
            extra_dense_vectors=((TEXT_VECTOR_NAME, self.text_vector_size),),
        )

    async def ensure(self, vector_size: int) -> None:
        """Create the collection with ``vector_size`` image dimensions if it does not exist yet."""

        await self._engine.ensure_collection(self.collection_spec(vector_size))

    async def vector_size(self, vector_name: str = IMAGE_VECTOR_NAME) -> int | None:
        """Dimension of one vector of the existing collection, or None when it (or the vector) does not exist."""

        if not await self._engine.collection_exists(self.collection):
            return None
        return await self._engine.dense_vector_size(self.collection, vector_name)

    async def upsert(self, points: list[ImagePoint]) -> None:
        if not points:
            return
        await self.ensure(len(points[0].vector))
        await self._engine.upsert(
            self.collection,
            [
                qmodels.PointStruct(
                    id=point.image_id,
                    vector=_vectors(point.vector, point.text_vector),
                    payload=self._engine.safe_payload(point.payload),
                )
                for point in points
            ],
        )

    async def set_payload(self, image_id: str, payload: dict, *, text_vector: list[float] | None = None) -> None:
        """Replace payload fields of an existing point, keeping its image vector; optionally set its text vector."""

        if text_vector is None:
            await self._engine.set_payload(self.collection, image_id, payload)
            return
        await self._engine.batch_update(
            self.collection,
            [
                qmodels.SetPayloadOperation(
                    set_payload=qmodels.SetPayload(payload=self._engine.safe_payload(payload), points=[image_id])
                ),
                qmodels.UpdateVectorsOperation(
                    update_vectors=qmodels.UpdateVectors(
                        points=[qmodels.PointVectors(id=image_id, vector={TEXT_VECTOR_NAME: text_vector})]
                    )
                ),
            ],
        )

    async def get_many(self, project_id: str, image_ids: list[str]) -> list[QdrantRecord]:
        if not image_ids or not await self._engine.collection_exists(self.collection):
            return []
        records = await self._engine.retrieve(self.collection, image_ids)
        return [record for record in records if record.payload.get("project_id") == project_id]

    async def list(
        self, project_id: str, *, filter_: qmodels.Filter | None = None, limit: int = 50
    ) -> list[QdrantRecord]:
        if not await self._engine.collection_exists(self.collection):
            return []
        records, _ = await self._engine.scroll(
            self.collection, scroll_filter=self._engine.project_filter(project_id, filter_=filter_), limit=limit
        )
        return records

    async def list_all(self, project_id: str, *, filter_: qmodels.Filter | None = None) -> list[QdrantRecord]:
        """Every point of the project matching ``filter_`` (paged scroll)."""

        if not await self._engine.collection_exists(self.collection):
            return []
        scroll_filter = self._engine.project_filter(project_id, filter_=filter_)
        records: list[QdrantRecord] = []
        offset: Any | None = None
        while True:
            page, offset = await self._engine.scroll(
                self.collection, scroll_filter=scroll_filter, limit=_SCROLL_PAGE, offset=offset
            )
            records.extend(page)
            if offset is None or not page:
                return records

    async def count(self, project_id: str, *, filter_: qmodels.Filter | None = None) -> int:
        if not await self._engine.collection_exists(self.collection):
            return 0
        return await self._engine.count(
            self.collection, count_filter=self._engine.project_filter(project_id, filter_=filter_)
        )

    async def delete_many(self, project_id: str, image_ids: list[str]) -> None:
        """Delete points by id, but only where they belong to ``project_id``.

        The project condition rides along as defense in depth: callers derive the ids from a
        project-scoped scroll, and the filter keeps a future caller (or a reused id) from
        deleting another project's point.
        """

        if image_ids and await self._engine.collection_exists(self.collection):
            await self._engine.delete_by_filter(
                self.collection,
                self._engine.project_filter(
                    project_id, conditions=[qmodels.HasIdCondition(has_id=image_ids)]
                ),
            )

    async def search(
        self,
        project_id: str,
        vector: list[float],
        *,
        using: str = IMAGE_VECTOR_NAME,
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
            using=using,
            query_filter=self._engine.project_filter(project_id, filter_=filter_),
            limit=limit,
            score_threshold=score_threshold,
        )


def _vectors(image_vector: list[float], text_vector: list[float] | None) -> dict[str, list[float]]:
    vectors = {IMAGE_VECTOR_NAME: image_vector}
    if text_vector is not None:
        vectors[TEXT_VECTOR_NAME] = text_vector
    return vectors
