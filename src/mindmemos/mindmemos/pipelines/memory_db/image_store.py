"""Memory DB boundary for the multimodal image collection."""

from __future__ import annotations

from ...infra.db import DatabaseClients, resolve_database_clients
from ...logging import traced
from ...mappers import search_filter_to_qdrant, to_image_point, to_image_record
from ...typing import FieldCondition, ImageRecord, ImageSearchHit, MemoryRequestContext, SearchFilter


class ImageMemoryStore:
    """Project-scoped image storage and similarity search."""

    def __init__(self, *, clients: DatabaseClients | None = None) -> None:
        self._clients = resolve_database_clients(clients)

    async def collection_vector_size(self) -> int | None:
        """Dimension of the existing image collection, or None when it has not been created."""

        return await self._clients.qdrant.image.vector_size()

    async def ensure_collection(self, vector_size: int) -> None:
        await self._clients.qdrant.image.ensure(vector_size)

    @traced("memory_db.get_images", record_args=False)
    async def get_images(self, ctx: MemoryRequestContext, image_ids: list[str]) -> list[ImageRecord]:
        records = await self._clients.qdrant.image.get_many(ctx.project_id, image_ids)
        return [to_image_record(record.payload) for record in records]

    @traced("memory_db.upsert_images", record_args=False)
    async def upsert_images(self, ctx: MemoryRequestContext, items: list[tuple[ImageRecord, list[float]]]) -> None:
        for record, _ in items:
            if record.project_id != ctx.project_id:
                raise ValueError("image record belongs to another project")
        await self._clients.qdrant.image.upsert([to_image_point(record, vector) for record, vector in items])

    @traced("memory_db.update_images", record_args=False)
    async def update_images(self, ctx: MemoryRequestContext, records: list[ImageRecord]) -> None:
        """Rewrite the payload of already indexed images (their vectors are unchanged)."""

        for record in records:
            if record.project_id != ctx.project_id:
                raise ValueError("image record belongs to another project")
            await self._clients.qdrant.image.set_payload(record.image_id, to_image_point(record, []).payload)

    @traced("memory_db.list_session_images", record_args=False)
    async def list_session_images(
        self, ctx: MemoryRequestContext, session_id: str, *, user_id: str | None, limit: int = 50
    ) -> list[ImageRecord]:
        """Images seen in one session (of one user), oldest first."""

        conditions = [FieldCondition(field="session_id", op="match", value=session_id)]
        if user_id:
            conditions.append(FieldCondition(field="user_id", op="match", value=user_id))
        records = await self._clients.qdrant.image.list(
            ctx.project_id,
            filter_=search_filter_to_qdrant(ctx, SearchFilter(must=conditions), target="image"),
            limit=limit,
        )
        images = [to_image_record(record.payload) for record in records]
        return sorted(images, key=lambda image: _first_time(image, session_id))

    @traced("memory_db.search_images", record_args=False)
    async def search_images(
        self,
        ctx: MemoryRequestContext,
        vector: list[float],
        *,
        filters: SearchFilter | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[ImageSearchHit]:
        hits = await self._clients.qdrant.image.search(
            ctx.project_id,
            vector,
            filter_=search_filter_to_qdrant(ctx, filters, target="image"),
            limit=limit,
            score_threshold=score_threshold,
        )
        return [ImageSearchHit(record=to_image_record(hit.payload), score=hit.score) for hit in hits]


def _first_time(image: ImageRecord, session_id: str) -> int:
    times = [occ.event_timestamp_ms or 0 for occ in image.occurrences if occ.session_id == session_id]
    return min(times) if times else 0
