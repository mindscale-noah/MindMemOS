"""Memory DB boundary for the multimodal image collection."""

from __future__ import annotations

from collections.abc import Mapping

from ...infra.blob import BlobStore
from ...infra.db import DatabaseClients, resolve_database_clients
from ...logging import get_logger, traced
from ...mappers import search_filter_to_qdrant, to_image_point, to_image_record
from ...typing import FieldCondition, ImageRecord, ImageSearchHit, MemoryRequestContext, SearchFilter

logger = get_logger(__name__)

IMAGE_CHANNEL = "image"
TEXT_CHANNEL = "text"


class ImageMemoryStore:
    """Project-scoped image storage and similarity search."""

    def __init__(self, *, clients: DatabaseClients | None = None) -> None:
        self._clients = resolve_database_clients(clients)

    async def collection_vector_size(self, channel: str = IMAGE_CHANNEL) -> int | None:
        """Dimension of one vector (``image`` or ``text``) of the existing collection, or None when absent."""

        return await self._clients.qdrant.image.vector_size(channel)

    @property
    def text_vector_size(self) -> int:
        """Dimension the ``text`` vector must have (the text memory embedding dimension)."""

        return self._clients.qdrant.image.text_vector_size

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
    async def update_images(
        self,
        ctx: MemoryRequestContext,
        records: list[ImageRecord],
        *,
        text_vectors: Mapping[str, list[float]] | None = None,
    ) -> None:
        """Rewrite the payload of already indexed images; ``text_vectors`` (by image id) sets their text vector."""

        for record in records:
            if record.project_id != ctx.project_id:
                raise ValueError("image record belongs to another project")
            await self._clients.qdrant.image.set_payload(
                record.image_id,
                to_image_point(record, []).payload,
                text_vector=(text_vectors or {}).get(record.image_id),
            )

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
        channel: str = IMAGE_CHANNEL,
        filters: SearchFilter | None = None,
        limit: int = 5,
        score_threshold: float | None = None,
    ) -> list[ImageSearchHit]:
        """Nearest images by the ``image`` vector (multimodal query) or the ``text`` vector (text query)."""

        hits = await self._clients.qdrant.image.search(
            ctx.project_id,
            vector,
            using=channel,
            filter_=search_filter_to_qdrant(ctx, filters, target="image"),
            limit=limit,
            score_threshold=score_threshold,
        )
        return [ImageSearchHit(record=to_image_record(hit.payload), score=hit.score) for hit in hits]

    @traced("memory_db.blob_owned_by", record_args=False)
    async def blob_owned_by(self, ctx: MemoryRequestContext, blob_id: str, user_id: str) -> bool:
        """Whether this project has an image record of ``blob_id`` owned by ``user_id``."""

        conditions = [
            FieldCondition(field="blob_id", op="match", value=blob_id),
            FieldCondition(field="user_id", op="match", value=user_id),
        ]
        return (
            await self._clients.qdrant.image.count(
                ctx.project_id, filter_=search_filter_to_qdrant(ctx, SearchFilter(must=conditions), target="image")
            )
            > 0
        )

    @traced("memory_db.purge_images", record_args=False)
    async def purge_images(self, ctx: MemoryRequestContext, blobs: BlobStore, *, user_id: str | None = None) -> int:
        """Delete the image records of the request project (or of one user in it) and their unused blobs.

        Blobs are shared by the users of a project (content addressed), so a user purge deletes a
        blob only when no other image record of the project still uses it; a project purge drops the
        whole blob namespace, including images that were stored but never indexed. Returns the
        number of image records deleted.
        """

        repo = self._clients.qdrant.image
        user_filter = None
        if user_id is not None:
            user_filter = search_filter_to_qdrant(ctx, _match("user_id", user_id), target="image")
        records = await repo.list_all(ctx.project_id, filter_=user_filter)
        await repo.delete_many(ctx.project_id, [record.point_id for record in records])
        if user_id is None:
            await blobs.delete_namespace(ctx.project_id)
        else:
            for blob_id in sorted({str(record.payload.get("blob_id")) for record in records}):
                still_used = await repo.count(
                    ctx.project_id, filter_=search_filter_to_qdrant(ctx, _match("blob_id", blob_id), target="image")
                )
                if not still_used:
                    await blobs.delete(ctx.project_id, blob_id)
        logger.info("images_purged", project_id=ctx.project_id, user_scoped=user_id is not None, records=len(records))
        return len(records)


def _match(field: str, value: str) -> SearchFilter:
    return SearchFilter(must=[FieldCondition(field=field, op="match", value=value)])


def _first_time(image: ImageRecord, session_id: str) -> int:
    times = [occ.event_timestamp_ms or 0 for occ in image.occurrences if occ.session_id == session_id]
    return min(times) if times else 0
