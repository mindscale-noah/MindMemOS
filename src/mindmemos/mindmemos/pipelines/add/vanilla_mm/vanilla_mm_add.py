"""vanilla_mm add pipeline: vanilla text memory plus stored, embedded images.

A request without images is handed to ``VanillaAddPipeline`` untouched. For a request with
images, the text part (images removed, image-only turns dropped) still goes through vanilla
unchanged; each image is loaded, stored once per project in the blob store (content hash), and,
when ``mm_embed_model_router`` is configured, embedded into the image collection with the
dialogue context it appeared in. Without that router images are only stored.
"""

from __future__ import annotations

import asyncio
import zlib
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ....components.multimodal import AttachedImage, ImageLoader, attached_images, has_images, without_images
from ....components.multimodal.extraction import (
    IMAGE_BLOB_IDS_KEY,
    ExtractionImage,
    MultimodalMemoryExtractor,
    RequestImages,
    request_images,
)
from ....config import VanillaMmConfig, get_config
from ....errors import BadRequestError, ConfigNotInitializedError
from ....infra.blob import BlobStore, StoredBlob, get_blob_store
from ....llm import ImageInput, MmEmbedClient, get_mm_embed_client
from ....logging import get_logger, traced
from ....typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    DialogueMessage,
    ImageOccurrence,
    ImageRecord,
    ImageRef,
    MemoryAddEventItem,
    MemoryRequestContext,
)
from ...memory_db import ImageMemoryStore, suppress_recording_errors
from ...registry import register
from ..vanilla import VanillaAddPipeline

logger = get_logger(__name__)
_UNSET = object()
_LOCK_STRIPE_COUNT = 64
_LOCK_STRIPES: dict[asyncio.AbstractEventLoop, list[asyncio.Lock]] = {}


def _lock_stripes() -> list[asyncio.Lock]:
    loop = asyncio.get_running_loop()
    stripes = _LOCK_STRIPES.get(loop)
    if stripes is None:
        for stale in [other for other in _LOCK_STRIPES if other.is_closed()]:
            del _LOCK_STRIPES[stale]
        stripes = _LOCK_STRIPES[loop] = [asyncio.Lock() for _ in range(_LOCK_STRIPE_COUNT)]
    return stripes


def image_point_id(project_id: str, user_id: str | None, blob_id: str) -> str:
    """Images are deduplicated per project and user by content."""

    return str(uuid5(NAMESPACE_URL, f"mindmemos:image:{project_id}:{user_id or ''}:{blob_id}"))


def _default_vanilla_mm_config() -> VanillaMmConfig:
    try:
        return get_config().algo_config.vanilla_mm
    except ConfigNotInitializedError:
        return VanillaMmConfig()


@register(type="add", name="vanilla_mm_add")
class VanillaMmAddPipeline(VanillaAddPipeline):
    """Vanilla add for text, plus raw-image storage and multimodal image indexing."""

    def __init__(
        self,
        *,
        image_store: ImageMemoryStore | None = None,
        blob_store: BlobStore | None = None,
        image_loader: ImageLoader | None = None,
        mm_embed_client: MmEmbedClient | None | object = _UNSET,
        vanilla_mm_config: VanillaMmConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._image_store = image_store
        self._blob_store = blob_store
        self._image_loader = image_loader
        self._explicit_mm_client = mm_embed_client
        self._explicit_vanilla_mm_config = vanilla_mm_config

    @traced("add.vanilla_mm_add.sync")
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        if not has_images(inp):
            return await super().add_sync(inp, context, add_record_id=add_record_id)

        cfg = self._config()
        image_events, stored = await self._add_images(inp, context)
        text_inp = without_images(inp)
        if not text_inp.messages:
            result = AddPipelineSyncResult(status="ok")
        elif cfg.enhanced:
            # The wrapped extractor sees this request's images and anchors facts to them.
            images = RequestImages(
                by_message=_images_by_text_index(inp, stored),
                detail=cfg.extraction_image_detail,
                max_images=cfg.max_extraction_images,
            )
            with request_images(images):
                result = await super().add_sync(text_inp, context, add_record_id=None)
            await self._link_images(context, result, images)
        else:
            result = await super().add_sync(text_inp, context, add_record_id=None)
        result = result.model_copy(update={"memories": [*result.memories, *image_events]})
        if add_record_id is not None:
            await suppress_recording_errors(
                self.recorder.mark_add_completed(context, add_record_id, result),
                operation="add.vanilla_mm_add.sync",
            )
        return result

    @traced("add.vanilla_mm_add.async")
    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> AddPipelineAsyncResult:
        """Queue the request with images already stored, so queue messages carry blob ids only."""

        if has_images(inp):
            inp = await self._with_stored_blobs(inp, context)
        return await super().add_async(inp, context, add_record_id=add_record_id, record_metadata=record_metadata)

    def _memory_extractor_for(self, llm_client):
        # Inert unless an enhanced request publishes its images (see ``request_images``).
        return MultimodalMemoryExtractor(super()._memory_extractor_for(llm_client), llm_client=llm_client)

    async def _link_images(
        self, context: MemoryRequestContext, result: AddPipelineSyncResult, images: RequestImages
    ) -> None:
        """Store image descriptions and the memories anchored to each image on the image records."""

        memory_ids = [event.memory_id for event in result.memories if event.memory_id]
        anchored: dict[str, list[str]] = {}
        if memory_ids:
            for view in await self.db_reader.get_memories(context, memory_ids):
                for blob_id in (view.metadata or {}).get(IMAGE_BLOB_IDS_KEY) or []:
                    anchored.setdefault(blob_id, []).append(view.memory_id)
        blob_ids = set(anchored) | set(images.descriptions)
        if not blob_ids or self._mm_client() is None:
            return
        store = self._images()
        ids = {image_point_id(context.project_id, context.user_id, blob_id): blob_id for blob_id in blob_ids}
        updates = []
        for record in await store.get_images(context, list(ids)):
            blob_id = ids[record.image_id]
            merged = list(dict.fromkeys([*record.anchored_memory_ids, *anchored.get(blob_id, [])]))[:100]
            updates.append(
                record.model_copy(
                    update={
                        "anchored_memory_ids": merged,
                        "description": images.descriptions.get(blob_id) or record.description,
                    }
                )
            )
        await store.update_images(context, updates)

    async def _add_images(
        self, inp: AddPipelineInput, context: MemoryRequestContext
    ) -> tuple[list[MemoryAddEventItem], list[tuple[StoredBlob, bytes, AttachedImage]]]:
        cfg = self._config()
        images = attached_images(inp, context_chars=cfg.image_context_chars)
        if len(images) > cfg.max_images_per_request:
            raise BadRequestError(
                f"at most {cfg.max_images_per_request} images per add request", code="add.too_many_images"
            )
        stored = await self._store_blobs(images, context)
        client = self._mm_client()
        if client is None:
            logger.info("vanilla_mm_images_stored_unindexed", count=len(stored), request_id=context.request_id)
            return [
                MemoryAddEventItem(operation="add", content=f"[image {blob.blob_id[:12]} stored]", memory_type="image")
                for blob, _, _ in stored
            ], stored
        return await self._index(stored, context, client, cfg), stored

    async def _store_blobs(
        self, images: list[AttachedImage], context: MemoryRequestContext
    ) -> list[tuple[StoredBlob, bytes, AttachedImage]]:
        loader = self._loader()
        blobs = self._blobs()
        loaded = await asyncio.gather(*(loader.load(image.ref, namespace=context.project_id) for image in images))
        stored = await asyncio.gather(
            *(blobs.put(context.project_id, item.data, mime_type=item.mime_type) for item in loaded)
        )
        return [(blob, item.data, image) for blob, item, image in zip(stored, loaded, images, strict=True)]

    async def _index(
        self,
        stored: list[tuple[StoredBlob, bytes, AttachedImage]],
        context: MemoryRequestContext,
        client: MmEmbedClient,
        cfg: VanillaMmConfig,
    ) -> list[MemoryAddEventItem]:
        now = datetime.now(UTC)
        by_id: dict[str, tuple[StoredBlob, bytes, list[AttachedImage]]] = {}
        for blob, data, image in stored:
            image_id = image_point_id(context.project_id, context.user_id, blob.blob_id)
            by_id.setdefault(image_id, (blob, data, []))[2].append(image)

        store = self._images()
        events: list[MemoryAddEventItem] = []
        stripes = _lock_stripes()
        locks = sorted({zlib.crc32(image_id.encode()) % len(stripes) for image_id in by_id})
        # Serialize read-modify-write of the same image across concurrent requests in this process.
        async with _MultiLock([stripes[i] for i in locks]):
            existing = {record.image_id: record for record in await store.get_images(context, list(by_id))}
            new_ids = [image_id for image_id in by_id if image_id not in existing]
            vectors = (
                await client.embed_images(
                    [
                        ImageInput(data=by_id[i][1], mime_type=by_id[i][0].mime_type, url=_source_url(by_id[i][2]))
                        for i in new_ids
                    ],
                    task="vanilla_mm.add.image",
                )
                if new_ids
                else []
            )
            inserts: list[tuple[ImageRecord, list[float]]] = []
            for image_id, vector in zip(new_ids, vectors, strict=True):
                blob, _, occurrences = by_id[image_id]
                record = ImageRecord(
                    image_id=image_id,
                    blob_id=blob.blob_id,
                    mime_type=blob.mime_type,
                    size_bytes=blob.size_bytes,
                    account_id=context.account_id,
                    project_id=context.project_id,
                    api_key_uuid=context.api_key_uuid,
                    user_id=context.user_id,
                    app_id=context.app_id,
                    created_at=now,
                    updated_at=now,
                    event_time=datetime.fromtimestamp(occurrences[0].timestamp_ms / 1000, tz=UTC),
                    embed_model=client.model,
                    source_url=_source_url(occurrences),
                    occurrences=[_occurrence(image, context) for image in occurrences][: cfg.max_occurrences_per_image],
                )
                inserts.append((record, vector))
                events.append(_event(record, occurrences[0], "add"))
            updates: list[ImageRecord] = []
            for image_id, record in existing.items():
                occurrences = by_id[image_id][2]
                merged = [*record.occurrences, *(_occurrence(image, context) for image in occurrences)]
                updates.append(
                    record.model_copy(
                        update={"occurrences": merged[: cfg.max_occurrences_per_image], "updated_at": now}
                    )
                )
                events.append(_event(record, occurrences[0], "reinforcement"))
            await store.upsert_images(context, inserts)
            await store.update_images(context, updates)
        return events

    async def _with_stored_blobs(self, inp: AddPipelineInput, context: MemoryRequestContext) -> AddPipelineInput:
        loader = self._loader()
        blobs = self._blobs()
        messages = []
        for message in inp.messages:
            if isinstance(message, DialogueMessage) and message.images:
                refs = []
                for ref in message.images:
                    loaded = await loader.load(ref, namespace=context.project_id)
                    blob = await blobs.put(context.project_id, loaded.data, mime_type=loaded.mime_type)
                    refs.append(
                        ImageRef(
                            blob_id=blob.blob_id, mime_type=blob.mime_type, image_id=ref.image_id, caption=ref.caption
                        )
                    )
                message = message.model_copy(update={"images": refs})
            messages.append(message)
        return inp.model_copy(update={"messages": messages})

    def _config(self) -> VanillaMmConfig:
        return self._explicit_vanilla_mm_config or _default_vanilla_mm_config()

    def _mm_client(self) -> MmEmbedClient | None:
        if self._explicit_mm_client is not _UNSET:
            return self._explicit_mm_client  # type: ignore[return-value]
        return get_mm_embed_client()

    def _blobs(self) -> BlobStore:
        if self._blob_store is None:
            self._blob_store = get_blob_store()
        return self._blob_store

    def _loader(self) -> ImageLoader:
        if self._image_loader is None:
            self._image_loader = ImageLoader(get_config().blob_store, self._blobs())
        return self._image_loader

    def _images(self) -> ImageMemoryStore:
        if self._image_store is None:
            self._image_store = ImageMemoryStore()
        return self._image_store


class _MultiLock:
    def __init__(self, locks: list[asyncio.Lock]) -> None:
        self._locks = locks

    async def __aenter__(self) -> None:
        for lock in self._locks:
            await lock.acquire()

    async def __aexit__(self, *exc: object) -> None:
        for lock in reversed(self._locks):
            lock.release()


def _source_url(images: list[AttachedImage]) -> str | None:
    return next((image.ref.url for image in images if image.ref.url), None)


def _occurrence(image: AttachedImage, context: MemoryRequestContext) -> ImageOccurrence:
    return ImageOccurrence(
        session_id=context.session_id,
        request_id=context.request_id,
        agent_id=context.agent_id,
        role=image.role,
        message_index=image.message_index,
        event_timestamp_ms=image.timestamp_ms,
        context=image.context,
        caller_image_id=image.ref.image_id,
        caption=image.ref.caption,
    )


def _event(record: ImageRecord, image: AttachedImage, operation: str) -> MemoryAddEventItem:
    label = image.ref.caption or image.context or "image"
    return MemoryAddEventItem(
        operation=operation,  # type: ignore[arg-type]
        content=f"[image] {label[:200]}",
        memory_id=record.image_id,
        memory_type="image",
    )


def _images_by_text_index(
    inp: AddPipelineInput, stored: list[tuple[StoredBlob, bytes, AttachedImage]]
) -> dict[int, list[ExtractionImage]]:
    """Map images to the message indices of the text-only input vanilla receives.

    Image-only turns are dropped from that input, so their images go to the next kept turn
    (the reply that discusses them), or to the previous one when they come last.
    """

    kept = [
        index
        for index, message in enumerate(inp.messages)
        if not (isinstance(message, DialogueMessage) and message.images and not message.content.strip())
    ]
    position = {original: text_index for text_index, original in enumerate(kept)}
    result: dict[int, list[ExtractionImage]] = {}
    for blob, data, image in stored:
        if image.message_index in position:
            target = position[image.message_index]
        else:
            later = [position[i] for i in kept if i > image.message_index]
            earlier = [position[i] for i in kept if i < image.message_index]
            if not later and not earlier:
                continue
            target = later[0] if later else earlier[-1]
        images = result.setdefault(target, [])
        if all(existing.blob_id != blob.blob_id for existing in images):
            images.append(ExtractionImage(blob_id=blob.blob_id, mime_type=blob.mime_type, data=data))
    return result
