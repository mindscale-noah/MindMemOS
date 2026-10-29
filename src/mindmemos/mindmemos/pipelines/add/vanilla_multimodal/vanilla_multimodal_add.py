"""Vanilla add with images (``algo_config.multimodal.mode``): the add pipeline bound to ``vanilla``.

Mode ``off``, and any request without images, is handed to ``VanillaAddPipeline`` untouched. For
a request with images, the text part (images removed, image-only turns dropped) still goes through
vanilla; each image is loaded, stored once per project in the blob store (content hash), and,
when ``mm_embed_model_router`` is configured, embedded into the image collection with the dialogue
context it appeared in. Without that router images are only stored.

Mode ``enhanced`` also lets the extractor read the images (``MultimodalMemoryExtractor``): dialogue
facts reach text memory through vanilla, anchored to their images; the descriptions and visual
facts the VLM read from the pixels are stored on the image records with a text vector, so they
never enter text memory (vanilla search, related-memory recall, dreaming and feedback).
"""

from __future__ import annotations

import asyncio
import zlib
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from ....components.multimodal import (
    AttachedImage,
    ImageLoader,
    attached_images,
    has_images,
    is_image_only_turn,
    without_images,
)
from ....components.multimodal.extraction import (
    IMAGE_BLOB_IDS_KEY,
    ExtractionImage,
    MultimodalMemoryExtractor,
    RequestImages,
    request_images,
)
from ....config import MultimodalConfig, get_config
from ....errors import BadRequestError
from ....infra.blob import BlobStore, StoredBlob
from ....llm import ImageInput, MmEmbedClient, get_embed_client
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
from ...multimodal_deps import MM_CLIENT_UNSET, MultimodalDepsMixin
from ...registry import register
from ..vanilla import VanillaAddPipeline
from ..vanilla.vanilla_add import _CLIENT_UNSET

logger = get_logger(__name__)
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


_IMAGE_IO_LIMIT = 8
_IMAGE_IO_SEMAPHORES: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _image_io_semaphore() -> asyncio.Semaphore:
    """Cap the in-flight load+put jobs of image-heavy adds process-wide.

    Every job hops to a thread (URL download, base64 decode, S3 upload of up to
    ``max_image_bytes``) on the default executor that text searches and blob reads share;
    without a cap a few concurrent 32-image adds monopolize it. Per event loop like the lock
    stripes: the server has one loop, test suites a fresh one per test.
    """

    loop = asyncio.get_running_loop()
    semaphore = _IMAGE_IO_SEMAPHORES.get(loop)
    if semaphore is None:
        for stale in [other for other in _IMAGE_IO_SEMAPHORES if other.is_closed()]:
            del _IMAGE_IO_SEMAPHORES[stale]
        semaphore = _IMAGE_IO_SEMAPHORES[loop] = asyncio.Semaphore(_IMAGE_IO_LIMIT)
    return semaphore


def image_point_id(project_id: str, user_id: str | None, blob_id: str) -> str:
    """Images are deduplicated per project and user by content."""

    return str(uuid5(NAMESPACE_URL, f"mindmemos:image:{project_id}:{user_id or ''}:{blob_id}"))


@register(type="add", name="vanilla_multimodal_add")
class VanillaMultimodalAddPipeline(MultimodalDepsMixin, VanillaAddPipeline):
    """Vanilla add for text, plus raw-image storage and image indexing when multimodal mode is on."""

    def __init__(
        self,
        *,
        image_store: ImageMemoryStore | None = None,
        blob_store: BlobStore | None = None,
        image_loader: ImageLoader | None = None,
        mm_embed_client: MmEmbedClient | None | object = MM_CLIENT_UNSET,
        multimodal_config: MultimodalConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._image_store = image_store
        self._blob_store = blob_store
        self._image_loader = image_loader
        self._explicit_mm_client = mm_embed_client
        self._explicit_multimodal_config = multimodal_config

    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        if not has_images(inp):
            return await super().add_sync(inp, context, add_record_id=add_record_id)
        return await self._add_sync_with_images(inp, context, add_record_id=add_record_id)

    @traced("add.vanilla_multimodal_add.sync")
    async def _add_sync_with_images(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None,
    ) -> AddPipelineSyncResult:
        cfg = self.multimodal_config()
        _require_multimodal(cfg)
        image_events, stored = await self._add_images(inp, context)
        text_inp = without_images(inp)
        images: RequestImages | None = None
        if cfg.mode == "enhanced":
            # The wrapped extractor sees this request's images, anchors dialogue facts to them and
            # collects what each image shows (stored on the image records below).
            images = RequestImages(
                by_message=_images_by_text_index(inp, stored),
                detail=cfg.extraction_image_detail,
                max_images=cfg.max_extraction_images,
                max_facts=cfg.max_visual_facts_per_image,
                max_concurrent_vlm_calls=cfg.max_concurrent_vlm_calls,
            )
        if not text_inp.messages:
            result = AddPipelineSyncResult(status="ok")
        elif images is not None:
            with request_images(images):
                result = await super().add_sync(text_inp, context, add_record_id=None)
            await self._link_images(context, result, images, cfg)
        else:
            result = await super().add_sync(text_inp, context, add_record_id=None)
        result = result.model_copy(update={"memories": [*result.memories, *image_events]})
        if add_record_id is not None:
            await suppress_recording_errors(
                self.recorder.mark_add_completed(context, add_record_id, result),
                operation="add.vanilla_multimodal_add.sync",
            )
        return result

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
            _require_multimodal(self.multimodal_config())
            inp = await self._with_stored_blobs(inp, context)
        return await super().add_async(inp, context, add_record_id=add_record_id, record_metadata=record_metadata)

    def _memory_extractor_for(self, llm_client):
        # Inert unless an enhanced request with images publishes them (see ``request_images``).
        return MultimodalMemoryExtractor(super()._memory_extractor_for(llm_client), llm_client=llm_client)

    async def _link_images(
        self,
        context: MemoryRequestContext,
        result: AddPipelineSyncResult,
        images: RequestImages,
        cfg: MultimodalConfig,
    ) -> None:
        """Store on each image record its description, visual facts, anchored memories and text vector."""

        memory_ids = [event.memory_id for event in result.memories if event.memory_id]
        anchored: dict[str, list[str]] = {}
        if memory_ids:
            for view in await self.db_reader.get_memories(context, memory_ids):
                for blob_id in (view.metadata or {}).get(IMAGE_BLOB_IDS_KEY) or []:
                    anchored.setdefault(blob_id, []).append(view.memory_id)
        blob_ids = set(anchored) | set(images.visuals)
        if not blob_ids or self.mm_client() is None:
            if images.visuals:
                logger.info("multimodal_visual_facts_dropped", reason="images_not_indexed", count=len(images.visuals))
            return
        store = self.images()
        ids = {image_point_id(context.project_id, context.user_id, blob_id): blob_id for blob_id in blob_ids}
        updates: list[ImageRecord] = []
        for record in await store.get_images(context, list(ids)):
            blob_id = ids[record.image_id]
            merged = list(dict.fromkeys([*record.anchored_memory_ids, *anchored.get(blob_id, [])]))[:100]
            update: dict[str, Any] = {"anchored_memory_ids": merged}
            visuals = images.visuals.get(blob_id)
            if visuals is not None:
                update["description"] = visuals.description or record.description
                facts = list(dict.fromkeys([*visuals.facts, *record.visual_facts]))
                update["visual_facts"] = facts[: cfg.max_visual_facts_per_image]
            updates.append(record.model_copy(update=update))
        text_vectors = await self._text_vectors(
            context, [record for record in updates if record.blob_id in images.visuals], store
        )
        await store.update_images(context, updates, text_vectors=text_vectors)

    async def _text_vectors(
        self, context: MemoryRequestContext, records: list[ImageRecord], store: ImageMemoryStore
    ) -> dict[str, list[float]]:
        """Text-embed what each image shows, for the text channel of image search."""

        texts = {record.image_id: image_index_text(record) for record in records}
        texts = {image_id: text for image_id, text in texts.items() if text}
        if not texts:
            return {}
        try:
            embed_client = self._text_embed_client()
            response = await embed_client.embed(task="multimodal.add.image_text", text=list(texts.values()))
        except Exception:  # noqa: BLE001 - the image stays searchable by its image vector
            logger.warning("multimodal_image_text_embed_failed", request_id=context.request_id, exc_info=True)
            return {}
        vectors = dict(zip(texts, response.embeddings, strict=False))
        size = store.text_vector_size
        if any(len(vector) != size for vector in vectors.values()):
            # e.g. a project bound to another embedding provider: keep the image vector only.
            logger.warning("multimodal_image_text_vector_size_mismatch", expected=size)
            return {}
        return vectors

    def _text_embed_client(self):
        if self._explicit_embed_client is not _CLIENT_UNSET:
            return self._explicit_embed_client
        return get_embed_client()

    async def _add_images(
        self, inp: AddPipelineInput, context: MemoryRequestContext
    ) -> tuple[list[MemoryAddEventItem], list[tuple[StoredBlob, bytes, AttachedImage]]]:
        cfg = self.multimodal_config()
        images = attached_images(inp, context_chars=cfg.image_context_chars)
        if len(images) > cfg.max_images_per_request:
            raise BadRequestError(
                f"at most {cfg.max_images_per_request} images per add request", code="add.too_many_images"
            )
        stored = await self._store_blobs(images, context)
        client = self.mm_client()
        if client is None:
            logger.info("multimodal_images_stored_unindexed", count=len(stored), request_id=context.request_id)
            return [
                MemoryAddEventItem(operation="add", content=f"[image {blob.blob_id[:12]} stored]", memory_type="image")
                for blob, _, _ in stored
            ], stored
        return await self._index(stored, context, client, cfg), stored

    async def _store_blobs(
        self, images: list[AttachedImage], context: MemoryRequestContext
    ) -> list[tuple[StoredBlob, bytes, AttachedImage]]:
        loader = self._loader()
        blobs = self.blobs()
        semaphore = _image_io_semaphore()

        async def store(image: AttachedImage) -> tuple[StoredBlob, bytes, AttachedImage]:
            async with semaphore:
                item = await loader.load(image.ref, namespace=context.project_id)
                blob = await blobs.put(context.project_id, item.data, mime_type=item.mime_type)
            return blob, item.data, image

        return list(await asyncio.gather(*(store(image) for image in images)))

    async def _index(
        self,
        stored: list[tuple[StoredBlob, bytes, AttachedImage]],
        context: MemoryRequestContext,
        client: MmEmbedClient,
        cfg: MultimodalConfig,
    ) -> list[MemoryAddEventItem]:
        now = datetime.now(UTC)
        by_id: dict[str, tuple[StoredBlob, bytes, list[AttachedImage]]] = {}
        for blob, data, image in stored:
            image_id = image_point_id(context.project_id, context.user_id, blob.blob_id)
            by_id.setdefault(image_id, (blob, data, []))[2].append(image)

        store = self.images()
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
                    task="multimodal.add.image",
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
                    update_at=now,
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
                    record.model_copy(update={"occurrences": merged[: cfg.max_occurrences_per_image], "update_at": now})
                )
                events.append(_event(record, occurrences[0], "reinforcement"))
            await store.upsert_images(context, inserts)
            await store.update_images(context, updates)
        return events

    async def _with_stored_blobs(self, inp: AddPipelineInput, context: MemoryRequestContext) -> AddPipelineInput:
        loader = self._loader()
        blobs = self.blobs()
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

    def _loader(self) -> ImageLoader:
        if self._image_loader is None:
            self._image_loader = ImageLoader(get_config().blob_store, self.blobs())
        return self._image_loader


class _MultiLock:
    def __init__(self, locks: list[asyncio.Lock]) -> None:
        self._locks = locks

    async def __aenter__(self) -> None:
        acquired: list[asyncio.Lock] = []
        try:
            for lock in self._locks:
                await lock.acquire()
                acquired.append(lock)
        except BaseException:
            # A task cancelled while waiting for a later stripe must not strand the earlier
            # ones: `async with` never calls __aexit__ when __aenter__ raises, and a leaked
            # stripe lock would hang every future add hashed to it for the loop's lifetime.
            for lock in reversed(acquired):
                lock.release()
            raise

    async def __aexit__(self, *exc: object) -> None:
        for lock in reversed(self._locks):
            lock.release()


def _require_multimodal(cfg: MultimodalConfig) -> None:
    # The API rejects these first; this guards direct pipeline calls and queued requests.
    if cfg.mode == "off":
        raise BadRequestError(
            "images require algo_config.multimodal.mode basic or enhanced", code="add.images_unsupported"
        )


def image_index_text(record: ImageRecord) -> str:
    """The text an image is found by in the text channel: its description, visual facts and dialogue."""

    context = next((occ.context for occ in record.occurrences if occ.context), "")
    parts = [record.description or "", *record.visual_facts, context]
    return "\n".join(part.strip() for part in parts if part and part.strip())


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

    kept = [index for index, message in enumerate(inp.messages) if not is_image_only_turn(message)]
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
