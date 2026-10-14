"""vanilla_mm search engine: vanilla text retrieval plus a text-to-image similarity channel.

Text candidates come from ``VanillaSearchEngine`` unchanged and go through the pipeline's usual
final filter (rerank, threshold, ``top_k``). The image channel runs after that, in
``augment_results``: the query text is embedded by the multimodal model and matched against the
user's stored images. Matching images are appended after the text results (up to
``image_top_k``), so the text results are never displaced. Each image item references the
original image by ``blob_id`` (readable via ``GET /v1/memory/images/{blob_id}``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ....components.multimodal import text_only_search
from ....config import VanillaMmConfig, get_config
from ....errors import ConfigNotInitializedError, InvalidFilterError
from ....llm import MmEmbedClient, get_mm_embed_client
from ....logging import get_logger, traced
from ....mappers import parse_search_dsl
from ....typing import (
    FieldCondition,
    ImageSearchHit,
    MemoryRequestContext,
    MemorySearchItem,
    SearchFilter,
    SearchPipelineInput,
)
from ...memory_db import ImageMemoryStore
from ...utils import format_datetime
from ..base import SearchEngineOptions
from ..vanilla import VanillaSearchEngine

logger = get_logger(__name__)
_UNSET = object()
IMAGE_MEMORY_TYPE = "image"
IMAGE_FILTER_FIELDS = frozenset({"user_id", "app_id", "session_id", "agent_id", "request_id", "created_at"})
"""Public filter fields that image items carry; a filter on any other field skips the image channel."""


class VanillaMmSearchEngine(VanillaSearchEngine):
    """Vanilla hybrid text retrieval, plus images found by text-image similarity."""

    name = "vanilla_mm"

    def __init__(
        self,
        *,
        image_store: ImageMemoryStore | None = None,
        mm_embed_client: MmEmbedClient | None | object = _UNSET,
        vanilla_mm_config: VanillaMmConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._image_store = image_store
        self._explicit_mm_client = mm_embed_client
        self._explicit_vanilla_mm_config = vanilla_mm_config

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Text candidates, exactly as vanilla computes them."""

        return await super().search_candidates(text_only_search(inp), context, options=options)

    @traced("search.vanilla_mm.images")
    async def augment_results(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        memories: list[MemorySearchItem],
    ) -> list[MemorySearchItem]:
        """Append image results to the final text results."""

        cfg = self._config()
        client = self._mm_client()
        if not cfg.image_search or client is None or not inp.query.strip():
            return memories
        filters = _image_filter(inp.filters)
        if filters is None:
            logger.info("vanilla_mm_image_channel_skipped", reason="filter_fields_not_on_images")
            return memories
        vector = (await client.embed_texts([inp.query], task="vanilla_mm.search.query"))[0]
        hits = await self._images().search_images(
            context,
            vector,
            filters=filters,
            limit=cfg.image_top_k,
            score_threshold=cfg.image_score_threshold,
        )
        return [*memories, *(_to_search_item(hit) for hit in hits)]

    def _config(self) -> VanillaMmConfig:
        if self._explicit_vanilla_mm_config is not None:
            return self._explicit_vanilla_mm_config
        try:
            return get_config().algo_config.vanilla_mm
        except ConfigNotInitializedError:
            return VanillaMmConfig()

    def _mm_client(self) -> MmEmbedClient | None:
        if self._explicit_mm_client is not _UNSET:
            return self._explicit_mm_client  # type: ignore[return-value]
        return get_mm_embed_client()

    def _images(self) -> ImageMemoryStore:
        if self._image_store is None:
            self._image_store = ImageMemoryStore()
        return self._image_store


def _image_filter(filters: dict[str, Any] | None) -> SearchFilter | None:
    """The request filter if every field it uses exists on image items, else None."""

    try:
        parsed = parse_search_dsl(filters)
    except InvalidFilterError:
        return None
    return parsed if _fields(parsed) <= IMAGE_FILTER_FIELDS else None


def _fields(node: SearchFilter | FieldCondition) -> set[str]:
    if isinstance(node, FieldCondition):
        return {node.field}
    return {field for child in [*node.must, *node.should, *node.must_not] for field in _fields(child)}


def _to_search_item(hit: ImageSearchHit) -> MemorySearchItem:
    record = hit.record
    first = record.occurrences[0] if record.occurrences else None
    caption = next((occ.caption for occ in record.occurrences if occ.caption), None)
    text = caption or (first.context if first else "") or "image"
    event_time = (
        datetime.fromtimestamp(first.event_timestamp_ms / 1000, tz=UTC)
        if first and first.event_timestamp_ms
        else record.event_time
    )
    return MemorySearchItem(
        id=record.image_id,
        memory=f"[image] {text}",
        memory_type=IMAGE_MEMORY_TYPE,
        last_update_at=format_datetime(record.updated_at),
        event_time=format_datetime(event_time) if event_time else None,
        source_timestamp=format_datetime(event_time) if event_time else None,
        metadata={
            "modality": IMAGE_MEMORY_TYPE,
            "score": hit.score,
            "blob_id": record.blob_id,
            "mime_type": record.mime_type,
            "image_path": f"/v1/memory/images/{record.blob_id}",
            "source_url": record.source_url,
            "session_ids": record.session_ids,
            "caller_image_ids": [occ.caller_image_id for occ in record.occurrences if occ.caller_image_id],
            "occurrences": [occ.model_dump(exclude_none=True) for occ in record.occurrences],
            "embed_model": record.embed_model,
        },
        status="active",
    )
