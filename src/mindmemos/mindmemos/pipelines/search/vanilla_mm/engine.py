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
from itertools import zip_longest
from typing import Any

from ....components.multimodal import ImageLoader, text_only_search
from ....components.multimodal.extraction import IMAGE_BLOB_IDS_KEY
from ....config import VanillaMmConfig, get_config
from ....errors import ConfigNotInitializedError, InvalidFilterError
from ....infra.blob import BlobStore, get_blob_store
from ....llm import ImageInput, MmEmbedClient, get_llm_client, get_mm_embed_client
from ....logging import get_logger, traced
from ....mappers import parse_search_dsl
from ....typing import (
    FieldCondition,
    ImageSearchHit,
    MemoryDbSearchHit,
    MemoryRequestContext,
    MemorySearchItem,
    SearchFilter,
    SearchPipelineInput,
)
from ...add.vanilla_mm import image_point_id
from ...memory_db import ImageMemoryStore
from ...utils import format_datetime
from ..base import SearchEngineOptions
from ..vanilla import VanillaSearchEngine
from ..vanilla.engine import _to_memory_search_item

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
        llm_client: Any = None,
        blob_store: BlobStore | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._explicit_llm_client = llm_client
        self._blob_store = blob_store
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
        if cfg.enhanced:
            return await self._augment_enhanced(inp, context, memories, cfg, client)
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

    async def _augment_enhanced(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        memories: list[MemorySearchItem],
        cfg: VanillaMmConfig,
        client: MmEmbedClient | None,
    ) -> list[MemorySearchItem]:
        """Text results, facts recalled through matching images, and the evidence images.

        Only image hits that stand out from the rest (``_strong``) recall facts; evidence images
        mix images anchored to the facts with the best similarity matches.
        """

        text_views = await self.db_reader.get_memories(context, [m.id for m in memories])
        view_by_id = {view.memory_id: view for view in text_views}
        text_hits, query_image_hits = await self.image_hits(inp, context, cfg)
        strong_text = _strong(text_hits, cfg.image_hit_margin)
        strong_query = _strong(query_image_hits, cfg.image_hit_margin)

        seen = {m.id for m in memories}
        recalled_ids: list[str] = []
        for hit in [*strong_query, *strong_text]:
            for memory_id in hit.record.anchored_memory_ids:
                if memory_id not in seen and len(recalled_ids) < cfg.anchored_fact_limit:
                    seen.add(memory_id)
                    recalled_ids.append(memory_id)
        recalled = await self.db_reader.get_memories(context, recalled_ids) if recalled_ids else []
        recalled_by_id = {view.memory_id: view for view in recalled}
        recalled_items = [
            _to_memory_search_item(MemoryDbSearchHit(memory_id=i, score=0.0, memory=recalled_by_id[i], source="image"))
            for i in recalled_ids
            if i in recalled_by_id
        ]

        # Evidence images: standout query-image matches, then anchored images of the retrieved facts
        # interleaved with the best text-to-image matches (similarity often identifies the right
        # scene where text facts alone are ambiguous), capped at ``evidence_images``.
        anchored: list[tuple[str, str | None, float | None, str]] = []
        for item in [*memories, *recalled_items]:
            view = view_by_id.get(item.id) or recalled_by_id.get(item.id)
            for blob_id in ((view.metadata or {}).get(IMAGE_BLOB_IDS_KEY) or []) if view else []:
                anchored.append((blob_id, view.user_id, None, "anchored"))
        similar = [(h.record.blob_id, h.record.user_id, h.score, "similar") for h in text_hits]
        ordered = [(h.record.blob_id, h.record.user_id, h.score, "query_image") for h in strong_query]
        for pair in zip_longest(anchored, similar):
            ordered.extend(entry for entry in pair if entry is not None)
        chosen: dict[str, tuple[str | None, float | None, str]] = {}
        for blob_id, user_id, score, evidence in ordered:
            if blob_id not in chosen and len(chosen) < cfg.evidence_images:
                chosen[blob_id] = (user_id, score, evidence)
        image_items = await self._evidence_items(context, chosen, [*text_hits, *query_image_hits], memories)
        return [*memories, *recalled_items, *image_items]

    async def image_hits(
        self, inp: SearchPipelineInput, context: MemoryRequestContext, cfg: VanillaMmConfig
    ) -> tuple[list[ImageSearchHit], list[ImageSearchHit]]:
        """Image hits for the query text and for the query images (best score per image)."""

        client = self._mm_client()
        filters = _image_filter(inp.filters)
        if client is None or not cfg.image_search or filters is None:
            return [], []
        limit = max(cfg.evidence_images * 2, 8)
        text_hits: list[ImageSearchHit] = []
        if inp.query.strip():
            vector = (await client.embed_texts([inp.query], task="vanilla_mm.search.query"))[0]
            text_hits = await self._images().search_images(context, vector, filters=filters, limit=limit)
        query_hits: list[ImageSearchHit] = []
        if inp.query_images:
            loader = ImageLoader(get_config().blob_store, self.blobs())
            loaded = [await loader.load(ref, namespace=context.project_id) for ref in inp.query_images[:4]]
            vectors = await client.embed_images(
                [ImageInput(data=item.data, mime_type=item.mime_type) for item in loaded],
                task="vanilla_mm.search.query_image",
            )
            for vector in vectors:
                query_hits.extend(await self._images().search_images(context, vector, filters=filters, limit=limit))
            query_hits = _best_per_image(query_hits)
        return text_hits, query_hits

    async def _evidence_items(
        self,
        context: MemoryRequestContext,
        chosen: dict[str, tuple[str | None, float | None, str]],
        hits: list[ImageSearchHit],
        memories: list[MemorySearchItem],
    ) -> list[MemorySearchItem]:
        known = {hit.record.blob_id: hit.record for hit in hits}
        missing = {
            image_point_id(context.project_id, user_id, blob_id): blob_id
            for blob_id, (user_id, _, _) in chosen.items()
            if blob_id not in known
        }
        if missing:
            for record in await self._images().get_images(context, list(missing)):
                known[record.blob_id] = record
        items = []
        for blob_id, (_, score, evidence) in chosen.items():
            record = known.get(blob_id)
            if record is None:
                items.append(_bare_image_item(blob_id, evidence))
                continue
            item = _to_search_item(ImageSearchHit(record=record, score=score or 0.0))
            item.metadata["evidence"] = evidence
            items.append(item)
        return items

    def owns_agentic(self) -> bool:
        """Enhanced mode replaces the generic agentic wrapper with the multimodal agent."""

        return self._config().enhanced

    @traced("search.vanilla_mm.agentic")
    async def agentic_refine(
        self, inp: SearchPipelineInput, context: MemoryRequestContext, fast: list[MemorySearchItem]
    ) -> list[MemorySearchItem]:
        from .agentic import MultimodalAgent

        try:
            llm = self._explicit_llm_client or get_llm_client()
            return await MultimodalAgent(self, llm, self._config()).refine(inp, context, fast)
        except Exception:  # noqa: BLE001 - the fast-path result is always a valid answer
            logger.warning("vanilla_mm_agentic_failed", exc_info=True)
            return fast

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

    def blobs(self) -> BlobStore:
        if self._blob_store is None:
            self._blob_store = get_blob_store()
        return self._blob_store

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


def _strong(hits: list[ImageSearchHit], margin: float) -> list[ImageSearchHit]:
    """Hits that stand out: at least ``margin`` above the median similarity of the retrieved hits.

    Near-identical images (screenshots of one game, one dashboard) score alike for any query; none
    of them stands out, so none is used. A single retrieved hit stands out by definition.
    """

    if len(hits) <= 1:
        return list(hits)
    scores = sorted(hit.score for hit in hits)
    median = scores[len(scores) // 2]
    return [hit for hit in hits if hit.score - median >= margin]


def _best_per_image(hits: list[ImageSearchHit]) -> list[ImageSearchHit]:
    best: dict[str, ImageSearchHit] = {}
    for hit in hits:
        if hit.record.image_id not in best or hit.score > best[hit.record.image_id].score:
            best[hit.record.image_id] = hit
    return sorted(best.values(), key=lambda hit: -hit.score)


def _bare_image_item(blob_id: str, evidence: str) -> MemorySearchItem:
    return MemorySearchItem(
        id=blob_id,
        memory="[image]",
        memory_type=IMAGE_MEMORY_TYPE,
        last_update_at="",
        metadata={
            "modality": IMAGE_MEMORY_TYPE,
            "blob_id": blob_id,
            "evidence": evidence,
            "image_path": f"/v1/memory/images/{blob_id}",
        },
        status="active",
    )


def _to_search_item(hit: ImageSearchHit) -> MemorySearchItem:
    record = hit.record
    first = record.occurrences[0] if record.occurrences else None
    caption = next((occ.caption for occ in record.occurrences if occ.caption), None)
    text = caption or record.description or (first.context if first else "") or "image"
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
            "description": record.description,
            "anchored_memory_ids": record.anchored_memory_ids[:20],
        },
        status="active",
    )
