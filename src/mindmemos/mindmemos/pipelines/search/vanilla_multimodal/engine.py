"""The search engine of ``vanilla``: vanilla text retrieval, plus stored images when multimodal.

A search is multimodal when ``algo_config.multimodal.mode`` is basic or enhanced and the request
carries ``query_images`` or sets ``multimodal=true``. Any other search is exactly vanilla: the
image methods return at once, with no extra embedding, metadata read or image query.

Text candidates always come from ``VanillaSearchEngine`` unchanged and go through the pipeline's
usual final filter (rerank, threshold, ``top_k``). The image channel runs after that, in
``augment_results``:

- basic: the query text is embedded by the multimodal model and matched against the user's stored
  images; matches are appended after the text results (up to ``image_top_k``).
- enhanced: image hits are scored by text similarity (query vs. each image's description and
  visual facts) mixed with image similarity. Appended after the text results: facts recalled
  through standout image matches, the visual facts of the best-matching images most relevant to
  the query (reranked, one result each), then evidence images (session of the best match,
  anchored to the facts, similar) carrying their descriptions.

Each image item references the original image by ``blob_id`` (``GET /v1/memory/images/{blob_id}``).
"""

from __future__ import annotations

import asyncio
import statistics
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import zip_longest
from typing import Any

from ....components.multimodal import ImageLoader, text_only_search
from ....components.multimodal.extraction import IMAGE_BLOB_IDS_KEY
from ....config import MultimodalConfig, get_config
from ....errors import ConfigNotInitializedError, InvalidFilterError
from ....infra.blob import BlobStore, get_blob_store
from ....llm import ImageInput, MmEmbedClient, RerankClient, get_llm_client, get_mm_embed_client, get_rerank_client
from ....logging import get_logger, traced
from ....mappers import parse_search_dsl
from ....typing import (
    FieldCondition,
    ImageRecord,
    ImageSearchHit,
    MemoryDbSearchHit,
    MemoryRequestContext,
    MemorySearchItem,
    SearchFilter,
    SearchPipelineInput,
)
from ...add.vanilla_multimodal import image_point_id
from ...memory_db import ImageMemoryStore
from ...memory_db.image_store import IMAGE_CHANNEL, TEXT_CHANNEL
from ...utils import format_datetime
from ..base import SearchEngineOptions
from ..vanilla import VanillaSearchEngine
from ..vanilla.engine import _to_memory_search_item

logger = get_logger(__name__)
_UNSET = object()
IMAGE_MEMORY_TYPE = "image"
VISUAL_FACT_MEMORY_TYPE = "visual_fact"
IMAGE_FILTER_FIELDS = frozenset({"user_id", "app_id", "session_id", "agent_id", "request_id", "created_at"})
"""Public filter fields that image items carry; a filter on any other field skips the image channel."""

_QUERY_TEXT_VECTORS: ContextVar[dict[str, list[float]] | None] = ContextVar(
    "multimodal_query_text_vectors", default=None
)
"""Text embeddings of the queries of the current multimodal search, reused by the image text channel."""


@dataclass
class ImageHits:
    """Image matches of one query."""

    ranked: list[ImageSearchHit] = field(default_factory=list)
    """Query text vs. images: text and image similarity mixed (enhanced), best first."""
    by_image: list[ImageSearchHit] = field(default_factory=list)
    """Query text vs. images by image similarity only (raw scores)."""
    by_query_image: list[ImageSearchHit] = field(default_factory=list)
    """Query images vs. images (best score per image)."""


class VanillaMultimodalSearchEngine(VanillaSearchEngine):
    """Vanilla hybrid text retrieval, plus stored images for multimodal searches."""

    def __init__(
        self,
        *,
        image_store: ImageMemoryStore | None = None,
        mm_embed_client: MmEmbedClient | None | object = _UNSET,
        multimodal_config: MultimodalConfig | None = None,
        llm_client: Any = None,
        blob_store: BlobStore | None = None,
        rerank_client: RerankClient | None | object = _UNSET,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._explicit_llm_client = llm_client
        self._explicit_rerank_client = rerank_client
        self._blob_store = blob_store
        self._image_store = image_store
        self._explicit_mm_client = mm_embed_client
        self._explicit_multimodal_config = multimodal_config

    def is_multimodal(self, inp: SearchPipelineInput) -> bool:
        """Whether this search uses images (query images, or ``multimodal=true``, with a mode on)."""

        if not (inp.query_images or inp.multimodal):
            return False
        return self._config().mode != "off"

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Text candidates, exactly as vanilla computes them."""

        if not self.is_multimodal(inp):
            return await super().search_candidates(inp, context, options=options)
        if _QUERY_TEXT_VECTORS.get() is None:
            _QUERY_TEXT_VECTORS.set({})
        return await super().search_candidates(text_only_search(inp), context, options=options)

    async def _encode_dense(self, query: str) -> list[float] | None:
        vector = await super()._encode_dense(query)
        holder = _QUERY_TEXT_VECTORS.get()
        if holder is not None and vector is not None:
            holder[query] = vector
        return vector

    async def augment_results(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        memories: list[MemorySearchItem],
    ) -> list[MemorySearchItem]:
        """Append image results to the final text results of a multimodal search."""

        if not self.is_multimodal(inp):
            return memories
        return await self._augment(inp, context, memories)

    @traced("search.vanilla_multimodal.images")
    async def _augment(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        memories: list[MemorySearchItem],
    ) -> list[MemorySearchItem]:
        cfg = self._config()
        if cfg.mode == "enhanced":
            return await self._augment_enhanced(inp, context, memories, cfg)
        client = self._mm_client()
        if not cfg.image_search or client is None or not inp.query.strip():
            return memories
        filters = _image_filter(inp.filters)
        if filters is None:
            logger.info("multimodal_image_channel_skipped", reason="filter_fields_not_on_images")
            return memories
        vector = (await client.embed_texts([inp.query], task="multimodal.search.query"))[0]
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
        cfg: MultimodalConfig,
    ) -> list[MemorySearchItem]:
        """Text results, facts recalled through matching images, visual facts, and the evidence images.

        Only image hits that stand out by image similarity (``_strong``) recall facts; evidence
        images mix images anchored to the facts with the best (text + image) matches.
        """

        text_views = await self.db_reader.get_memories(context, [m.id for m in memories])
        view_by_id = {view.memory_id: view for view in text_views}
        hits = await self.image_hits(inp, context, cfg)
        strong_text = _strong(hits.by_image, cfg.image_hit_margin)
        strong_query = _strong(hits.by_query_image, cfg.image_hit_margin)

        seen = {m.id for m in memories}
        recalled_ids: list[str] = []
        for hit in [*strong_query, *strong_text]:
            for memory_id in hit.record.anchored_memory_ids:
                if memory_id not in seen and len(recalled_ids) < cfg.anchored_fact_limit:
                    seen.add(memory_id)
                    recalled_ids.append(memory_id)
        recalled = await self.db_reader.get_memories(context, recalled_ids) if recalled_ids else []
        # Deleted (archived) memories stay anchored on image records; they are never recalled.
        recalled_by_id = {view.memory_id: view for view in recalled if view.status == "active"}
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
        similar = [(h.record.blob_id, h.record.user_id, h.score, "similar") for h in hits.ranked]
        # Session expansion: the best matching image usually identifies an episode (a route, a game,
        # a dashboard series); its sibling images in that session, in order, carry the rest of it.
        # The seed is the best image-similarity match: text similarity of descriptions tends to pick
        # a look-alike episode (another paint test, another dashboard).
        seeds = strong_query[:1] + hits.by_image[:1]
        expansion, positions = await self._session_expansion(context, seeds, cfg)
        ordered = [(h.record.blob_id, h.record.user_id, h.score, "query_image") for h in strong_query]
        ordered += expansion
        for pair in zip_longest(anchored, similar):
            ordered.extend(entry for entry in pair if entry is not None)
        chosen: dict[str, tuple[str | None, float | None, str]] = {}
        for blob_id, user_id, score, evidence in ordered:
            if blob_id not in chosen and len(chosen) < cfg.evidence_images:
                chosen[blob_id] = (user_id, score, evidence)
        with_facts = cfg.visual_fact_limit <= 0
        image_items, evidence_records = await self._evidence_items(
            context, chosen, [*hits.ranked, *hits.by_query_image], with_facts=with_facts
        )
        for item in image_items:
            position = positions.get(item.metadata.get("blob_id"))
            if position is not None:
                item.metadata["session_position"] = position
                item.memory = f"[image {position}] {item.memory.removeprefix('[image] ')}"
        pool = [*evidence_records, *(hit.record for hit in hits.ranked[: cfg.visual_fact_images])]
        fact_items = await self._visual_fact_items(inp, pool, cfg)
        return [*memories, *recalled_items, *fact_items, *image_items]

    async def _visual_fact_items(
        self, inp: SearchPipelineInput, records: list[ImageRecord], cfg: MultimodalConfig
    ) -> list[MemorySearchItem]:
        """The visual facts of ``records`` most relevant to the query, one result each.

        Visual facts are kept out of text memory; this is how a multimodal search still returns
        them as individual facts (ranked by the reranker when the request asks for reranking).
        """

        if cfg.visual_fact_limit <= 0:
            return []
        unique = {record.blob_id: record for record in reversed(records)}
        ordered = [unique[blob_id] for blob_id in dict.fromkeys(record.blob_id for record in records)]
        candidates = [(record, index) for record in ordered for index in range(len(record.visual_facts))]
        if not candidates:
            return []
        documents = [record.visual_facts[index] for record, index in candidates]
        order: list[tuple[int, float | None]] = [(i, None) for i in range(len(candidates))]
        reranker = self._rerank_client() if inp.rerank else None
        if reranker is not None and inp.query.strip():
            try:
                response = await reranker.rerank(inp.query, documents, top_n=cfg.visual_fact_limit)
                order = [
                    (hit.index, hit.relevance_score) for hit in response.results if 0 <= hit.index < len(documents)
                ]
            except Exception:  # noqa: BLE001 - fall back to image rank order
                logger.warning("multimodal_visual_fact_rerank_failed", exc_info=True)
        return [_visual_fact_item(*candidates[i], score) for i, score in order[: cfg.visual_fact_limit]]

    def _rerank_client(self) -> RerankClient | None:
        if self._explicit_rerank_client is not _UNSET:
            return self._explicit_rerank_client  # type: ignore[return-value]
        try:
            return get_rerank_client()
        except Exception:  # noqa: BLE001 - no reranker: image rank order
            return None

    async def _session_expansion(
        self, context: MemoryRequestContext, seeds: list[ImageSearchHit], cfg: MultimodalConfig
    ) -> tuple[list[tuple[str, str | None, float | None, str]], dict[str, str]]:
        """Seed images plus their session siblings in order; returns entries and position labels."""

        entries: list[tuple[str, str | None, float | None, str]] = []
        positions: dict[str, str] = {}
        if cfg.session_expansion_images <= 0:
            return entries, positions
        for seed in seeds:
            session_id = next((occ.session_id for occ in seed.record.occurrences if occ.session_id), None)
            if session_id is None:
                continue
            siblings = await self._images().list_session_images(context, session_id, user_id=seed.record.user_id)
            ids = [image.blob_id for image in siblings]
            if seed.record.blob_id not in ids:
                continue
            pos = ids.index(seed.record.blob_id)
            size = cfg.session_expansion_images
            start = max(0, min(pos - size // 2, len(ids) - size))
            for index in range(start, min(len(ids), start + size)):
                blob_id = ids[index]
                evidence = "similar" if blob_id == seed.record.blob_id else "session"
                score = seed.score if blob_id == seed.record.blob_id else None
                entries.append((blob_id, siblings[index].user_id, score, evidence))
                positions.setdefault(blob_id, f"{index + 1} of {len(ids)} in conversation {session_id}")
        return entries, positions

    async def image_hits(
        self, inp: SearchPipelineInput, context: MemoryRequestContext, cfg: MultimodalConfig
    ) -> ImageHits:
        """Image matches for the query text (mixed and image-only scores) and for the query images."""

        client = self._mm_client()
        filters = _image_filter(inp.filters)
        if client is None or not cfg.image_search or filters is None:
            return ImageHits()
        limit = max(cfg.evidence_images * 2, 8)
        store = self._images()
        hits = ImageHits()
        if inp.query.strip():
            image_vector = (await client.embed_texts([inp.query], task="multimodal.search.query"))[0]
            text_vector = await self._query_text_vector(inp.query) if cfg.image_text_weight > 0 else None
            by_image, by_text = await asyncio.gather(
                store.search_images(context, image_vector, channel=IMAGE_CHANNEL, filters=filters, limit=limit),
                store.search_images(context, text_vector, channel=TEXT_CHANNEL, filters=filters, limit=limit)
                if text_vector is not None
                else _no_hits(),
            )
            hits.by_image = by_image
            hits.ranked = _mix(by_text, by_image, cfg.image_text_weight)
        if inp.query_images:
            loader = ImageLoader(get_config().blob_store, self.blobs())
            loaded = [await loader.load(ref, namespace=context.project_id) for ref in inp.query_images[:4]]
            vectors = await client.embed_images(
                [ImageInput(data=item.data, mime_type=item.mime_type) for item in loaded],
                task="multimodal.search.query_image",
            )
            query_hits: list[ImageSearchHit] = []
            for vector in vectors:
                query_hits.extend(await store.search_images(context, vector, filters=filters, limit=limit))
            hits.by_query_image = _best_per_image(query_hits)
        return hits

    async def _query_text_vector(self, query: str) -> list[float] | None:
        """The query's text embedding: reused from the text search when it ran for this query."""

        cached = (_QUERY_TEXT_VECTORS.get() or {}).get(query)
        if cached is not None:
            return cached
        return await super()._encode_dense(query)

    async def _evidence_items(
        self,
        context: MemoryRequestContext,
        chosen: dict[str, tuple[str | None, float | None, str]],
        hits: list[ImageSearchHit],
        *,
        with_facts: bool = True,
    ) -> tuple[list[MemorySearchItem], list[ImageRecord]]:
        """Result items of the chosen images, and their records."""

        known = {hit.record.blob_id: hit.record for hit in hits}
        missing = {
            image_point_id(context.project_id, user_id, blob_id): blob_id
            for blob_id, (user_id, _, _) in chosen.items()
            if blob_id not in known
        }
        if missing:
            for record in await self._images().get_images(context, list(missing)):
                known[record.blob_id] = record
        items, records = [], []
        for blob_id, (_, score, evidence) in chosen.items():
            record = known.get(blob_id)
            if record is None:
                items.append(_bare_image_item(blob_id, evidence))
                continue
            item = _to_search_item(ImageSearchHit(record=record, score=score or 0.0), with_facts=with_facts)
            item.metadata["evidence"] = evidence
            items.append(item)
            records.append(record)
        return items, records

    def owns_agentic(self, inp: SearchPipelineInput) -> bool:
        """Enhanced multimodal searches replace the generic agentic wrapper with the multimodal agent."""

        return self.is_multimodal(inp) and self._config().mode == "enhanced"

    @traced("search.vanilla_multimodal.agentic")
    async def agentic_refine(
        self, inp: SearchPipelineInput, context: MemoryRequestContext, fast: list[MemorySearchItem]
    ) -> list[MemorySearchItem]:
        from .agentic import MultimodalAgent

        try:
            llm = self._explicit_llm_client or get_llm_client()
            return await MultimodalAgent(self, llm, self._config()).refine(inp, context, fast)
        except Exception:  # noqa: BLE001 - the fast-path result is always a valid answer
            logger.warning("multimodal_agentic_failed", exc_info=True)
            return fast

    def _config(self) -> MultimodalConfig:
        if self._explicit_multimodal_config is not None:
            return self._explicit_multimodal_config
        try:
            return get_config().algo_config.multimodal
        except ConfigNotInitializedError:
            return MultimodalConfig()

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


async def _no_hits() -> list[ImageSearchHit]:
    return []


def _mix(by_text: list[ImageSearchHit], by_image: list[ImageSearchHit], text_weight: float) -> list[ImageSearchHit]:
    """Rank images by ``text_weight`` x text similarity + the rest x image similarity.

    The two channels score on different scales (text-text vs. text-image cosine), so each is first
    normalized over its own hits (distribution-based: mean +/- 3 standard deviations -> 0..1); an
    image missing from a channel scores 0 there. Without text hits this is the image ranking.
    """

    text_norm, image_norm = _normalized(by_text), _normalized(by_image)
    records = {hit.record.image_id: hit.record for hit in [*by_text, *by_image]}
    mixed = [
        ImageSearchHit(
            record=record,
            score=text_weight * text_norm.get(image_id, 0.0) + (1 - text_weight) * image_norm.get(image_id, 0.0),
        )
        for image_id, record in records.items()
    ]
    return sorted(mixed, key=lambda hit: -hit.score)


def _normalized(hits: list[ImageSearchHit]) -> dict[str, float]:
    if not hits:
        return {}
    scores = [hit.score for hit in hits]
    spread = 6 * statistics.pstdev(scores)
    if spread == 0:
        return {hit.record.image_id: 1.0 for hit in hits}
    low = statistics.fmean(scores) - spread / 2
    return {hit.record.image_id: min(1.0, max(0.0, (hit.score - low) / spread)) for hit in hits}


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


def _image_event_time(record: ImageRecord) -> datetime | None:
    first = record.occurrences[0] if record.occurrences else None
    if first and first.event_timestamp_ms:
        return datetime.fromtimestamp(first.event_timestamp_ms / 1000, tz=UTC)
    return record.event_time


def _visual_fact_item(record: ImageRecord, index: int, score: float | None) -> MemorySearchItem:
    event_time = _image_event_time(record)
    return MemorySearchItem(
        id=f"{record.image_id}#{index}",
        memory=record.visual_facts[index],
        memory_type=VISUAL_FACT_MEMORY_TYPE,
        last_update_at=format_datetime(record.updated_at),
        event_time=format_datetime(event_time) if event_time else None,
        source_timestamp=format_datetime(event_time) if event_time else None,
        metadata={
            "modality": VISUAL_FACT_MEMORY_TYPE,
            "score": score,
            "image_id": record.image_id,
            "blob_id": record.blob_id,
            "image_path": f"/v1/memory/images/{record.blob_id}",
        },
        status="active",
    )


def _to_search_item(hit: ImageSearchHit, *, with_facts: bool = True) -> MemorySearchItem:
    record = hit.record
    first = record.occurrences[0] if record.occurrences else None
    caption = next((occ.caption for occ in record.occurrences if occ.caption), None)
    text = caption or record.description or (first.context if first else "") or "image"
    if with_facts and record.visual_facts:
        # What the VLM read from the pixels lives here (enhanced), not in text memory.
        text += "\n" + "\n".join(f"- {fact}" for fact in record.visual_facts)
    event_time = _image_event_time(record)
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
            "visual_facts": record.visual_facts,
            "anchored_memory_ids": record.anchored_memory_ids[:20],
        },
        status="active",
    )
