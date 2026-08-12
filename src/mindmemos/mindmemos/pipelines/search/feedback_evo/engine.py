"""Independent feedback_evo search engine.

Implements flat-memory hybrid recall (dense + sparse) with entity-tag score
weights and archived-version lineage recall inside the feedback_evo component
namespace — no dependency on the vanilla search engine.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ....components.text import (
    SparseVectorEncoder,
    TextPreprocessor,
    get_text_preprocessor,
)
from ....config import TextProcessingConfig, get_config
from ....llm import EmbedClient, get_embed_client
from ....logging import get_logger
from ....mappers import parse_search_dsl
from ....typing import (
    FieldCondition,
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemorySearchItem,
    MemoryView,
    SearchFilter,
    SearchPipelineInput,
    SparseVector,
)
from ...base import MemoryDbPipelineMixin
from ..base import SearchEngineOptions
from ...utils import format_datetime, format_memory_event_time, format_source_timestamp

logger = get_logger(__name__)


@dataclass(frozen=True)
class _LineageCandidate:
    memory_id: str
    score: float
    memory: MemoryView


class FeedbackEvoSearchEngine(MemoryDbPipelineMixin):
    """Flat-memory hybrid retrieval with tag weights and lineage recall."""

    name = "feedback_evo"

    def __init__(
        self,
        *,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        embed_client: EmbedClient | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        cfg = text_config or get_config().algo_config.text_processing
        self._text_preprocessor = text_preprocessor or get_text_preprocessor(cfg)
        self._sparse_encoder = sparse_encoder or SparseVectorEncoder(cfg)
        self._embed_client = embed_client
        if self._embed_client is None:
            try:
                self._embed_client = get_embed_client()
            except Exception:
                self._embed_client = None

    def _search_config(self):
        return get_config().algo_config.search.feedback_evo

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Search flat memories via hybrid dense+sparse recall."""

        scfg = self._search_config()
        preprocessed = self._text_preprocessor.preprocess_query(
            inp.query,
            include_entities=False,
        )
        if not preprocessed.tokens:
            return []

        dense_vector, sparse_vector = await asyncio.gather(
            self._encode_dense(inp.query),
            self._encode_sparse(preprocessed.tokens),
        )
        filters = _request_filter(inp, context)
        request_top_k = options.result_top_n if options and options.result_top_n is not None else inp.top_k
        recall_size = (
            options.recall_top_k
            if options and options.recall_top_k is not None
            else scfg.recall_size
        )
        if request_top_k is not None:
            recall_size = max(recall_size, request_top_k)

        if dense_vector is not None:
            prefetch_limit = max(
                recall_size * scfg.hybrid_prefetch_factor,
                scfg.hybrid_prefetch_min,
            )
            prefetch_limit = min(prefetch_limit, scfg.hybrid_prefetch_max)
            result = await self.db_reader.search_hybrid(
                context,
                MemoryDbSearchQuery(
                    query=inp.query,
                    top_k=recall_size,
                    filters=filters,
                    mode="rrf",
                    ranking="hybrid",
                ),
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                dense_limit=prefetch_limit,
                sparse_limit=prefetch_limit,
            )
        else:
            result = await self.db_reader.search_sparse(
                context,
                MemoryDbSearchQuery(
                    query=inp.query,
                    top_k=recall_size,
                    filters=filters,
                    mode="bm25",
                    ranking="score",
                ),
                indices=list(sparse_vector.indices),
                values=list(sparse_vector.values),
            )

        ranked = _rank_by_score(result.hits)
        if scfg.tag_weights:
            ranked = _apply_tag_weights(ranked, scfg.tag_weights)
        lineage_candidates = await self._lineage_candidates(ranked, context)

        candidates = [
            _to_memory_search_item(hit)
            for hit in ranked
        ]
        by_id = {item.id: item for item in candidates}
        for cand in lineage_candidates:
            if cand.memory_id in by_id:
                continue
            by_id[cand.memory_id] = _to_memory_search_item(
                MemoryDbSearchHit(
                    memory_id=cand.memory_id,
                    score=cand.score,
                    memory=cand.memory,
                )
            )
        return list(by_id.values())

    async def _lineage_candidates(
        self,
        ranked: list[MemoryDbSearchHit],
        context: MemoryRequestContext,
    ) -> list[_LineageCandidate]:
        """Append archived versions of recalled memories with decayed scores."""

        seed_ids = list(dict.fromkeys(hit.memory_id for hit in ranked if hit.memory_id))
        if not seed_ids:
            return []
        try:
            lineage_by_id = await self.db_reader.get_memory_lineage(context, seed_ids)
            archived_ids = _ordered_archived_ids(lineage_by_id, existing_ids=set(seed_ids))
            if not archived_ids:
                return []
            archived = await self.db_reader.get_memories(context, archived_ids)
            by_id = {m.memory_id: m for m in archived}
            return [
                _LineageCandidate(
                    memory_id=memory_id,
                    score=0.5,
                    memory=by_id[memory_id],
                )
                for memory_id in archived_ids
                if memory_id in by_id
            ]
        except Exception:
            logger.warning("feedback_evo_lineage_recall_failed", exc_info=True)
            return []

    async def _encode_dense(self, query: str) -> list[float] | None:
        """Generate a dense query embedding; None when unavailable."""

        if self._embed_client is None:
            return None
        try:
            resp = await self._embed_client.embed(task="search.query", text=query)
            return resp.embeddings[0] if resp.embeddings else None
        except Exception:
            logger.debug("feedback_evo_dense_unavailable", exc_info=True)
            return None

    async def _encode_sparse(self, tokens: list[str]) -> SparseVector:
        """Generate the sparse BM25 query vector."""

        return self._sparse_encoder.encode_query(tokens)


def _request_filter(inp: SearchPipelineInput, ctx: MemoryRequestContext) -> SearchFilter:
    """Combine user DSL with the always-on active-memory scope."""

    base = parse_search_dsl(inp.filters)
    defaults = [FieldCondition(field="status", op="match", value="active")]
    return SearchFilter(
        must=[*defaults, *base.must],
        should=base.should,
        must_not=base.must_not,
    )


def _rank_by_score(hits: list[MemoryDbSearchHit]) -> list[MemoryDbSearchHit]:
    """Sort hits by score descending, tie-broken by rank then position."""

    return [
        hit
        for _, hit in sorted(
            enumerate(hits),
            key=lambda item: (
                -(item[1].score or 0.0),
                item[1].rank if item[1].rank is not None else item[0],
                item[0],
            ),
        )
    ]


def _apply_tag_weights(
    hits: list[MemoryDbSearchHit],
    weights: dict[str, float],
) -> list[MemoryDbSearchHit]:
    """Multiply scores by entity_type/mem_type weights and re-rank."""

    weighted: list[MemoryDbSearchHit] = []
    for hit in hits:
        memory = hit.memory
        tag = getattr(memory, "entity_type", None) or getattr(memory, "mem_type", None)
        multiplier = weights.get(tag, 1.0) if tag is not None else 1.0
        weighted.append(
            hit.model_copy(update={"score": (hit.score or 0.0) * multiplier})
        )
    return _rank_by_score(weighted)


def _ordered_archived_ids(
    lineage_by_id: dict[str, list[str]],
    *,
    existing_ids: set[str],
) -> list[str]:
    """Flatten archived ancestor ids, skipping seeds and duplicates."""

    seen = set(existing_ids)
    result: list[str] = []
    for derived_from_ids in lineage_by_id.values():
        for memory_id in derived_from_ids:
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)
            result.append(memory_id)
    return result


def _to_memory_search_item(hit: MemoryDbSearchHit) -> MemorySearchItem:
    """Project one DB hit into the public search item shape."""

    memory = hit.memory
    return MemorySearchItem(
        id=hit.memory_id,
        memory=memory.content if memory else "",
        memory_type=memory.mem_type if memory else "fact",
        last_update_at=format_datetime((memory.update_at or memory.created_at) if memory else None),
        event_time=format_memory_event_time(memory, fallback_to_source_timestamp=True) if memory else None,
        source_timestamp=format_source_timestamp(memory) if memory else None,
    )
