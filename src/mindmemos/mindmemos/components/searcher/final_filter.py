"""Final search result filtering shared by all search pipelines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Awaitable

from ...llm import RerankClient
from ...logging import get_logger
from ...typing import MemorySearchItem
from .rerank import rerank as rerank_documents
from .rerank import rerank_with_scores as rerank_documents_with_scores

logger = get_logger(__name__)
RerankClientFactory = Callable[[], RerankClient | None]


@dataclass(frozen=True, slots=True)
class FinalFilterOutcome:
    """Filter output plus per-candidate rerank scores for retention scoring."""

    candidates: list[MemorySearchItem]
    rerank_scores: list[float | None] = field(default_factory=list)
    rerank_outcome: str = "disabled"


class SearchFinalFilter:
    """Apply final rerank and top-k truncation to search candidates."""

    def __init__(
        self,
        rerank_client: RerankClient | None = None,
        rerank_client_factory: RerankClientFactory | None = None,
        rerank_fn: Callable[[RerankClient, str, list[str], int], Awaitable[list[int]]] = rerank_documents,
        rerank_with_scores_fn: Callable[
            [RerankClient, str, list[str], int], Awaitable[list[tuple[int, float]]]
        ] = rerank_documents_with_scores,
    ) -> None:
        self._rerank_client = rerank_client
        self._rerank_client_factory = rerank_client_factory
        self._rerank_fn = rerank_fn
        self._rerank_with_scores_fn = rerank_with_scores_fn

    async def apply(
        self,
        *,
        query: str,
        candidates: list[MemorySearchItem],
        top_k: int | None,
        rerank: bool,
        score_threshold: float | None = None,
    ) -> list[MemorySearchItem]:
        """Return final search results after optional rerank and truncation."""

        if not candidates:
            return []

        result = candidates
        if rerank:
            rerank_client = self._ensure_rerank_client()
            if (
                rerank_client is None
                or not rerank_client.available
                or not getattr(rerank_client, "has_external_model", True)
            ):
                logger.debug("search_final_rerank_unavailable")
                return _truncate(candidates, top_k)
            documents = [item.memory for item in candidates]
            limit = len(candidates) if top_k is None else min(top_k, len(candidates))
            try:
                if score_threshold is not None:
                    scored = await self._rerank_with_scores_fn(rerank_client, query, documents, limit)
                    scored = [(idx, score) for idx, score in scored if score >= score_threshold]
                    indices = [idx for idx, _ in scored]
                else:
                    indices = await self._rerank_fn(rerank_client, query, documents, limit)
            except Exception:
                logger.warning("search_final_rerank_failed", exc_info=True)
                return _truncate(candidates, top_k)
            result = [candidates[i] for i in indices if 0 <= i < len(candidates)]

        return _truncate(result, top_k)

    async def apply_with_outcome(
        self,
        *,
        query: str,
        candidates: list[MemorySearchItem],
        top_k: int | None,
        rerank: bool,
        score_threshold: float | None = None,
        truncate: bool = True,
    ) -> FinalFilterOutcome:
        """Filter like :meth:`apply` while also exposing rerank scores.

        Used by the token-budget retention path, which needs per-candidate
        relevance scores and defers truncation until after retention packing.
        Always routes through ``rerank_with_scores_fn`` so scores are available
        even when no ``score_threshold`` is set.
        """

        if not candidates:
            return FinalFilterOutcome(candidates=[], rerank_scores=[], rerank_outcome="disabled")

        result = candidates
        scores: list[float | None] = [None] * len(candidates)
        outcome = "disabled"
        if rerank:
            rerank_client = self._ensure_rerank_client()
            if (
                rerank_client is None
                or not rerank_client.available
                or not getattr(rerank_client, "has_external_model", True)
            ):
                logger.debug("search_final_rerank_unavailable")
                outcome = "skipped_unavailable"
            else:
                documents = [item.memory for item in candidates]
                limit = len(candidates) if top_k is None else min(top_k, len(candidates))
                try:
                    scored = await self._rerank_with_scores_fn(rerank_client, query, documents, limit)
                except Exception:
                    logger.warning("search_final_rerank_failed", exc_info=True)
                    outcome = "failed"
                else:
                    outcome = "applied"
                    if score_threshold is not None:
                        scored = [(idx, score) for idx, score in scored if score >= score_threshold]
                    pairs = [(candidates[idx], score) for idx, score in scored if 0 <= idx < len(candidates)]
                    result = [item for item, _ in pairs]
                    scores = [score for _, score in pairs]
        if not truncate:
            return FinalFilterOutcome(candidates=result, rerank_scores=scores, rerank_outcome=outcome)
        kept = _truncate(result, top_k)
        return FinalFilterOutcome(
            candidates=kept,
            rerank_scores=scores[: len(kept)],
            rerank_outcome=outcome,
        )

    def _ensure_rerank_client(self) -> RerankClient | None:
        # Never cache the factory result: this filter is held by a process-wide
        # singleton (SearchPipelineImpl._final_filter), and rerank clients are
        # project-scoped (resolved per request via get_config()). Caching would pin
        # the first project's client for all subsequent projects. An explicitly
        # injected client (tests) still wins and is returned as-is.
        if self._rerank_client is not None:
            return self._rerank_client
        if self._rerank_client_factory is not None:
            return self._rerank_client_factory()
        return None


def _truncate(candidates: list[MemorySearchItem], top_k: int | None) -> list[MemorySearchItem]:
    if top_k is None:
        return candidates
    return candidates[:top_k]
