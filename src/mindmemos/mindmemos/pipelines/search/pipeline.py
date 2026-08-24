"""HTTP-facing search pipeline."""

from __future__ import annotations

from typing import Any

from ...components.searcher import (
    MemoryConsolidator,
    MemoryRetentionSelector,
    ScoredSearchCandidate,
    SearchFinalFilter,
)
from ...components.text import get_text_preprocessor
from ...config import get_config
from ...config.algo.search import MemoryRetentionConfig
from ...llm import RerankClient
from ...logging import get_logger
from ...typing import MemoryRequestContext, MemorySearchItem, SearchPipelineInput, SearchPipelineResult
from ..base import MemoryDbPipelineMixin
from ..registry import register
from .agentic.wrapper import AgenticSearchWrapper
from .base import SearchEngine
from .default import DefaultSearchEngine
from .schema import SchemaSearchEngine
from .vanilla import VanillaSearchEngine

logger = get_logger(__name__)

_DEFAULT_ENGINE_NAMES = frozenset({"default", "vanilla", "schema"})


@register(type="search", name="search_pipeline")
class SearchPipelineImpl(MemoryDbPipelineMixin):
    """Select a search engine, optionally wrap it in agentic orchestration, then final-filter."""

    def __init__(
        self,
        *,
        engines: dict[str, SearchEngine] | None = None,
        agentic_wrapper: AgenticSearchWrapper | None = None,
        final_filter: SearchFinalFilter | None = None,
        rerank_client: RerankClient | None = None,
        retention_config: MemoryRetentionConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._engines = dict(engines or {})
        self._use_default_engines = engines is None
        self._agentic = agentic_wrapper
        self._retention_config = retention_config
        if final_filter is not None:
            self._final_filter = final_filter
        else:
            self._final_filter = SearchFinalFilter(
                rerank_client=rerank_client,
                rerank_client_factory=None if rerank_client is not None else _optional_rerank_client,
            )

    async def search(self, inp: SearchPipelineInput, context: MemoryRequestContext) -> SearchPipelineResult:
        """Run search according to the request controls."""

        strategy = inp.search_pipeline
        engine = self._engine(strategy)
        if engine is None:
            available = ", ".join(sorted(self._available_engine_names()))
            raise ValueError(f"Unknown search strategy {strategy!r}. Available strategies: {available}")

        if inp.agentic:
            candidates = await self._agentic_wrapper().run(inp, context, engine)
        elif inp.token_budget is not None:
            # Retention packs from a wider pool than top_k; raise engine recall accordingly.
            retention_config = self._resolve_retention_config()
            engine_top_k = max(inp.top_k or 0, retention_config.max_candidates)
            candidates = await engine.search_candidates(inp.model_copy(update={"top_k": engine_top_k}), context)
        else:
            candidates = await engine.search_candidates(inp, context)
        if inp.token_budget is None:
            memories = await self._final_filter.apply(
                query=inp.query,
                candidates=candidates,
                top_k=inp.top_k,
                rerank=inp.rerank and _strategy_allows_rerank(strategy),
                score_threshold=inp.score_threshold,
            )
            return SearchPipelineResult(status="ok", memories=memories)
        memories = await self._search_with_retention(inp, strategy=strategy, candidates=candidates)
        return SearchPipelineResult(status="ok", memories=memories)

    async def _search_with_retention(
        self,
        inp: SearchPipelineInput,
        *,
        strategy: str,
        candidates: list[MemorySearchItem],
    ) -> list[MemorySearchItem]:
        """Final-filter without truncation, then pack candidates under the token budget."""

        retention_config = self._resolve_retention_config()
        bounded = candidates[: retention_config.max_candidates]
        filter_result = await self._final_filter.apply_with_outcome(
            query=inp.query,
            candidates=bounded,
            top_k=inp.top_k,
            rerank=inp.rerank and _strategy_allows_rerank(strategy),
            score_threshold=inp.score_threshold,
            truncate=False,
        )
        scored = [
            ScoredSearchCandidate(
                item=item,
                original_rank=rank,
                rank=rank,
                rerank_score=score,
                relevance_score=score if score is not None else 0.0,
                final_score_source="rerank" if score is not None else "rank_fallback",
            )
            for rank, (item, score) in enumerate(zip(filter_result.candidates, filter_result.rerank_scores))
        ]
        retention_input = scored
        metrics: dict[str, Any] = {"rerank_outcome": filter_result.rerank_outcome}
        if retention_config.consolidation_enabled:
            consolidator = MemoryConsolidator(
                text_preprocessor=get_text_preprocessor(),
                max_memories=retention_config.consolidation_max_memories,
                cluster_threshold=retention_config.consolidation_cluster_threshold,
                near_dup_threshold=retention_config.consolidation_near_dup_threshold,
                stitch_max_members=retention_config.consolidation_stitch_max_members,
                max_chars=retention_config.consolidation_max_chars,
            )
            consol = consolidator.consolidate(scored)
            retention_input = consol.candidates
            metrics.update(
                {
                    "consolidation_enabled": True,
                    "consolidation_input_count": consol.input_count,
                    "consolidation_cluster_count": consol.cluster_count,
                    "consolidation_output_count": consol.output_count,
                }
            )
        selection = MemoryRetentionSelector(config=retention_config).select(
            query=inp.query,
            candidates=retention_input,
            token_budget=inp.token_budget,
        )
        metrics.update(
            {
                "token_budget": inp.token_budget,
                "candidate_count_before_retention": len(retention_input),
                "candidate_count_after_retention": len(selection.candidates),
                "estimated_tokens_before": selection.estimated_tokens_before,
                "estimated_tokens_after": selection.estimated_tokens_after,
                "budget_utilization": selection.estimated_tokens_after / inp.token_budget,
                "budget_induced_empty": selection.budget_induced_empty,
                "retention_selector_version": retention_config.selector_version,
                "token_estimator_version": retention_config.estimator_version,
            }
        )
        logger.info("search_retention_metrics", **metrics)
        selected = selection.candidates if inp.top_k is None else selection.candidates[: inp.top_k]
        return [candidate.item for candidate in selected]

    def _resolve_retention_config(self) -> MemoryRetentionConfig:
        if self._retention_config is not None:
            return self._retention_config
        return get_config().algo_config.search.retention

    def _engine(self, name: str) -> SearchEngine | None:
        engine = self._engines.get(name)
        if engine is not None or not self._use_default_engines:
            return engine
        if name not in _DEFAULT_ENGINE_NAMES:
            return None

        common = {"db_reader": self.db_reader, "db_writer": self.db_writer}
        if name == "default":
            engine = DefaultSearchEngine(**common)
        elif name == "vanilla":
            engine = VanillaSearchEngine(**common)
        else:
            engine = SchemaSearchEngine(**common)
        self._engines[name] = engine
        return engine

    def _agentic_wrapper(self) -> AgenticSearchWrapper:
        if self._agentic is None:
            self._agentic = AgenticSearchWrapper()
        return self._agentic

    def _available_engine_names(self) -> set[str]:
        if self._use_default_engines:
            return set(_DEFAULT_ENGINE_NAMES)
        return set(self._engines)


def _optional_rerank_client() -> RerankClient | None:
    try:
        from ...llm import get_rerank_client

        return get_rerank_client()
    except Exception:
        return None


def _strategy_allows_rerank(strategy: str) -> bool:
    if strategy != "vanilla":
        return True
    return get_config().algo_config.search.vanilla.use_reranker
