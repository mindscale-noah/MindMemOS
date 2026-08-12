"""Feedback-driven self-evolution search pipeline (``feedback_evo`` mode)."""

from __future__ import annotations

from ....components.feedback_evo import ensure_evolution_state
from ....components.searcher import SearchFinalFilter
from ....llm import RerankClient
from ....config import bind_config_overrides
from ....infra.db import EvolutionStateStore
from ....typing import MemoryRequestContext, SearchPipelineInput, SearchPipelineResult
from ...base import MemoryDbPipelineMixin
from ...registry import register
from ..agentic.wrapper import AgenticSearchWrapper
from .engine import FeedbackEvoSearchEngine


@register(type="search", name="feedback_evo_search")
class FeedbackEvoSearchPipeline(MemoryDbPipelineMixin):
    """Independent feedback_evo search pipeline.

    Owns its orchestration (feedback_evo engine + optional agentic wrap + final
    filter) and reads the evolved ``search_config`` live from the evolution
    state store — same isolation level as vanilla vs schema.
    """

    def __init__(
        self,
        *,
        state_store: EvolutionStateStore | None = None,
        engine: FeedbackEvoSearchEngine | None = None,
        final_filter: SearchFinalFilter | None = None,
        agentic_wrapper: AgenticSearchWrapper | None = None,
        rerank_client: RerankClient | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._state_store = state_store or EvolutionStateStore()
        self._engine = engine
        if final_filter is not None:
            self._final_filter = final_filter
        else:
            self._final_filter = SearchFinalFilter(
                rerank_client=rerank_client,
                rerank_client_factory=None if rerank_client is not None else _optional_rerank_client,
            )
        self._agentic_wrapper = agentic_wrapper

    def _engine_impl(self) -> FeedbackEvoSearchEngine:
        """Lazily create the feedback_evo engine (requires initialized config)."""

        if self._engine is None:
            self._engine = FeedbackEvoSearchEngine(
                db_reader=self.db_reader,
                db_writer=self.db_writer,
            )
        return self._engine

    def _agentic_impl(self) -> AgenticSearchWrapper:
        """Lazily create the agentic wrapper (requires initialized config)."""

        if self._agentic_wrapper is None:
            self._agentic_wrapper = AgenticSearchWrapper()
        return self._agentic_wrapper

    async def search(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
    ) -> SearchPipelineResult:
        """Search with the project's current evolved configuration."""

        state = await ensure_evolution_state(self._state_store, context.project_id)
        cfg = state.search_config
        modified = _apply_input_overrides(inp, cfg)

        engine_overrides = dict(cfg.get("engine") or {})
        weights = cfg.get("weights")
        if weights:
            engine_overrides["tag_weights"] = weights

        if not engine_overrides:
            candidates = await self._candidates(modified, context)
        else:
            project_config = {
                "algo_config": {"search": {"feedback_evo": engine_overrides}}
            }
            with bind_config_overrides(project_config=project_config):
                candidates = await self._candidates(modified, context)

        memories = await self._final_filter.apply(
            query=modified.query,
            candidates=candidates,
            top_k=modified.top_k,
            rerank=modified.rerank,
            score_threshold=modified.score_threshold,
        )
        return SearchPipelineResult(status="ok", memories=memories)

    async def _candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
    ):
        """Retrieve candidates through the feedback_evo engine."""

        if inp.agentic:
            return await self._agentic_impl().run(inp, context, self._engine_impl())
        return await self._engine_impl().search_candidates(inp, context)


def _apply_input_overrides(
    inp: SearchPipelineInput,
    cfg: dict,
) -> SearchPipelineInput:
    """Map evolved search config onto the public search input fields."""

    update: dict = {}
    if cfg.get("top_k") is not None:
        update["top_k"] = int(cfg["top_k"])
    if cfg.get("rerank") is not None:
        update["rerank"] = bool(cfg["rerank"])
    if cfg.get("score_threshold") is not None:
        update["score_threshold"] = float(cfg["score_threshold"])
    strategy = cfg.get("search_strategy")
    if strategy == "agentic":
        update["agentic"] = True
    if not update:
        return inp
    return inp.model_copy(update=update)


def _optional_rerank_client() -> RerankClient | None:
    """Resolve the project-scoped rerank client when configured, else None."""

    try:
        from ....llm import get_rerank_client

        return get_rerank_client()
    except Exception:
        return None
