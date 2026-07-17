"""Schema-aware single-pass search engine."""

from __future__ import annotations

from typing import Any

from ....components.memory_modeling.schema import EntityManager, get_entity_manager
from ....components.searcher.schema import SchemaSearchExpander, SchemaSearchQueryBuilder
from ....components.text import SparseVectorEncoder, TextPreprocessor, detect_prompt_language, get_text_preprocessor
from ....config import get_config
from ....config.algo.search import SearchConfig
from ....llm import EmbedClient, LLMClient, RerankClient, get_embed_client, get_llm_client
from ....mappers import parse_schema_search_filters
from ....prompts import SearchPromptSet, get_search_prompts
from ....typing import (
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemorySearchItem,
    SearchFilter,
    SearchPipelineInput,
)
from ...base import MemoryDbPipelineMixin
from ...utils import format_datetime, format_memory_event_time, format_source_timestamp
from ..base import SearchEngineOptions


class SchemaSearchEngine(MemoryDbPipelineMixin):
    """Run one schema-aware entity/property retrieval pass."""

    name = "schema"

    def __init__(
        self,
        *,
        search_config: SearchConfig | None = None,
        expander: SchemaSearchExpander | None = None,
        query_builder: SchemaSearchQueryBuilder | None = None,
        llm_client: LLMClient | None = None,
        embed_client: EmbedClient | None = None,
        rerank_client: RerankClient | None = None,
        entity_manager: EntityManager | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        prompts: SearchPromptSet | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # This engine is held by a process-wide singleton (SearchPipelineImpl._engines),
        # so it MUST stay project-agnostic: all project-scoped deps (LLM/embed/rerank
        # clients, text preprocessor, sparse encoder, prompts, entity manager, schema
        # search config) are resolved per request from the request-scoped ContextVar
        # config (see get_config()). The explicit injections below are overrides for
        # tests only; production leaves them None.
        self._explicit_search_config = search_config
        self._expander = expander
        self._query_builder = query_builder
        self._explicit_llm = llm_client
        self._explicit_embed = embed_client
        self._explicit_rerank = rerank_client
        self._explicit_entity_manager = entity_manager
        self._explicit_text_preprocessor = text_preprocessor
        self._explicit_sparse_encoder = sparse_encoder
        self._explicit_prompts = prompts

    def _get_search_config(self) -> SearchConfig:
        if self._explicit_search_config is not None:
            return self._explicit_search_config
        return get_config().algo_config.search

    def _get_schema_search_config(self):
        return self._get_search_config().schema_search

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Search schema entities and project them to public memory items."""

        schema_cfg = self._get_schema_search_config()

        detected_lang = detect_prompt_language(
            inp.query,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = self._explicit_prompts or get_search_prompts(detected_lang)

        # Resolve project-scoped deps from the request-scoped config (ContextVar).
        llm = self._explicit_llm or get_llm_client()
        embed = self._explicit_embed or get_embed_client()
        rerank = self._explicit_rerank if self._explicit_rerank is not None else _optional_rerank_client()
        text_preprocessor = self._explicit_text_preprocessor or get_text_preprocessor()
        sparse_encoder = self._explicit_sparse_encoder or SparseVectorEncoder(
            get_config().algo_config.text_processing
        )
        project_em = self._explicit_entity_manager or get_entity_manager(project_id=context.project_id)
        project_entity_schema = project_em.get_all_dicts() if project_em else []

        query_builder = self._query_builder or SchemaSearchQueryBuilder(
            llm=llm,
            prompts=request_prompts,
            entity_schema=project_entity_schema,
            current_time_mode=schema_cfg.current_time_mode,
            min_time_window_days=schema_cfg.min_time_window_days,
        )

        parsed_filters = parse_schema_search_filters(inp.filters, context)
        property_filter = query_builder.all_property_filter(entity_schema=project_entity_schema)
        initial_time_window = None
        if not parsed_filters.has_time_filter:
            initial_time_window = await query_builder.extract_time_from_query(
                inp.query, prompts=request_prompts
            )

        expander = self._expander or SchemaSearchExpander(
            db_reader=self.db_reader,
            embed_client=embed,
            rerank_client=rerank,
            text_preprocessor=text_preprocessor,
            sparse_encoder=sparse_encoder,
            config=schema_cfg,
        )
        entities = await expander.search(
            ctx=parsed_filters.context,
            query=inp.query,
            entity_types=list(property_filter.keys()) or None,
            property_filter=property_filter,
            time_window=initial_time_window,
            search_filter=parsed_filters.memory_filter,
            entity_search_filter=parsed_filters.entity_filter,
            num_hops=options.num_hops if options and options.num_hops is not None else schema_cfg.multi_hop,
            use_reranker=options.use_reranker if options else None,
            top_k=options.recall_top_k if options else None,
            top_n=options.result_top_n if options and options.result_top_n is not None else inp.top_k,
        )
        if not entities:
            return await self._search_memory_fallback(
                inp, parsed_filters.context, parsed_filters.memory_filter, options
            )

        return [
            MemorySearchItem(
                id=entity.entity_id,
                memory=entity.format_entity_prompt(
                    ignore_edge_num=schema_cfg.output_max_edge_num,
                    include_description=False,
                    include_edges=schema_cfg.include_edges,
                ),
                memory_type="fact",
                last_update_at="",
            )
            for entity in entities
        ]

    async def _search_memory_fallback(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        filters: SearchFilter | None,
        options: SearchEngineOptions | None,
    ) -> list[MemorySearchItem]:
        """Fallback to direct memory recall when schema entity recall is empty."""

        text_preprocessor = self._explicit_text_preprocessor or get_text_preprocessor()
        sparse_encoder = self._explicit_sparse_encoder or SparseVectorEncoder(
            get_config().algo_config.text_processing
        )
        preprocessed = text_preprocessor.preprocess_query(inp.query, include_entities=False)
        if not preprocessed.tokens:
            return []

        sparse = sparse_encoder.encode_query(preprocessed.tokens)
        top_k = _memory_fallback_top_k(inp, options)
        query = MemoryDbSearchQuery(
            query=inp.query,
            top_k=top_k or self._get_search_config().default.top_k,
            filters=filters,
            mode="bm25",
            ranking="score",
        )
        result = await self.db_reader.search_sparse(
            context,
            query,
            indices=list(sparse.indices),
            values=list(sparse.values),
        )
        return [_to_memory_search_item(hit) for hit in result.hits]


def _optional_rerank_client() -> RerankClient | None:
    try:
        from ....llm import get_rerank_client

        return get_rerank_client()
    except Exception:
        return None


def _to_memory_search_item(hit: MemoryDbSearchHit) -> MemorySearchItem:
    memory = hit.memory
    return MemorySearchItem(
        id=hit.memory_id,
        memory=memory.content if memory else "",
        memory_type=memory.mem_type if memory else "fact",
        last_update_at=format_datetime((memory.update_at or memory.created_at) if memory else None),
        event_time=format_memory_event_time(memory, fallback_to_source_timestamp=True) if memory else None,
        source_timestamp=format_source_timestamp(memory) if memory else None,
    )


def _memory_fallback_top_k(inp: SearchPipelineInput, options: SearchEngineOptions | None) -> int | None:
    if options and options.recall_top_k is not None:
        return options.recall_top_k
    if options and options.result_top_n is not None:
        return options.result_top_n
    return inp.top_k
