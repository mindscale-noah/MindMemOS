import asyncio
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.searcher.final_filter import SearchFinalFilter
from mindmemos.config import (
    TextProcessingConfig,
    VanillaSearchConfig,
    bind_config_overrides,
    init_config,
    reset_config,
)
from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
from mindmemos.pipelines.search.vanilla import VanillaSearchEngine
from mindmemos.pipelines.search.vanilla import engine as vanilla_engine_module
from mindmemos.typing.llm import EmbeddingResponse
from mindmemos.typing.memory import (
    FieldCondition,
    GraphNeighborScope,
    MemoryRequestContext,
    MemoryView,
    SearchFilter,
)
from mindmemos.typing.memory_db import MemoryDbSearchHit, MemoryDbSearchResult
from mindmemos.typing.service import SearchPipelineInput


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def fake_search_config():
    return SimpleNamespace(
        algo_config=SimpleNamespace(
            text_processing=TextProcessingConfig(
                bm25_use_spacy_lemma=False,
                spacy_en_model="missing_en_model",
                spacy_zh_model="missing_zh_model",
                sparse_hash_dim=128,
            ),
            search=SimpleNamespace(vanilla=VanillaSearchConfig(recall_size=4, use_reranker=False)),
        )
    )


def memory(
    memory_id: str,
    content: str,
    *,
    mem_type: str = "fact",
    status: str = "active",
    user_id: str | None = None,
    app_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    validate_from: datetime | None = None,
    metadata: dict | None = None,
    parent_ids: list[str] | None = None,
    created_at: datetime | None = None,
    update_at: datetime | None = None,
) -> MemoryView:
    return MemoryView(
        memory_id=memory_id,
        project_id="proj-1",
        content=content,
        mem_type=mem_type,
        status=status,
        metadata=metadata or {},
        user_id=user_id,
        app_id=app_id,
        session_id=session_id,
        agent_id=agent_id,
        parent_ids=parent_ids or [],
        validate_from=validate_from,
        created_at=created_at or datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        update_at=update_at,
    )


class FakeEmbedClient:
    async def embed(self, *, task: str, text):
        if isinstance(text, list):
            return EmbeddingResponse(embeddings=[[1.0, 0.0] for _ in text])
        return EmbeddingResponse(embeddings=[[1.0, 0.0]])


class FailingEmbedClient:
    async def embed(self, *, task: str, text):
        raise RuntimeError("embedding unavailable")


class FakeReader:
    def __init__(
        self,
        hits: list[MemoryDbSearchHit],
        *,
        related_ids: list[str | dict[str, str]] | Exception | None = None,
        shared_entity_scopes: list[GraphNeighborScope] | Exception | None = None,
        graph_memories: list[MemoryView] | None = None,
        lineage_by_id: dict[str, list[str]] | None = None,
        archived_memories: list[MemoryView] | None = None,
    ) -> None:
        self.hits = hits
        self.related_ids = related_ids or []
        self.shared_entity_scopes = shared_entity_scopes or []
        self.graph_memories = graph_memories or []
        self.lineage_by_id = lineage_by_id or {}
        self.archived_memories = archived_memories or []
        self.calls = []
        self.related_calls = []
        self.shared_entity_calls = []
        self.get_memories_calls = []
        self.lineage_calls = []

    async def search_hybrid(
        self,
        context: MemoryRequestContext,
        req,
        *,
        dense_vector,
        sparse_vector,
        dense_limit=None,
        sparse_limit=None,
    ):
        self.calls.append(
            SimpleNamespace(
                context=context,
                req=req,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
            )
        )
        return MemoryDbSearchResult(query=req.query, hits=self.hits, total=len(self.hits))

    async def search_sparse(self, context: MemoryRequestContext, req, *, indices, values):
        raise AssertionError("vanilla search should use hybrid retrieval when dense embedding is available")

    async def get_related_memory_ids(
        self,
        context: MemoryRequestContext,
        memory_ids: list[str],
        *,
        limit_per_memory: int,
        max_candidates: int,
    ):
        self.related_calls.append(
            SimpleNamespace(
                context=context,
                memory_ids=memory_ids,
                limit_per_memory=limit_per_memory,
                max_candidates=max_candidates,
            )
        )
        if isinstance(self.related_ids, Exception):
            raise self.related_ids
        return list(self.related_ids)

    async def list_memories_by_shared_entities(
        self,
        context: MemoryRequestContext,
        memory_ids: list[str],
        *,
        include_seed: bool = True,
        active_only: bool = True,
        limit_per_entity: int = 50,
    ):
        self.shared_entity_calls.append(
            SimpleNamespace(
                context=context,
                memory_ids=memory_ids,
                include_seed=include_seed,
                active_only=active_only,
                limit_per_entity=limit_per_entity,
            )
        )
        if isinstance(self.shared_entity_scopes, Exception):
            raise self.shared_entity_scopes
        return list(self.shared_entity_scopes)

    async def get_memories(self, context: MemoryRequestContext, memory_ids: list[str]):
        self.get_memories_calls.append(SimpleNamespace(context=context, memory_ids=memory_ids))
        by_id = {memory.memory_id: memory for memory in [*self.graph_memories, *self.archived_memories]}
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    async def get_memory_lineage(self, context: MemoryRequestContext, memory_ids: list[str]):
        self.lineage_calls.append(SimpleNamespace(context=context, memory_ids=memory_ids))
        return {memory_id: self.lineage_by_id.get(memory_id, []) for memory_id in memory_ids}


def make_engine(reader: FakeReader, search_config: VanillaSearchConfig | None = None) -> VanillaSearchEngine:
    return VanillaSearchEngine(
        db_reader=reader,
        db_writer=SimpleNamespace(),
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        search_config=search_config or VanillaSearchConfig(recall_size=4),
        embed_client=FakeEmbedClient(),
    )


@pytest.mark.asyncio
async def test_vanilla_dedup_keeps_highest_scored_copy():
    first = MemoryDbSearchHit(
        memory_id="low",
        score=0.6,
        memory=memory("low", "User joined an advanced investment course online."),
        source="rrf",
        rank=2,
    )
    second = MemoryDbSearchHit(
        memory_id="high",
        score=0.9,
        memory=memory("high", "User joined an advanced investment course online."),
        source="rrf",
        rank=1,
    )
    distinct = MemoryDbSearchHit(
        memory_id="distinct",
        score=0.5,
        memory=memory("distinct", "User enjoys hiking on weekends."),
        source="rrf",
        rank=3,
    )
    reader = FakeReader([first, second, distinct])

    result = await make_engine(reader).search_candidates(SearchPipelineInput(query="course"), make_context())

    assert [item.id for item in result] == ["high", "distinct"]


@pytest.mark.asyncio
async def test_vanilla_dedup_keeps_same_text_for_different_users():
    first = MemoryDbSearchHit(
        memory_id="alice",
        score=0.9,
        memory=memory("alice", "User joined an advanced investment course online.", user_id="alice"),
        source="rrf",
        rank=1,
    )
    second = MemoryDbSearchHit(
        memory_id="bob",
        score=0.8,
        memory=memory("bob", "User joined an advanced investment course online.", user_id="bob"),
        source="rrf",
        rank=2,
    )
    reader = FakeReader([first, second])

    result = await make_engine(reader).search_candidates(SearchPipelineInput(query="course"), make_context())

    assert [item.id for item in result] == ["alice", "bob"]


@pytest.mark.asyncio
async def test_vanilla_dedup_keeps_same_text_for_current_and_archived_lineage():
    current = MemoryDbSearchHit(
        memory_id="current",
        score=0.9,
        memory=memory("current", "User joined an advanced investment course online.", user_id="alice"),
        source="rrf",
        rank=1,
    )
    archived = MemoryDbSearchHit(
        memory_id="archived",
        score=0.8,
        memory=memory(
            "archived",
            "User joined an advanced investment course online.",
            user_id="alice",
            status="archived",
        ),
        source="lineage_archived",
        rank=2,
    )
    reader = FakeReader([current, archived])

    result = await make_engine(reader).search_candidates(SearchPipelineInput(query="course"), make_context())

    assert [item.id for item in result] == ["current", "archived"]
    assert [item.lineage.role for item in result if item.lineage is not None] == ["current", "archived"]


@pytest.mark.asyncio
async def test_vanilla_dedup_does_not_block_the_event_loop(monkeypatch):
    dedup_started = threading.Event()
    release_dedup = threading.Event()
    dedup_finished = threading.Event()

    def blocking_dedup(candidates, **kwargs):
        dedup_started.set()
        if not release_dedup.wait(timeout=1):
            raise TimeoutError("test did not release dedup")
        dedup_finished.set()
        return candidates

    monkeypatch.setattr(vanilla_engine_module, "dedup_by_text_similarity", blocking_dedup)
    hit = MemoryDbSearchHit(
        memory_id="one",
        score=0.9,
        memory=memory("one", "User joined an advanced investment course online."),
        source="rrf",
        rank=1,
    )
    heartbeat_ran_during_dedup = False

    async def heartbeat():
        nonlocal heartbeat_ran_during_dedup
        await asyncio.sleep(0.02)
        heartbeat_ran_during_dedup = dedup_started.is_set() and not dedup_finished.is_set()

    def release_after_dedup_starts():
        if dedup_started.wait(timeout=1):
            time.sleep(0.2)
            release_dedup.set()

    release_thread = threading.Thread(target=release_after_dedup_starts, daemon=True)
    release_thread.start()
    try:
        await asyncio.gather(
            make_engine(FakeReader([hit])).search_candidates(
                SearchPipelineInput(query="course"),
                make_context(),
            ),
            heartbeat(),
        )
    finally:
        release_dedup.set()
        release_thread.join(timeout=1)

    assert heartbeat_ran_during_dedup is True


def make_graph_engine(reader: FakeReader) -> VanillaSearchEngine:
    return make_engine(
        reader,
        VanillaSearchConfig(
            recall_size=4,
            graph_enabled=True,
            graph_seed_memory_limit=2,
            graph_related_per_seed=3,
            graph_max_candidates=4,
            graph_decay=0.5,
        ),
    )


def make_shared_entity_graph_engine(reader: FakeReader) -> VanillaSearchEngine:
    return make_engine(
        reader,
        VanillaSearchConfig(
            recall_size=4,
            shared_entity_graph_enabled=True,
            graph_seed_memory_limit=2,
            shared_entity_graph_limit_per_entity=3,
            graph_max_candidates=4,
            graph_decay=0.5,
        ),
    )


def make_combined_graph_engine(reader: FakeReader) -> VanillaSearchEngine:
    return make_engine(
        reader,
        VanillaSearchConfig(
            recall_size=4,
            graph_enabled=True,
            shared_entity_graph_enabled=True,
            graph_seed_memory_limit=2,
            graph_related_per_seed=3,
            shared_entity_graph_limit_per_entity=3,
            graph_max_candidates=2,
            graph_decay=0.5,
        ),
    )


class SparseFallbackReader(FakeReader):
    def __init__(self, hits: list[MemoryDbSearchHit]) -> None:
        super().__init__(hits)
        self.sparse_calls = []

    async def search_hybrid(self, context: MemoryRequestContext, req, *, dense_vector, sparse_vector):
        raise AssertionError("dense embedding failure should use sparse search")

    async def search_sparse(self, context: MemoryRequestContext, req, *, indices, values):
        self.sparse_calls.append(SimpleNamespace(context=context, req=req, indices=indices, values=values))
        return MemoryDbSearchResult(query=req.query, hits=self.hits, total=len(self.hits))


def _flatten_field_conditions(filters: SearchFilter) -> list[FieldCondition]:
    conditions: list[FieldCondition] = []
    for clause in [*filters.must, *filters.should, *filters.must_not]:
        if isinstance(clause, FieldCondition):
            conditions.append(clause)
        else:
            conditions.extend(_flatten_field_conditions(clause))
    return conditions


@pytest.mark.asyncio
async def test_vanilla_search_uses_rrf_and_preserves_event_time() -> None:
    hits = [
        MemoryDbSearchHit(
            memory_id="mem-1",
            score=0.6,
            memory=memory("mem-1", "Kai likes Redis.", metadata={"source_timestamp_ms": 1700000000000}),
            source="rrf",
            rank=1,
        ),
        MemoryDbSearchHit(
            memory_id="mem-2",
            score=0.9,
            memory=memory(
                "mem-2",
                "Kai uses Qdrant.",
                validate_from=datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC),
            ),
            source="rrf",
            rank=2,
        ),
    ]
    reader = FakeReader(hits)
    engine = make_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=2), make_context())

    assert [item.id for item in result] == ["mem-2", "mem-1"]
    assert result[0].memory_type == "fact"
    assert result[0].event_time == "2023-11-14 22:13:20"
    assert result[0].source_timestamp == "2023-11-14 22:13:20"
    assert result[1].event_time == "2023-11-14 22:13:20"
    assert result[1].source_timestamp == "2023-11-14 22:13:20"

    assert len(reader.calls) == 1
    call = reader.calls[0]
    assert call.req.mode == "rrf"
    assert call.req.top_k == 4
    assert call.dense_vector == [1.0, 0.0]
    assert call.sparse_vector.indices
    must_fields = [c.field for c in call.req.filters.must if isinstance(c, FieldCondition)]
    assert must_fields == ["status"]
    assert call.req.filters.should == []
    assert call.req.filters.must_not == []
    # recall_size=4 -> dense/sparse prefetch = max(4 * 3, 30) = 30 (floor dominates)
    assert call.dense_limit == 30
    assert call.sparse_limit == 30


@pytest.mark.asyncio
async def test_vanilla_search_threads_rrf_prefetch_limits() -> None:
    """Vanilla computes dense/sparse prefetch via max(recall_size * factor, min) and threads it down."""

    # floor dominates: recall_size=4 -> max(4 * 3, 30) = 30
    reader_floor = FakeReader([])
    await make_engine(reader_floor).search_candidates(
        SearchPipelineInput(query="qdrant memory", top_k=2),
        make_context(),
    )
    assert reader_floor.calls[0].dense_limit == 30
    assert reader_floor.calls[0].sparse_limit == 30

    # factor dominates: recall_size=20 -> max(20 * 3, 30) = 60
    reader_factor = FakeReader([])
    await make_engine(reader_factor, VanillaSearchConfig(recall_size=20)).search_candidates(
        SearchPipelineInput(query="qdrant memory", top_k=2),
        make_context(),
    )
    assert reader_factor.calls[0].dense_limit == 60
    assert reader_factor.calls[0].sparse_limit == 60

    # configured ceiling dominates: max(100 * 3, 30) is capped at 250
    reader_cap = FakeReader([])
    await make_engine(
        reader_cap,
        VanillaSearchConfig(recall_size=100, hybrid_prefetch_max=250),
    ).search_candidates(
        SearchPipelineInput(query="qdrant memory", top_k=2),
        make_context(),
    )
    assert reader_cap.calls[0].dense_limit == 250
    assert reader_cap.calls[0].sparse_limit == 250


@pytest.mark.asyncio
async def test_vanilla_search_enforces_runtime_recall_and_prefetch_caps() -> None:
    reader = FakeReader([])
    engine = make_engine(
        reader,
        VanillaSearchConfig(
            recall_size=500,
            hybrid_prefetch_factor=50,
            hybrid_prefetch_min=500,
            hybrid_prefetch_max=500,
        ),
    )

    await engine.search_candidates(
        SearchPipelineInput(query="qdrant memory", top_k=500),
        make_context(),
    )

    call = reader.calls[0]
    assert call.req.top_k == 100
    assert call.dense_limit == 300
    assert call.sparse_limit == 300


@pytest.mark.parametrize(("configured_cap", "expected_cap"), [(17, 17), (500, 128)])
@pytest.mark.asyncio
async def test_vanilla_search_passes_dedup_candidate_cap(monkeypatch, configured_cap: int, expected_cap: int) -> None:
    recorded_kwargs = {}

    class RecordingDedupExecutor:
        async def run(self, func, candidates, **kwargs):
            recorded_kwargs.update(kwargs)
            return func(candidates, **kwargs)

    monkeypatch.setattr(vanilla_engine_module, "vanilla_dedup_executor", RecordingDedupExecutor())
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="one",
                score=0.9,
                memory=memory("one", "User joined an advanced investment course online."),
                source="rrf",
                rank=1,
            )
        ]
    )

    await make_engine(reader, VanillaSearchConfig(dedup_max_candidates=configured_cap)).search_candidates(
        SearchPipelineInput(query="course", top_k=1),
        make_context(),
    )

    assert recorded_kwargs["max_candidates"] == expected_cap


@pytest.mark.asyncio
async def test_vanilla_search_appends_archived_lineage_candidates() -> None:
    hits = [
        MemoryDbSearchHit(
            memory_id="current",
            score=0.9,
            memory=memory("current", "Current Qdrant preference."),
            source="rrf",
            rank=1,
        )
    ]
    reader = FakeReader(
        hits,
        lineage_by_id={"current": ["old-1", "old-2"], "old-1": ["root-1"], "old-2": []},
        archived_memories=[
            memory(
                "old-1",
                "Old Qdrant preference.",
                status="archived",
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            memory(
                "old-2",
                "Newer archived Qdrant preference.",
                status="archived",
                created_at=datetime(2026, 1, 3, tzinfo=UTC),
            ),
        ],
    )
    engine = make_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=3), make_context())

    assert [item.id for item in result] == ["current", "old-2", "old-1"]
    assert result[0].lineage is not None
    assert result[0].lineage.role == "current"
    assert result[0].lineage.derived_from_memory_ids == ["old-2", "old-1"]
    assert result[1].lineage is not None
    assert result[1].lineage.role == "archived"
    assert result[1].lineage.derived_to_memory_ids == ["current"]
    assert result[2].lineage is not None
    assert result[2].lineage.role == "archived"
    assert result[2].lineage.derived_from_memory_ids == ["root-1"]
    assert result[2].lineage.derived_to_memory_ids == ["current"]
    assert [call.memory_ids for call in reader.lineage_calls] == [["current"], ["old-1", "old-2"]]
    assert reader.get_memories_calls[-1].memory_ids == ["old-1", "old-2"]


@pytest.mark.asyncio
async def test_vanilla_search_prefers_resolved_event_date_over_source_time() -> None:
    hits = [
        MemoryDbSearchHit(
            memory_id="mem-1",
            score=0.9,
            memory=memory(
                "mem-1",
                "On 2023-01-19 (yesterday), Caroline visited an LGBTQ support group.",
                validate_from=datetime(2023, 1, 20, tzinfo=UTC),
                metadata={
                    "resolved_event_date": "2023-01-19",
                    "event_time_text": "yesterday",
                    "source_timestamp_ms": 1674172800000,
                },
            ),
            source="rrf",
            rank=1,
        )
    ]
    reader = FakeReader(hits)
    engine = make_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="support group", top_k=1), make_context())

    assert result[0].event_time == "2023-01-19 00:00:00"
    assert result[0].source_timestamp == "2023-01-20 00:00:00"


@pytest.mark.asyncio
async def test_vanilla_search_filter_matches_fastapi_filter_without_implicit_clauses() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="mem-1",
                score=0.8,
                memory=memory("mem-1", "Kai uses Qdrant."),
                source="rrf",
                rank=1,
            )
        ]
    )
    engine = make_engine(reader)

    await engine.search_candidates(
        SearchPipelineInput(
            query="Qdrant",
            top_k=1,
            filters={
                "mem_type": {"in": ["fact"]},
                "content": {"icontains": "qdrant"},
            },
        ),
        make_context(),
    )

    call = reader.calls[0]
    assert call.req.filters.should == []
    assert call.req.filters.must_not == []
    conditions = _flatten_field_conditions(call.req.filters)
    fields = [c.field for c in conditions]
    assert "status" in fields
    assert "user_id" not in fields
    assert "app_id" not in fields
    assert "agent_id" not in fields
    assert "session_id" not in fields
    assert FieldCondition(field="mem_type", op="any", values=["fact"]) in conditions
    assert FieldCondition(field="content", op="text", value="qdrant") in conditions


@pytest.mark.asyncio
async def test_vanilla_search_uses_user_scope_only_when_filter_supplies_it() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="mem-1",
                score=0.8,
                memory=memory("mem-1", "Kai uses Qdrant."),
                source="rrf",
                rank=1,
            )
        ]
    )
    engine = make_engine(reader)

    await engine.search_candidates(
        SearchPipelineInput(
            query="Qdrant",
            top_k=1,
            filters={
                "user_id": "user-1",
                "app_id": "app-1",
                "agent_id": "agent-1",
                "session_id": "session-1",
            },
        ),
        make_context(),
    )

    conditions = _flatten_field_conditions(reader.calls[0].req.filters)
    assert FieldCondition(field="status", op="match", value="active") in conditions
    assert FieldCondition(field="user_id", op="match", value="user-1") in conditions
    assert FieldCondition(field="app_id", op="match", value="app-1") in conditions
    assert FieldCondition(field="agent_id", op="match", value="agent-1") in conditions
    assert FieldCondition(field="session_id", op="match", value="session-1") in conditions


@pytest.mark.asyncio
async def test_vanilla_search_appends_one_hop_related_memories_with_one_batch_read() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="seed",
                score=0.8,
                memory=memory("seed", "Kai uses Qdrant.", user_id="user-1"),
                source="rrf",
                rank=1,
            )
        ],
        related_ids=[
            {"memory_id": "graph-ok", "seed_memory_id": "seed"},
            {"memory_id": "archived", "seed_memory_id": "seed"},
            {"memory_id": "wrong-user", "seed_memory_id": "seed"},
            {"memory_id": "text-miss", "seed_memory_id": "seed"},
            {"memory_id": "seed", "seed_memory_id": "seed"},
        ],
        graph_memories=[
            memory("graph-ok", "Graph neighbor also mentions Qdrant.", user_id="user-1"),
            memory("archived", "Archived Qdrant neighbor.", status="archived", user_id="user-1"),
            memory("wrong-user", "Wrong user Qdrant neighbor.", user_id="user-2"),
            memory("text-miss", "Graph neighbor about Redis.", user_id="user-1"),
        ],
    )
    engine = make_graph_engine(reader)

    result = await engine.search_candidates(
        SearchPipelineInput(
            query="Qdrant",
            top_k=2,
            filters={"user_id": "user-1", "content": {"icontains": "qdrant"}},
        ),
        make_context(),
    )

    assert [item.id for item in result] == ["seed", "graph-ok"]
    assert len(reader.related_calls) == 1
    assert reader.related_calls[0].memory_ids == ["seed"]
    assert reader.related_calls[0].limit_per_memory == 3
    assert reader.related_calls[0].max_candidates == 4
    assert len(reader.get_memories_calls) == 1
    assert reader.get_memories_calls[0].memory_ids == ["graph-ok", "archived", "wrong-user", "text-miss"]


@pytest.mark.asyncio
async def test_vanilla_search_appends_shared_entity_memories_with_one_batch_read() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="seed",
                score=0.8,
                memory=memory("seed", "Kai uses Qdrant.", user_id="user-1"),
                source="rrf",
                rank=1,
            ),
            MemoryDbSearchHit(
                memory_id="existing",
                score=0.7,
                memory=memory("existing", "Existing Qdrant hit.", user_id="user-1"),
                source="rrf",
                rank=2,
            ),
        ],
        shared_entity_scopes=[
            GraphNeighborScope(
                seed_memory_id="seed",
                entity_id="entity-qdrant",
                entity_name="Qdrant",
                entity_type="software",
                memory_ids=("seed", "shared-ok", "existing", "archived", "wrong-user", "text-miss", "shared-ok"),
            )
        ],
        graph_memories=[
            memory("shared-ok", "Shared entity neighbor also mentions Qdrant.", user_id="user-1"),
            memory("archived", "Archived Qdrant neighbor.", status="archived", user_id="user-1"),
            memory("wrong-user", "Wrong user Qdrant neighbor.", user_id="user-2"),
            memory("text-miss", "Shared entity neighbor about Redis.", user_id="user-1"),
        ],
    )
    engine = make_shared_entity_graph_engine(reader)

    result = await engine.search_candidates(
        SearchPipelineInput(
            query="Qdrant",
            top_k=3,
            filters={"user_id": "user-1", "content": {"icontains": "qdrant"}},
        ),
        make_context(),
    )

    assert [item.id for item in result] == ["seed", "existing", "shared-ok"]
    assert reader.related_calls == []
    assert len(reader.shared_entity_calls) == 1
    assert reader.shared_entity_calls[0].memory_ids == ["seed", "existing"]
    assert reader.shared_entity_calls[0].include_seed is False
    assert reader.shared_entity_calls[0].active_only is True
    assert reader.shared_entity_calls[0].limit_per_entity == 3
    assert len(reader.get_memories_calls) == 1
    assert reader.get_memories_calls[0].memory_ids == ["shared-ok", "archived", "wrong-user", "text-miss"]


@pytest.mark.asyncio
async def test_vanilla_search_caps_combined_graph_candidates_before_hydration() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="seed",
                score=0.8,
                memory=memory("seed", "Kai uses Qdrant."),
                source="rrf",
                rank=1,
            )
        ],
        related_ids=[{"memory_id": "direct-1", "seed_memory_id": "seed"}],
        shared_entity_scopes=[
            GraphNeighborScope(
                seed_memory_id="seed",
                entity_id="entity-qdrant",
                entity_name="Qdrant",
                entity_type="software",
                memory_ids=("shared-1", "shared-2"),
            )
        ],
        graph_memories=[
            memory("direct-1", "Direct graph neighbor."),
            memory("shared-1", "First shared entity neighbor."),
            memory("shared-2", "Second shared entity neighbor."),
        ],
    )
    engine = make_combined_graph_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=3), make_context())

    assert [item.id for item in result] == ["seed", "direct-1", "shared-1"]
    assert len(reader.get_memories_calls) == 1
    assert reader.get_memories_calls[0].memory_ids == ["direct-1", "shared-1"]


@pytest.mark.asyncio
async def test_vanilla_search_shared_entity_expansion_failure_falls_back_to_qdrant_hits() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="seed",
                score=0.8,
                memory=memory("seed", "Kai uses Qdrant."),
                source="rrf",
                rank=1,
            )
        ],
        shared_entity_scopes=RuntimeError("neo4j unavailable"),
    )
    engine = make_shared_entity_graph_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=2), make_context())

    assert [item.id for item in result] == ["seed"]
    assert len(reader.shared_entity_calls) == 1
    assert reader.get_memories_calls == []


@pytest.mark.asyncio
async def test_vanilla_search_ranks_graph_hits_by_seed_score_decay() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="seed-high",
                score=0.9,
                memory=memory("seed-high", "Strong seed."),
                source="rrf",
                rank=1,
            ),
            MemoryDbSearchHit(
                memory_id="base-low",
                score=0.2,
                memory=memory("base-low", "Low base hit."),
                source="rrf",
                rank=2,
            ),
        ],
        related_ids=[{"memory_id": "graph-from-high", "seed_memory_id": "seed-high"}],
        graph_memories=[memory("graph-from-high", "Related to strong seed.")],
    )
    engine = make_graph_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=2), make_context())

    assert [item.id for item in result] == ["seed-high", "graph-from-high", "base-low"]


@pytest.mark.asyncio
async def test_vanilla_search_graph_expansion_failure_falls_back_to_qdrant_hits() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="seed",
                score=0.8,
                memory=memory("seed", "Kai uses Qdrant."),
                source="rrf",
                rank=1,
            )
        ],
        related_ids=RuntimeError("neo4j unavailable"),
    )
    engine = make_graph_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=2), make_context())

    assert [item.id for item in result] == ["seed"]
    assert len(reader.related_calls) == 1
    assert reader.get_memories_calls == []


@pytest.mark.asyncio
async def test_vanilla_search_returns_stored_memory_type() -> None:
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="mem-1",
                score=0.8,
                memory=memory("mem-1", "Kai prefers concise answers.", mem_type="profile"),
                source="rrf",
                rank=1,
            )
        ]
    )
    engine = make_engine(reader)

    result = await engine.search_candidates(SearchPipelineInput(query="preferences", top_k=1), make_context())

    assert result[0].memory_type == "profile"


@pytest.mark.asyncio
async def test_vanilla_search_falls_back_to_sparse_when_dense_embedding_fails() -> None:
    reader = SparseFallbackReader(
        [
            MemoryDbSearchHit(
                memory_id="mem-1",
                score=0.8,
                memory=memory("mem-1", "Kai uses Qdrant."),
                source="bm25",
                rank=1,
            )
        ]
    )
    engine = VanillaSearchEngine(
        db_reader=reader,
        db_writer=SimpleNamespace(),
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
        search_config=VanillaSearchConfig(recall_size=4),
        embed_client=FailingEmbedClient(),
    )

    result = await engine.search_candidates(SearchPipelineInput(query="Qdrant", top_k=1), make_context())

    assert [item.id for item in result] == ["mem-1"]
    assert len(reader.sparse_calls) == 1
    call = reader.sparse_calls[0]
    assert call.req.mode == "bm25"
    assert call.req.ranking == "score"
    assert call.indices
    assert call.values


@pytest.mark.asyncio
async def test_search_pipeline_lazy_loads_vanilla_engine_and_final_filters(monkeypatch) -> None:
    monkeypatch.setattr(
        "mindmemos.pipelines.search.pipeline.get_config",
        fake_search_config,
    )
    monkeypatch.setattr(
        "mindmemos.pipelines.search.vanilla.engine.get_config",
        fake_search_config,
    )
    monkeypatch.setattr(
        "mindmemos.pipelines.search.vanilla.engine.get_embed_client",
        lambda: FakeEmbedClient(),
    )
    # VanillaSearchEngine.__init__ falls back to get_text_preprocessor() when text_config
    # is not supplied (lazy-load path), which reads its own module-level get_config.
    monkeypatch.setattr(
        "mindmemos.components.text.preprocessor.get_config",
        fake_search_config,
    )
    reader = FakeReader(
        [
            MemoryDbSearchHit(
                memory_id="mem-1",
                score=0.8,
                memory=memory("mem-1", "Kai uses Qdrant."),
                source="rrf",
                rank=1,
            ),
            MemoryDbSearchHit(
                memory_id="mem-2",
                score=0.7,
                memory=memory("mem-2", "Kai uses Redis."),
                source="rrf",
                rank=2,
            ),
        ]
    )
    pipeline = SearchPipelineImpl(
        db_reader=reader,
        db_writer=SimpleNamespace(),
        final_filter=SearchFinalFilter(),
        rerank_client=None,
    )

    result = await pipeline.search(
        SearchPipelineInput(query="Qdrant", search_pipeline="vanilla", top_k=1, rerank=True),
        make_context(),
    )

    assert result.status == "ok"
    assert [item.id for item in result.memories] == ["mem-1"]
    assert isinstance(pipeline._engines["vanilla"], VanillaSearchEngine)


@pytest.mark.asyncio
async def test_cached_vanilla_engine_uses_each_project_search_config(monkeypatch) -> None:
    try:
        init_config(config_path="config/mindmemos/dev.example.yaml")
        monkeypatch.setattr(
            "mindmemos.pipelines.search.vanilla.engine.get_embed_client",
            lambda: FakeEmbedClient(),
        )
        reader = FakeReader(
            [
                MemoryDbSearchHit(
                    memory_id="mem-1",
                    score=0.9,
                    memory=memory("mem-1", "Kai uses Qdrant for vector search."),
                    source="rrf",
                    rank=1,
                ),
                MemoryDbSearchHit(
                    memory_id="mem-2",
                    score=0.8,
                    memory=memory("mem-2", "Kai uses Qdrant for vector search."),
                    source="rrf",
                    rank=2,
                ),
            ]
        )
        pipeline = SearchPipelineImpl(
            db_reader=reader,
            db_writer=SimpleNamespace(),
            final_filter=SearchFinalFilter(),
        )
        request = SearchPipelineInput(
            query="Qdrant",
            search_pipeline="vanilla",
            top_k=2,
            rerank=False,
        )

        with bind_config_overrides(
            project_config={
                "algo_config": {
                    "search": {
                        "vanilla": {
                            "dedup_enabled": True,
                            "dedup_threshold": 0.6,
                        }
                    }
                }
            }
        ):
            project_a = await pipeline.search(request, make_context())

        cached_engine = pipeline._engines["vanilla"]
        with bind_config_overrides(
            project_config={
                "algo_config": {
                    "search": {
                        "vanilla": {
                            "dedup_enabled": False,
                            "dedup_threshold": 1.0,
                        }
                    }
                }
            }
        ):
            project_b = await pipeline.search(request, make_context())

        assert pipeline._engines["vanilla"] is cached_engine
        assert [item.id for item in project_a.memories] == ["mem-1"]
        assert [item.id for item in project_b.memories] == ["mem-1", "mem-2"]
    finally:
        reset_config()
