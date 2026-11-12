from __future__ import annotations

from types import SimpleNamespace

import pytest
from mindmemos.components.searcher.final_filter import SearchFinalFilter
from mindmemos.components.text import estimate_tokens
from mindmemos.config.algo.search import MemoryRetentionConfig
from mindmemos.pipelines.search.base import SearchEngineOptions
from mindmemos.pipelines.search.pipeline import SearchPipelineImpl
from mindmemos.typing.memory import MemoryRequestContext
from mindmemos.typing.service import MemorySearchItem, SearchPipelineInput


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


class FakeEngine:
    name = "default"

    def __init__(self) -> None:
        self.inputs: list[SearchPipelineInput] = []

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        self.inputs.append(inp)
        return [MemorySearchItem(id="mem-1", memory=f"{inp.search_pipeline}:{inp.query}", last_update_at="")]


class ExplodingAgenticWrapper:
    async def run(self, inp, context, engine):
        raise AssertionError("agentic wrapper should not run for non-agentic search")


@pytest.mark.asyncio
async def test_search_pipeline_uses_selected_engine_without_agentic_wrapper() -> None:
    engine = FakeEngine()
    pipeline = SearchPipelineImpl(
        engines={"default": engine},
        agentic_wrapper=ExplodingAgenticWrapper(),
        final_filter=SearchFinalFilter(),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    result = await pipeline.search(SearchPipelineInput(query="Qdrant", search_pipeline="default"), make_context())

    assert result.memories[0].id == "mem-1"
    assert engine.inputs[0].agentic is False


@pytest.mark.asyncio
async def test_search_pipeline_rejects_unknown_strategy_with_available_names() -> None:
    pipeline = SearchPipelineImpl(
        engines={"default": FakeEngine()},
        final_filter=SearchFinalFilter(),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="Available strategies: default"):
        await pipeline.search(SearchPipelineInput(query="Qdrant", search_pipeline="schema"), make_context())


class FakeTextPreprocessor:
    def preprocess_query(self, text: str, *, include_entities: bool = False):
        return SimpleNamespace(tokens=text.lower().split())

    def preprocess_text(self, text: str, *, include_entities: bool = False):
        return SimpleNamespace(tokens=text.lower().split())


class MultiItemEngine:
    name = "default"

    def __init__(self, memories: list[str]) -> None:
        self.memories = memories
        self.inputs: list[SearchPipelineInput] = []

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        self.inputs.append(inp)
        return [
            MemorySearchItem(id=f"mem-{i}", memory=text, last_update_at="2026-01-01 00:00:00")
            for i, text in enumerate(self.memories)
        ]


def _patch_fake_preprocessors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeTextPreprocessor()
    monkeypatch.setattr("mindmemos.components.searcher.memory_retention.get_text_preprocessor", lambda: fake)


@pytest.mark.asyncio
async def test_search_pipeline_without_token_budget_keeps_legacy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_preprocessors(monkeypatch)
    engine = MultiItemEngine(["alpha beta", "gamma delta"])
    pipeline = SearchPipelineImpl(
        engines={"default": engine},
        final_filter=SearchFinalFilter(),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    result = await pipeline.search(SearchPipelineInput(query="alpha gamma", search_pipeline="default"), make_context())

    assert [m.id for m in result.memories] == ["mem-0", "mem-1"]
    assert engine.inputs[0].top_k == 10


@pytest.mark.asyncio
async def test_search_pipeline_token_budget_packs_under_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_preprocessors(monkeypatch)
    engine = MultiItemEngine(["alpha beta gamma delta epsilon", "zeta eta theta iota kappa", "lambda mu nu xi omicron"])
    pipeline = SearchPipelineImpl(
        engines={"default": engine},
        final_filter=SearchFinalFilter(),
        retention_config=MemoryRetentionConfig(max_candidates=50),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    result = await pipeline.search(
        SearchPipelineInput(query="alpha gamma", search_pipeline="default", token_budget=10000), make_context()
    )

    assert [m.id for m in result.memories] == ["mem-0", "mem-1", "mem-2"]
    # Retention raises engine recall to max_candidates instead of the request top_k.
    assert engine.inputs[0].top_k == 50

    tight = await pipeline.search(
        SearchPipelineInput(query="alpha gamma", search_pipeline="default", token_budget=8), make_context()
    )
    assert 0 < len(tight.memories) < 3
    assert sum(estimate_tokens(m.memory) for m in tight.memories) <= 8


@pytest.mark.asyncio
async def test_search_pipeline_token_budget_reranks_full_pool_before_packing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_preprocessors(monkeypatch)
    engine = MultiItemEngine(["alpha beta", "gamma delta", "epsilon zeta", "eta theta", "iota kappa"])
    rerank_top_n_calls: list[int] = []

    async def fake_rerank_with_scores(client, query, docs, top_n):
        rerank_top_n_calls.append(top_n)
        return [(i, 1.0 - i * 0.1) for i in range(min(top_n, len(docs)))]

    pipeline = SearchPipelineImpl(
        engines={"default": engine},
        final_filter=SearchFinalFilter(
            rerank_client=SimpleNamespace(available=True, has_external_model=True),
            rerank_with_scores_fn=fake_rerank_with_scores,
        ),
        retention_config=MemoryRetentionConfig(max_candidates=50),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    result = await pipeline.search(
        SearchPipelineInput(
            query="alpha gamma",
            search_pipeline="default",
            top_k=3,
            rerank=True,
            token_budget=10000,
        ),
        make_context(),
    )

    # Retention raises engine recall to max_candidates and the rerank call
    # scores that whole pool instead of being pre-narrowed to the request top_k.
    assert engine.inputs[0].top_k == 50
    assert rerank_top_n_calls == [5]
    # top_k still caps the final packed result count.
    assert [m.id for m in result.memories] == ["mem-0", "mem-1", "mem-2"]


@pytest.mark.asyncio
async def test_search_pipeline_token_budget_keeps_real_engine_ids_for_near_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Retention must never merge or rewrite results: even a pool containing
    # near-duplicate memories must come back as the engine's own items, with
    # ids usable by get/update/delete and feedback.
    _patch_fake_preprocessors(monkeypatch)
    engine = MultiItemEngine(
        [
            "alice likes coffee in seattle",
            "alice likes coffee in seattle mornings",
            "bob plays tennis every sunday",
        ]
    )
    pipeline = SearchPipelineImpl(
        engines={"default": engine},
        final_filter=SearchFinalFilter(),
        retention_config=MemoryRetentionConfig(max_candidates=50),
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
    )

    result = await pipeline.search(
        SearchPipelineInput(query="alice coffee", search_pipeline="default", token_budget=10000), make_context()
    )

    assert sorted(m.id for m in result.memories) == ["mem-0", "mem-1", "mem-2"]
