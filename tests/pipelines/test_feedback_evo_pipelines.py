"""Tests for the feedback_evo add/search pipeline wiring helpers."""

from __future__ import annotations

import pytest

from mindmemos.pipelines.search.feedback_evo.pipeline import _apply_input_overrides
from mindmemos.pipelines.search.vanilla.engine import _apply_tag_weights
from mindmemos.typing import MemoryDbSearchHit, MemoryView, SearchPipelineInput


def test_feedback_evo_pipelines_are_independent_from_vanilla():
    """feedback_evo is a sibling mode: no pipeline-level inheritance/delegation."""

    from mindmemos.pipelines.add.feedback_evo import FeedbackEvoAddPipeline
    from mindmemos.pipelines.add.vanilla import VanillaAddPipeline
    from mindmemos.pipelines.search.feedback_evo import FeedbackEvoSearchPipeline
    from mindmemos.pipelines.search.pipeline import SearchPipelineImpl

    assert not issubclass(FeedbackEvoAddPipeline, VanillaAddPipeline)
    assert not issubclass(FeedbackEvoSearchPipeline, SearchPipelineImpl)


def _hit(memory_id: str, score: float, *, mem_type: str = "fact", entity_type: str | None = None) -> MemoryDbSearchHit:
    return MemoryDbSearchHit(
        memory_id=memory_id,
        score=score,
        memory=MemoryView(
            memory_id=memory_id,
            project_id="p",
            content="content",
            mem_type=mem_type,
            status="active",
            entity_type=entity_type,
        ),
    )


def test_apply_input_overrides_maps_evolved_search_config():
    inp = SearchPipelineInput(query="q", top_k=10, rerank=False)
    modified = _apply_input_overrides(
        inp,
        {"top_k": 15, "rerank": True, "score_threshold": 0.3, "search_strategy": "agentic"},
    )

    assert modified.top_k == 15
    assert modified.rerank is True
    assert modified.score_threshold == 0.3
    assert modified.agentic is True


def test_apply_input_overrides_fast_strategy_only_toggles_agentic():
    inp = SearchPipelineInput(query="q", top_k=10, search_pipeline="default")
    modified = _apply_input_overrides(inp, {"search_strategy": "fast"})
    assert modified.search_pipeline == "default"
    assert modified.agentic is False


def test_apply_input_overrides_empty_config_is_noop():
    inp = SearchPipelineInput(query="q", top_k=10)
    assert _apply_input_overrides(inp, {}) is inp


def test_apply_tag_weights_prefers_entity_type_and_reranks():
    hits = [
        _hit("a", 0.9, mem_type="fact"),
        _hit("b", 0.8, mem_type="fact", entity_type="user"),
    ]
    ranked = _apply_tag_weights(hits, {"fact": 0.5, "user": 2.0})

    assert [h.memory_id for h in ranked] == ["b", "a"]
    assert ranked[0].score == pytest.approx(1.6)
    assert ranked[1].score == pytest.approx(0.45)


def test_apply_tag_weights_unknown_tag_keeps_score():
    hits = [_hit("a", 0.9, mem_type="tool_trace")]
    ranked = _apply_tag_weights(hits, {"fact": 0.5})
    assert ranked[0].score == pytest.approx(0.9)
