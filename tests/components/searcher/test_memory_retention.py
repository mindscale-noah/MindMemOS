import math
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.searcher.memory_retention import MemoryRetentionSelector
from mindmemos.components.searcher.scored_candidate import ScoredSearchCandidate
from mindmemos.config.algo.search import MemoryRetentionConfig
from mindmemos.typing.service import MemorySearchItem


class FakeTextPreprocessor:
    def preprocess_query(self, text: str, *, include_entities: bool = False):
        return SimpleNamespace(tokens=text.lower().split())

    def preprocess_text(self, text: str, *, include_entities: bool = False):
        return SimpleNamespace(tokens=text.lower().split())


def candidate(
    memory_id: str,
    text: str,
    *,
    rank: int,
    relevance: float,
    last_update_at: str = "",
) -> ScoredSearchCandidate:
    return ScoredSearchCandidate(
        item=MemorySearchItem(id=memory_id, memory=text, last_update_at=last_update_at),
        rank=rank,
        relevance_score=relevance,
    )


def _selector(now: Callable[[], datetime] | None = None, **config_overrides: object) -> MemoryRetentionSelector:
    return MemoryRetentionSelector(
        config=MemoryRetentionConfig(**config_overrides),
        text_preprocessor=FakeTextPreprocessor(),
        now=now,
    )


def test_score_computes_documented_mixed_features() -> None:
    selector = _selector(
        now=lambda: datetime(2026, 1, 31, tzinfo=UTC),
        relevance_weight=0.50,
        query_overlap_weight=0.25,
        recency_weight=0.15,
        cost_weight=0.10,
        recency_half_life_days=30.0,
        missing_recency_score=0.5,
    )
    scored = selector.score(
        query="alpha beta",
        candidates=[
            candidate(
                "a",
                "alpha beta gamma",
                rank=0,
                relevance=0.8,
                last_update_at="2026-01-01 00:00:00",
            )
        ],
        token_budget=1000,
    )

    assert len(scored) == 1
    entry = scored[0]
    assert entry.query_overlap == 1.0  # both query terms appear in the memory
    assert entry.recency == pytest.approx(0.5)  # exactly one half-life old
    expected_priority = 0.50 * 0.8 + 0.25 * 1.0 + 0.15 * 0.5 - 0.10 * entry.cost_ratio
    assert entry.priority == pytest.approx(expected_priority)
    assert entry.estimated_tokens > 0


def test_score_clamps_nonfinite_and_out_of_range_relevance() -> None:
    selector = _selector(
        relevance_weight=1.0,
        query_overlap_weight=0.0,
        recency_weight=0.0,
        cost_weight=0.0,
    )
    scored = selector.score(
        query="alpha",
        candidates=[
            candidate("nan", "alpha", rank=0, relevance=math.nan),
            candidate("big", "alpha", rank=1, relevance=1.7),
        ],
        token_budget=1000,
    )

    priorities = {entry.candidate.id: entry.priority for entry in scored}
    assert priorities["nan"] == pytest.approx(0.0)
    assert priorities["big"] == pytest.approx(1.0)


def test_score_uses_missing_recency_score_without_timestamp() -> None:
    selector = _selector(recency_weight=1.0, missing_recency_score=0.25)
    scored = selector.score(
        query="alpha",
        candidates=[candidate("a", "alpha", rank=0, relevance=0.0)],
        token_budget=1000,
    )

    assert scored[0].recency == pytest.approx(0.25)


def test_mixed_v1_skips_oversized_candidates_and_keeps_fitting_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    costs = {"huge memory text": 100, "small one": 20, "small two": 20}
    monkeypatch.setattr(
        "mindmemos.components.searcher.memory_retention.estimate_tokens",
        lambda text: costs[text],
    )
    selector = _selector(
        relevance_weight=1.0,
        query_overlap_weight=0.0,
        recency_weight=0.0,
        cost_weight=0.0,
    )
    candidates = [
        candidate("huge", "huge memory text", rank=0, relevance=0.9),
        candidate("a", "small one", rank=1, relevance=0.5),
        candidate("b", "small two", rank=2, relevance=0.4),
    ]

    result = selector.select(query="q", candidates=candidates, token_budget=50)

    # The 100-token leader does not fit; both 20-token candidates do.
    assert [c.id for c in result.candidates] == ["a", "b"]
    assert result.estimated_tokens_after == 40
    assert result.budget_induced_empty is False


def test_mixed_v1_reports_budget_induced_empty_when_nothing_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mindmemos.components.searcher.memory_retention.estimate_tokens",
        lambda text: 100,
    )
    selector = _selector(relevance_weight=1.0)
    candidates = [candidate("a", "alpha", rank=0, relevance=0.9)]

    result = selector.select(query="q", candidates=candidates, token_budget=10)

    assert result.candidates == []
    assert result.estimated_tokens_after == 0
    assert result.budget_induced_empty is True


def test_select_returns_empty_for_no_candidates() -> None:
    selector = _selector()

    result = selector.select(query="q", candidates=[], token_budget=100)

    assert result.candidates == []
    assert result.budget_induced_empty is False


def test_mixed_v2_guarantee_outranks_cheaper_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    costs = {"expensive top": 10, "cheap bargain": 2}
    monkeypatch.setattr(
        "mindmemos.components.searcher.memory_retention.estimate_tokens",
        lambda text: costs[text],
    )
    common = {
        "relevance_weight": 1.0,
        "query_overlap_weight": 0.0,
        "recency_weight": 0.0,
        "cost_weight": 1.0,
    }
    candidates = [
        candidate("top", "expensive top", rank=0, relevance=1.0),
        candidate("bargain", "cheap bargain", rank=1, relevance=0.5),
    ]

    v1 = _selector(**common).select(query="q", candidates=candidates, token_budget=10)
    # Greedy priority prefers the cheap item (0.5 - 0.2) over the top item (1.0 - 1.0).
    assert [c.id for c in v1.candidates] == ["bargain"]

    v2 = _selector(**common, selector_version="mixed-v2", top_m_guarantee=1).select(
        query="q", candidates=candidates, token_budget=10
    )
    # Phase 1 force-keeps the highest-relevance candidate despite its cost.
    assert [c.id for c in v2.candidates] == ["top"]


def test_score_bounds_candidates_to_max_candidates() -> None:
    selector = _selector(max_candidates=2)
    candidates = [candidate(f"c{i}", "alpha", rank=i, relevance=0.5) for i in range(5)]

    scored = selector.score(query="q", candidates=candidates, token_budget=1000)

    assert len(scored) == 2
