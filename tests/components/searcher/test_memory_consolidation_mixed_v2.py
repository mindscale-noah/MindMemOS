from datetime import UTC, datetime
from types import SimpleNamespace

from mindmemos.components.searcher.memory_consolidation import MemoryConsolidator
from mindmemos.components.searcher.memory_retention import MemoryRetentionSelector
from mindmemos.components.searcher.scored_candidate import ScoredSearchCandidate
from mindmemos.config.algo.search import MemoryRetentionConfig
from mindmemos.typing.service import MemorySearchItem


class FakeTextPreprocessor:
    def preprocess_query(self, text: str, *, include_entities: bool = False):
        return SimpleNamespace(tokens=text.lower().split())

    def preprocess_text(self, text: str, *, include_entities: bool = False):
        return SimpleNamespace(tokens=text.lower().split())


def candidate(memory_id: str, text: str, *, rank: int, relevance: float) -> ScoredSearchCandidate:
    return ScoredSearchCandidate(
        item=MemorySearchItem(id=memory_id, memory=text, last_update_at="2026-01-01 00:00:00"),
        rank=rank,
        relevance_score=relevance,
    )


def test_consolidation_merges_similar_memories_under_cap() -> None:
    consolidator = MemoryConsolidator(
        text_preprocessor=FakeTextPreprocessor(),
        max_memories=2,
        cluster_threshold=0.5,
        stitch_max_members=3,
        max_chars=600,
    )
    candidates = [
        candidate("a", "alice likes coffee in seattle", rank=0, relevance=0.9),
        candidate("b", "alice likes coffee in seattle mornings", rank=1, relevance=0.8),
        candidate("c", "bob plays tennis every sunday", rank=2, relevance=0.7),
    ]
    result = consolidator.consolidate(candidates)
    assert result.output_count == 2
    assert result.cluster_count == 2
    # First cluster should stitch complementary alice memories.
    assert "coffee" in result.candidates[0].memory
    assert "tennis" in result.candidates[1].memory


def test_consolidation_returns_zero_counts_for_empty_input() -> None:
    consolidator = MemoryConsolidator(text_preprocessor=FakeTextPreprocessor())

    result = consolidator.consolidate([])

    assert result.candidates == []
    assert result.input_count == 0
    assert result.cluster_count == 0
    assert result.output_count == 0


def test_consolidation_digit_fingerprint_blocks_false_near_duplicates() -> None:
    # Same words except one digit token: Jaccard ~0.857 >= 0.85, but the digit
    # fingerprints ("100" vs "200") differ, so both members must be stitched.
    shared = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima"
    consolidator = MemoryConsolidator(
        text_preprocessor=FakeTextPreprocessor(),
        cluster_threshold=0.5,
        near_dup_threshold=0.85,
        stitch_max_members=3,
        max_chars=600,
    )

    result = consolidator.consolidate(
        [
            candidate("a", f"{shared} 100", rank=0, relevance=0.9),
            candidate("b", f"{shared} 200", rank=1, relevance=0.8),
        ]
    )

    assert result.cluster_count == 1
    assert "100" in result.candidates[0].memory
    assert "200" in result.candidates[0].memory

    # Control: identical digits mean identical fingerprints -> true near-duplicate.
    control = consolidator.consolidate(
        [
            candidate("a", f"{shared} 100", rank=0, relevance=0.9),
            candidate("b", f"{shared} 100", rank=1, relevance=0.8),
        ]
    )
    assert control.output_count == 1
    assert control.candidates[0].id == "a"


def test_consolidation_truncates_stitched_text_to_max_chars() -> None:
    consolidator = MemoryConsolidator(
        text_preprocessor=FakeTextPreprocessor(),
        cluster_threshold=0.5,
        near_dup_threshold=0.85,
        stitch_max_members=3,
        max_chars=20,
    )

    result = consolidator.consolidate(
        [
            candidate("a", "alpha beta gamma", rank=0, relevance=0.9),
            candidate("b", "alpha beta delta", rank=1, relevance=0.8),
        ]
    )

    assert result.cluster_count == 1
    stitched = result.candidates[0].memory
    assert len(stitched) <= 20
    assert stitched == stitched.rstrip()


def test_mixed_v2_guarantees_top_relevance_then_mmr() -> None:
    selector = MemoryRetentionSelector(
        config=MemoryRetentionConfig(
            selector_version="mixed-v2",
            relevance_weight=1.0,
            query_overlap_weight=0.0,
            recency_weight=0.0,
            cost_weight=0.0,
            top_m_guarantee=1,
            mmr_lambda=0.7,
        ),
        text_preprocessor=FakeTextPreprocessor(),
        now=lambda: datetime(2026, 1, 31, tzinfo=UTC),
    )
    candidates = [
        candidate("rel", "alpha beta", rank=0, relevance=1.0),
        candidate("dup", "alpha beta gamma", rank=1, relevance=0.5),
        candidate("novel", "zeta eta", rank=2, relevance=0.4),
    ]
    result = selector.select(query="alpha", candidates=candidates, token_budget=10)
    ids = [c.id for c in result.candidates]
    assert ids[0] == "rel"
    # Novel item should be preferred over near-duplicate once rel is kept.
    assert "novel" in ids
