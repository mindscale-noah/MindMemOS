"""Tests for near-duplicate folding of search candidates."""

from mindmemos.components.searcher.vanilla import _dedup as dedup_module
from mindmemos.components.searcher.vanilla import dedup_by_text_similarity
from mindmemos.typing.service import MemorySearchItem


def item(memory_id: str, text: str) -> MemorySearchItem:
    return MemorySearchItem(id=memory_id, memory=text, last_update_at="")


# Real-world near-duplicates: the same fact restated with reworded phrasing.
_COURSE_A = "User participated in an online course on advanced investment strategies that offered flexibility, challenging content, interactive discussions, and actionable insights."
_COURSE_B = "User participated in an online course on advanced investment strategies that offered flexibility, challenging content, and interactive discussions, resulting in actionable insights."
_COMMUNITY = "User joined a new online investment community, actively participating in discussions that enhance knowledge, confidence, and motivation."
_HIKING = "User enjoys hiking in the mountains on weekends."


def test_folds_reworded_restatement_of_same_fact():
    kept = dedup_by_text_similarity([item("1", _COURSE_A), item("2", _COURSE_B)], threshold=0.6)
    assert [k.id for k in kept] == ["1"]


def test_keeps_related_but_distinct_memories():
    # Same topic (investment) but different facts must survive.
    kept = dedup_by_text_similarity([item("1", _COURSE_A), item("2", _COMMUNITY)], threshold=0.6)
    assert [k.id for k in kept] == ["1", "2"]


def test_keeps_completely_unrelated_memories():
    kept = dedup_by_text_similarity([item("1", _COURSE_A), item("2", _HIKING)], threshold=0.6)
    assert [k.id for k in kept] == ["1", "2"]


def test_preserves_input_order_of_survivors():
    kept = dedup_by_text_similarity(
        [item("1", _HIKING), item("2", _COURSE_A), item("3", _COURSE_B), item("4", _COMMUNITY)],
        threshold=0.6,
    )
    # 3 folds into 2; order of survivors preserved.
    assert [k.id for k in kept] == ["1", "2", "4"]


def test_folds_a_burst_of_duplicates_to_one():
    dups = [item(str(i), _COURSE_A) for i in range(9)]
    kept = dedup_by_text_similarity(dups, threshold=0.6)
    assert [k.id for k in kept] == ["0"]


def test_empty_and_single_are_returned_as_is():
    assert dedup_by_text_similarity([], threshold=0.6) == []
    single = [item("1", _COURSE_A)]
    assert dedup_by_text_similarity(single, threshold=0.6) == single


def test_threshold_of_one_folds_only_token_identical():
    # Reworded restatement is not token-identical, so threshold=1.0 keeps both.
    kept = dedup_by_text_similarity([item("1", _COURSE_A), item("2", _COURSE_B)], threshold=1.0)
    assert [k.id for k in kept] == ["1", "2"]


def test_folds_reordered_cjk_restatement():
    first = "用户参加了高级投资策略在线课程"
    second = "用户参加了在线高级投资策略课程"

    kept = dedup_by_text_similarity([item("1", first), item("2", second)], threshold=0.6)

    assert [entry.id for entry in kept] == ["1"]


def test_keeps_short_distinct_facts_with_generic_overlap():
    first = "User bought a red car"
    second = "User bought a blue car"

    kept = dedup_by_text_similarity([item("1", first), item("2", second)], threshold=0.6)

    assert [entry.id for entry in kept] == ["1", "2"]


def test_keeps_short_distinct_cjk_facts_with_generic_overlap():
    first = "用户买了红色的车"
    second = "用户买了蓝色的车"

    kept = dedup_by_text_similarity([item("1", first), item("2", second)], threshold=0.6)

    assert [entry.id for entry in kept] == ["1", "2"]


def test_folds_duplicates_only_within_the_same_actor_group():
    candidates = [item("alice-1", _COURSE_A), item("bob", _COURSE_A), item("alice-2", _COURSE_A)]

    kept = dedup_by_text_similarity(
        candidates,
        threshold=0.6,
        group_keys=[("alice", "current"), ("bob", "current"), ("alice", "current")],
    )

    assert [entry.id for entry in kept] == ["alice-1", "bob"]


def test_keeps_identical_text_from_current_and_archived_lineage_groups():
    candidates = [item("current", _COURSE_A), item("archived", _COURSE_A)]

    kept = dedup_by_text_similarity(
        candidates,
        threshold=0.6,
        group_keys=[("alice", "current"), ("alice", "archived")],
    )

    assert [entry.id for entry in kept] == ["current", "archived"]


def test_near_dedup_token_fingerprint_is_bounded():
    text = " ".join(f"token{index}" for index in range(600))

    assert len(dedup_module._tokens(text)) == 512


def test_short_text_classification_is_cached_per_candidate(monkeypatch):
    original = dedup_module._is_short_for_near_dedup
    calls = 0

    def counting_is_short(text, tokens):
        nonlocal calls
        calls += 1
        return original(text, tokens)

    monkeypatch.setattr(dedup_module, "_is_short_for_near_dedup", counting_is_short)
    candidates = [
        item(str(index), " ".join(f"group{index}token{token}" for token in range(10)))
        for index in range(4)
    ]

    kept = dedup_by_text_similarity(candidates, threshold=1.0)

    assert len(kept) == len(candidates)
    assert calls == len(candidates)


def test_approximate_comparisons_are_limited_to_leading_candidates(monkeypatch):
    comparisons = 0
    original = dedup_module._jaccard

    def counting_jaccard(left, right):
        nonlocal comparisons
        comparisons += 1
        return original(left, right)

    monkeypatch.setattr(dedup_module, "_jaccard", counting_jaccard)
    candidates = [
        item(str(index), " ".join(f"candidate{index}token{token}" for token in range(10))) for index in range(3)
    ]

    kept = dedup_by_text_similarity(candidates, threshold=0.6, max_candidates=2)

    assert [entry.id for entry in kept] == ["0", "1", "2"]
    assert comparisons == 1


def test_exact_duplicate_in_uncompared_tail_is_still_folded():
    candidates = [
        item("0", _COURSE_A),
        item("1", _COMMUNITY),
        item("2", _COURSE_A),
    ]

    kept = dedup_by_text_similarity(candidates, threshold=0.6, max_candidates=2)

    assert [entry.id for entry in kept] == ["0", "1"]
