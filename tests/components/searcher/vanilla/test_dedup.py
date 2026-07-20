"""Tests for near-duplicate folding of search candidates."""

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
