"""Unit tests for PersonaMem scope isolation and answer extraction."""

from __future__ import annotations

from mindmemos_eval.memory.envs.personamem.env import (
    PersonaMemItem,
    PersonaMemScope,
    _extract_predicted_option,
    build_personamem_scope,
)


def _scope(ctx_id: str, end_index: int, persona_id: str = "0") -> PersonaMemScope:
    return build_personamem_scope(ctx_id, end_index, persona_id)


def _item(index: int, ctx_id: str, end_index: int, persona_id: str = "0") -> PersonaMemItem:
    return PersonaMemItem(
        index=index,
        persona_id=persona_id,
        question_id=f"q-{index}",
        question_type="recall",
        topic="test",
        question=f"question {index}",
        correct_answer="(a)",
        all_options="(a)\n(b)\n(c)\n(d)",
        scope=_scope(ctx_id, end_index, persona_id),
    )


# ---------- build_personamem_scope: session_id isolation ----------


def test_scope_user_id_derived_from_shared_context():
    scope = build_personamem_scope("ctx-A", 5, "3")
    assert scope.user_id == "personamem-ctx-A"


def test_scope_session_id_derived_from_shared_context():
    scope = build_personamem_scope("ctx-A", 5, "3")
    assert scope.session_id == "personamem-ctx-A"


def test_different_contexts_are_independent_scopes():
    """Each shared_context is an independent memory scope — user_id and session_id are both context-scoped."""
    scope1 = build_personamem_scope("ctx-A", 5, "3")
    scope2 = build_personamem_scope("ctx-B", 10, "3")
    assert scope1.user_id == "personamem-ctx-A"
    assert scope2.user_id == "personamem-ctx-B"
    assert scope1.user_id != scope2.user_id
    assert scope1.session_id == "personamem-ctx-A"
    assert scope2.session_id == "personamem-ctx-B"
    assert scope1.session_id != scope2.session_id


def test_same_context_same_user_id_regardless_of_persona():
    """user_id is derived from shared_context_id only, not persona_id."""
    scope1 = build_personamem_scope("ctx-A", 5, "3")
    scope2 = build_personamem_scope("ctx-A", 5, "7")
    assert scope1.user_id == scope2.user_id == "personamem-ctx-A"


def test_scope_id_unique_per_context_and_end_index():
    scope1 = build_personamem_scope("ctx-A", 5, "3")
    scope2 = build_personamem_scope("ctx-A", 10, "3")
    scope3 = build_personamem_scope("ctx-B", 5, "3")
    assert scope1.scope_id != scope2.scope_id
    assert scope1.scope_id != scope3.scope_id


def test_37_contexts_produce_37_distinct_user_and_session_ids():
    """Each of the 37 shared_contexts must map to a unique user_id and session_id."""
    expected_ids = {f"personamem-ctx-{i:02d}" for i in range(37)}
    scopes = [build_personamem_scope(f"ctx-{i:02d}", 100, "0") for i in range(37)]
    assert {s.session_id for s in scopes} == expected_ids
    assert {s.user_id for s in scopes} == expected_ids
    assert len({s.user_id for s in scopes}) == 37


# ---------- _extract_predicted_option ----------


def test_extract_accepts_parenthesized_option_in_tag():
    assert _extract_predicted_option("<final_answer>(a)</final_answer>") == "a"
    assert _extract_predicted_option("<final_answer>(b)</final_answer>") == "b"
    assert _extract_predicted_option("<final_answer>(d)</final_answer>") == "d"


def test_extract_accepts_bare_option_in_tag():
    assert _extract_predicted_option("<final_answer>c</final_answer>") == "c"


def test_extract_accepts_surrounding_whitespace_in_tag():
    assert _extract_predicted_option("<final_answer>  (a)  </final_answer>") == "a"
    assert _extract_predicted_option("<final_answer>\n(b)\n</final_answer>") == "b"


def test_extract_rejects_word_after_option_in_tag():
    # "apple" must NOT be matched as 'a'.
    assert _extract_predicted_option("<final_answer>apple</final_answer>") is None


def test_extract_rejects_reasoning_inside_tag():
    # The two reported silent-misparse regressions: the first option letter
    # appearing inside the tag must NOT be read as the answer.
    assert _extract_predicted_option(
        "<final_answer>This is a difficult choice; option c is best</final_answer>"
    ) is None
    assert _extract_predicted_option(
        "<final_answer>Between (b) and (c), I choose (c)</final_answer>"
    ) is None


def test_extract_rejects_trailing_punctuation_in_tag():
    assert _extract_predicted_option("<final_answer>(a).</final_answer>") is None
    assert _extract_predicted_option("<final_answer>a.</final_answer>") is None


def test_extract_rejects_option_before_tag():
    # An option placed before/outside the tag is not the tag's content.
    assert _extract_predicted_option("(c)<final_answer>") is None
    assert _extract_predicted_option("c <final_answer>") is None


def test_extract_rejects_option_after_open_tag_without_close():
    assert _extract_predicted_option("<final_answer> the answer is (b)") is None


def test_extract_rejects_reasoning_text_before_tag():
    """(c) separated from <final_answer> by other text is reasoning, not an answer."""
    assert _extract_predicted_option("(c) because blah <final_answer>") is None


def test_extract_no_tag_no_guess():
    """Without <final_answer>, return None - don't guess from reasoning text."""
    assert _extract_predicted_option("I choose (d)") is None


def test_extract_empty_response():
    assert _extract_predicted_option("") is None
    assert _extract_predicted_option(None) is None  # type: ignore[arg-type]


def test_extract_no_option_in_response():
    assert _extract_predicted_option("I don't know the answer") is None
