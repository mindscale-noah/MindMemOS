"""Unit tests for PersonaMem scope isolation and answer extraction."""

from __future__ import annotations

from typing import Any

import pytest

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


def test_scope_user_id_derived_from_persona():
    scope = build_personamem_scope("ctx-A", 5, "3")
    assert scope.user_id == "personamem-3"


def test_scope_session_id_derived_from_shared_context():
    scope = build_personamem_scope("ctx-A", 5, "3")
    assert scope.session_id == "personamem-ctx-A"


def test_same_persona_different_contexts_share_user_id():
    """Two contexts of the same persona share user_id but have different session_ids."""
    scope1 = build_personamem_scope("ctx-A", 5, "3")
    scope2 = build_personamem_scope("ctx-B", 10, "3")
    assert scope1.user_id == scope2.user_id == "personamem-3"
    assert scope1.session_id != scope2.session_id
    assert scope1.session_id == "personamem-ctx-A"
    assert scope2.session_id == "personamem-ctx-B"


def test_different_personas_different_user_ids():
    scope1 = build_personamem_scope("ctx-A", 5, "3")
    scope2 = build_personamem_scope("ctx-A", 5, "7")
    assert scope1.user_id != scope2.user_id


def test_scope_id_unique_per_context_and_end_index():
    scope1 = build_personamem_scope("ctx-A", 5, "3")
    scope2 = build_personamem_scope("ctx-A", 10, "3")
    scope3 = build_personamem_scope("ctx-B", 5, "3")
    assert scope1.scope_id != scope2.scope_id
    assert scope1.scope_id != scope3.scope_id


def test_37_contexts_produce_37_distinct_session_ids():
    """The benchmark has 37 shared_contexts; each must map to a unique session_id."""
    session_ids = {f"personamem-ctx-{i:02d}" for i in range(37)}
    scopes = [build_personamem_scope(f"ctx-{i:02d}", 100, "0") for i in range(37)]
    assert {s.session_id for s in scopes} == session_ids
    assert len({s.session_id for s in scopes}) == 37


# ---------- _extract_predicted_option ----------


def test_extract_pattern1_with_closing_tag():
    assert _extract_predicted_option("<final_answer>(a)</final_answer>") == "a"
    assert _extract_predicted_option("<final_answer>(b)</final_answer>") == "b"
    assert _extract_predicted_option("<final_answer>c</final_answer>") == "c"


def test_extract_pattern1_rejects_word_after_tag():
    # apple should NOT be matched as 'a' (would have been a false positive without closing-tag requirement)
    assert _extract_predicted_option("<final_answer>apple</final_answer>") is None


def test_extract_pattern2_after_token_no_closing_tag():
    assert _extract_predicted_option("<final_answer> the answer is (b)") == "b"


def test_extract_pattern3_before_token():
    assert _extract_predicted_option("(c) because blah <final_answer>") == "c"


def test_extract_pattern4_no_token():
    assert _extract_predicted_option("I choose (d)") == "d"


def test_extract_empty_response():
    assert _extract_predicted_option("") is None
    assert _extract_predicted_option(None) is None  # type: ignore[arg-type]


def test_extract_no_option_in_response():
    assert _extract_predicted_option("I don't know the answer") is None
