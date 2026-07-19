"""Tests for the single question-type-agnostic PersonaMem answering prompt."""

from __future__ import annotations

from mindmemos_eval.memory.envs.personamem.env import (
    PersonaMemItem,
    PersonaMemScope,
    build_personamem_prompt,
)


def _item(question_type: str) -> PersonaMemItem:
    return PersonaMemItem(
        index=0,
        persona_id="0",
        question_id="q",
        question_type=question_type,
        topic="music",
        question="What changed?",
        correct_answer="(a)",
        all_options='["(a) x", "(b) y", "(c) z", "(d) w"]',
        scope=PersonaMemScope(
            shared_context_id="ctx",
            end_index=1,
            scope_id="ctx:1",
            user_id="u",
            session_id="u",
        ),
    )


def test_memory_rag_uses_single_unified_prompt() -> None:
    prompt = build_personamem_prompt(
        _item("suggest_new_ideas"),
        retrieved_memories=["(2026-01-05) User liked jazz."],
    )
    # One user message containing the universal-rules prompt; no system/minimal pair.
    assert [m["role"] for m in prompt] == ["user"]
    assert "UNIVERSAL RULES" in prompt[0]["content"]
    assert "(2026-01-05) User liked jazz." in prompt[0]["content"]


def test_prompt_is_identical_across_question_types() -> None:
    # The question_type label is never read: the instruction body is the same.
    a = build_personamem_prompt(_item("track_full_preference_evolution"), retrieved_memories=["m"])
    b = build_personamem_prompt(_item("suggest_new_ideas"), retrieved_memories=["m"])
    assert a[0]["content"] == b[0]["content"]
