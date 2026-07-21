"""PersonaMem answer-prompt routing by memory algorithm: schema=CoT, vanilla=unified."""

from __future__ import annotations

from mindmemos_eval.memory.envs.personamem.env import (
    PERSONAMEM_COT_PROMPT,
    PERSONAMEM_UNIFIED_PROMPT,
    build_personamem_items,
    build_personamem_prompt,
    personamem_answer_prompt,
)


def _item():
    row = {
        "shared_context_id": "ctx-1",
        "end_index_in_shared_context": "0",
        "persona_id": "p1",
        "question_id": "q1",
        "question_type": "recommendation",
        "topic": "books",
        "user_question_or_message": "What should I read next?",
        "correct_answer": "(a)",
        "all_options": "(a) X (b) Y (c) Z (d) W",
    }
    return build_personamem_items([row])[0]


def test_answer_prompt_routes_schema_to_cot_and_vanilla_to_unified():
    assert personamem_answer_prompt("schema") is PERSONAMEM_COT_PROMPT
    assert personamem_answer_prompt("vanilla") is PERSONAMEM_UNIFIED_PROMPT


def test_only_schema_forces_cot_others_use_unified():
    assert personamem_answer_prompt("default") is PERSONAMEM_UNIFIED_PROMPT


def test_build_prompt_honors_selected_answer_prompt():
    item = _item()
    cot_text = build_personamem_prompt(item, retrieved_memories=["m1"], answer_prompt=PERSONAMEM_COT_PROMPT)[0][
        "content"
    ]
    unified_text = build_personamem_prompt(item, retrieved_memories=["m1"], answer_prompt=PERSONAMEM_UNIFIED_PROMPT)[0][
        "content"
    ]
    assert "Chain-of-Thought" in cot_text
    assert "UNIVERSAL RULES" in unified_text
    assert cot_text != unified_text


def test_build_prompt_defaults_to_cot():
    prompt = build_personamem_prompt(_item(), retrieved_memories=["m1"])
    assert "Chain-of-Thought" in prompt[0]["content"]
