"""Regression tests for PersonaMem answer parsing and retry accounting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mindmemos_eval.llm import LLMCompletion
from mindmemos_eval.memory.envs.personamem import env as personamem_env
from mindmemos_eval.memory.envs.personamem.env import (
    PersonaMemEnv,
    PersonaMemItem,
    PersonaMemScope,
    _extract_predicted_option,
)


def _item(question_id: str = "question-1") -> PersonaMemItem:
    return PersonaMemItem(
        index=0,
        persona_id="persona",
        question_id=question_id,
        question_type="suggest_new_ideas",
        topic="music",
        question="What should I try?",
        correct_answer="(a)",
        all_options='["(a) x", "(b) y", "(c) z", "(d) w"]',
        scope=PersonaMemScope(
            shared_context_id="context",
            end_index=0,
            scope_id="context:0",
            user_id="user",
            session_id="session",
        ),
    )


class _FakeAnswerLlm:
    def __init__(self, responses: list[str]) -> None:
        self.config = SimpleNamespace(model="test-model")
        self._responses = iter(responses)
        self.prompts: list[list[dict[str, object]]] = []

    async def complete(self, prompt):
        self.prompts.append(prompt)
        return LLMCompletion(
            content=next(self._responses),
            prompt_tokens=2,
            completion_tokens=3,
            total_tokens=5,
        )


def _env(answer_llm: _FakeAnswerLlm) -> PersonaMemEnv:
    return PersonaMemEnv(
        object(),
        answer_llm=answer_llm,
        context_store=SimpleNamespace(visible=lambda scope: []),
        evaluation_mode="official_full_context",
    )


class _FakeMemory:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []

    async def search(self, query: str, **kwargs: object) -> SimpleNamespace:
        self.search_calls.append({"query": query, **kwargs})
        return SimpleNamespace(memories=[])


def _memory_rag_env(memory: _FakeMemory, answer_llm: _FakeAnswerLlm) -> PersonaMemEnv:
    return PersonaMemEnv(
        memory,
        answer_llm=answer_llm,
        context_store=SimpleNamespace(visible=lambda scope: []),
        evaluation_mode="memory_rag",
    )


def test_extracts_last_complete_final_answer_tag() -> None:
    response = "Draft <final_answer>(a)</final_answer>. Correction <final_answer>(c)</final_answer>."

    assert _extract_predicted_option(response) == "c"


def test_extracts_parenthesized_option_after_prose_inside_final_answer_tag() -> None:
    assert _extract_predicted_option("<final_answer>Answer: (b)</final_answer>") == "b"


def test_accepts_option_after_final_answer_token_without_closing_tag() -> None:
    assert _extract_predicted_option("<final_answer> (a)") == "a"


def test_rejects_unformatted_option_letters_in_prose() -> None:
    assert _extract_predicted_option("The answer is c, and that is a fact.") is None


@pytest.mark.asyncio
async def test_memory_rag_search_filters_to_the_current_scope_user() -> None:
    memory = _FakeMemory()
    item = _item()

    result = await _memory_rag_env(memory, _FakeAnswerLlm(["<final_answer>(a)</final_answer>"]))._answer_item(
        item, build_error=None
    )

    assert result.answer is not None
    assert memory.search_calls == [
        {
            "query": item.question,
            "user_id": item.scope.user_id,
            "session_id": item.scope.session_id,
            "top_k": 50,
            "search_strategy": "fast",
            "rerank": False,
            "filters": {"user_id": item.scope.user_id},
        }
    ]


@pytest.mark.asyncio
async def test_retry_records_final_prompt_and_actual_llm_usage() -> None:
    llm = _FakeAnswerLlm(["I choose c", "<final_answer>(a)</final_answer>"])

    result = await _env(llm)._answer_item(_item(), build_error=None)

    assert result.answer is not None
    assert result.answer.is_correct is True
    assert result.answer.llm_calls == 2
    assert result.answer.total_tokens == 10
    assert len(result.prompt) == 3
    assert "previous response" in result.prompt[-1]["content"]


@pytest.mark.asyncio
async def test_unparseable_answers_are_scored_as_failures() -> None:
    first = await _env(_FakeAnswerLlm(["no answer"] * 3))._answer_item(_item(), build_error=None)
    second = await _env(_FakeAnswerLlm(["no answer"] * 3))._answer_item(_item(), build_error=None)

    assert first.answer is not None
    assert second.answer is not None
    assert first.answer.parse_failed is True
    assert first.answer.extracted_answer == ""
    assert first.answer.is_correct is False
    assert second.answer.parse_failed is True
    assert first.answer.llm_calls == 3


class _FakeProgress:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.postfixes: list[dict[str, object]] = []
        self.closed = False

    def update(self, amount: int = 1) -> None:
        self.updates.append(amount)

    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        self.postfixes.append(dict(ordered_dict or kwargs))

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_run_dataset_displays_live_personamem_score(monkeypatch) -> None:
    progress_instances: list[_FakeProgress] = []

    def fake_tqdm(**kwargs) -> _FakeProgress:
        progress = _FakeProgress(**kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr(personamem_env, "tqdm", fake_tqdm)
    result = await _env(
        _FakeAnswerLlm(["<final_answer>(a)</final_answer>", "no answer", "no answer", "no answer"])
    ).run_dataset(
        [_item("one"), _item("two")],
        show_progress=True,
    )

    progress = progress_instances[0]
    assert progress.updates == [1, 1]
    assert progress.postfixes[-1] == {
        "correct": 1,
        "acc_done": "0.5000",
        "acc_all": "0.5000",
        "parse_fail": 1,
        "search_fail": 0,
        "answer_fail": 0,
    }
    assert result.metrics["overall_accuracy"] == 0.5
