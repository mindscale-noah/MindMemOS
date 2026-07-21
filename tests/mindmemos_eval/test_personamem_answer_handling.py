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
    def __init__(self, responses: list[str | Exception]) -> None:
        self.config = SimpleNamespace(model="test-model")
        self._responses = iter(responses)
        self.prompts: list[list[dict[str, object]]] = []

    async def complete(self, prompt):
        self.prompts.append(prompt)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return LLMCompletion(
            content=response,
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


def test_rejects_multiple_options_inside_final_answer_tag() -> None:
    assert _extract_predicted_option("<final_answer>(a) or (c)</final_answer>") is None


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
    llm = _FakeAnswerLlm(["I considered (a) and (c).", "<final_answer>(a)</final_answer>"])

    result = await _env(llm)._answer_item(_item(), build_error=None)

    assert result.answer is not None
    assert result.answer.is_correct is True
    assert result.answer.llm_calls == 2
    assert result.answer.total_tokens == 10
    assert len(result.prompt) == 3
    assert "previous response" in result.prompt[-1]["content"]


@pytest.mark.asyncio
async def test_official_fallback_accepts_one_untagged_option_without_retry() -> None:
    env = _env(_FakeAnswerLlm(["The answer is (a)."]))
    result = await env.run_dataset(
        [_item()],
        show_progress=False,
    )

    answer = result.qa_results[0].answer
    assert answer is not None
    assert answer.extracted_answer == "a"
    assert answer.is_correct is True
    assert answer.llm_calls == 1
    assert answer.parse_failed is False
    assert answer.format_compliant is False
    assert result.metrics["answer_parse_failure_count"] == 0
    assert result.metrics["answer_format_failure_count"] == 1


@pytest.mark.asyncio
async def test_tagged_answer_is_reported_as_format_compliant() -> None:
    result = await _env(_FakeAnswerLlm(["<final_answer>(a)</final_answer>"])).run_dataset(
        [_item()],
        show_progress=False,
    )

    answer = result.qa_results[0].answer
    assert answer is not None
    assert answer.format_compliant is True
    assert result.metrics["answer_format_failure_count"] == 0


@pytest.mark.asyncio
async def test_retry_failure_preserves_usage_from_successful_completions() -> None:
    result = await _env(_FakeAnswerLlm(["no answer", ConnectionError("offline")])).run_dataset(
        [_item()],
        show_progress=False,
    )

    qa_result = result.qa_results[0]
    assert qa_result.error == "answer failed: ConnectionError: offline"
    assert qa_result.answer is not None
    assert qa_result.answer.response == "no answer"
    assert qa_result.answer.extracted_answer == ""
    assert qa_result.answer.is_correct is False
    assert qa_result.answer.parse_failed is False
    assert qa_result.answer.llm_calls == 1
    assert qa_result.answer.total_tokens == 5
    assert result.metrics["answer_llm_calls"] == 1
    assert result.metrics["answer_total_tokens"] == 5
    assert result.metrics["answer_failure_count"] == 1
    assert result.metrics["answer_parse_failure_count"] == 0


@pytest.mark.asyncio
async def test_initial_answer_failure_has_no_usage_to_preserve() -> None:
    result = await _env(_FakeAnswerLlm([ConnectionError("offline")])).run_dataset(
        [_item()],
        show_progress=False,
    )

    qa_result = result.qa_results[0]
    assert qa_result.error == "answer failed: ConnectionError: offline"
    assert qa_result.answer is None
    assert result.metrics["answer_llm_calls"] == 0
    assert result.metrics["answer_total_tokens"] == 0
    assert result.metrics["answer_failure_count"] == 1


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
