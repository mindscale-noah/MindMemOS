"""MemEye adapter offline: build, retrieval, shared answerer, scoring and metrics."""

import argparse
import io
import json
from types import SimpleNamespace

import pytest
from mindmemos_eval.llm import LLMCompletion
from mindmemos_eval.memory.base import BenchmarkSpec, RunContext, RunnerConfig
from mindmemos_eval.memory.envs.memeye import MemEyeAdapter
from mindmemos_eval.memory.envs.memeye.scoring import bleu1, extract_choice, parse_judge
from mindmemos_eval.memory.identity import new_identity


def _png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()


def _dataset(root):
    (root / "dialog").mkdir(parents=True)
    (root / "image" / "T").mkdir(parents=True)
    (root / "image" / "T" / "a.png").write_bytes(PNG)
    sessions = [
        {
            "session_id": "S1",
            "date": "2024-01-05",
            "dialogues": [
                {
                    "round": "S1:R1",
                    "user": "Here is my red sofa",
                    "assistant": "Nice red sofa.",
                    "input_image": ["T/a.png"],
                    "image_id": ["IMG1"],
                    "image_caption": ["gold caption"],
                },
                {"round": "S1:R2", "user": "", "assistant": "", "input_image": ["../image/T/a.png"]},
            ],
        },
    ]
    mcq = {
        "multi_session_dialogues": sessions,
        "human-annotated QAs": [
            {
                "question_id": "Q1",
                "question": "Sofa color?",
                "answer": "A",
                "options": [{"A": "red", "B": "blue", "answer": "A"}, {"A": "blue", "B": "red", "answer": "B"}],
            },
        ],
    }
    open_ = {
        "multi_session_dialogues": sessions,
        "human-annotated QAs": [{"question_id": "Q1", "question": "Sofa color?", "answer": "red"}],
    }
    (root / "dialog" / "T.json").write_text(json.dumps(mcq))
    (root / "dialog" / "T_Open.json").write_text(json.dumps(open_))


class FakeMemory:
    def __init__(self):
        self.adds, self.searches = [], []

    async def add(self, messages, **kwargs):
        self.adds.append((messages, kwargs))

    async def search(self, query, **kwargs):
        self.searches.append((query, kwargs))
        return SimpleNamespace(
            memories=[
                SimpleNamespace(
                    id="m1",
                    memory="User has a red sofa",
                    memory_type="fact",
                    metadata={},
                    event_time="2024-01-05 00:00:00",
                    source_timestamp=None,
                    last_update_at="x",
                ),
                SimpleNamespace(
                    id="i1",
                    memory="[image] user: Here is my red sofa",
                    memory_type="image",
                    metadata={"blob_id": "b" * 64, "user_id": "user-1", "score": 0.2},
                    event_time=None,
                    source_timestamp=None,
                    last_update_at="x",
                ),
            ]
        )

    async def get_image(self, blob_id, user_id=None):
        return PNG, "image/png"


class FakeLlm:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    async def complete(self, messages, **kwargs):
        self.calls.append(messages)
        text = json.dumps(messages)
        answer = next(value for key, value in self.answers.items() if key in text)
        return LLMCompletion(content=answer, total_tokens=10)


async def _run(tmp_path, *, send_images=True, search_params=None, runner=None):
    _dataset(tmp_path / "data")
    memory = FakeMemory()
    # MCQ rotation 1: red is A; rotation 2: red is B; the fake answerer always says A -> EM 0.5.
    answer = FakeLlm({"Answer with only the option letter": "A", "QUESTION": "red"})
    judge = FakeLlm({"Ground Truth": '{"score": 1, "reasoning": "ok"}'})
    spec = BenchmarkSpec(
        name="memeye",
        dataset=str(tmp_path / "data"),
        memory_algorithm="vanilla",
        raw={"tasks": ["T"], "output_root": str(tmp_path / "runs")},
        execution_params={"send_images": send_images},
        search_params=dict(search_params or {}),
    )
    args = argparse.Namespace(runner_config=runner or RunnerConfig(), limit=None)
    ctx = RunContext(identity=new_identity("memeye", "vanilla"))
    result = await MemEyeAdapter().run(
        memory=memory, answer_llm=answer, judge_llm=judge, ctx=ctx, bench_config=spec, args=args
    )
    return result, memory, answer, judge


@pytest.mark.asyncio
async def test_images_are_sent_and_the_answerer_sees_them(tmp_path):
    result, memory, answer, _ = await _run(tmp_path, search_params={"multimodal": True})

    [(messages, kwargs)] = memory.adds
    assert kwargs["session_id"] == "S1"
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[0].images[0].image_id == "IMG1" and messages[0].images[0].caption is None
    assert messages[0].timestamp < messages[2].timestamp
    assert result["metrics"]["EM"] == 50.0 and result["metrics"]["Judge"] == 100.0
    assert result["metrics"]["B-1"] == 100.0
    assert "image_url" in json.dumps(answer.calls[0])
    assert result["cost"]["add"]["calls"] == 1 and result["cost"]["search"]["calls"] == 2
    # Runner defaults apply (no hardcoded top_k); the manifest contract fields metrics.py
    # expects (locomo names) are present.
    assert memory.searches[0][1] == {
        "user_id": "memeye-t",
        "top_k": 50,
        "rerank": True,
        "search_strategy": "fast",
        "multimodal": True,
    }
    assert result["metrics"]["overall_accuracy"] == 0.5
    assert result["metrics"]["total_questions"] == 2
    assert result["metrics"]["judge_runs_used"] == 1
    assert result["metrics"]["answer_llm_calls"] == result["cost"]["answer_calls"]
    assert result["metrics"]["judge_total_tokens"] == result["cost"]["judge_tokens"]
    assert "answer_prompt_tokens" in result["cost"]


@pytest.mark.asyncio
async def test_runner_top_k_and_rerank_apply(tmp_path):
    # CLI/runner-level --top-k/--rerank reach the search call instead of a hardcoded default.
    result, memory, _, _ = await _run(tmp_path, runner=RunnerConfig(top_k=7, rerank=False))
    assert memory.searches[0][1]["top_k"] == 7 and memory.searches[0][1]["rerank"] is False
    assert result["valid"]


@pytest.mark.asyncio
async def test_spec_search_params_win_over_runner_defaults(tmp_path):
    # Benchmark/profile search_params still win over the runner defaults.
    _, memory, _, _ = await _run(
        tmp_path, runner=RunnerConfig(top_k=7, rerank=False), search_params={"multimodal": True, "top_k": 9}
    )
    assert memory.searches[0][1]["top_k"] == 9 and memory.searches[0][1]["rerank"] is False


@pytest.mark.asyncio
async def test_agentic_search_param_maps_to_agentic_strategy(tmp_path):
    # `agentic: true` is a supported search_params key on every benchmark; dropping it silently
    # would mislabel an enhanced_agentic ablation as fast retrieval.
    _, memory, _, _ = await _run(tmp_path, search_params={"agentic": True})
    assert memory.searches[0][1]["search_strategy"] == "agentic"


@pytest.mark.asyncio
async def test_judge_runs_average_scores(tmp_path):
    result, _, _, judge = await _run(tmp_path, runner=RunnerConfig(judge_runs=3))
    # One OPEN question judged three times, all scores 1.0.
    assert len(judge.calls) == 3
    assert result["metrics"]["Judge"] == 100.0 and result["metrics"]["judge_runs_used"] == 3


@pytest.mark.asyncio
async def test_text_only_baseline_sends_no_images(tmp_path):
    _, memory, _, _ = await _run(tmp_path, send_images=False)
    [(messages, _)] = memory.adds
    assert [m.role for m in messages] == ["user", "assistant"]
    assert all(not m.images for m in messages)
    assert all("query_images" not in kwargs for _, kwargs in memory.searches)


def test_scoring_helpers():
    assert extract_choice("B.", {"A", "B"}) == "B"
    assert extract_choice("The answer is A", {"A", "B"}) == "A"
    assert extract_choice("none", {"A", "B"}) == "INVALID"
    assert parse_judge('{"score": 0.75, "reasoning": "x"}') == 0.75
    assert parse_judge("score: 0.3") is None
    assert bleu1("The red sofa", "red sofa") == 1.0
