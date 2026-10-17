"""Mem-Gallery adapter offline: build, retrieval, Mem-Gallery answer protocol, scoring and metrics."""

import argparse
import io
import json
from types import SimpleNamespace

import pytest
from mindmemos_eval.llm import LLMCompletion
from mindmemos_eval.memory.base import BenchmarkSpec, RunContext, RunnerConfig
from mindmemos_eval.memory.envs.memgallery import MemGalleryAdapter
from mindmemos_eval.memory.envs.memgallery.scoring import f1, parse_judge
from mindmemos_eval.memory.identity import new_identity
from mindmemos_eval.memory.runner import default_adapters


def _png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return buffer.getvalue()


PNG = _png()
BLOB, OTHER_BLOB = "b" * 64, "c" * 64


def _dataset(root):
    (root / "dialog").mkdir(parents=True)
    (root / "image" / "T").mkdir(parents=True)
    for name in ("D1_IMG_001.jpg", "QA_IMG_001.png"):
        (root / "image" / "T" / name).write_bytes(PNG)
    data = {
        "character_profile": {"name": "Ann Lee"},
        "multi_session_dialogues": [
            {
                "session_id": "D1",
                "date": "2024-06-17",
                "dialogues": [
                    {"round": "D1:1", "user": "Hi", "assistant": "Hello."},
                    {
                        "round": "D1:2",
                        "user": "This is my robot tutor",
                        "assistant": "A robot tutor.",
                        "image_id": ["D1:IMG_001"],
                        "input_image": ["../image/T/D1_IMG_001.jpg"],
                        "image_caption": ["gold caption, never sent"],
                    },
                ],
            }
        ],
        "human-annotated QAs": [
            {"point": "VS", "question": "Which image shows a robot tutor?", "answer": "D1:IMG_001"},
            {"point": "AR", "question": "Did Ann mention Mars?", "answer": "Not mentioned."},
            {
                "point": "TTL",
                "question": "What is in the picture?",
                "question_image": "../image/T/QA_IMG_001.png",
                "answer": "running robots",
            },
        ],
    }
    (root / "dialog" / "T.json").write_text(json.dumps(data))


def _item(id_, memory, memory_type, metadata):
    return SimpleNamespace(
        id=id_,
        memory=memory,
        memory_type=memory_type,
        metadata=metadata,
        event_time="2024-06-17 00:01:00",
        source_timestamp=None,
        last_update_at="x",
    )


class FakeMemory:
    def __init__(self):
        self.adds, self.searches = [], []

    async def add(self, messages, **kwargs):
        self.adds.append((messages, kwargs))

    async def search(self, query, **kwargs):
        self.searches.append((query, kwargs))
        return SimpleNamespace(
            memories=[
                _item("m1", "User showed a robot tutor", "fact", {"image_blob_ids": [BLOB]}),
                _item("m2", "User likes tea", "fact", {"image_blob_ids": [OTHER_BLOB]}),
                _item("v1", "A white robot at a desk", "visual_fact", {"blob_id": BLOB}),
                _item("i1", "[image] robot tutor", "image", {"blob_id": BLOB, "caller_image_ids": ["D1:IMG_001"]}),
            ]
        )

    async def get_image(self, blob_id):
        return PNG, "image/png"


class FakeLlm:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        text = json.dumps(messages, ensure_ascii=False)
        return LLMCompletion(content=next(v for k, v in self.answers.items() if k in text), total_tokens=10)


async def _run(tmp_path, *, send_images=True):
    _dataset(tmp_path / "data")
    memory = FakeMemory()
    answer = FakeLlm({"Return the image_id": "D1:IMG_001", "Not mentioned": "Not mentioned.", "QUESTION": "run robot"})
    # 0.75 snaps to 1 and 0.25 to 0.5, as the benchmark's judge parser does.
    judge = FakeLlm({"Mars": '{"score": 0.25, "reasoning": "x"}', "Question:": '{"score": 0.75, "reasoning": "ok"}'})
    spec = BenchmarkSpec(
        name="memgallery",
        dataset=str(tmp_path / "data"),
        memory_algorithm="vanilla",
        raw={"output_root": str(tmp_path / "runs")},
        execution_params={"send_images": send_images},
        search_params={"multimodal": True},
    )
    args = argparse.Namespace(runner_config=RunnerConfig(), limit=None)
    ctx = RunContext(identity=new_identity("memgallery", "vanilla"))
    result = await MemGalleryAdapter().run(
        memory=memory, answer_llm=answer, judge_llm=judge, ctx=ctx, bench_config=spec, args=args
    )
    return result, memory, answer, judge


@pytest.mark.asyncio
async def test_build_search_answer_and_metrics(tmp_path):
    result, memory, answer, judge = await _run(tmp_path)

    [(messages, kwargs)] = memory.adds
    assert kwargs == {"user_id": "memgallery-t", "mode": "sync", "session_id": "D1"}
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    image = messages[2].images[0]
    assert image.image_id == "D1:IMG_001" and image.caption is None and image.path.endswith("D1_IMG_001.jpg")
    assert "gold caption" not in json.dumps([m.model_dump() for m in messages])

    queries = {query: kwargs for query, kwargs in memory.searches}
    assert set(queries) == {"Which image shows a robot tutor?", "Did Ann mention Mars?", "What is in the picture?"}
    assert queries["Did Ann mention Mars?"] == {
        "user_id": "memgallery-t",
        "top_k": 20,
        "rerank": True,
        "search_strategy": "fast",
        "multimodal": True,
    }
    assert queries["What is in the picture?"]["query_images"][0].path.endswith("QA_IMG_001.png")

    vs = next(call for call in answer.calls if "Which image shows" in json.dumps(call[0]))
    system, user = vs[0][0]["content"], json.dumps(vs[0][1]["content"])
    assert system.startswith("You are an AI assistant evaluated on multimodal long-term conversational memory")
    assert vs[1]["max_tokens"] == 256
    assert "between user (Ann Lee) and assistant" in user and "Return the image_id of the image(s)" in user
    assert "User showed a robot tutor (image_id: D1:IMG_001)" in user  # anchored fact
    assert "A white robot at a desk (image_id: D1:IMG_001)" in user  # visual fact
    assert "User likes tea" in user and "User likes tea (image_id" not in user  # image not returned: no id
    assert "Image of memory [4] (image_id: D1:IMG_001): [image] robot tutor" in user and "image_url" in user

    # The judge sees the question as asked, not the answer prompt.
    judged = [json.dumps(call[0]) for call in judge.calls]
    assert all("Your task is to answer" not in text for text in judged)

    metrics = result["metrics"]
    assert result["valid"] and metrics["n"] == 3 and metrics["unjudged"] == 0
    assert metrics["Judge"] == pytest.approx(100 * (1 + 0.5 + 1) / 3, abs=0.01)
    assert metrics["F1"] == pytest.approx(100.0, abs=0.01)  # "run robot" vs "running robots" after stemming
    assert metrics["per_point"]["AR"]["Judge"] == 50.0 and metrics["per_task"]["T"]["n"] == 3
    assert any((tmp_path / "runs").rglob("summary.json"))


@pytest.mark.asyncio
async def test_text_only_baseline_sends_no_images(tmp_path):
    _, memory, _, _ = await _run(tmp_path, send_images=False)
    [(messages, _)] = memory.adds
    assert all(not m.images for m in messages)
    assert all("query_images" not in kwargs for _, kwargs in memory.searches)


def test_registered():
    assert isinstance(default_adapters()["memgallery"], MemGalleryAdapter)


def test_scoring_helpers():
    assert f1("Running dogs.", "the run dog") == 1.0
    assert f1("D1:IMG_002, D1:IMG_003", "D1:IMG_002") == pytest.approx(2 / 3)
    assert f1("", "x") == 0.0
    assert parse_judge('{"score": 0.75, "reasoning": "x"}') == 1.0
    assert parse_judge('{"score": 0.25}') == 0.5
    assert parse_judge('{"score": 0}') == 0.0
    assert parse_judge("score: 0.6") == 0.5
    assert parse_judge("no score here") is None
