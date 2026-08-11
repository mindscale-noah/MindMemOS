from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest
from experiments import experience_memory_embedding as SCRIPT
from mindmemos_skill.algos.evolve.skill_grpo_without_replay_buffer.contracts import (
    ExperienceSource,
    ReplayFreeExtractedExperience,
)
from mindmemos_skill.typing import Task


def task(task_id: str, descriptor: str) -> Task:
    return Task(
        task_id=task_id,
        instruction="complete the task",
        metadata={
            "task_type": descriptor.split("-", 1)[0],
            "gamefile": f"json_2.1.1/train/{descriptor}/trial/game.tw-pddl",
        },
    )


def experience_set() -> ReplayFreeExtractedExperience:
    content = {
        "experiences": [
            {
                "topic": "transform before delivery",
                "lesson": "Apply the required transformation before final placement.",
                "reason": "Successful attempts transformed the held item before moving it.",
                "evidence": [
                    {"task_id": "train:heat", "rollout": 2, "observation": "heated, then moved"},
                ],
            },
            {
                "topic": "exact action syntax",
                "lesson": "Use the action form exposed by the environment.",
                "reason": "Unsupported alternatives produced no progress.",
                "evidence": [],
            },
        ]
    }
    return ReplayFreeExtractedExperience(
        task_id="success-mini-batch-1",
        task_ids=["train:heat", "train:place"],
        source=ExperienceSource.SUCCESS,
        content=json.dumps(content),
        rollout_count=2,
    )


def test_atomic_experiences_preserves_current_payload_and_provenance() -> None:
    tasks = [
        task("train:heat", "pick_heat_then_place_in_recep-Mug-None-Cabinet-1"),
        task("train:place", "pick_and_place_simple-Book-None-Shelf-1"),
    ]
    bank = SCRIPT.atomic_experiences([experience_set()], tasks=tasks)

    assert len(bank) == 2
    assert bank[0].topic == "transform before delivery"
    assert bank[0].task_ids == ["train:heat"]
    assert "pick heat then place in recep" in bank[0].retrieval_document
    assert bank[1].task_ids == ["train:heat", "train:place"]
    assert bank[0].source is ExperienceSource.SUCCESS


def test_cli_defaults_to_requested_qwen_embedding_model(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT.__file__),
            "--data-root",
            str(tmp_path / "data"),
            "--split-dir",
            str(tmp_path / "split"),
            "--initial-skill",
            str(tmp_path / "SKILL.md"),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-id",
            "run",
        ],
    )

    args = SCRIPT.parse_args()

    assert args.embedding_model == "openai/qwen3-embedding-4b"
    assert args.top_k == 3
    assert args.train_rollouts == 4


def test_rendered_guidance_marks_retrieved_items_as_conditional() -> None:
    bank = SCRIPT.atomic_experiences(
        [experience_set()],
        tasks=[
            task("train:heat", "pick_heat_then_place_in_recep-Mug-None-Cabinet-1"),
            task("train:place", "pick_and_place_simple-Book-None-Shelf-1"),
        ],
    )
    record = SCRIPT.ExperienceRetrievalRecord(
        task_id="test:0",
        query="heat mug",
        embedding_model="openai/qwen3-embedding-4b",
        experiences=[SCRIPT.RetrievedExperience(rank=1, similarity=0.9, memory=bank[0])],
    )

    guidance = SCRIPT.render_retrieved_guidance(record)

    assert "conditional lessons" in guidance
    assert "use, adapt, or ignore" in guidance
    assert "Apply the required transformation before final placement" in guidance
    assert "evidence" not in guidance.lower()


class FakeEmbeddingModel:
    async def embed(self, task, text, **kwargs):
        del task, kwargs
        texts = [text] if isinstance(text, str) else text
        return {
            "embeddings": [[1.0, 0.0] if "transform" in value or "heat" in value else [0.0, 1.0] for value in texts]
        }


@pytest.mark.asyncio
async def test_embedding_retrieval_uses_cosine_top_k_and_persists_vectors() -> None:
    tasks = [
        task("train:heat", "pick_heat_then_place_in_recep-Mug-None-Cabinet-1"),
        task("train:place", "pick_and_place_simple-Book-None-Shelf-1"),
    ]
    bank = SCRIPT.atomic_experiences([experience_set()], tasks=tasks)
    query_task = task("test:heat", "pick_heat_then_place_in_recep-Egg-None-CounterTop-9")

    records = await SCRIPT.retrieve_experiences(
        [query_task],
        bank,
        embedding_model=FakeEmbeddingModel(),
        embedding_model_name="openai/qwen3-embedding-4b",
        embedding_batch_size=1,
        top_k=1,
    )

    assert records[0].embedding_model == "openai/qwen3-embedding-4b"
    assert records[0].experiences[0].memory.topic == "transform before delivery"
    assert records[0].experiences[0].similarity == pytest.approx(1.0)
    assert all(item.embedding for item in bank)


def test_recovers_experience_sets_from_recorded_llm_calls(tmp_path: Path) -> None:
    database = tmp_path / "state.db"
    request = {
        "messages": [
            {"role": "system", "content": "extract"},
            {
                "role": "user",
                "content": "## Task trajectory 1\n\n### Task ID\n\ntrain:0001\n\n### Task\n\nDo it",
            },
        ]
    }
    content = {"experiences": [{"topic": "syntax", "lesson": "Use exact verbs.", "reason": "Observed."}]}
    response = {"choices": [{"message": {"content": json.dumps(content)}}]}
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE llm_calls (task TEXT, request TEXT, response TEXT, status TEXT, started_at TEXT)"
        )
        connection.execute(
            "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?)",
            (
                "skill_grpo.experience.success.0",
                json.dumps(request),
                json.dumps(response),
                "succeeded",
                "2026-08-11T00:00:00Z",
            ),
        )

    recovered = SCRIPT.recover_experience_sets(database)

    assert len(recovered) == 1
    assert recovered[0].source is ExperienceSource.SUCCESS
    assert recovered[0].task_ids == ["train:0001"]
    assert recovered[0].rollout_count == 1
    assert json.loads(recovered[0].content) == content
