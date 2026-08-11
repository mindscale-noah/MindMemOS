from __future__ import annotations

import json

import pytest
from mindmemos_skill.algos.evolve.trajectory_memory.contracts import TrajectorySnapshot
from mindmemos_skill.algos.evolve.trajectory_memory.memory import (
    TrajectoryMemoryBankBuilder,
    select_trajectory_snapshots,
    task_retrieval_key,
)
from mindmemos_skill.algos.evolve.trajectory_memory.prompts import render_retrieved_memories
from mindmemos_skill.typing import Task


class FakeChatModel:
    async def chat(self, task, messages, **kwargs):
        del task, kwargs
        request = json.loads(messages[-1]["content"])
        return {
            "content": json.dumps(
                {
                    "title": f"Plan for {request['task_id']}",
                    "task_summary": request["task_query"],
                    "strategy": "Follow the required transformation before delivery.",
                    "key_steps": ["find the exact object", "transform it", "deliver it"],
                    "transferable_lessons": ["Use only admissible actions."],
                    "cautions": ["Do not assume the same object location."],
                }
            )
        }


class FakeEmbeddingModel:
    async def embed(self, task, text, **kwargs):
        del task, kwargs
        texts = [text] if isinstance(text, str) else text
        return {
            "embeddings": [
                [1.0, 0.0] if "heat" in value else [0.0, 1.0]
                for value in texts
            ]
        }


def alfworld_task(task_id: str, descriptor: str) -> Task:
    task_type = descriptor.split("-", 1)[0]
    return Task(
        task_id=task_id,
        instruction=f"Complete {descriptor}",
        tags=["train"],
        metadata={
            "task_type": task_type,
            "gamefile": f"json_2.1.1/train/{descriptor}/trial/game.tw-pddl",
        },
    )


def snapshot(task: Task, rollout_id: str, reward: float, turns: int) -> TrajectorySnapshot:
    return TrajectorySnapshot(
        task=task,
        rollout_id=rollout_id,
        query="put a hot mug in coffeemachine",
        events=[
            {"role": "user", "content": "Your task is to: put a hot mug in coffeemachine."},
            {"role": "assistant", "content": "<think>heat first</think><action>heat mug 1</action>"},
        ],
        reward_score=reward,
        n_turn=turns,
    )


def test_select_trajectory_snapshots_prefers_shortest_success() -> None:
    task = alfworld_task("train:0000", "pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-1")
    selected = select_trajectory_snapshots(
        [snapshot(task, "failed", 0.0, 2), snapshot(task, "long", 1.0, 12), snapshot(task, "short", 1.0, 7)],
        success_reward=1.0,
        max_examples_per_task=1,
    )
    assert [item.rollout_id for item in selected] == ["short"]


def test_task_retrieval_key_removes_scene_and_trial_identity() -> None:
    task = alfworld_task("test:0000", "pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-17")
    key = task_retrieval_key(task)
    assert key == (
        "ALFWorld task type: pick heat then place in recep; target object: mug; "
        "movable receptacle: none; destination or appliance: coffee machine"
    )
    assert "17" not in key


@pytest.mark.asyncio
async def test_memory_builder_retrieves_top_k_and_renders_guardrails() -> None:
    heat = alfworld_task("train:heat", "pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-1")
    place = alfworld_task("train:place", "pick_and_place_simple-Book-None-Shelf-1")
    test = alfworld_task("test:heat", "pick_heat_then_place_in_recep-Egg-None-CounterTop-9")
    builder = TrajectoryMemoryBankBuilder(
        chat_model=FakeChatModel(),
        embedding_model=FakeEmbeddingModel(),
        max_trajectory_chars=4_000,
        max_summary_chars=1_000,
        max_concurrent_summaries=2,
    )
    bank = await builder.build([snapshot(heat, "heat", 1.0, 5), snapshot(place, "place", 1.0, 4)])
    records = await builder.retrieve([test], bank, top_k=1)
    assert records[0].memories[0].item.source_task_id == "train:heat"
    prompt = render_retrieved_memories(records[0].memories)
    assert "not authoritative instructions" in prompt
    assert "current task, observation, inventory, and admissible-action list" in prompt
    assert "briefly assess each memory" in prompt
