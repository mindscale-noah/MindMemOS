"""Concurrent relation batches preserve ordered merging and cancel failed siblings."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from mindmemos_skill.algos.manage.skill_manage.algorithm import SkillManageAlgorithm
from mindmemos_skill.algos.manage.skill_manage.config import SkillManageConfig
from mindmemos_skill.algos.manage.skill_manage.errors import SceneGraphError
from mindmemos_skill.algos.manage.skill_manage.typing import SceneGraphContent


@pytest.fixture
def relation_algorithm():
    """Create an offline algorithm and validated candidate nodes."""
    fixture = json.loads((Path(__file__).parent / "fixtures/scene_graph_d0.json").read_text())
    nodes = SceneGraphContent.model_validate(fixture["content"]).nodes
    config = SkillManageConfig(
        embedding={"model": "fixture", "base_url": "http://fixture.invalid", "dimensions": 3},
        llm_linking=False,
        llm_batch_size=1,
    )
    algorithm = SkillManageAlgorithm(config, store=SimpleNamespace(scope={"graph_id": "fixture"}))
    return algorithm, nodes


async def test_all_relation_batches_overlap_and_merge_in_input_order(relation_algorithm, monkeypatch):
    """Reverse completion order without changing duplicate selection or edge order."""
    algorithm, nodes = relation_algorithm
    pairs = [(nodes[0], nodes[1]), (nodes[0], nodes[1]), (nodes[1], nodes[2])]
    releases = [asyncio.Event() for _ in pairs]
    all_started = asyncio.Event()
    calls = []
    completed = []

    async def infer_relations(system, payload):
        index = len(calls)
        calls.append(payload)
        if len(calls) == len(pairs):
            all_started.set()
        await releases[index].wait()
        completed.append(index)
        releases[index - 1].set()
        left, right = pairs[index]
        return json.dumps(
            {
                "relations": [
                    {
                        "source_id": left.node_id,
                        "target_id": right.node_id,
                        "edge_type": "semantic",
                        "confidence": 0.8,
                        "evidence": f"batch-{index}",
                    }
                ]
            }
        )

    monkeypatch.setattr(algorithm, "_chat", infer_relations)
    task = asyncio.create_task(algorithm._infer_joint_relations(iter(pairs)))
    try:
        await asyncio.wait_for(all_started.wait(), timeout=2)
        releases[-1].set()
        edges = await asyncio.wait_for(task, timeout=2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert completed == [2, 1, 0]
    assert [edge["description"] for edge in edges] == ["batch-0", "batch-2"]
    assert all(edge["weight"] == pytest.approx(0.32) for edge in edges)


@pytest.mark.parametrize("failure", ["transport", "invalid_output", "cancel"])
async def test_relation_failure_or_cancellation_drains_all_batches(relation_algorithm, monkeypatch, failure):
    """Leave no active sibling model calls after failure or caller cancellation."""
    algorithm, nodes = relation_algorithm
    all_started = asyncio.Event()
    calls = 0
    active = 0

    async def infer_relations(system, payload):
        nonlocal calls, active
        index = calls
        calls += 1
        active += 1
        if calls == 3:
            all_started.set()
        try:
            await all_started.wait()
            if index == 0 and failure != "cancel":
                if failure == "transport":
                    raise RuntimeError("Controlled transport failure")
                return "invalid JSON"
            await asyncio.Event().wait()
        finally:
            active -= 1

    monkeypatch.setattr(algorithm, "_chat", infer_relations)
    task = asyncio.create_task(algorithm._infer_joint_relations(iter([(nodes[0], nodes[1])] * 3)))
    try:
        await asyncio.wait_for(all_started.wait(), timeout=2)
        if failure == "cancel":
            task.cancel()
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else SceneGraphError):
            await asyncio.wait_for(task, timeout=2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert calls == 3
    assert active == 0


async def test_empty_relation_pairs_skip_chat(relation_algorithm, monkeypatch):
    """Avoid model calls and a zero concurrency limit for empty candidates."""
    algorithm, _ = relation_algorithm

    async def reject_chat_call(system, payload):
        pytest.fail("Empty candidates must not invoke chat")

    monkeypatch.setattr(algorithm, "_chat", reject_chat_call)
    assert await algorithm._infer_joint_relations(iter(())) == []
