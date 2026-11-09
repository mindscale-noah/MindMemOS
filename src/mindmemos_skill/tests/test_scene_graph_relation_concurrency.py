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

    async def infer_relations(system, payload, allowed):
        index = len(calls)
        calls.append(payload)
        if len(calls) == len(pairs):
            all_started.set()
        await releases[index].wait()
        completed.append(index)
        releases[index - 1].set()
        left, right = pairs[index]
        return algorithm._parse_relations(
            json.dumps(
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
            ),
            allowed=allowed,
        )

    monkeypatch.setattr(algorithm, "_request_relations", infer_relations)
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

    async def infer_relations(system, payload, allowed):
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
                return algorithm._parse_relations("invalid JSON", allowed=allowed)
            await asyncio.Event().wait()
        finally:
            active -= 1

    monkeypatch.setattr(algorithm, "_request_relations", infer_relations)
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

    async def reject_chat_call(system, payload, allowed):
        pytest.fail("Empty candidates must not invoke chat")

    monkeypatch.setattr(algorithm, "_request_relations", reject_chat_call)
    assert await algorithm._infer_joint_relations(iter(())) == []


@pytest.mark.parametrize("retry_mode", ["direct", "feedback"])
@pytest.mark.parametrize("failure", ["invalid_endpoints", "self_link", "invalid_schema", "missing_weight"])
async def test_relation_validation_retries_only_failed_batch(relation_algorithm, monkeypatch, retry_mode, failure):
    """Repair invalid output with independent parsers while successful siblings run only once."""
    from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig

    algorithm, nodes = relation_algorithm
    algorithm.chat_model = OpenAIChatAgent(
        OpenAIChatConfig(model="fixture", parse_max_attempts=3, parse_retry_mode=retry_mode)
    )
    if failure == "missing_weight":
        algorithm.config = algorithm.config.model_copy(
            update={
                "joint_linking": algorithm.config.joint_linking.model_copy(
                    update={"relation_weighting": "model_confidence"}
                )
            }
        )
    pairs = [(nodes[0], nodes[1]), (nodes[1], nodes[2])]
    calls = [0, 0]

    async def generate_response(self, *, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        index = 0 if payload["pairs"][0][0]["node_id"] == nodes[0].node_id else 1
        calls[index] += 1
        await asyncio.sleep(0)
        left, right = pairs[index]
        relation = {
            "source_id": left.node_id,
            "target_id": right.node_id,
            "edge_type": "semantic",
            "confidence": 0.8,
            "weight": 0.5,
            "evidence": "Supported by public capabilities",
        }
        if index == 0 and calls[index] == 1:
            if failure == "invalid_schema":
                return AgentResult(output='{"relations": "private-invalid-payload"}')
            if failure == "missing_weight":
                relation.pop("weight")
            else:
                relation["target_id"] = left.node_id if failure == "self_link" else nodes[2].node_id
        return AgentResult(output=json.dumps({"relations": [relation]}))

    monkeypatch.setattr(OpenAIChatAgent, "run_messages", generate_response)
    edges = await algorithm._infer_joint_relations(iter(pairs))
    assert calls == [2, 1]
    assert [(edge["source"], edge["target"]) for edge in edges] == [
        (left.node_id, right.node_id) for left, right in pairs
    ]
    assert algorithm.chat_model.output_parser is None


async def test_relation_retry_exhaustion_remains_strict_and_private(relation_algorithm, monkeypatch, caplog):
    """Exhaust the configured bound without leaking provider payloads or accepting invalid edges."""
    from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig

    algorithm, nodes = relation_algorithm
    algorithm.chat_model = OpenAIChatAgent(OpenAIChatConfig(model="fixture", parse_max_attempts=3))
    calls = 0

    async def generate_response(self, **kwargs):
        nonlocal calls
        calls += 1
        return AgentResult(output='{"relations":"private-invalid-payload"}')

    monkeypatch.setattr(OpenAIChatAgent, "run_messages", generate_response)
    with pytest.raises(SceneGraphError):
        await algorithm._infer_joint_relations(iter([(nodes[0], nodes[1])]))
    assert calls == 3
    assert "private-invalid-payload" not in caplog.text
    assert "private-invalid-payload" not in str([record.__dict__ for record in caplog.records])


async def test_explicit_drop_policy_preserves_valid_relations(relation_algorithm, monkeypatch):
    """Keep the legacy opt-in dropping policy without relaxing the default strict policy."""
    from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig

    algorithm, nodes = relation_algorithm
    algorithm.chat_model = OpenAIChatAgent(OpenAIChatConfig(model="fixture"))
    algorithm.config = algorithm.config.model_copy(
        update={
            "joint_linking": algorithm.config.joint_linking.model_copy(
                update={"relation_validation": "drop_invalid_endpoints"}
            )
        }
    )
    calls = 0

    async def generate_response(self, **kwargs):
        nonlocal calls
        calls += 1
        return AgentResult(
            output=json.dumps(
                {
                    "relations": [
                        {
                            "source_id": nodes[0].node_id,
                            "target_id": node.node_id,
                            "edge_type": "semantic",
                            "evidence": "Public evidence",
                        }
                        for node in nodes[:3]
                    ]
                }
            )
        )

    monkeypatch.setattr(OpenAIChatAgent, "run_messages", generate_response)
    edges = await algorithm._infer_joint_relations(iter([(nodes[0], nodes[1])]))
    assert len(edges) == 1
    assert edges[0]["target"] == nodes[1].node_id
    assert calls == 1
