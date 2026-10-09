"""Verify unlabeled batching, bounded concurrency, merging, and evidence clipping."""

import json
import threading
import time
from types import SimpleNamespace

import pytest
from mindmemos_skill.algos.evolve.simple_evolve import SimpleEvolverConfig, SimpleSkillEvolver
from mindmemos_skill.algos.evolve.simple_evolve.trajectory_rendering import clip_message
from mindmemos_skill.algos.evolve.typing import SkillTrajectory
from mindmemos_skill.envs.typing import Reward, Task
from mindmemos_skill.skill import Skill, SkillContent


def make_skill():
    """Create a detached Skill with an auxiliary resource."""
    return Skill(
        SkillContent(
            content="# Workflow\nCheck inputs.\n",
            name="Workflow",
            description="Check inputs",
            standard_format=False,
            resources={"guide.md": b"preserve"},
        )
    )


@pytest.mark.asyncio
async def test_parallel_batches_and_pairwise_merge():
    """Forty traces produce five extraction calls and four merges against one base."""
    skill = make_skill()
    calls = []
    lock = threading.Lock()
    active = 0
    peak = 0

    class Agent:
        def __init__(self, config):
            pass

        def run(self, *, messages):
            nonlocal active, peak
            payload = json.loads(messages[1]["content"])
            with lock:
                calls.append(payload)
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return SimpleNamespace(
                output=json.dumps(
                    {
                        "reasoning": "Observed repeated retry errors",
                        "edits": [{"op": "append", "content": "Stop unchanged retries."}],
                    }
                )
            )

    traces = [
        SkillTrajectory(
            trajectory_id=f"trace-{index}",
            task=Task(id=str(index), instruction="Check"),
            skills={},
            trajectory=[{"role": "tool", "content": "Repeated error"}],
            reward=Reward(score=1, success=True, detail="label"),
            status="succeeded",
            observed_score=1,
        )
        for index in range(40)
    ]
    evolver = SimpleSkillEvolver(
        SimpleEvolverConfig(optimizer={"model": "offline"}, concurrency=2), agent_factory=Agent
    )
    result = await evolver.evolve(skill=skill, trajectories=traces)
    extracts = [call for call in calls if "executions" in call]
    merges = [call for call in calls if "patches" in call]
    assert len(extracts) == 5
    assert all(len(call["executions"]) == 8 for call in extracts)
    assert len(merges) == 4
    assert all(len(call["patches"]) == 2 for call in merges)
    assert sorted(sum(len(group["trajectory_ids"]) for group in call["patches"]) for call in merges) == [16, 16, 32, 40]
    assert all(call["skill"] == skill.content for call in calls)
    assert 1 < peak <= 2
    assert all(
        "reward" not in trace and "status" not in trace and "observed_score" not in trace
        for call in extracts
        for trace in call["executions"]
    )
    assert result.content.count("Stop unchanged retries.") == 1
    assert result.resources == skill.resources
    assert "Stop unchanged retries." not in skill.content


@pytest.mark.parametrize("role,limit", [("user", 60000), ("assistant", 12000), ("tool", 6000)])
def test_role_clipping_preserves_head_tail_and_structure(role, limit):
    """Long content preserves both ends and short content is unchanged."""
    config = SimpleEvolverConfig(optimizer={"model": "offline"})
    text = "HEAD" + "中" * 70000 + "TAIL"
    message = {"role": role, "tool_call_id": "call-1", "content": [{"text": text}]}
    clipped = clip_message(message, config)
    retained = clipped["content"][0]["text"]
    assert len(retained) == limit
    assert retained.startswith("HEAD") and retained.endswith("TAIL")
    assert "middle omitted" in retained
    assert clipped["role"] == role and clipped["tool_call_id"] == "call-1"
    assert message["content"][0]["text"] == text
    short = {"role": role, "content": "Keep all of this."}
    assert clip_message(short, config) == short


def test_assistant_tool_arguments_are_clipped():
    """Nested tool arguments share the assistant text budget."""
    config = SimpleEvolverConfig(optimizer={"model": "offline"})
    result = clip_message(
        {
            "role": "assistant",
            "content": "Planning",
            "tool_calls": [{"function": {"name": "execute", "arguments": "A" * 20000 + "END"}}],
        },
        config,
    )
    assert result["tool_calls"][0]["function"]["name"] == "execute"
    assert result["content"] == "Planning"
    arguments = result["tool_calls"][0]["function"]["arguments"]
    assert len(arguments) < config.assistant_message_chars
    assert arguments.endswith("END") and "middle omitted" in arguments


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,match",
    [
        ({"edits": [{"op": "replace", "target": "missing", "content": "replacement"}]}, "exactly once"),
        ({"edits": [{"op": "append", "content": "x"}] * 6}, "max_edits"),
    ],
)
async def test_invalid_patch_is_rejected(response, match):
    """Reject invalid exact targets and excess edits before publication."""

    class Agent:
        def __init__(self, config):
            pass

        def run(self, *, messages):
            return SimpleNamespace(output=json.dumps(response))

    evolver = SimpleSkillEvolver(SimpleEvolverConfig(optimizer={"model": "offline"}), agent_factory=Agent)
    with pytest.raises(ValueError, match=match):
        await evolver.evolve(
            skill=make_skill(),
            trajectories=[SkillTrajectory(task=Task(id="1", instruction="Check"), skills={}, trajectory=[])],
        )


def test_default_sampling_and_extraction_settings():
    """Expose forty samples and eight-item groups through the public configuration."""
    from mindmemos_skill.pipelines.builtin_pipelines.evolve.config import SimpleEvolutionPipelineConfig

    config = SimpleEvolutionPipelineConfig(
        evolver={"class_name": "simple_evolve", "config": {"optimizer": {"model": "offline"}}},
        work_dir="unused",
    )
    assert config.traj_collect.sample_size == 40
    assert config.traj_collect.max_page_size == 200
    assert config.evolver.config.batch_size == 8
    assert config.evolver.config.concurrency == 5


def test_trajectory_collection_config_rejects_removed_options_and_invalid_pool():
    """Keep removed collector options out of configuration and enforce a viable sample pool."""
    from mindmemos_skill.pipelines.builtin_pipelines.evolve.config import TrajCollectConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="sample_size <= trajectory_limit"):
        TrajCollectConfig(sample_size=40, trajectory_limit=20)
    with pytest.raises(ValidationError, match="Extra inputs"):
        TrajCollectConfig(status="completed")
