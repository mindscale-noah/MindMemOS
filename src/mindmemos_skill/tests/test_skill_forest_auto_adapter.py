"""Automatic adapter matching, run isolation, and logged selection without provider calls."""

import asyncio
import json
import logging
from pathlib import Path

import pytest
from mindmemos_skill.agents import AgentResult
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter, SkillForestTrainer, SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.env_specific.selection import select_env_adapter
from mindmemos_skill.algos.train.skill_forest.registry import BUILTIN_ADAPTER_NAMES
from mindmemos_skill.envs import Environment, RolloutResult
from mindmemos_skill.envs.builtin_envs.livemath.typing import LiveMathTask
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.skill import Skill, SkillContent


@pytest.mark.parametrize("name", BUILTIN_ADAPTER_NAMES)
def test_registered_environment_name_selects_builtin_and_logs(name, caplog):
    # Selection must not initialize an agent, simulator, or workbook backend.
    env = object.__new__(Environment.resolve(name))
    with caplog.at_level(logging.INFO, logger="mindmemos_skill"):
        first = select_env_adapter(env)
        second = select_env_adapter(env)
    assert type(first) is SkillForestEnvAdapter.resolve(name)
    assert first is not second
    assert f"environment={name} adapter={name}" in caplog.text


def test_unmatched_or_unregistered_environment_uses_generic(caplog):
    class UnregisteredEnvironment:
        pass

    with caplog.at_level(logging.INFO, logger="mindmemos_skill"):
        for env in (UnregisteredEnvironment(), object.__new__(Environment.resolve("agentskillos"))):
            assert type(select_env_adapter(env)) is SkillForestEnvAdapter.resolve("generic")
    assert "environment=agentskillos adapter=generic" in caplog.text
    assert "UnregisteredEnvironment adapter=generic" in caplog.text


@pytest.mark.asyncio
async def test_same_trainer_uses_run_adapter_for_reflection_and_synthesis(tmp_path, monkeypatch, caplog):
    cases = json.loads((Path(__file__).parent / "fixtures/skill_forest/livemath_reflection.json").read_text())["cases"]
    failed = cases[0]
    samples = [
        DataSample(
            task=LiveMathTask.model_validate({**failed["task"], "id": f"task-{index}"}),
            reward_data=RewardData(),
        )
        for index in range(4)
    ]
    messages_seen = []
    environments_seen = set()
    merged_by = []

    class Optimizer:
        def __init__(self, config):
            pass

        async def run(self, *, messages):
            messages_seen.append(messages)
            if messages[1]["content"].startswith("## Skill Candidate 1"):
                return AgentResult(output="# Skill\nimproved synthesized\n")
            return AgentResult(output=json.dumps({"patch": {"edits": [{"op": "append", "content": "improved"}]}}))

    async def rollout(self, *, task, reward_data, skill_set, work_dir, resume):
        environments_seen.add(Environment.name_for(type(self)))
        improved = "improved" in skill_set.skills["candidate"].content
        result = RolloutResult.model_validate(failed["result"])
        result.reward.success = improved
        result.reward.score = float(improved)
        await asyncio.sleep(0)
        return result

    for name in ("livemath", "agentskillos"):
        monkeypatch.setattr(Environment.resolve(name), "rollout", rollout)
        monkeypatch.setattr(Environment.resolve(name), "input_identity", Environment.input_identity)
    original_merge = SkillForestEnvAdapter.build_skill_merge_messages

    def capture_merge(self, context):
        merged_by.append(type(self))
        return original_merge(self, context)

    monkeypatch.setattr(SkillForestEnvAdapter, "build_skill_merge_messages", capture_merge)
    trainer = SkillForestTrainer(
        SkillForestTrainerConfig(
            optimizer={"model": "offline"},
            num_trees=2,
            expansion_budget=2,
            candidates_per_node=1,
            ensemble_strategy="hierarchical_merge",
        ),
        agent_factory=Optimizer,
    )
    skill = Skill(
        source=SkillContent(
            content="# Skill\nInitial rules.\n",
            name="test-skill",
            description="Task instructions",
            standard_format=False,
        )
    )

    async def train(name):
        env = object.__new__(Environment.resolve(name))
        env.config = SkillForestTrainerConfig(optimizer={"model": "offline"})
        return await trainer.train(
            skill=skill,
            env=env,
            train_samples=samples,
            validation_samples=[],
            work_dir=tmp_path / name,
        )

    with caplog.at_level(logging.INFO, logger="mindmemos_skill"):
        selected = await asyncio.gather(train("livemath"), train("agentskillos"))
    assert all("synthesized" in candidate.content for candidate in selected)
    assert environments_seen == {"livemath", "agentskillos"}
    assert set(merged_by) == {SkillForestEnvAdapter.resolve("livemath"), SkillForestEnvAdapter.resolve("generic")}
    reflection_systems = {
        request[0]["content"]
        for request in messages_seen
        if not request[1]["content"].startswith("## Skill Candidate 1")
    }
    for name, adapter_name in (("livemath", "livemath"), ("agentskillos", "generic")):
        assert SkillForestEnvAdapter.build(adapter_name, {}).prompts.analyst_error in reflection_systems
        saved = json.loads((tmp_path / name / "config.json").read_text())
        assert saved["adapter"]["class"].endswith(SkillForestEnvAdapter.resolve(adapter_name).__name__)
        assert f"environment={name} adapter={adapter_name}" in caplog.text
    assert skill.content == "# Skill\nInitial rules.\n"
