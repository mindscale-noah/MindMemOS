"""Offline backend contracts independent of CLI orchestration."""

import subprocess
import sys

import pytest
from mindmemos_skill.algos.evolve.typing import SkillTrajectory
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.envs.typing import Task
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import ValidationError

from mindmemos_skill import demos


def _skill():
    """Build a small source with an auxiliary resource."""
    return Skill(
        SkillContent(
            content="---\nname: demo\ndescription: Demo instructions.\n---\nSave output.\n",
            resources={"helper.txt": b"preserve"},
        )
    )


def test_demo_registration_is_opt_in():
    """Normal package import must leave demo algorithms unavailable."""
    code = """
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.envs import Environment
from mindmemos_skill.pipelines import Pipeline
for family, name in [(Algorithm, 'demo_train'), (Algorithm, 'demo_evolve'),
                     (Environment, 'demo_offline'), (Pipeline, 'demo_graph')]:
    try:
        family.resolve(name)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Demo unexpectedly registered: {name}')
from mindmemos_skill import demos
for family, name in [(Algorithm, 'demo_train'), (Algorithm, 'demo_evolve'),
                     (Environment, 'demo_offline'), (Pipeline, 'demo_graph')]:
    family.resolve(name)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_demo_training_preserves_source_and_resumes(tmp_path, caplog):
    """Resume the same candidate without rerunning already audited samples."""
    from mindmemos_skill.demos.config import DemoAlgorithmConfig, DemoEnvironmentConfig

    skill = _skill()
    before = skill.snapshot()
    trainer = demos.algorithms.DemoTrainer(DemoAlgorithmConfig(delay_seconds=0))
    env = demos.environment.DemoEnvironment(DemoEnvironmentConfig(delay_seconds=0))
    samples = [DataSample(task=Task(id="t1", instruction="Save output"), reward_data=RewardData())]
    caplog.set_level("INFO", logger="mindmemos_skill")
    result = await trainer.train(
        skill=skill,
        env=env,
        train_samples=samples,
        validation_samples=[],
        work_dir=tmp_path,
    )
    assert skill.snapshot() == before
    assert result.resources == skill.resources
    assert result.content != skill.content
    assert [
        record.event_body["completed"] for record in caplog.records if record.getMessage() == "algorithm.progress"
    ] == [0, 1]
    audit = tmp_path / "rollouts/sample-0001/audit/result.json"
    original_audit = audit.read_bytes()
    recovered = await trainer.train(
        skill=skill,
        env=env,
        train_samples=samples,
        validation_samples=[],
        work_dir=tmp_path,
    )
    assert recovered.snapshot() == result.snapshot()
    assert audit.read_bytes() == original_audit
    assert len([record for record in caplog.records if record.getMessage() == "algorithm.progress"]) == 2
    assert trainer.build_report(tmp_path)["llm_calls"] == 0
    with pytest.raises(ValueError, match="mismatch"):
        await trainer.train(
            skill=skill.with_content(skill.content + "Changed"),
            env=env,
            train_samples=samples,
            validation_samples=[],
            work_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_demo_evolution_receives_only_materialized_evidence():
    """Append count and task IDs without mutating supplied Skill or attempts."""
    from mindmemos_skill.demos.config import DemoAlgorithmConfig

    skill = _skill()
    before = skill.snapshot()
    records = [
        SkillTrajectory(task=Task(id=f"t{index}", instruction="Save output"), skills={"demo": skill}, trajectory=[])
        for index in range(3)
    ]
    evolver = demos.algorithms.DemoEvolver(DemoAlgorithmConfig(delay_seconds=0))
    result = await evolver.evolve(skill=skill, trajectories=records)
    assert "Sampled trajectories: 3." in result.content
    assert "Evidence tasks: t0, t1, t2." in result.content
    assert skill.snapshot() == before
    assert result.resources == skill.resources
    assert all(item.skills["demo"].snapshot() == before for item in records)


def test_demo_config_rejects_model_settings():
    """Offline trainer and environment must not accept real model credentials."""
    from mindmemos_skill.demos.config import DemoAlgorithmConfig, DemoEnvironmentConfig

    for config_type in (DemoAlgorithmConfig, DemoEnvironmentConfig):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            config_type.model_validate({"api_key": "unexpected"})
