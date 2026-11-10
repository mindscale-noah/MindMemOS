"""TreeSkill selects run-local adapters and preserves them in dynamic Skill snapshots."""

import asyncio
import logging

import pytest
from mindmemos_skill.algos.train.treeskill import (
    TreeSkill,
    TreeSkillEnvAdapter,
    TreeSkillTrainer,
    TreeSkillTrainerConfig,
)
from mindmemos_skill.algos.train.treeskill.env_specific.selection import select_env_adapter
from mindmemos_skill.algos.train.treeskill.typing import TrainingRollout
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import ValidationError


@pytest.mark.parametrize(
    "name, expected",
    [
        ("spreadsheetbench_text_action", "spreadsheetbench_text_action"),
        ("searchqa", "generic"),
        ("spreadsheetbench_tool_calling", "generic"),
    ],
)
def test_auto_adapter_matches_registration_and_logs(name, expected, caplog):
    env = object.__new__(Environment.resolve(name))
    with caplog.at_level(logging.INFO, logger="mindmemos_skill"):
        adapter = select_env_adapter(env)
    assert type(adapter) is TreeSkillEnvAdapter.resolve(expected)
    assert f"environment={name} adapter={expected}" in caplog.text


def test_unregistered_environment_uses_generic(caplog):
    class CustomEnvironment:
        pass

    with caplog.at_level(logging.INFO, logger="mindmemos_skill"):
        assert type(select_env_adapter(CustomEnvironment())) is TreeSkillEnvAdapter.resolve("generic")
    assert "CustomEnvironment adapter=generic" in caplog.text


def test_training_config_rejects_manual_adapter():
    with pytest.raises(ValidationError, match="env_adapter"):
        TreeSkillTrainerConfig(
            optimizer={"model": "offline"}, router={"model": "offline"}, env_adapter={"class_name": "generic"}
        )


@pytest.mark.asyncio
async def test_concurrent_training_preserves_selected_analysis_and_routing(tmp_path, monkeypatch):
    from mindmemos_skill.algos.train.treeskill import trainer as module

    analyzed = {}
    sample = DataSample(task=Task(id="task", instruction="Inspect output"), reward_data=RewardData())

    async def sample_tasks(*, env, skill, samples, directory, repeats, concurrency):
        await asyncio.sleep(0)
        return [
            TrainingRollout(
                trajectory_id="trace",
                sample=sample,
                work_dir=directory,
                result=RolloutResult(
                    output="done",
                    duration_s=0.0,
                    started_at="2026-09-30T00:00:00Z",
                    finished_at="2026-09-30T00:00:00Z",
                    trajectory=[{"role": "assistant", "content": "Checked output"}],
                    reward=Reward(score=1, success=True, detail="Passed"),
                ),
            )
        ], []

    def create_analyzer(self, runtime):
        name = TreeSkillEnvAdapter.name_for(type(self))

        class Analyzer:
            async def __call__(self, inputs):
                await asyncio.sleep(0)
                analyzed[inputs.workspace.parents[4].name] = name
                from mindmemos_skill.algos.train.treeskill.typing import AnalysisOutcome, TrajectoryAnalysisRecord

                return AnalysisOutcome(
                    record=TrajectoryAnalysisRecord(
                        instance_id=inputs.evidence.trajectory_id,
                        task_id=inputs.evidence.task_id,
                        record_source="success",
                    )
                )

        return Analyzer()

    monkeypatch.setattr(module, "sample_tasks", sample_tasks)
    for name in ("generic", "spreadsheetbench_text_action"):
        monkeypatch.setattr(TreeSkillEnvAdapter.resolve(name), "create_analyzer", create_analyzer)
    trainer = TreeSkillTrainer(
        TreeSkillTrainerConfig(optimizer={"model": "offline"}, router={"model": "offline"}, require_skill_match=False)
    )
    source = Skill(
        source=SkillContent(
            content="# Workflow\nInspect output.\n",
            name="test-skill",
            description="Task instructions",
            standard_format=False,
        )
    )

    async def train(name):
        env = object.__new__(Environment.resolve(name))
        env.config = Environment.resolve(name).config_type.model_validate({"agent": {"model": "offline"}})
        return await trainer.train(skill=source, env=env, train_samples=[sample], work_dir=tmp_path / name)

    names = ["spreadsheetbench_text_action", "searchqa"]
    results = await asyncio.gather(*(train(name) for name in names))
    for name, result in zip(names, results, strict=True):
        expected = name if name == "spreadsheetbench_text_action" else "generic"
        assert analyzed[name] == expected
        assert result.config.env_adapter.class_name == expected
        restored = TreeSkill.from_snapshot(result.snapshot())
        assert type(restored.env_adapter) is TreeSkillEnvAdapter.resolve(expected)
        before = (tmp_path / name / "completed.json").read_text()
        resumed = await train(name)
        assert resumed.config.env_adapter.class_name == expected
        assert (tmp_path / name / "completed.json").read_text() == before
        assert expected in (tmp_path / name / "final_skill.json").read_text()
    assert source.content == "# Workflow\nInspect output.\n"
