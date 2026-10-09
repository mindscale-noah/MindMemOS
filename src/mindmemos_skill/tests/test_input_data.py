"""Verify hash-free user inputs and internally checked external dependencies."""

import asyncio
import json

import pytest
from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.agents.buitin_agents.codex.config import CodexConfig
from mindmemos_skill.algos.train.recovery import sample_identity
from mindmemos_skill.common.checkpoint import boundary
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.builtin_envs.agentskillos import AgentSkillOSConfig, AgentSkillOSEnvironment
from mindmemos_skill.envs.builtin_envs.agentskillos.typing import (
    AgentSkillOSRewardData,
    AgentSkillOSTask,
    RetrievalRewardData,
)
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history import ALFWorldConfig, ALFWorldEnvironment
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.typing import ALFWorldRewardData, ALFWorldTask
from mindmemos_skill.envs.builtin_envs.livemath.typing import LiveMathRewardData
from mindmemos_skill.envs.builtin_envs.officeqa.typing import OfficeQARewardData
from mindmemos_skill.envs.builtin_envs.searchqa.typing import SearchQARewardData
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
    WorkbookInput,
    WorkbookReference,
)
from mindmemos_skill.envs.data import DataSample, RewardData
from pydantic import BaseModel, ValidationError


def test_materialized_data_needs_only_business_fields(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("input")
    models = [
        SearchQARewardData(answers=["42"]),
        OfficeQARewardData(ground_truth="42"),
        LiveMathRewardData(correct_choice={"label": "A", "text": "42"}, source_file=source, choice_seed=None),
        ALFWorldRewardData(),
        WorkbookInput(number="1", path=source),
        WorkbookReference(number="1", path=source),
        SpreadsheetBenchRewardData(cases=[{"number": "1", "path": source}], answer_position="Sheet!A1"),
        AgentSkillOSTask(id="one", instruction="Run", files={"input.txt": source}),
        AgentSkillOSRewardData(task_config={"task_id": "one"}, relevance_source="official_task_skills"),
        RetrievalRewardData(core_gt_ids=["alpha"]),
    ]
    for model in models:
        assert not any("hash" in name or "sha256" in name for name in type(model).model_fields)
        assert type(model).model_validate_json(model.model_dump_json()) == model


class Options(BaseModel):
    """Deterministic environment settings."""


class FileEnvironment(Environment[Options]):
    """Read public and reference files without any model calls."""

    config_type = Options
    calls = 0
    change_reference = False

    def _run(self, *, task, reward_data, skills, workspace):
        self.calls += 1
        if self.change_reference:
            reward_data.cases[0].path.write_text("changed during execution")
        return RolloutResult(output="done", trajectory=[], reward=Reward(score=1, success=True, detail=""))


def file_sample(tmp_path):
    public, private = tmp_path / "input.xlsx", tmp_path / "reference.xlsx"
    public.write_bytes(b"public")
    private.write_bytes(b"private")
    return (
        SpreadsheetBenchTask(id="one", instruction="Run", cases=[WorkbookInput(number="1", path=public)]),
        SpreadsheetBenchRewardData(cases=[WorkbookReference(number="1", path=private)], answer_position="Sheet!A1"),
    )


def test_json_inputs_resume_without_hashes_and_detect_reference_changes(tmp_path):
    task, private = file_sample(tmp_path)
    task = SpreadsheetBenchTask.model_validate_json(task.model_dump_json())
    private = SpreadsheetBenchRewardData.model_validate_json(private.model_dump_json())
    env = FileEnvironment(Options())
    directory = tmp_path / "run"
    result = asyncio.run(env.rollout(task, directory, reward_data=private))
    assert asyncio.run(env.rollout(task, directory, reward_data=private)) == result
    assert env.calls == 1
    request = json.loads((directory / "audit/request.json").read_text())
    assert set(request["input_identity"]) == {str(task.cases[0].path), str(private.cases[0].path)}
    private.cases[0].path.write_bytes(b"new reference")
    with pytest.raises(ValueError, match="inputs changed"):
        asyncio.run(env.rollout(task, directory, reward_data=private))
    assert env.calls == 1


def test_changed_reference_during_execution_is_never_cached(tmp_path):
    task, private = file_sample(tmp_path)
    env = FileEnvironment(Options())
    env.change_reference = True
    with pytest.raises(ValueError, match="files changed during execution"):
        asyncio.run(env.rollout(task, tmp_path / "run", reward_data=private))
    assert not (tmp_path / "run/audit/result.json").exists()


def test_trainer_checkpoint_rejects_file_changes_without_user_hashes(tmp_path):
    task, private = file_sample(tmp_path)
    sample = DataSample(task=task, reward_data=private)
    env = FileEnvironment(Options())
    checkpoint = tmp_path / "completed.json"
    identity = sample_identity(sample, env)
    boundary(checkpoint, identity, payload={"complete": True})
    assert boundary(checkpoint, sample_identity(sample, env)) == {"complete": True}
    task.cases[0].path.write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Recovery checkpoint mismatch"):
        boundary(checkpoint, sample_identity(sample, env))


def test_alfworld_support_files_are_computed_internally(tmp_path):
    game = tmp_path / "game.json"
    game.write_text("{}")
    support = tmp_path / "game.pddl"
    support.write_text("initial")
    env = ALFWorldEnvironment(ALFWorldConfig(agent=OpenAIChatConfig(model="offline"), simulator={}))
    task = ALFWorldTask(id="one", instruction="Run", gamefile=game)
    private = ALFWorldRewardData()
    before = env.input_identity(task, private)
    support.write_text("changed")
    assert env.input_identity(task, private) != before


def test_evaluator_config_has_no_hash_and_source_changes_are_checked(tmp_path):
    source = tmp_path / "benchmark/AgentSkillOS_bench/orchestrator.py"
    source.parent.mkdir(parents=True)
    source.write_text("# evaluator fixture\n")
    config = AgentSkillOSConfig(agent=CodexConfig(model="offline"), benchmark_root=tmp_path)
    assert "evaluator_sha256" not in config.model_dump()
    assert "evaluator_sha256" not in AgentSkillOSConfig.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="Extra inputs"):
        AgentSkillOSConfig.model_validate({**config.model_dump(), "evaluator_sha256": "obsolete"})
    env = AgentSkillOSEnvironment(config)
    task = Task(id="one", instruction="Run")
    private = RewardData()
    assert env.input_identity(task, private)
    source.write_text("# changed evaluator\n")
    with pytest.raises(ValueError, match="evaluator sources changed"):
        env.input_identity(task, private)
