"""Verify local submission naming and honest export of native TreeSkill test results."""

import pytest
from mindmemos_eval.cli import main
from mindmemos_eval.skills import treeskill
from mindmemos_eval.skills.config import TreeSkillEvalConfig
from mindmemos_skill.api.service import SkillService
from mindmemos_skill.api.typing import AlgoSkillEvolveResult
from mindmemos_skill.common.checkpoint import boundary
from mindmemos_skill.common.json import read_json, read_jsonl, write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.envs.typing import Reward, RolloutResult, Task, Usage
from pydantic import BaseModel


class StubConfig(BaseModel):
    pass


class StubEnvironment(Environment[StubConfig]):
    config_type = StubConfig

    async def _run(self, *, task, reward_data, skills, workspace):
        raise AssertionError("Client must never execute model rollouts")


def samples():
    return [DataSample(task=Task(id=str(i), instruction="Task"), reward_data=RewardData()) for i in range(3)]


def complete(root, position, success):
    result = RolloutResult(
        output="done",
        trajectory=[{"role": "assistant", "content": "done"}],
        reward=Reward(score=float(success), success=success, detail="fixture"),
        usage=Usage(input_tokens=10, output_tokens=2, total_tokens=12),
    )
    write_json(root / "training/selected_skill.json", {"fixture": True})
    env = StubEnvironment(StubConfig())
    identity = {
        "skill": {"fixture": True},
        "environment": f"{type(env).__module__}.{type(env).__qualname__}",
        "config": {},
        "samples": [sample.model_dump(mode="json") for sample in samples()],
    }
    boundary(root / "training/test/inputs.json", identity, payload={"protocol": 1})
    boundary(
        root / f"training/test/task-{position:06d}/completed.json",
        identity,
        payload={"result": result.model_dump(mode="json")},
    )


@pytest.mark.asyncio
async def test_export_separates_execution_errors_from_unstarted_and_zero_reward(tmp_path):
    complete(tmp_path, 0, False)
    (tmp_path / "training/test/task-000001/attempts/one").mkdir(parents=True)
    report = await treeskill.export_test(
        tmp_path, samples(), StubEnvironment(StubConfig()), 4, "spreadsheetbench_text_action"
    )
    assert (report.attempted, report.completed, report.errors, report.not_started) == (2, 1, 1, 1)
    assert report.success_rate == 0 and report.total_tokens is None
    assert report.reported_usage["total_tokens"] == 12
    assert len(list(read_jsonl(report.trajectory_file))) == 3
    await treeskill.export_test(tmp_path, samples(), StubEnvironment(StubConfig()), 4, "spreadsheetbench_text_action")
    assert len(list(read_jsonl(report.trajectory_file))) == 3
    assert (tmp_path / "report.md").is_file()


@pytest.mark.asyncio
async def test_local_run_uses_named_registration_and_service_test(monkeypatch, tmp_path, capsys):
    source = tmp_path / "initial"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: xlsx\ndescription: Sheets.\n---\nOriginal body.\n")
    config = TreeSkillEvalConfig(
        service_config=tmp_path / "service.yaml",
        data_root=tmp_path,
        split_file=tmp_path / "split.json",
        initial_skill=source,
        output_dir=tmp_path / "run",
        concurrency=16,
        poll_interval=1,
    )
    write_json(config.split_file, {"train": ["train"], "test": ["0", "1", "2"]})
    monkeypatch.setattr(
        treeskill,
        "_load_samples",
        lambda config, env: [
            DataSample(task=Task(id="train", instruction="Task"), reward_data=RewardData()),
            *samples(),
        ],
    )
    monkeypatch.setattr(Environment, "resolve", lambda name: StubEnvironment)
    write_json(
        config.service_config,
        {
            "service": {
                "home": str(tmp_path / "unused-service-home"),
                "training_pipelines": {
                    config.profile: {
                        "class_name": "training",
                        "config": {
                            "trainer": {
                                "class_name": "treeskill",
                                "config": {
                                    "optimizer": {"model": "fixture"},
                                    "router": {"model": "fixture"},
                                    "rollout_concurrency": 16,
                                    "analysis_concurrency": 16,
                                    "localization_concurrency": 16,
                                },
                            },
                            "evaluation_concurrency": 16,
                        },
                    }
                },
            },
            "fastapi": {"auth": {"mode": "none", "default_project_id": "unused"}},
        },
    )
    config_file = tmp_path / "treeskill.yaml"
    write_json(config_file, config)
    assert main(["skill", "evaluate", "train", "--algo", "tree-skill", "--config", str(config_file), "--check"]) == 0
    assert "1 train / 3 test" in capsys.readouterr().out
    assert not config.output_dir.exists()

    calls = {}
    register_skill = SkillService.register_skill

    async def register(self, request):
        calls["registration"] = request
        return await register_skill(self, request)

    async def train(self, request):
        calls["training"] = request
        for i in range(3):
            complete(config.output_dir, i, i != 1)
        registered = await self.get_skill_version(request.init_skill_version_id, request.project_id)
        return AlgoSkillEvolveResult(
            skill_id=registered.skill_id,
            base_version_id=registered.version_id,
            version_id="trained",
            run_id="fixture",
            content=registered.content,
            resources={},
            content_hash="content",
            skill_hash="skill",
        )

    monkeypatch.setattr(SkillService, "register_skill", register)
    monkeypatch.setattr(SkillService, "_run_training", train)
    summary = await treeskill.run(config)
    registered = calls["registration"]
    assert registered.alias.startswith("xlsx-mindmemos-eval-")
    assert registered.skill.name == registered.alias
    assert f"name: {registered.alias}" in registered.skill.content
    assert "Original body." in registered.skill.content
    assert calls["training"].algo == "treeskill_mindmemos_eval"
    assert calls["training"].env.class_name == "spreadsheetbench_text_action"
    assert summary.completed == 3 and summary.success_rate == 2 / 3
    assert read_json(config.output_dir / "task.json")["result"]["version_id"] == "trained"
    assert (config.output_dir / "state/skills.db").is_file()
    assert not (tmp_path / "unused-service-home").exists()


@pytest.mark.asyncio
async def test_export_rejects_task_order_drift(tmp_path):
    complete(tmp_path, 0, True)
    with pytest.raises(ValueError, match="requested tasks"):
        await treeskill.export_test(
            tmp_path, list(reversed(samples())), StubEnvironment(StubConfig()), 4, "spreadsheetbench_text_action"
        )


def test_other_registered_environment_prepares(tmp_path):
    source = tmp_path / "initial"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: math\ndescription: Math.\n---\nSolve the task.\n")
    rows = [
        {
            "task": {"id": str(i), "instruction": "Choose one", "choices": [{"label": "A", "text": "One"}]},
            "reward_data": {
                "correct_choice": {"label": "A", "text": "One"},
                "source_file": "fixture.json",
                "choice_seed": None,
            },
        }
        for i in range(3)
    ]
    write_json(tmp_path / "samples.json", rows)
    write_json(tmp_path / "split.json", {"train": ["0"], "val": ["1"], "test": ["2"]})
    write_json(
        tmp_path / "service.yaml",
        {
            "service": {
                "home": str(tmp_path),
                "training_pipelines": {
                    "treeskill_mindmemos_eval": {
                        "class_name": "training",
                        "config": {
                            "trainer": {
                                "class_name": "treeskill",
                                "config": {"optimizer": {"model": "fixture"}, "router": {"model": "fixture"}},
                            }
                        },
                    }
                },
            },
            "fastapi": {"auth": {"mode": "none", "default_project_id": "unused"}},
        },
    )
    config = TreeSkillEvalConfig(
        environment={"class_name": "livemath", "config": {"agent": {"model": "fixture", "api_key": "test"}}},
        samples_file=tmp_path / "samples.json",
        split_file=tmp_path / "split.json",
        initial_skill=source,
        service_config=tmp_path / "service.yaml",
    )
    env, pools, skill, _ = treeskill.prepare(config)
    assert Environment.name_for(type(env)) == "livemath"
    assert {name: [sample.task.id for sample in values] for name, values in pools.items()} == {
        "train": ["0"],
        "val": ["1"],
        "test": ["2"],
    }
    assert skill.source.name == "math"
