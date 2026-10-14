"""Verify HTTP submission naming and honest export of native TreeSkill test results."""

from types import SimpleNamespace

import pytest
from mindmemos_eval.skills import treeskill
from mindmemos_eval.skills.config import TreeSkillEvalConfig
from mindmemos_skill.common.checkpoint import boundary
from mindmemos_skill.common.json import read_json, read_jsonl, write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.envs.typing import Reward, RolloutResult, Task, Usage
from mindmemos_skill.skill import Skill
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
async def test_http_run_uses_named_registration_and_service_test(monkeypatch, tmp_path):
    source = tmp_path / "initial"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: xlsx\ndescription: Sheets.\n---\nOriginal body.\n")
    config = TreeSkillEvalConfig(
        reference_run=tmp_path / "reference",
        service_config=tmp_path / "service.yaml",
        sdk_home=tmp_path / "sdk",
        data_root=tmp_path,
        split_file=tmp_path / "split.json",
        initial_skill=source,
        output_dir=tmp_path / "run",
        concurrency=16,
    )
    reference_algorithm = {
        "seed": 41,
        "rollout_concurrency": 32,
        "analysis_concurrency": 32,
        "localization_concurrency": 32,
    }
    write_json(config.reference_run / "training/train/config.json", {"algorithm": reference_algorithm})
    monkeypatch.setattr(
        treeskill,
        "prepare",
        lambda config: (
            StubEnvironment(StubConfig()),
            {"train": samples()[:1], "val": [], "test": samples()},
            Skill.from_path(source),
        ),
    )
    monkeypatch.setattr(
        treeskill.ConfigManager,
        "load",
        lambda self: SimpleNamespace(
            resolve_skill_connection=lambda: ("http://service", "test-key"), network=SimpleNamespace(timeout_seconds=10)
        ),
    )
    calls = {}

    class Transport:
        def __init__(self, **kwargs):
            calls["connection"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class Client:
        def __init__(self, transport):
            pass

        async def register(self, request):
            calls["registration"] = request
            return {"skill_id": "family", "skill_version_id": "initial"}

        async def train(self, request):
            calls["training"] = request
            for i in range(3):
                complete(config.output_dir, i, i != 1)
            write_json(
                config.output_dir / "training/train/config.json",
                {
                    "algorithm": {
                        **reference_algorithm,
                        "rollout_concurrency": 16,
                        "analysis_concurrency": 16,
                        "localization_concurrency": 16,
                    }
                },
            )
            return {"task_id": "task", "project_id": "ui-project", "status": "queued"}

        async def task_progress(self, request):
            return []

        async def show_task(self, request):
            return {"task_id": "task", "status": "succeeded", "result": {"version_id": "trained"}}

    monkeypatch.setattr(treeskill, "AsyncHttpTransport", Transport)
    monkeypatch.setattr(treeskill, "AsyncSkillClient", Client)
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


@pytest.mark.asyncio
async def test_export_rejects_task_order_drift(tmp_path):
    complete(tmp_path, 0, True)
    with pytest.raises(ValueError, match="requested tasks"):
        await treeskill.export_test(
            tmp_path, list(reversed(samples())), StubEnvironment(StubConfig()), 4, "spreadsheetbench_text_action"
        )


def test_protocol_keeps_model_and_sampling_settings():
    assert treeskill._protocol({"api_key": "secret", "base_url": "private", "model": "qwen", "temperature": 0.7}) == {
        "model": "qwen",
        "temperature": 0.7,
    }


def test_other_registered_environment_prepares_without_reference(tmp_path, capsys):
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
                "training_pipelines": {"treeskill_mindmemos_eval": {"config": {"trainer": {"class_name": "treeskill"}}}}
            }
        },
    )
    config = TreeSkillEvalConfig(
        environment={"class_name": "livemath", "config": {"agent": {"model": "fixture", "api_key": "test"}}},
        samples_file=tmp_path / "samples.json",
        split_file=tmp_path / "split.json",
        initial_skill=source,
        service_config=tmp_path / "service.yaml",
        sdk_home=tmp_path / "sdk",
    )
    env, pools, skill = treeskill.prepare(config)
    assert Environment.name_for(type(env)) == "livemath"
    assert {name: [sample.task.id for sample in values] for name, values in pools.items()} == {
        "train": ["0"],
        "val": ["1"],
        "test": ["2"],
    }
    assert skill.source.name == "math"
    assert "no reference score guarantee" in capsys.readouterr().out


def test_reference_protocol_restores_agent_defaults_without_ignoring_model_drift():
    recorded = {"model": "qwen", "timeout_s": 600.0, "response_validation": "content"}
    current = {"model": "qwen", "timeout_s": 600.0, "class_name": "openai_chat", "sdk_max_retries": 0}
    assert treeskill._protocol(recorded) == treeskill._protocol(current)
    assert treeskill._protocol(recorded) != treeskill._protocol({**current, "model": "different"})
    assert treeskill._protocol({**recorded, "response_validation": "strict"}) != treeskill._protocol(current)
