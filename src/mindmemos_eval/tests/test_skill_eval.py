"""Verify fresh eval protocols against native environments and in-process storage."""

import pytest
from mindmemos_eval.skills import runner
from mindmemos_eval.skills.config import EvalConfig, RunOptions, load_config
from mindmemos_eval.skills.evaluation import evaluate
from mindmemos_eval.skills.report import read_report, write_report
from mindmemos_eval.skills.typing import ArmResult, BenchmarkReport, CaseResult, DatasetInputs, SuiteReport
from mindmemos_skill.algos.train.skill_forest.trainer import SkillForestTrainer
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.livemath.rollout import LiveMathEnvironment
from mindmemos_skill.envs.builtin_envs.livemath.typing import (
    Choice,
    LiveMathResult,
    LiveMathReward,
    LiveMathRewardData,
    LiveMathTask,
)
from mindmemos_skill.envs.data import DataSample
from mindmemos_skill.envs.typing import Usage


@pytest.fixture
def config():
    return EvalConfig.model_validate(
        {
            "models": {
                name: {"model": "offline", "api_key": "private-test-key", "base_url": "https://offline.invalid/v1"}
                for name in ("target", "optimizer")
            },
            "skill_forest": {"num_trees": 1, "expansion_budget": 1},
            "environments": {"livemath": {}},
        }
    )


@pytest.fixture
def inputs(tmp_path, config):
    source = tmp_path / "source.json"
    source.write_text("[]")
    skill = tmp_path / "initial"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: offline\ndescription: Answer tasks.\n---\nChoose carefully.\n")
    splits = {}
    for name, count in (("train", 4), ("val", 2), ("test", 2)):
        splits[name] = [
            DataSample(
                task=LiveMathTask(id=f"{name}-{i}", instruction="Choose", choices=[Choice(label="A", text="Yes")]),
                reward_data=LiveMathRewardData(
                    correct_choice=Choice(label="A", text="Yes"), source_file=source, choice_seed=None
                ),
            )
            for i in range(count)
        ]
    return DatasetInputs(
        splits=splits,
        environment=config.environment("livemath", {}),
        initial_skill=skill,
        split_file=tmp_path / "splits.json",
        source=tmp_path,
    )


@pytest.fixture
def offline(monkeypatch):
    calls = []

    async def execute(self, *, task, reward_data, skills, workspace):
        calls.append((task.id, workspace, bool(skills)))
        return LiveMathResult(
            output="A",
            trajectory=[],
            usage=Usage(input_tokens=2, output_tokens=1, total_tokens=3),
            reward=LiveMathReward(
                score=1,
                success=True,
                detail="offline",
                predicted_label="A",
                predicted_text="Yes",
                correct_label="A",
                correct_text="Yes",
            ),
        )

    async def train(self, *, skill, env, train_samples, validation_samples, work_dir):
        assert [s.task.id for s in train_samples] == [f"train-{i}" for i in range(4)]
        assert [s.task.id for s in validation_samples] == ["val-0", "val-1"]
        write_json(work_dir / "result.json", {"status": "completed", "training": {"algorithm_report": {}}})
        return skill

    monkeypatch.setattr(LiveMathEnvironment, "_run", execute)
    monkeypatch.setattr(SkillForestTrainer, "train", train)
    return calls


async def test_three_arms_use_pipeline_first_test_and_frozen_repeats(tmp_path, monkeypatch, config, inputs, offline):
    monkeypatch.setattr(runner, "load_dataset", lambda *args: inputs)
    suite = await runner.run(config, RunOptions(output_dir=tmp_path / "run", smoke=True, test_repeats=2))
    assert suite.benchmarks[0].error is None
    assert len(offline) == 14  # Six test repeats plus one external validation, each with two tasks.
    assert len({workspace for _, workspace, _ in offline}) == 14
    assert not (tmp_path / "run/livemath/trained_skill/repeat-001").exists()
    summary = read_json(tmp_path / "run/summary.json")
    assert not summary["incomplete"]
    assert len(summary["rows"]) == 6
    assert all(row["planned"] == row["completed"] == 2 for row in summary["rows"])
    assert "private-test-key" not in (tmp_path / "run/config.json").read_text()
    published = read_json(tmp_path / "run/livemath/trained_skill.json")
    assert published["version_id"] == suite.benchmarks[0].trained_version_id
    await_again = read_report(tmp_path / "run")
    write_report(await_again, tmp_path / "run")
    assert len(offline) == 14


async def test_control_only_skips_service_and_optimizer_connection(tmp_path, monkeypatch, config, inputs, offline):
    monkeypatch.setattr(runner, "load_dataset", lambda *args: inputs)
    config.models.optimizer.api_key = None

    async def forbidden(*args):
        pytest.fail("Control-only runs must not create a training service")

    monkeypatch.setattr(runner, "create_service", forbidden)
    suite = await runner.run(config, RunOptions(output_dir=tmp_path / "run", arms=["no_skill", "initial_skill"]))
    assert suite.benchmarks[0].error is None
    assert len(offline) == 4


async def test_preflight_collects_errors_before_creating_service(tmp_path, monkeypatch, config):
    checked = []

    def invalid(benchmark, *args):
        checked.append(benchmark)
        raise ValueError("missing input")

    monkeypatch.setattr(runner, "load_dataset", invalid)
    with pytest.raises(ValueError, match="Evaluation preflight failed"):
        await runner.run(config, RunOptions(benchmark="all", output_dir=tmp_path / "run"))
    assert len(checked) == 5
    assert len(read_json(tmp_path / "run/diagnostics.json")["errors"]) == 5
    assert not (tmp_path / "run/livemath/state").exists()


async def test_nonempty_output_is_rejected_before_preflight(tmp_path, config):
    (tmp_path / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        await runner.run(config, RunOptions(output_dir=tmp_path))
    assert (tmp_path / "keep.txt").read_text() == "keep"


async def test_execution_error_remains_error_and_other_tasks_continue(tmp_path, monkeypatch, config, inputs, offline):
    original = LiveMathEnvironment._run

    async def fail_one(self, **kwargs):
        if kwargs["task"].id == "test-0":
            raise RuntimeError("scoring unavailable")
        return await original(self, **kwargs)

    monkeypatch.setattr(LiveMathEnvironment, "_run", fail_one)
    results = await evaluate(
        Environment.build("livemath", inputs.environment.config), inputs.splits["test"], None, tmp_path / "test", 2
    )
    assert results[0].score is None and "scoring unavailable" in results[0].error
    assert results[1].score == 1
    arm = ArmResult(arm="no_skill", repeat=1, task_ids=[s.task.id for s in inputs.splits["test"]], cases=results)
    write_report(
        SuiteReport(output_dir=tmp_path, smoke=True, benchmarks=[BenchmarkReport(benchmark="livemath", arms=[arm])]),
        tmp_path,
    )
    row = read_json(tmp_path / "summary.json")["rows"][0]
    assert row["incomplete"] and row["mean_score"] is None and row["coverage"] == 0.5


def test_missing_usage_and_interrupted_arms_are_not_zero(tmp_path):
    case = CaseResult(task_id="a", score=0, success=False, result_path=tmp_path / "raw")
    arm = ArmResult(arm="no_skill", repeat=1, task_ids=["a"], cases=[case])
    report = SuiteReport(
        output_dir=tmp_path, smoke=False, benchmarks=[BenchmarkReport(benchmark="livemath", arms=[arm])]
    )
    write_report(report, tmp_path)
    row = read_json(tmp_path / "summary.json")["rows"][0]
    assert row["mean_score"] == 0 and row["total_tokens"] is None
    arm.task_ids.append("b")
    write_report(report, tmp_path)
    assert read_json(tmp_path / "summary.json")["incomplete"]


def test_native_environment_fields_and_smoke_config(config):
    assert config.forest(True).num_trees == config.forest(True).max_depth == 1
    document = config.model_dump()
    document["environments"]["officeqa"] = {"max_turns": 24}
    with pytest.raises(ValueError):
        EvalConfig.model_validate(document)
    with pytest.raises(ValueError):
        RunOptions(arms=["no_skill", "no_skill"])
    with pytest.raises(ValueError):
        RunOptions(test_repeats=0)


def test_synchronized_config_templates_use_native_officeqa_field():
    from mindmemos_eval.skills.config import ROOT

    example = load_config(ROOT / "config/mindmemos_eval/dev.example.yaml")
    local = load_config(ROOT / "config/mindmemos_eval/dev.yaml")
    assert example.environments["officeqa"]["max_tool_turns"] == local.environments["officeqa"]["max_tool_turns"] == 24


async def test_suite_continues_after_benchmark_failure(tmp_path, monkeypatch, config, inputs):
    monkeypatch.setattr(runner, "load_dataset", lambda *args: inputs)
    called = []

    async def execute(config, options, inputs, directory, report):
        called.append(report.benchmark)
        if report.benchmark == "alfworld":
            raise RuntimeError("simulator stopped")

    monkeypatch.setattr(runner, "_run_skill_forest", execute)
    suite = await runner.run(config, RunOptions(benchmark="all", output_dir=tmp_path / "run"))
    assert len(called) == 5
    assert "simulator stopped" in suite.benchmarks[1].error
    assert (tmp_path / "run/officeqa/results.json").is_file()
    assert read_json(tmp_path / "run/summary.json")["incomplete"]


async def test_pipeline_adoption_rejects_corrupt_index_and_wrong_version(
    tmp_path, monkeypatch, config, inputs, offline
):
    from mindmemos_eval.skills.evaluation import read_pipeline_test
    from mindmemos_skill.skill import Skill, SkillSnapshot

    monkeypatch.setattr(runner, "load_dataset", lambda *args: inputs)
    suite = await runner.run(config, RunOptions(output_dir=tmp_path / "run", arms=["trained_skill"]))
    assert suite.benchmarks[0].error is None
    directory = tmp_path / "run/livemath/training/test"
    snapshot = SkillSnapshot.model_validate(read_json(directory.parent / "selected_skill.json"))
    env = Environment.build("livemath", inputs.environment.config)
    ids = [sample.task.id for sample in inputs.splits["test"]]
    data = read_json(directory / "results.json")
    data["tasks"][0]["score"] = 0
    write_json(directory / "results.json", data)
    with pytest.raises(ValueError, match="disagrees"):
        read_pipeline_test(directory, ids, snapshot, env, inputs.splits["test"])
    wrong = Skill.from_path(inputs.initial_skill).snapshot().model_dump(mode="json")
    wrong["description"] = "Other version"
    write_json(directory.parent / "selected_skill.json", wrong)
    with pytest.raises(ValueError, match="snapshot"):
        read_pipeline_test(directory, ids, snapshot, env, inputs.splits["test"])


async def test_report_marks_missing_original_rollouts_incomplete(tmp_path, monkeypatch, config, inputs, offline):
    monkeypatch.setattr(runner, "load_dataset", lambda *args: inputs)
    suite = await runner.run(config, RunOptions(output_dir=tmp_path / "run", arms=["no_skill"]))
    suite.benchmarks[0].arms[0].cases[0].result_path.unlink()
    report = read_report(tmp_path / "run")
    write_report(report, tmp_path / "run")
    summary = read_json(tmp_path / "run/summary.json")
    assert summary["incomplete"] and summary["rows"][0]["mean_score"] is None
    assert len(offline) == 2
