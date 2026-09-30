"""End-to-end training artifacts, held-out separation and reference analysis gates."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainer, TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.env_specific.config import SpreadsheetBenchReferenceConfig
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench import (
    SpreadsheetBenchReferenceAnalyzer,
    SpreadsheetBenchReferenceEnvironment,
)
from mindmemos_skill.algos.train.treeskill.experiment import run_experiment
from mindmemos_skill.algos.train.treeskill.typing import ChatResponse, TraceEvidence, TrainingRollout
from mindmemos_skill.envs import Reward, RolloutResult, Task
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
    WorkbookInput,
    WorkbookReference,
)
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.skill import Skill, SkillBundle, SkillSet

spec = importlib.util.spec_from_file_location(
    "treeskill_smoke", Path(__file__).resolve().parents[2] / "scripts/treeskill/smoke.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def make_trainer():
    model = OpenAIChatConfig(model="offline", api_key="must-not-be-recorded")
    return TreeSkillTrainer(TreeSkillTrainerConfig(optimizer=model, router=model), agent_factory=smoke.SmokeAgent)


def make_sample(identifier):
    return DataSample(task=Task(id=identifier, instruction="Save and verify"), reward_data=RewardData())


@pytest.mark.asyncio
async def test_smoke_preserves_patch_history_and_only_tests_final_skill(tmp_path):
    root = tmp_path / "run"
    summary = await smoke.run_smoke(root)
    assert summary["test"]["completed"] == 1
    assert summary["training"]["input_trajectories"] == 1
    patch = next((root / "training/patches").glob("*/patch.json"))
    payload = json.loads(patch.read_text())
    assert payload["result"]["accepted"]
    assert payload["proposal"]["content"] == "Save the output.\nVerify the saved output."
    assert "+Verify the saved output." in patch.with_name("change.diff").read_text()
    assert "Verify the saved output." not in patch.with_name("before.md").read_text()
    assert "Verify the saved output." in patch.with_name("after.md").read_text()
    assert summary["model_usage"]["calls"] == 4
    assert summary["duration_s"] >= sum(summary["stages"].values())
    config = json.loads((root / "config.json").read_text())
    assert set(config) == {"algorithm"}
    assert config["algorithm"]["optimizer"]["model"] == "offline-smoke"
    training_config = json.loads((root / "training/config.json").read_text())
    assert training_config == config
    training_calls = list((root / "training/model_calls/optimizer").glob("**/attempt-*.json"))
    assert all('"task_id": "test"' not in path.read_text() for path in training_calls)


@pytest.mark.asyncio
async def test_empty_test_and_val_do_not_evaluate_or_route(tmp_path):
    trainer = make_trainer()
    root = tmp_path / "empty-test"
    await run_experiment(
        trainer=trainer,
        skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"})),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=root,
    )
    summary = json.loads((root / "summary.json").read_text())
    assert summary["test"] is None and summary["val"] is None
    assert summary["model_usage"]["calls"] == 3
    assert not (root / "test").exists()
    assert not (root / "training/model_calls/router").exists()
    assert "must-not-be-recorded" not in "\n".join(path.read_text() for path in root.rglob("*.json"))
    with pytest.raises(FileExistsError):
        await run_experiment(
            trainer=trainer,
            skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# X\nx"})),
            env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
            train_samples=[make_sample("train")],
            work_dir=root,
        )


@pytest.mark.asyncio
async def test_overlap_is_rejected_before_any_rollout(tmp_path):
    with pytest.raises(ValueError, match="Duplicate task ID"):
        await run_experiment(
            trainer=make_trainer(),
            skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# X\nx"})),
            env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
            train_samples=[make_sample("same")],
            test_samples=[make_sample("same")],
            work_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.asyncio
async def test_collection_failures_are_not_zero_score_evidence(tmp_path):
    class FailingEnvironment(smoke.SmokeEnvironment):
        def _run(self, **kwargs):
            raise RuntimeError("execution failed")

    root = tmp_path / "failed"
    await run_experiment(
        trainer=make_trainer(),
        skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# X\nx"})),
        env=FailingEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=root,
    )
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "partial"
    assert summary["train"]["mean_score"] is None
    assert summary["training"]["input_trajectories"] == 0
    assert summary["model_usage"]["calls"] == 0
    assert list((root / "training/rollouts").glob("**/error.json"))


def spreadsheet_sample(tmp_path, identifier="sheet"):
    import hashlib

    import openpyxl

    paths = []
    for name in ("input.xlsx", "reference.xlsx"):
        path = tmp_path / name
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = 42
        workbook.save(path)
        workbook.close()
        paths.append(path)
    task = SpreadsheetBenchTask(
        id=identifier,
        instruction="Preserve A1.",
        answer_position="Sheet!A1",
        cases=[WorkbookInput(number="1", path=paths[0], sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest())],
    )
    reward = SpreadsheetBenchRewardData(
        cases=[WorkbookReference(number="1", path=paths[1], sha256=hashlib.sha256(paths[1].read_bytes()).hexdigest())],
        answer_position="Sheet!A1",
        source_sha256="fixture",
    )
    return DataSample(task=task, reward_data=reward)


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    async def chat(self, task, messages, **kwargs):
        self.calls.append({"task": task, "messages": messages})
        return ChatResponse(content=next(self.outputs))


@pytest.mark.asyncio
@pytest.mark.parametrize("case_directory", ["", "1"])
async def test_reference_failure_analysis_requires_pass_and_keeps_gold_out_of_policy_workspace(
    tmp_path, case_directory
):
    sample = spreadsheet_sample(tmp_path)
    work = tmp_path / "rollout"
    (work / "workspace" / case_directory).mkdir(parents=True)
    import shutil

    shutil.copyfile(sample.task.cases[0].path, work / "workspace" / case_directory / "input.xlsx")
    report = "# Failure Memory Item 1\n\n## Title\nVerify\n\n## Description\nCheck saved values.\n\n## Content\nVerify preserved cells after saving.\n\n## Skill Reflection\nAdd verification.\n\nACTION: TASK_COMPLETE"
    model = ScriptedModel(
        [
            report,
            'Action:\n{"name":"bash","arguments":{"command":"cp agent_work/input.xlsx agent_work/output.xlsx"}}',
            'Action:\n{"name":"evaluate_output","arguments":{"output_file":"agent_work/output.xlsx","ground_truth":"agent_work/gold.xlsx"}}',
            report,
        ]
    )
    trajectory = TrainingRollout(
        trajectory_id="train-1",
        sample=sample,
        work_dir=work,
        result=RolloutResult(
            output="missing",
            trajectory=[{"role": "assistant", "content": "Did not save."}],
            reward=Reward(score=0, success=False, detail="Missing output"),
        ),
    )
    analyzer = SpreadsheetBenchReferenceAnalyzer(
        chat_model=model,
        task="analysis",
        output_root=tmp_path / "analysis",
        concurrency=1,
        success_score_threshold=1,
        temperature=1,
        max_tokens=4096,
        max_turns=5,
    )
    records, failures = await analyzer.analyze(
        [TraceEvidence(trajectory_id="train-1", task_id="sheet", transcript="", score=0)],
        trajectories_by_id={"train-1": trajectory},
    )
    assert not failures and len(records) == 1
    assert "[System Check]" in model.calls[1]["messages"][-1]["content"]
    assert (tmp_path / "analysis/error/train-1/evaluate_passed.flag").exists()
    assert not (work / "workspace/gold.xlsx").exists()
    assert (tmp_path / "analysis/error/train-1/agent_work/gold.xlsx").exists()


def test_reference_policy_uses_original_messages_and_private_scoring(tmp_path):
    sample = spreadsheet_sample(tmp_path)
    config = OpenAIChatConfig(model="offline")

    class Agent(OpenAIChatAgent):
        def __init__(self):
            super().__init__(config)
            self.calls = []

        def run(self, *, messages):
            self.calls.append(list(messages))
            output = (
                'Action:\n{"name":"bash","arguments":{"command":"cp input.xlsx output.xlsx"}}'
                if len(self.calls) == 1
                else "ACTION: TASK_COMPLETE"
            )
            return AgentResult(output=output, trajectory=[], usage={})

    agent = Agent()
    env = SpreadsheetBenchReferenceEnvironment(
        SpreadsheetBenchReferenceConfig(agent=config, transactional_recalculation=False), agent=agent
    )
    result = env.rollout(
        sample.task,
        tmp_path / "run",
        reward_data=sample.reward_data,
        skill_set=SkillSet(
            {"0": Skill(bundle=SkillBundle(files={"SKILL.md": "# Workbook\nPreserve cells."}))}, max_concurrency=1
        ),
        resume=False,
    )
    assert result.reward.score == 1
    assert all("reference.xlsx" not in json.dumps(call) and "gold.xlsx" not in json.dumps(call) for call in agent.calls)
    assert (tmp_path / "run/workspace/1/SKILL.md").exists()
    assert not (tmp_path / "run/workspace/gold.xlsx").exists()


@pytest.mark.asyncio
async def test_mixed_collection_failure_skips_only_that_data(tmp_path):
    class MixedEnvironment(smoke.SmokeEnvironment):
        def _run(self, *, task, **kwargs):
            if task.id == "bad":
                raise RuntimeError("Provider unavailable")
            return super()._run(task=task, **kwargs)

    root = tmp_path / "mixed"
    final = await run_experiment(
        trainer=make_trainer(),
        skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"})),
        env=MixedEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("bad"), make_sample("good")],
        test_samples=[make_sample("test")],
        work_dir=root,
    )
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "partial"
    assert summary["train"]["completed"] == 1 and summary["train"]["skipped"] == 1
    assert summary["train"]["score_sum"] == 0 and summary["train"]["mean_score"] == 0
    assert summary["test"]["mean_score"] == 1
    assert summary["training"]["input_trajectories"] == 1
    assert "Verify the saved output." in final.bundle.files["SKILL.md"]
    evidence = json.loads((root / "training/stages/evidence.json").read_text())
    assert [item["task_id"] for item in evidence] == ["good"]


@pytest.mark.asyncio
async def test_reference_analysis_request_failure_does_not_cancel_other_records(tmp_path):
    class MixedModel:
        async def chat(self, task, messages, **kwargs):
            if "FAIL_PROVIDER" in messages[-1]["content"]:
                raise RuntimeError("Provider unavailable")
            return ChatResponse(
                content="# Success Memory Item 1\n\n## Title\nVerify\n\n## Description\nChecked the result.\n\n## Content\nVerify saved output.\n"
            )

    trajectories = {}
    evidence = []
    for name, text in (("bad", "FAIL_PROVIDER"), ("good", "Saved the output")):
        trajectories[name] = TrainingRollout(
            trajectory_id=name,
            sample=make_sample(name),
            work_dir=tmp_path,
            result=RolloutResult(
                output="done",
                trajectory=[{"role": "assistant", "content": text}],
                reward=Reward(score=1, success=True, detail="Passed"),
            ),
        )
        evidence.append(TraceEvidence(trajectory_id=name, task_id=name, transcript=text, score=1))
    analyzer = SpreadsheetBenchReferenceAnalyzer(
        chat_model=MixedModel(),
        task="analysis",
        output_root=tmp_path / "analysis",
        concurrency=2,
        success_score_threshold=1,
        temperature=1,
        max_tokens=4096,
    )
    records, failures = await analyzer.analyze(evidence, trajectories_by_id=trajectories)
    assert [record.instance_id for record in records] == ["good"]
    assert failures == ["bad"]
    assert "Provider unavailable" in (tmp_path / "analysis/failures/bad.txt").read_text()


@pytest.mark.asyncio
async def test_router_provider_failure_is_a_skipped_test_not_full_skill_fallback(tmp_path):
    class FailingRouter(smoke.SmokeAgent):
        def run_messages(self, *, messages, **kwargs):
            if messages[-1]["content"].startswith("Route"):
                raise RuntimeError("Router unavailable")
            return super().run_messages(messages=messages, **kwargs)

    trainer = make_trainer()
    trainer.agent_factory = FailingRouter
    root = tmp_path / "router-failure"
    await run_experiment(
        trainer=trainer,
        skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"})),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        test_samples=[make_sample("test")],
        work_dir=root,
    )
    summary = json.loads((root / "summary.json").read_text())
    assert summary["test"]["mean_score"] is None and summary["test"]["skipped"] == 1
    assert summary["status"] == "partial"
    assert summary["model_usage"]["failed_calls"] == 1


def test_reference_config_and_split_are_pinned_without_live_dependencies():
    import yaml
    from mindmemos_skill.algos.train.treeskill.config import TreeSkillExperimentConfig

    root = Path(__file__).resolve().parents[2]
    config = TreeSkillExperimentConfig.model_validate(
        yaml.safe_load((root / "config/mindmemos_skill/treeskill/spreadsheetbench_reference.yaml").read_text())
    )
    splits = json.loads((root / config.split_file).read_text())
    assert len(splits["train"]) == 200 and len(splits["test"]) == 200 and splits["val"] == []
    assert not set(splits["train"]) & set(splits["test"])
    assert config.algorithm.analysis_concurrency == 128 and config.algorithm.localization_concurrency == 16
    assert config.algorithm.optimizer.extra_body["chat_template_kwargs"]["enable_thinking"]
    assert config.algorithm.failure_optimizer.extra_body["chat_template_kwargs"]["enable_thinking"] is False
    assert config.strict_reference_skill


def test_transactional_recalculation_preserves_literal_workbook(tmp_path):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import recalculate_workbook

    sample = spreadsheet_sample(tmp_path)
    path = sample.task.cases[0].path
    before = path.read_bytes()
    result = recalculate_workbook(path)
    assert result["status"] == "not_needed"
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_trained_snapshot_reloads_credentials_without_persisting_masks(tmp_path, monkeypatch):
    from mindmemos_skill.agents.constants import AGENT_API_KEY_ENV
    from mindmemos_skill.algos.train.treeskill import TreeSkill

    trainer = make_trainer()
    final = await trainer.train(
        skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"})),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=tmp_path / "run",
    )
    snapshot = final.snapshot()
    assert "api_key" not in snapshot.config["router"]
    monkeypatch.setenv(AGENT_API_KEY_ENV, "restored-secret")
    restored = TreeSkill.from_snapshot(snapshot)
    assert restored.config.router.api_key.get_secret_value() == "restored-secret"
    legacy = snapshot.model_copy(deep=True)
    legacy.config["router"]["api_key"] = "**********"
    assert TreeSkill.from_snapshot(legacy).config.router.api_key.get_secret_value() == "restored-secret"
    injected = smoke.SmokeAgent(trainer.config.router)
    restored = TreeSkill.from_snapshot(snapshot, router=injected)
    assert restored.router is injected
    assert (
        "Verify the saved output."
        in (await asyncio.to_thread(restored.assemble, make_sample("new-task").task)).bundle.files["SKILL.md"]
    )


def test_standalone_uno_worker_has_no_package_import_dependency():
    import subprocess
    import sys

    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import recalculation

    worker = Path(recalculation.__file__).with_name("runtime") / "uno_worker.py"
    result = subprocess.run([sys.executable, "-I", str(worker), "--help"], capture_output=True, text=True, check=True)
    assert "--pipe-name" in result.stdout


def test_treeskill_environment_inherits_scoring_and_rejects_changed_workbooks(tmp_path):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import SpreadsheetBenchEnvironment

    sample = spreadsheet_sample(tmp_path)
    env = SpreadsheetBenchReferenceEnvironment(
        SpreadsheetBenchReferenceConfig(agent=OpenAIChatConfig(model="offline"), transactional_recalculation=False)
    )
    assert type(env)._run is SpreadsheetBenchEnvironment._run
    assert type(env)._interact is SpreadsheetBenchEnvironment._interact
    assert env.config.stagnation_limit == 5
    sample.task.cases[0].path.write_bytes(b"changed after loading")
    with pytest.raises(ValueError, match="workbook changed"):
        env.rollout(sample.task, tmp_path / "run", reward_data=sample.reward_data)


def test_reference_config_accepts_previous_stagnation_option():
    config = SpreadsheetBenchReferenceConfig(agent=OpenAIChatConfig(model="offline"), reference_stagnation_limit=7)
    assert config.stagnation_limit == 7
    assert config.model_dump()["stagnation_limit"] == 7
