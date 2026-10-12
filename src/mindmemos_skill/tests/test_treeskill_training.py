"""End-to-end training artifacts, held-out separation and reference analysis gates."""

import json
import logging
from pathlib import Path

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainer, TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_text_action import (
    SpreadsheetBenchReferenceAnalyzer,
)
from mindmemos_skill.algos.train.treeskill.trainer import run_experiment
from mindmemos_skill.algos.train.treeskill.typing import TraceEvidence, TrainingRollout
from mindmemos_skill.envs import Reward, RolloutResult, Task
from mindmemos_skill.envs.builtin_envs.spreadsheetbench import SpreadsheetBenchTextActionConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.environment import (
    SpreadsheetBenchTextActionEnvironment,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import (
    SpreadsheetBenchRewardData,
    SpreadsheetBenchTask,
    WorkbookInput,
    WorkbookReference,
)
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.logging.tracing import bind_tracer
from mindmemos_skill.skill import Skill, SkillContent, SkillSet
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from . import treeskill_smoke as smoke


def make_trainer():
    model = OpenAIChatConfig(model="offline", api_key="explicit-test-key")
    return TreeSkillTrainer(TreeSkillTrainerConfig(optimizer=model, router=model), agent_factory=smoke.SmokeAgent)


def make_sample(identifier):
    return DataSample(task=Task(id=identifier, instruction="Save and verify"), reward_data=RewardData())


@pytest.mark.asyncio
async def test_routed_empty_selection_accepts_a_standard_initial_skill(tmp_path):
    """An explicit no-Skill route is a valid execution, not a source validation error."""
    from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
    from mindmemos_skill.algos.train.treeskill.trainer import sample_tasks

    class EmptyRouter(OpenAIChatAgent):
        async def run(self, **kwargs):
            return AgentResult(output='{"selected_subtree_ids":[]}')

    router = EmptyRouter(OpenAIChatConfig(model="offline"))
    source = SkillContent(content="---\nname: workbook\ndescription: Edit workbooks.\n---\n# Save\nSave output.\n")
    skill = TreeSkill(TreeSkillConfig(router=router.config), source=source, router=router)
    records, failures = await sample_tasks(
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        skill=skill,
        samples=[make_sample("test")],
        directory=tmp_path / "test",
        repeats=1,
        concurrency=1,
    )
    assert len(records) == 1 and not failures
    assert records[0].result.reward.score == 0
    selection = next((tmp_path / "test").glob("*/attempt-*/routing/selection.json"))
    assert json.loads(selection.read_text())["content"] == ""


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
    assert config["algorithm"]["optimizer"]["model"] == "[REDACTED]"
    training_config = json.loads((root / "training/config.json").read_text())
    assert training_config == config
    training_calls = list((root / "training/items").glob("**/outcome.json"))
    assert len(training_calls) == 3
    assert {json.loads(path.read_text())["task"] for path in training_calls} == {
        "treeskill_trajectory_analysis",
        "treeskill_evidence_localization",
        "treeskill_node_fusion",
    }
    assert all('"task_id": "test"' not in path.read_text() for path in training_calls)


@pytest.mark.asyncio
async def test_empty_test_and_val_do_not_evaluate_or_route(tmp_path):
    trainer = make_trainer()
    root = tmp_path / "empty-test"
    await run_experiment(
        trainer=trainer,
        skill=Skill(
            source=SkillContent(
                content="# Workflow\nSave the output.\n",
                name="test-skill",
                description="Task instructions",
                standard_format=False,
            )
        ),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=root,
    )
    summary = json.loads((root / "summary.json").read_text())
    assert summary["test"] is None and summary["val"] is None
    assert summary["model_usage"]["calls"] == 3
    assert not (root / "test").exists()
    assert not (root / "training/model_calls/router").exists()
    assert "explicit-test-key" in "\n".join(path.read_text() for path in root.rglob("*.json"))
    with pytest.raises(FileExistsError):
        await run_experiment(
            trainer=trainer,
            skill=Skill(
                source=SkillContent(
                    content="# X\nx", name="test-skill", description="Task instructions", standard_format=False
                )
            ),
            env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
            train_samples=[make_sample("train")],
            work_dir=root,
        )


@pytest.mark.asyncio
async def test_overlap_is_rejected_before_any_rollout(tmp_path):
    with pytest.raises(ValueError, match="Duplicate task ID"):
        await run_experiment(
            trainer=make_trainer(),
            skill=Skill(
                source=SkillContent(
                    content="# X\nx", name="test-skill", description="Task instructions", standard_format=False
                )
            ),
            env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
            train_samples=[make_sample("same")],
            test_samples=[make_sample("same")],
            work_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.asyncio
async def test_collection_failures_are_not_zero_score_evidence(tmp_path):
    class FailingEnvironment(smoke.SmokeEnvironment):
        async def _run(self, **kwargs):
            raise RuntimeError("execution failed")

    root = tmp_path / "failed"
    await run_experiment(
        trainer=make_trainer(),
        skill=Skill(
            source=SkillContent(
                content="# X\nx", name="test-skill", description="Task instructions", standard_format=False
            )
        ),
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
        cases=[WorkbookInput(number="1", path=paths[0])],
    )
    reward = SpreadsheetBenchRewardData(
        cases=[WorkbookReference(number="1", path=paths[1])],
        answer_position="Sheet!A1",
    )
    return DataSample(task=task, reward_data=reward)


class ScriptedAgent(OpenAIChatAgent):
    def __init__(self, outputs):
        super().__init__(OpenAIChatConfig(model="offline"))
        self.outputs = iter(outputs)
        self.calls = []

    async def run_messages(self, *, messages, tools=None, response_format=None):
        self.calls.append({"messages": list(messages)})
        return AgentResult(output=next(self.outputs))


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
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation_cli import stage_task_local_helper

    stage_task_local_helper(work / "workspace" / case_directory)

    report = "# Failure Memory Item 1\n\n## Title\nVerify\n\n## Description\nCheck saved values.\n\n## Content\nVerify preserved cells after saving.\n\n## Skill Reflection\nAdd verification.\n\nACTION: TASK_COMPLETE"
    model = ScriptedAgent(
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
        agent=model,
        failure_agent=model,
        success_score_threshold=1,
        max_turns=5,
    )
    from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_text_action import (
        TreeSkillSpreadsheetBenchReferenceEnvAdapter,
    )

    trainer = make_trainer()
    adapter = TreeSkillSpreadsheetBenchReferenceEnvAdapter()
    records, failures = await trainer._analyze(
        [TraceEvidence(trajectory_id="train-1", task_id="sheet", transcript="", score=0)],
        [trajectory],
        analyzer,
        adapter,
        tmp_path / "analysis",
    )
    assert not failures and len(records) == 1
    assert "[System Check]" in model.calls[1]["messages"][-1]["content"]
    prompt = model.calls[0]["messages"][1]["content"]
    command = prompt.split("run this exact command before signaling completion:\n")[1].splitlines()[0]
    import shlex

    assert shlex.split(command)[1:3] == ["./.treeskill_recalculate.py", "agent_work/output.xlsx"]
    assert str(work / "workspace") not in command

    assert list((tmp_path / "analysis").rglob("evaluate_passed.flag"))
    assert not (work / "workspace/gold.xlsx").exists()
    assert list((tmp_path / "analysis").rglob("agent_work/gold.xlsx"))
    before_calls = len(model.calls)
    recovered, recovered_failures = await trainer._analyze(
        [TraceEvidence(trajectory_id="train-1", task_id="sheet", transcript="", score=0)],
        [trajectory],
        analyzer,
        adapter,
        tmp_path / "analysis",
    )
    assert recovered == records and not recovered_failures
    assert len(model.calls) == before_calls


async def test_reference_policy_uses_original_messages_and_private_scoring(tmp_path):
    sample = spreadsheet_sample(tmp_path)
    config = OpenAIChatConfig(model="offline")

    class Agent(OpenAIChatAgent):
        def __init__(self):
            super().__init__(config)
            self.calls = []

        async def run(self, *, messages):
            self.calls.append(list(messages))
            output = (
                'Action:\n{"name":"bash","arguments":{"command":"cp input.xlsx output.xlsx"}}'
                if len(self.calls) == 1
                else "ACTION: TASK_COMPLETE"
            )
            return AgentResult(output=output, trajectory=[], usage={})

    agent = Agent()
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=config, transactional_recalculation=False), agent=agent
    )
    result = await env.rollout(
        sample.task,
        tmp_path / "run",
        reward_data=sample.reward_data,
        skill_set=SkillSet(
            {
                "0": Skill(
                    source=SkillContent(
                        content="# Workbook\nPreserve cells.",
                        name="test-skill",
                        description="Task instructions",
                        standard_format=False,
                    )
                )
            },
            max_concurrency=1,
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
        async def _run(self, *, task, **kwargs):
            if task.id == "bad":
                raise RuntimeError("Provider unavailable")
            return await super()._run(task=task, **kwargs)

    root = tmp_path / "mixed"
    final = await run_experiment(
        trainer=make_trainer(),
        skill=Skill(
            source=SkillContent(
                content="# Workflow\nSave the output.\n",
                name="test-skill",
                description="Task instructions",
                standard_format=False,
            )
        ),
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
    assert "Verify the saved output." in final.content
    evidence = json.loads((root / "training/stages/evidence.json").read_text())
    assert [item["task_id"] for item in evidence] == ["good"]


@pytest.mark.asyncio
async def test_reference_analysis_request_failure_does_not_cancel_other_records(tmp_path):
    class MixedAgent(OpenAIChatAgent):
        async def run_messages(self, *, messages, tools=None, response_format=None):
            if "FAIL_PROVIDER" in messages[-1]["content"]:
                raise RuntimeError("Provider unavailable")
            return AgentResult(
                output="# Success Memory Item 1\n\n## Title\nVerify\n\n## Description\nChecked the result.\n\n## Content\nVerify saved output.\n"
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
        agent=MixedAgent(OpenAIChatConfig(model="offline")),
        failure_agent=MixedAgent(OpenAIChatConfig(model="offline")),
        success_score_threshold=1,
    )
    from mindmemos_skill.algos.train.treeskill.env_adapter import TreeSkillEnvAdapter

    records, failures = await make_trainer()._analyze(
        evidence, list(trajectories.values()), analyzer, TreeSkillEnvAdapter(), tmp_path / "analysis"
    )
    assert [record.instance_id for record in records] == ["good"]
    assert failures == ["bad"]
    assert "Provider unavailable" in "".join(path.read_text() for path in (tmp_path / "analysis").rglob("outcome.json"))


@pytest.mark.asyncio
async def test_router_provider_failure_uses_full_skill_reference_fallback(tmp_path):
    class FailingRouter(smoke.SmokeAgent):
        async def run_messages(self, *, messages, **kwargs):
            if messages[-1]["content"].startswith("Route"):
                raise RuntimeError("Router unavailable")
            return await super().run_messages(messages=messages, **kwargs)

    trainer = make_trainer()
    trainer.agent_factory = FailingRouter
    root = tmp_path / "router-failure"
    await run_experiment(
        trainer=trainer,
        skill=Skill(
            source=SkillContent(
                content="# Workflow\nSave the output.\n",
                name="test-skill",
                description="Task instructions",
                standard_format=False,
            )
        ),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        test_samples=[make_sample("test")],
        work_dir=root,
    )
    summary = json.loads((root / "summary.json").read_text())
    assert summary["test"]["mean_score"] == 1 and summary["test"]["skipped"] == 0
    assert summary["status"] == "completed"
    selection = next((root / "test").glob("*/attempt-*/routing/selection.json"))
    assert "Router unavailable" in json.loads(selection.read_text())["fallback_reason"]


def test_reference_split_is_pinned_without_live_dependencies():
    root = Path(__file__).resolve().parents[3]
    splits = json.loads(
        (root / "resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json").read_text()
    )
    assert len(splits["train"]) == 200 and len(splits["test"]) == 200 and splits["val"] == []
    assert not set(splits["train"]) & set(splits["test"])


async def test_transactional_recalculation_preserves_literal_workbook(tmp_path):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import recalculate_workbook

    sample = spreadsheet_sample(tmp_path)
    path = sample.task.cases[0].path
    before = path.read_bytes()
    result = await recalculate_workbook(path)
    assert result["status"] == "not_needed"
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_trained_snapshot_preserves_explicit_credentials(tmp_path):
    from mindmemos_skill.algos.train.treeskill import TreeSkill

    trainer = make_trainer()
    final = await trainer.train(
        skill=Skill(
            source=SkillContent(
                content="# Workflow\nSave the output.\n",
                name="test-skill",
                description="Task instructions",
                standard_format=False,
            )
        ),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=tmp_path / "run",
    )
    snapshot = final.snapshot()
    assert snapshot.config["router"]["api_key"] == "explicit-test-key"
    restored = TreeSkill.from_snapshot(snapshot)
    assert restored.config.router.api_key == "explicit-test-key"
    injected = smoke.SmokeAgent(final.config.router)
    restored = TreeSkill.from_snapshot(snapshot, router=injected)
    assert restored.router is injected
    assert "Verify the saved output." in (await restored.assemble(make_sample("new-task").task)).content


def test_treeskill_environment_inherits_scoring_and_rejects_changed_workbooks(tmp_path):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import SpreadsheetBenchToolCallingEnvironment

    sample = spreadsheet_sample(tmp_path)
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"), transactional_recalculation=False)
    )
    assert type(env)._run is SpreadsheetBenchToolCallingEnvironment._run
    assert type(env)._interact is SpreadsheetBenchToolCallingEnvironment._interact
    assert env.config.stagnation_limit == 5
    identity = env.input_identity(sample.task, sample.reward_data)
    sample.task.cases[0].path.write_bytes(b"changed after loading")
    assert env.input_identity(sample.task, sample.reward_data) != identity


def test_text_action_config_uses_explicit_stagnation_option():
    config = SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"), stagnation_limit=7)
    assert config.stagnation_limit == 7
    assert config.model_dump()["stagnation_limit"] == 7


@pytest.mark.asyncio
async def test_internal_stages_correlate_with_training_and_held_out_test(tmp_path, caplog):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("treeskill-test")
    try:
        with (
            caplog.at_level(logging.INFO),
            bind_tracer(tracer),
            tracer.start_as_current_span("algorithm.train") as parent,
        ):
            summary = await smoke.run_smoke(tmp_path / "staged")
        spans = [span for span in exporter.get_finished_spans() if span.name.startswith("treeskill.")]
        assert [span.name for span in spans] == [
            "treeskill.rollout",
            "treeskill.analyze",
            "treeskill.merge",
            "treeskill.test",
        ]
        for span in spans:
            assert span.parent.span_id == parent.get_span_context().span_id
            assert span.context.trace_id == parent.get_span_context().trace_id
            assert span.status.status_code.name == "OK"
            assert span.attributes["stage"] == span.name
            assert span.attributes["duration_ms"] >= 0
        records = [record for record in caplog.records if record.msg == "stage.completed"]
        assert {record.event_body["stage"] for record in records} == {span.name for span in spans}
        assert all(record.event_body["duration_ms"] >= 0 for record in records)
        assert summary["test"]["completed"] == 1
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_rollout_stage_records_uncaught_failure_and_stops_later_stages(tmp_path, monkeypatch, caplog):
    from mindmemos_skill.algos.train.treeskill import trainer as trainer_module

    async def fail_collection(**kwargs):
        raise RuntimeError("Collection infrastructure failed")

    monkeypatch.setattr(trainer_module, "sample_tasks", fail_collection)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with caplog.at_level(logging.INFO), bind_tracer(provider.get_tracer("treeskill-test")):
            with pytest.raises(RuntimeError, match="Collection infrastructure failed"):
                await smoke.run_smoke(tmp_path / "failed-stage")
        spans = exporter.get_finished_spans()
        assert [span.name for span in spans] == ["treeskill.rollout"]
        assert spans[0].status.status_code.name == "ERROR"
        assert any(event.name == "exception" for event in spans[0].events)
        failure = next(record for record in caplog.records if record.msg == "stage.failed")
        assert failure.event_body["stage"] == "treeskill.rollout"
        assert failure.event_body["duration_ms"] >= 0
        assert failure.exc_info or failure.exc_text
    finally:
        provider.shutdown()


async def test_training_router_applies_reference_budget_to_actual_requests(tmp_path):
    observed = []

    class CaptureAgent(smoke.SmokeAgent):
        async def run_messages(self, **kwargs):
            if kwargs["messages"][-1]["content"].startswith("Route"):
                observed.append((self.config.temperature, self.config.max_completion_tokens))
            return await super().run_messages(**kwargs)

    trainer = make_trainer()
    trainer.agent_factory = CaptureAgent
    source = Skill(
        source=SkillContent(
            content="# Workflow\nSave the output.\n", name="test", description="Test", standard_format=False
        )
    )
    final = await trainer.train(
        skill=source,
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[make_sample("train")],
        work_dir=tmp_path / "run",
    )
    assert "Verify" in (await final.assemble(Task(id="test", instruction="Verify"))).content
    assert observed == [(0.0, 16384)]
    assert trainer.config.router.max_completion_tokens != 16384
