"""Registered adapter extension, live call paths and snapshot compatibility."""

import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.skill_forest import (
    SkillForestEnvAdapter,
    SkillForestEnvAdapterConfig,
    SkillForestTrainer,
    SkillForestTrainerConfig,
)
from mindmemos_skill.algos.train.skill_forest.generation import generate_patch
from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime
from mindmemos_skill.algos.train.skill_forest.typing import SkillForestReflectionContext, SkillForestTrajectoryContext
from mindmemos_skill.algos.train.treeskill import (
    TreeSkill,
    TreeSkillConfig,
    TreeSkillEnvAdapter,
    TreeSkillTrainer,
    TreeSkillTrainerConfig,
)
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench import SpreadsheetBenchReferenceAnalyzer
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_prompts import ROUTING_SYSTEM_PROMPT
from mindmemos_skill.algos.train.treeskill.fusion_tree import compile_tree_metadata, parse_skill_markdown
from mindmemos_skill.algos.train.treeskill.typing import ChatResponse, TraceEvidence, TreeSkillAnalysisRuntime
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.skill import Skill, SkillBundle
from pydantic import ValidationError


class CustomForestConfig(SkillForestEnvAdapterConfig):
    """Test a user extension with one additional option."""

    prefix: str = "CUSTOM"
    """Text prepended by the user renderer."""


@SkillForestEnvAdapter.register("adapter_contract_test")
class SkillForestTestEnvAdapter(SkillForestEnvAdapter):
    """Override only rendering while inheriting all algorithm stages."""

    config_type = CustomForestConfig
    """Validate the extra renderer option through the existing registry."""

    def render_trajectory(self, context):
        return f"{self.config.prefix}: {context.task.instruction}"


@TreeSkillEnvAdapter.register("adapter_contract_test")
class TreeSkillTestEnvAdapter(TreeSkillEnvAdapter):
    """Use the same registration name in an independent algorithm family."""

    def render_task(self, task):
        return f"CUSTOM PUBLIC TASK: {task.instruction}"

    def render_trajectory(self, context):
        return f"CUSTOM TRACE: {context.transcript}"


class PrivateData(RewardData):
    """Sentinel that must never enter generic adapter contexts."""

    answer: str = "PRIVATE-REWARD-SENTINEL"
    """Private scoring answer, not an algorithm prompt."""


def sample():
    return DataSample(task=Task(id="task", instruction="Inspect output"), reward_data=PrivateData())


def result():
    return RolloutResult(
        output="done",
        trajectory=[{"role": "assistant", "content": "Checked output"}],
        reward=Reward(score=1, success=True, detail="PRIVATE-DETAIL-SENTINEL"),
    )


def test_default_and_registered_adapter_initialization_is_isolated():
    config = SkillForestTrainerConfig(
        optimizer={"model": "offline"},
        env_adapter={"class_name": "adapter_contract_test", "config": {"prefix": "FIRST"}},
    )
    first, second = SkillForestTrainer(config), SkillForestTrainer(config)
    config.env_adapter.config["prefix"] = "CHANGED"
    assert first.env_adapter is not second.env_adapter
    assert first.env_adapter.config.prefix == second.env_adapter.config.prefix == "FIRST"
    assert first.config.env_adapter.config["prefix"] == "FIRST"
    assert isinstance(SkillForestEnvAdapter.build("adapter_contract_test", {}), SkillForestTestEnvAdapter)
    assert isinstance(TreeSkillEnvAdapter.build("adapter_contract_test", {}), TreeSkillTestEnvAdapter)
    assert (
        type(SkillForestTrainer(SkillForestTrainerConfig(optimizer={"model": "offline"})).env_adapter).__name__
        == "SkillForestGenericEnvAdapter"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        SkillForestEnvAdapter.register("adapter_contract_test")(SkillForestTestEnvAdapter)
    with pytest.raises(TypeError, match="subclass"):
        SkillForestEnvAdapter.register("wrong_family")(TreeSkillTestEnvAdapter)


@pytest.mark.parametrize(
    "selection",
    [
        {"class_name": "missing_adapter"},
        {"class_name": "generic", "config": {"typo": True}},
        {"class_name": "adapter_contract_test", "config": {"prefix": 42}},
    ],
)
def test_invalid_selection_fails_when_trainer_is_constructed(selection):
    config = SkillForestTrainerConfig(optimizer={"model": "offline"}, env_adapter=selection)
    with pytest.raises((ValueError, ValidationError)):
        SkillForestTrainer(config)


def test_builtin_catalog_is_available_in_a_fresh_process():
    code = """
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter
from mindmemos_skill.algos.train.treeskill import TreeSkillEnvAdapter
for name in ('generic', 'livemath', 'officeqa', 'searchqa', 'spreadsheetbench', 'alfworld_bounded_history'):
    SkillForestEnvAdapter.build(name, {})
TreeSkillEnvAdapter.build('generic', {})
TreeSkillEnvAdapter.build('spreadsheetbench_reference', {})
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.asyncio
async def test_forest_custom_renderer_reaches_actual_model_transport(tmp_path):
    calls = []

    class Agent:
        def __init__(self, config):
            pass

        def run(self, *, messages):
            calls.append(messages)
            return AgentResult(output='{"patch": {"edits": []}}', trajectory=[], usage={})

    config = SkillForestTrainerConfig(
        optimizer={"model": "offline"},
        env_adapter={
            "class_name": "adapter_contract_test",
            "config": {"prefix": "CUSTOM", "analyst_success": "CUSTOM SYSTEM"},
        },
    )
    trainer = SkillForestTrainer(config, agent_factory=Agent)
    runtime = ForestRuntime(config, agent_factory=Agent, env_adapter=trainer.env_adapter)
    try:
        await generate_patch(
            skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Rules"})),
            samples=[sample()],
            results=[result()],
            runtime=runtime,
            directory=tmp_path,
            seed=42,
        )
    finally:
        await runtime.close()
    assert calls[0][0]["content"] == "CUSTOM SYSTEM"
    assert "CUSTOM: Inspect output" in calls[0][1]["content"]
    assert "PRIVATE" not in json.dumps(calls)


def test_forest_context_does_not_expose_private_sample_and_preserves_generic_prompt():
    context = SkillForestReflectionContext(
        skill="# Rules",
        success=True,
        edit_budget=4,
        trajectories=(SkillForestTrajectoryContext(task=sample().task, result=result()),),
    )
    before = context.model_dump_json()
    messages = SkillForestEnvAdapter().build_reflection_messages(context)
    payload = json.loads(messages[1]["content"])
    records = json.loads(payload["trajectories"])
    assert records[0]["task_id"] == "task"
    assert "PRIVATE" not in json.dumps(messages)
    assert context.model_dump_json() == before
    assert "reward_data" not in SkillForestTrajectoryContext.model_fields


class CaptureModel:
    async def chat(self, task, messages, **kwargs):
        self.messages = messages
        payload = json.loads(messages[1]["content"].split("\n\n", 1)[1])
        return ChatResponse(
            content=json.dumps(
                {
                    "instance_id": payload["instance_id"],
                    "task_id": payload["task_id"],
                    "record_source": payload["record_source"],
                    "items": [],
                }
            )
        )


@pytest.mark.asyncio
async def test_tree_default_analyzer_calls_custom_rendering_hook(tmp_path):
    adapter = TreeSkillEnvAdapter.build("adapter_contract_test", {})
    model = CaptureModel()
    config = TreeSkillTrainerConfig(optimizer={"model": "offline"}, router={"model": "offline"})
    analyzer = adapter.create_analyzer(TreeSkillAnalysisRuntime(config, model, model, tmp_path))
    evidence = TraceEvidence(trajectory_id="physical", task_id="task", transcript="observed", score=1)
    records, failures = await analyzer.analyze([evidence], trajectories_by_id={})
    assert len(records) == 1 and not failures
    assert "CUSTOM TRACE: observed" in model.messages[1]["content"]
    assert evidence.transcript == "observed"


def test_reference_adapter_builds_special_analyzer_and_rejects_inapplicable_overrides(tmp_path):
    adapter = TreeSkillEnvAdapter.build("spreadsheetbench_reference", {})
    config = TreeSkillTrainerConfig(optimizer={"model": "offline"}, router={"model": "offline"})
    model = CaptureModel()
    assert isinstance(
        adapter.create_analyzer(TreeSkillAnalysisRuntime(config, model, model, tmp_path)),
        SpreadsheetBenchReferenceAnalyzer,
    )
    with pytest.raises(ValidationError):
        TreeSkillEnvAdapter.build("spreadsheetbench_reference", {"analysis_system": "ignored override"})
    with pytest.raises(TypeError, match="SpreadsheetBenchReferenceEnvironment"):
        adapter.validate_environment(object())


class RoutingAgent(OpenAIChatAgent):
    """Record messages without performing network I/O."""

    def __init__(self):
        super().__init__(OpenAIChatConfig(model="offline", api_key=None))
        self.calls = []

    def run(self, *, messages):
        self.calls.append(messages)
        return AgentResult(output='{"selected_subtree_ids": ["001"], "rationale": "relevant"}', trajectory=[], usage={})


@pytest.mark.parametrize("stable_ids", [False, True])
def test_custom_tree_routing_survives_snapshot_and_bundle_replacement(stable_ids):
    agent = RoutingAgent()
    text = "# Rules\nInspect output.\n"
    tree = parse_skill_markdown(text)
    skill = TreeSkill(
        TreeSkillConfig(
            router=agent.config,
            reference_routing=stable_ids,
            tree_metadata=compile_tree_metadata(tree) if stable_ids else {},
            env_adapter={"class_name": "adapter_contract_test", "config": {"routing_system": "CUSTOM ROUTER"}},
        ),
        bundle=SkillBundle(files={"SKILL.md": text}),
        router=agent,
    )
    snapshot = skill.snapshot()
    assert snapshot.config_schema_version == 2
    assert snapshot.config["env_adapter"]["class_name"] == "adapter_contract_test"
    restored = TreeSkill.from_snapshot(snapshot, router=agent)
    for candidate in (skill, restored, restored.with_bundle(restored.bundle)):
        assert candidate.assemble(sample().task).bundle.files["SKILL.md"] == text
        assert agent.calls[-1][0]["content"] == "CUSTOM ROUTER"
        assert "CUSTOM PUBLIC TASK: Inspect output" in agent.calls[-1][1]["content"]
        assert "PRIVATE" not in json.dumps(agent.calls[-1])
    assert skill.bundle.files["SKILL.md"] == text


@pytest.mark.parametrize("stable_ids", [False, True])
def test_version_one_snapshot_restores_historical_routing_repeatedly(stable_ids):
    agent = RoutingAgent()
    text = "# Rules\nInspect output.\n"
    skill = TreeSkill(
        TreeSkillConfig(
            router=agent.config,
            reference_routing=stable_ids,
            tree_metadata=compile_tree_metadata(parse_skill_markdown(text)) if stable_ids else {},
        ),
        bundle=SkillBundle(files={"SKILL.md": text}),
        router=agent,
    )
    old = skill.snapshot()
    old.config_schema_version = 1
    old.config.pop("env_adapter")
    before = old.model_dump_json()
    for _ in range(2):
        restored = TreeSkill.from_snapshot(old, router=agent)
        restored.assemble(sample().task)
        assert agent.calls[-1][0]["content"] == ROUTING_SYSTEM_PROMPT
        payload = json.loads(agent.calls[-1][1]["content"].split("\n\n", 1)[1])
        assert payload["task"]["instance_id" if stable_ids else "id"] == "task"
        upgraded = TreeSkill.from_snapshot(restored.snapshot(), router=agent)
        upgraded.assemble(sample().task)
        assert agent.calls[-1] == agent.calls[-2]
    assert old.model_dump_json() == before


def test_missing_user_adapter_does_not_silently_fallback_on_restore():
    agent = RoutingAgent()
    skill = TreeSkill(TreeSkillConfig(router=agent.config), bundle=SkillBundle(files={"SKILL.md": "# Rules\nInspect."}))
    snapshot = skill.snapshot()
    snapshot.config["env_adapter"] = {"class_name": "uninstalled_user_adapter", "config": {}}
    with pytest.raises(ValueError, match="uninstalled_user_adapter"):
        TreeSkill.from_snapshot(snapshot)


@pytest.mark.asyncio
async def test_trainer_propagates_user_adapter_to_final_skill(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "adapter_smoke", Path(__file__).resolve().parents[2] / "scripts/treeskill/smoke.py"
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    config = TreeSkillTrainerConfig(
        optimizer={"model": "offline"}, router={"model": "offline"}, env_adapter={"class_name": "adapter_contract_test"}
    )
    trainer = TreeSkillTrainer(config, agent_factory=smoke.SmokeAgent)
    final = await trainer.train(
        skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Workflow\nSave the output.\n"})),
        env=smoke.SmokeEnvironment(smoke.SmokeConfig()),
        train_samples=[sample()],
        work_dir=tmp_path / "run",
    )
    assert final.config.env_adapter.class_name == "adapter_contract_test"
    assert isinstance(final.env_adapter, TreeSkillTestEnvAdapter)
    analysis_call = next(
        path
        for path in (tmp_path / "run/model_calls/optimizer").glob("**/attempt-*.json")
        if json.loads(path.read_text())["stage"] == "treeskill_trajectory_analysis"
    )
    assert "CUSTOM TRACE" in analysis_call.read_text()
    restored = TreeSkill.from_snapshot(final.snapshot())
    agent = RoutingAgent()
    restored.router = agent
    await asyncio.to_thread(restored.assemble, sample().task)
    assert "CUSTOM PUBLIC TASK" in agent.calls[0][1]["content"]


@pytest.mark.asyncio
async def test_one_rendering_failure_does_not_discard_other_tree_evidence(tmp_path):
    class SelectiveAdapter(TreeSkillEnvAdapter):
        def render_trajectory(self, context):
            if context.task_id == "bad":
                raise ValueError("Missing required evidence")
            return super().render_trajectory(context)

    model = CaptureModel()
    config = TreeSkillTrainerConfig(optimizer={"model": "offline"}, router={"model": "offline"})
    analyzer = SelectiveAdapter().create_analyzer(TreeSkillAnalysisRuntime(config, model, model, tmp_path))
    records, failures = await analyzer.analyze(
        [
            TraceEvidence(trajectory_id="bad-run", task_id="bad", transcript="", score=0),
            TraceEvidence(trajectory_id="good-run", task_id="good", transcript="observed", score=1),
        ],
        trajectories_by_id={},
    )
    assert failures == ["bad-run"]
    assert [record.instance_id for record in records] == ["good-run"]


@pytest.mark.asyncio
async def test_registered_analyzer_factory_is_used_by_trainer(tmp_path):
    calls = []

    class Analyzer:
        async def analyze(self, evidence, *, trajectories_by_id):
            calls.append((evidence, trajectories_by_id))
            return [], []

    @TreeSkillEnvAdapter.register("custom_analyzer_contract_test")
    class TreeSkillAnalyzerTestEnvAdapter(TreeSkillEnvAdapter):
        def create_analyzer(self, runtime):
            assert runtime.output_root == tmp_path / "run/stages/reference_analysis"
            return Analyzer()

    class Env(Environment):
        config_type = SkillForestEnvAdapterConfig

        def _run(self, *, task, reward_data, skills, workspace):
            return result()

    def no_model(config):
        raise AssertionError("The custom analyzer must not invoke the default model")

    trainer = TreeSkillTrainer(
        TreeSkillTrainerConfig(
            optimizer={"model": "offline"},
            router={"model": "offline"},
            env_adapter={"class_name": "custom_analyzer_contract_test"},
        ),
        agent_factory=no_model,
    )
    final = await trainer.train(
        skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Rules\nInspect output.\n"})),
        env=Env(SkillForestEnvAdapterConfig()),
        train_samples=[sample()],
        work_dir=tmp_path / "run",
    )
    assert len(calls) == 1
    assert calls[0][0][0].task_id == "task"
    assert final.config.env_adapter.class_name == "custom_analyzer_contract_test"
