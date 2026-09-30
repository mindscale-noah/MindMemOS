"""Offline verification of shared candidate ranking and its audit boundary."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from mindmemos_skill.agents import AgentResult
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter, SkillForestTrainer, SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.generation import InvalidPatchError, generate_patch
from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime
from mindmemos_skill.algos.train.skill_forest.typing import Patch, SkillForestRankingContext
from mindmemos_skill.envs.data.typing import DataSample, RewardData
from mindmemos_skill.envs.typing import Reward, RolloutResult, Task
from mindmemos_skill.skill import Skill, SkillBundle
from pydantic import ValidationError


class RankingOptimizer:
    """Return a four-edit reflection and a controlled ranking response."""

    def __init__(self, response):
        self.response = response
        self.messages = []

    def __call__(self, config):
        return self

    def run(self, *, messages):
        self.messages.append(messages)
        if messages[1]["content"].startswith("## Current Skill"):
            if isinstance(self.response, Exception):
                raise self.response
            return AgentResult(output=self.response)
        return AgentResult(
            output=json.dumps(
                {"patch": {"edits": [{"op": "append", "content": f"Rule {index}"} for index in range(4)]}}
            )
        )


def run_generation(directory, response, *, depth=0, trainer_entry=False, **options):
    """Exercise real generation, adapter and audited model transport offline."""
    settings = SkillForestTrainerConfig(optimizer={"model": "offline"}, **options)
    optimizer = RankingOptimizer(response)
    sample = DataSample(task=Task(id="task", instruction="Public task"), reward_data=RewardData())
    result = RolloutResult(
        output="Failed attempt", trajectory=[], reward=Reward(score=0, success=False, detail="Incorrect")
    )
    skill = Skill(bundle=SkillBundle(files={"SKILL.md": "# Existing rules\n"}))

    async def generate():
        runtime = ForestRuntime(settings, agent_factory=optimizer)
        try:
            if trainer_entry:
                trainer = SkillForestTrainer(settings, agent_factory=optimizer)
                return await trainer._generate_patch(
                    SimpleNamespace(id="parent", skill=skill, depth=depth),
                    0,
                    SimpleNamespace(train_ids=[sample.task.id]),
                    {sample.task.id: sample},
                    {sample.task.id: result},
                    runtime,
                    directory,
                )
            return await generate_patch(
                skill=skill,
                samples=[sample],
                results=[result],
                runtime=runtime,
                directory=directory,
                seed=42,
                depth=depth,
            )
        finally:
            await runtime.close()

    return asyncio.run(generate()), optimizer


@pytest.mark.parametrize("depth,budget", [(0, 3), (1, 2), (2, 1), (5, 1)])
def test_ranking_respects_depth_order_and_audits_original_pool(tmp_path, depth, budget):
    patch, optimizer = run_generation(
        tmp_path,
        '{"selected_indices": [3, 3, -1, 99, 2, 1, 0]}',
        depth=depth,
        trainer_entry=True,
        ranking_edit_budgets=(3, 2, 1),
    )
    assert [edit.content for edit in patch.edits] == [f"Rule {i}" for i in [3, 2, 1][:budget]]
    assert len(optimizer.messages) == 2
    assert f"budget={budget}" in optimizer.messages[1][1]["content"]
    before = json.loads((tmp_path / "pre_ranking_patch.json").read_text())
    assert [edit["content"] for edit in before["edits"]] == [f"Rule {i}" for i in range(4)]
    assert all(edit["source_type"] == "failure" for edit in before["edits"])
    assert json.loads((tmp_path / "patch.json").read_text())["edits"] == patch.model_dump(exclude_unset=True)["edits"]
    assert json.loads((tmp_path / "ranking/call.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("options", [{}, {"ranking_enabled": False, "ranking_edit_budgets": (1,)}])
def test_ranking_skips_within_budget_or_when_disabled(tmp_path, options):
    patch, optimizer = run_generation(tmp_path, RuntimeError("Ranking must not run"), **options)
    assert len(patch.edits) == 4
    assert len(optimizer.messages) == 1
    assert not (tmp_path / "ranking").exists()
    assert not (tmp_path / "pre_ranking_patch.json").exists()


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"selected_indices": []}',
        '{"selected_indices": [-1, 99]}',
        '{"selected_indices": [true]}',
        '{"selected_indices": [1.5]}',
    ],
)
def test_invalid_ranking_fails_proposal_without_truncation(tmp_path, response):
    with pytest.raises(InvalidPatchError):
        run_generation(tmp_path, response, ranking_edit_budgets=(2,))
    assert (tmp_path / "pre_ranking_patch.json").exists()
    assert not (tmp_path / "patch.json").exists()


def test_ranking_provider_failure_is_not_an_invalid_patch(tmp_path):
    with pytest.raises(RuntimeError, match="provider offline"):
        run_generation(tmp_path, RuntimeError("provider offline"), ranking_edit_budgets=(2,))
    assert json.loads((tmp_path / "ranking/call.json").read_text())["status"] == "failed"
    assert not (tmp_path / "patch.json").exists()


@pytest.mark.parametrize(
    "environment",
    [
        "generic",
        "livemath",
        "alfworld_bounded_history",
        "officeqa",
        "searchqa",
        "spreadsheetbench",
    ],
)
def test_all_adapters_share_ranking_and_support_literal_overrides(environment):
    context = SkillForestRankingContext(
        skill="# Rules",
        patch=Patch(edits=[{"op": "append", "content": "Verify output"}]),
        edit_budget=1,
    )
    adapter = SkillForestEnvAdapter.build(environment, {})
    assert adapter.build_ranking_messages(context)[0]["content"] == SkillForestEnvAdapter.RANKING_PROMPT
    custom = SkillForestEnvAdapter.build(environment, {"ranking": "Custom ranking instructions"})
    assert custom.build_ranking_messages(context)[0]["content"] == "Custom ranking instructions"
    assert adapter.prompts.ranking == SkillForestEnvAdapter.RANKING_PROMPT


@pytest.mark.parametrize("budgets", [(), (0,), (-1, 2)])
def test_ranking_budgets_require_nonempty_positive_limits(budgets):
    with pytest.raises(ValidationError):
        SkillForestTrainerConfig(optimizer={"model": "offline"}, ranking_edit_budgets=budgets)


def test_ranking_selects_from_final_merged_pool(tmp_path):
    class MergeOptimizer(RankingOptimizer):
        def run(self, *, messages):
            if messages[1]["content"].startswith("## Current Skill"):
                return super().run(messages=messages)
            request = json.loads(messages[1]["content"])
            if "groups" in request:
                self.messages.append(messages)
                return AgentResult(
                    output=json.dumps({"edits": [{"op": "append", "content": f"Merged {i}"} for i in range(6)]})
                )
            return super().run(messages=messages)

    optimizer = MergeOptimizer('{"selected_indices": [5, 1, 4, 0]}')
    settings = SkillForestTrainerConfig(optimizer={"model": "offline"})
    samples = [DataSample(task=Task(id=str(i), instruction="Public task"), reward_data=RewardData()) for i in range(2)]
    results = [
        RolloutResult(output="Attempt", trajectory=[], reward=Reward(score=i, success=bool(i), detail="Outcome"))
        for i in range(2)
    ]

    async def generate():
        runtime = ForestRuntime(settings, agent_factory=optimizer)
        try:
            return await generate_patch(
                skill=Skill(bundle=SkillBundle(files={"SKILL.md": "# Rules"})),
                samples=samples,
                results=results,
                runtime=runtime,
                directory=tmp_path,
                seed=42,
            )
        finally:
            await runtime.close()

    patch = asyncio.run(generate())
    assert [edit.content for edit in patch.edits] == [f"Merged {i}" for i in [5, 1, 4, 0]]
    assert len(optimizer.messages) == 4
    assert "[5] op=append" in optimizer.messages[-1][1]["content"]
    assert len(json.loads((tmp_path / "pre_ranking_patch.json").read_text())["edits"]) == 6
