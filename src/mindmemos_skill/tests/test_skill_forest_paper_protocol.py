"""Verify saved paper-run partitions and model-visible option order offline."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter, SkillForestTrainer, SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.runtime import ForestRuntime
from mindmemos_skill.algos.train.skill_forest.tree import apply_patch
from mindmemos_skill.algos.train.skill_forest.typing import Node, Patch, TreeSplit
from mindmemos_skill.envs.builtin_envs.livemath.typing import LiveMathRewardData, LiveMathTask
from mindmemos_skill.envs.data import DataSample
from mindmemos_skill.skill import Skill, SkillContent

REFERENCE = json.loads((Path(__file__).parent / "fixtures/skill_forest/livemath_paper_protocol.json").read_text())


def test_all_tree_partitions_match_saved_paper_run():
    trainer = SkillForestTrainer(SkillForestTrainerConfig(optimizer={"model": "offline"}))
    actual = [split.model_dump(mode="json") for split in trainer.create_splits(REFERENCE["training_pool_ids"])]
    assert actual == REFERENCE["splits"]


async def test_all_reflection_seeds_match_archived_candidate_manifests(monkeypatch, tmp_path):
    from mindmemos_skill.algos.train.skill_forest import trainer as trainer_module

    generate = AsyncMock(return_value=Patch())
    monkeypatch.setattr(trainer_module, "generate_patch", generate)
    config = SkillForestTrainerConfig(optimizer={"model": "offline"})
    trainer = SkillForestTrainer(config)
    runtime = ForestRuntime(config)
    skill = Skill(SkillContent(content="# Initial\n", name="test", description="test", standard_format=False))
    try:
        for case in REFERENCE["reflection_seeds"]:
            parent = Node(id="parent", tree_id=case["tree_id"], depth=case["depth"], skill=skill, scores={})
            await trainer._generate_patch(
                parent, case["candidate_index"], TreeSplit(train_ids=(), validation_ids=()), {}, {}, runtime, tmp_path
            )
            assert generate.await_args.kwargs["seed"] == case["seed"]
    finally:
        await runtime.close()


@pytest.mark.parametrize("case", REFERENCE["samples"])
def test_phase_options_and_answer_key_match_paper_predictions(case):
    sample = DataSample(
        task=LiveMathTask.model_validate(case["sample"]["task"]),
        reward_data=LiveMathRewardData.model_validate(case["sample"]["reward_data"]),
    )
    original = sample.model_dump()
    adapter = SkillForestEnvAdapter.build("livemath", {})
    prepared = adapter.prepare_sample(sample, seed=case["seed"])
    task = prepared.task
    user = f"## Question\n{task.instruction}\n\n## Choices\n" + "\n".join(
        f"{choice.label}. {choice.text}" for choice in task.choices
    )
    assert user == case["expected_user"]
    assert prepared.reward_data.correct_choice.model_dump() == case["expected_correct"]
    assert adapter.prepare_sample(prepared, seed=case["seed"]) == prepared
    assert adapter.prepare_sample(prepared, seed=42) == sample
    assert sample.model_dump() == original


def test_reference_patch_inserts_after_whole_line_and_retains_other_valid_edits():
    skill = Skill(
        SkillContent(content="# Heading with suffix\nold\n", name="test", description="test", standard_format=False)
    )
    patch = Patch.model_validate(
        {
            "edits": [
                {"op": "insert_after", "target": "# Heading", "content": "new"},
                {"op": "replace", "target": "missing", "content": "unused"},
                {"op": "append", "content": "last"},
            ]
        }
    )
    assert apply_patch(skill, patch, reference=True).content == "# Heading with suffix\n\nnew\nold\n\nlast\n"
    with pytest.raises(ValueError):
        apply_patch(skill, patch)
