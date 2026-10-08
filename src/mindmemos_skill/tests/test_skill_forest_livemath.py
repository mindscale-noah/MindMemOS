"""Historical LiveMath reflection parity through the unified result interface."""

import asyncio
import json
from pathlib import Path

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter
from mindmemos_skill.algos.train.skill_forest.config import SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.generation import generate_patch
from mindmemos_skill.algos.train.skill_forest.typing import SkillForestReflectionContext, SkillForestTrajectoryContext
from mindmemos_skill.envs.builtin_envs.livemath.typing import LiveMathTask
from mindmemos_skill.envs.data.typing import DataSample, RewardData
from mindmemos_skill.envs.typing import RolloutResult
from mindmemos_skill.skill import Skill, SkillBundle

CASES = json.loads((Path(__file__).parent / "fixtures/skill_forest/livemath_reflection.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=["failure", "success"])
def test_actual_reflection_request_matches_historical_input(case, tmp_path):
    sample = DataSample(task=LiveMathTask.model_validate(case["task"]), reward_data=RewardData())
    result = RolloutResult.model_validate(case["result"])
    assert (
        reflection_user(case["skill"], [(sample, result)], success=result.reward.success, edit_budget=4)
        == case["expected"]["user"]
    )
    assert (
        load_prompt("analyst_success" if result.reward.success else "analyst_error", "livemath")
        == case["expected"]["system"]
    )

    class CaptureRuntime:
        config = SkillForestTrainerConfig(optimizer={"model": "offline"}, env_adapter={"class_name": "livemath"})

        env_adapter = SkillForestEnvAdapter.build(config.env_adapter.class_name, config.env_adapter.config)

        async def chat(self, *, messages, directory):
            system, user = messages[0]["content"], messages[1]["content"]
            assert {"system": system, "user": user} == case["expected"]
            return '{"batch_size": 1, "patch": {"edits": []}}'

    patch = asyncio.run(
        generate_patch(
            skill=Skill(bundle=SkillBundle(files={"SKILL.md": case["skill"]})),
            samples=[sample],
            results=[result],
            runtime=CaptureRuntime(),
            directory=tmp_path,
            seed=42,
        )
    )
    assert not patch.edits


def test_livemath_requires_actual_input_context():
    case = CASES[0]
    sample = DataSample(task=LiveMathTask.model_validate(case["task"]), reward_data=RewardData())
    result = RolloutResult.model_validate(case["result"]).model_copy(update={"trajectory": []})
    with pytest.raises(ValueError, match="recorded system, user and assistant"):
        render_trajectories([(sample, result)])


def test_evaluation_detail_is_rendered_once_and_raw_private_fields_are_not_dumped():
    case = CASES[0]
    sample = DataSample(task=LiveMathTask.model_validate(case["task"]), reward_data=RewardData())
    result = RolloutResult.model_validate(case["result"])
    result.reward.detail = "PRIVATE-EXTRA-NOT-PART-OF-PAPER-PROTOCOL"
    result.trajectory.append({"role": "system", "content": result.reward.detail})
    text = render_trajectories([(sample, result)])
    assert text.count("[verification] [EVALUATION RESULT]") == 1
    assert "Correct label:" in text and "Correct text:" in text
    assert "PRIVATE-EXTRA" not in text


def render_trajectories(batch, environment="livemath"):
    adapter = SkillForestEnvAdapter.build(environment, {})
    return "\n\n---\n\n".join(
        adapter.render_trajectory(SkillForestTrajectoryContext(task=sample.task, result=result, index=index))
        for index, (sample, result) in enumerate(batch, 1)
    )


def load_prompt(name, environment="generic"):
    prompts = SkillForestEnvAdapter.build(environment, {}).prompts
    return {
        "analyst_error": prompts.analyst_error,
        "analyst_success": prompts.analyst_success,
        "merge_failure": prompts.merge_failure,
        "merge_success": prompts.merge_success,
        "merge_final": prompts.merge_final,
    }[name]


def reflection_user(skill, batch, *, success, edit_budget):
    return SkillForestEnvAdapter.build("livemath", {}).build_reflection_messages(
        SkillForestReflectionContext(
            skill=skill,
            trajectories=tuple(
                SkillForestTrajectoryContext(task=sample.task, result=result, index=index)
                for index, (sample, result) in enumerate(batch, 1)
            ),
            success=success,
            edit_budget=edit_budget,
        )
    )[1]["content"]
