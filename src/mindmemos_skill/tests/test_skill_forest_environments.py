"""Offline reference parity for all non-LiveMath forest reflection adapters."""

import asyncio
import json
from pathlib import Path

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter
from mindmemos_skill.algos.train.skill_forest.config import SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.generation import generate_patch
from mindmemos_skill.algos.train.skill_forest.trajectory_rendering import tool_conversation
from mindmemos_skill.algos.train.skill_forest.typing import SkillForestTrajectoryContext
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.typing import ALFWorldTask
from mindmemos_skill.envs.builtin_envs.officeqa.typing import OfficeQATask
from mindmemos_skill.envs.builtin_envs.searchqa.typing import SearchQATask
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import SpreadsheetBenchTask
from mindmemos_skill.envs.data.typing import DataSample, RewardData
from mindmemos_skill.envs.typing import RolloutResult
from mindmemos_skill.skill import Skill, SkillContent

FIXTURE = json.loads((Path(__file__).parent / "fixtures/skill_forest/environment_reflection.json").read_text())
CASES = FIXTURE["cases"]
TASK_TYPES = {
    "searchqa": SearchQATask,
    "officeqa": OfficeQATask,
    "spreadsheetbench_tool_calling": SpreadsheetBenchTask,
    "alfworld_bounded_history": ALFWorldTask,
}


def _pair(case):
    """Restore exactly the public task and persisted unified result."""
    return (
        DataSample(task=TASK_TYPES[case["environment"]].model_validate(case["task"]), reward_data=RewardData()),
        RolloutResult.model_validate_json(json.dumps(case["result"])),
    )


@pytest.mark.parametrize("case", CASES, ids=[case["task"]["id"] for case in CASES])
def test_reference_formatter_and_actual_optimizer_request(case, tmp_path):
    sample, result = _pair(case)
    original = result.model_dump_json()
    environment = case["environment"]
    assert render_trajectories([(sample, result)], environment) == case["expected_trajectory"]
    outcome = "Successful" if result.reward.success else "Failed"
    expected_user = (
        "## Current Skill\n# Rules\n\n## Edits Budget\nProduce at most L=4 edits.\n\n"
        f"## {outcome} Trajectories (1 total)\n{case['expected_trajectory']}"
    )
    requests = []

    class CaptureRuntime:
        config = SkillForestTrainerConfig(optimizer={"model": "offline"})

        env_adapter = SkillForestEnvAdapter.build(environment, {})

        async def chat(self, *, messages, directory):
            system, user = messages[0]["content"], messages[1]["content"]
            requests.append((system, user))
            return '{"patch": {"edits": []}}'

    asyncio.run(
        generate_patch(
            skill=Skill(
                source=SkillContent(
                    content="# Rules", name="test-skill", description="Task instructions", standard_format=False
                )
            ),
            samples=[sample],
            results=[result],
            runtime=CaptureRuntime(),
            directory=tmp_path,
            seed=42,
        )
    )
    assert requests == [(case["expected_system"], expected_user)]
    assert result.model_dump_json() == original


@pytest.mark.parametrize("environment", TASK_TYPES)
def test_outcome_reduction_uses_reference_merge_protocol(environment, tmp_path):
    pairs = [_pair(case) for case in CASES if case["environment"] == environment] * 2
    requests = []

    class CaptureRuntime:
        config = SkillForestTrainerConfig(
            optimizer={"model": "offline"},
            reflection_minibatch_size=1,
            merge_batch_size=2,
        )

        env_adapter = SkillForestEnvAdapter.build(environment, {})

        async def chat(self, *, messages, directory):
            system, user = messages[0]["content"], messages[1]["content"]
            requests.append((system, user))
            patch = {"edits": [{"op": "append", "content": "Verify the answer."}]}
            return json.dumps({"patch": patch} if "Trajectories (" in user else patch)

    patch = asyncio.run(
        generate_patch(
            skill=Skill(
                source=SkillContent(
                    content="# Rules", name="test-skill", description="Task instructions", standard_format=False
                )
            ),
            samples=[sample for sample, result in pairs],
            results=[result for sample, result in pairs],
            runtime=CaptureRuntime(),
            directory=tmp_path,
            seed=42,
        )
    )
    assert patch.edits
    assert len(requests) == 7
    for name in ("merge_failure", "merge_success"):
        user = next(user for system, user in requests if system == load_prompt(name, environment))
        assert user.startswith("## Current Skill\n# Rules\n\n## Patches to merge (2 total, merge level 1)\n")
        assert len(json.loads(user.split("\n", 4)[-1])) == 2
    final = next(user for system, user in requests if system == load_prompt("merge_final", environment))
    assert "Group 1 (failure, priority 1): 1 edits\nGroup 2 (success, priority 2): 1 edits" in final


@pytest.mark.parametrize("case", CASES, ids=[case["task"]["id"] for case in CASES])
def test_missing_evidence_is_not_reconstructed(case):
    sample, result = _pair(case)
    result.trajectory = []
    with pytest.raises(ValueError, match="recorded"):
        render_trajectories([(sample, result)], case["environment"])


def test_tool_pairing_multiple_calls_and_malformed_evidence():
    calls = [{"id": str(index), "function": {"name": "read", "arguments": str(index)}} for index in (1, 2)]
    assistant = {"role": "assistant", "content": "Read both.", "tool_calls": calls}
    events = [
        assistant,
        {"role": "tool", "tool_call_id": "2", "content": "second"},
        {"role": "tool", "tool_call_id": "1", "content": "first"},
    ]
    assert tool_conversation(events)[1:] == [
        {"type": "tool_call", "cmd": "read(1)", "obs": "first"},
        {"type": "tool_call", "cmd": "read(2)", "obs": "second"},
    ]
    with pytest.raises(ValueError, match="Missing"):
        tool_conversation([assistant])
    with pytest.raises(ValueError, match="Unmatched"):
        tool_conversation(events[1:])


def test_legacy_diagnostics_fallback_and_spreadsheet_verification_removed():
    for case in CASES:
        sample, result = _pair(case)
        if case["environment"] == "spreadsheetbench_tool_calling":
            result.reward.model_extra.pop("n_turns")
            result.trajectory.append({"role": "system", "content": result.reward.detail})
            text = render_trajectories([(sample, result)], case["environment"])
            assert "Steps: 3\n" in text
            assert "[POST-EXECUTION VERIFICATION]" not in text
            assert "[verification] [CASE 2]" in text
        elif case["environment"] == "alfworld_bounded_history":
            result.reward.model_extra.pop("task_description")
            result.reward.model_extra.pop("task_type")
            text = render_trajectories([(sample, result)], case["environment"])
            assert f"Task: {sample.task.instruction}\n" in text


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
