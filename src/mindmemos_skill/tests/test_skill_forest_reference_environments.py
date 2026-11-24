"""Cross-environment optimizer protocol checks against independent reference text."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter, SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.generation import _merge_group, _parse_reference
from mindmemos_skill.algos.train.skill_forest.typing import SkillForestFinalPatchContext

REFERENCE = json.loads((Path(__file__).parent / "fixtures/skill_forest/reference_optimizer_prompts.json").read_text())


@pytest.mark.parametrize("environment", REFERENCE)
def test_all_optimizer_system_prompts_match_reference(environment):
    adapter = SkillForestEnvAdapter.build(environment, {})
    actual = adapter.prompts.model_dump()
    assert {stage: actual[stage] for stage in REFERENCE[environment]} == REFERENCE[environment]


@pytest.mark.parametrize("environment", REFERENCE)
async def test_reference_merge_and_final_inputs_preserve_evidence(environment, tmp_path):
    adapter = SkillForestEnvAdapter.build(environment, {})
    config = SkillForestTrainerConfig(optimizer={"model": "offline"})
    response = '{"edits":[{"content":"Merged","op":"append"}],"reasoning":"merged"}'
    runtime = SimpleNamespace(config=config, env_adapter=adapter, chat=AsyncMock(return_value=response))
    patches = [
        _parse_reference(
            json.dumps(
                {
                    "batch_size": 3,
                    "patch": {"edits": [{"content": text, "op": "append"}], "deletions": [], "reasoning": text},
                }
            ),
            reflection=True,
            source="failure",
        )
        for text in ("second completed first", "first completed second")
    ]
    merged = await _merge_group(
        patches, name="failure", level=1, index=0, text="Parent", runtime=runtime, directory=tmp_path
    )
    expected_patches = [
        {
            "edits": [{"content": text, "op": "append", "source_type": "failure", "support_count": 3}],
            "deletions": [],
            "reasoning": text,
        }
        for text in ("second completed first", "first completed second")
    ]
    expected_user = "## Current Skill\nParent\n\n## Patches to merge (2 total, merge level 1)\n" + json.dumps(
        expected_patches, ensure_ascii=False, indent=2
    )
    assert runtime.chat.await_args.kwargs["messages"] == [
        {"role": "system", "content": REFERENCE[environment]["merge_failure"]},
        {"role": "user", "content": expected_user},
    ]
    expected_merged = {"edits": [{"content": "Merged", "op": "append", "merge_level": 1}], "reasoning": "merged"}
    success = _parse_reference('{"edits":[{"content":"Success","op":"append"}],"reasoning":"success"}')
    final = adapter.build_final_patch_messages(
        SkillForestFinalPatchContext(skill="Parent", failure=merged, success=success)
    )
    expected_final = (
        "## Current Skill\nParent\n\n## Pre-merged patch groups to combine (higher priority first)\nGroup order and priority:\nGroup 1 (failure, priority 1): 1 edits\nGroup 2 (success, priority 2): 1 edits\n\n"
        + json.dumps(
            [expected_merged, {"edits": [{"content": "Success", "op": "append"}], "reasoning": "success"}],
            ensure_ascii=False,
            indent=2,
        )
    )
    assert final == [
        {"role": "system", "content": REFERENCE[environment]["merge_final"]},
        {"role": "user", "content": expected_final},
    ]
