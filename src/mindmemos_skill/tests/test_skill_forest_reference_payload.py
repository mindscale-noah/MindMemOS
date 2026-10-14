"""Validate historical optimizer normalization using the single reference response path."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter, SkillForestTrainerConfig
from mindmemos_skill.algos.train.skill_forest.generation import (
    _merge_group,
    _parse_reference,
    _reflect_batch,
)
from mindmemos_skill.algos.train.skill_forest.typing import Patch


def test_reference_preserves_extra_metadata_and_original_key_order():
    response = '{"batch_size":3,"patch":{"edits":[{"content":"Check scope","op":"append"}],"deletions":[],"reasoning":"scope"}}'
    patch = _parse_reference(response, reflection=True, source="failure", edit_budget=4)
    expected = {
        "edits": [{"content": "Check scope", "op": "append", "source_type": "failure", "support_count": 3}],
        "deletions": [],
        "reasoning": "scope",
    }
    assert json.dumps(patch.optimizer_input()) == json.dumps(expected)
    assert patch.edits[0].support_count == 3


def test_reference_normalization_preserves_explicit_support_and_limits_edits():
    raw = {"patch": {"edits": [{"op": "append", "content": "A", "support_count": 7}, {"op": "append", "content": "B"}]}}
    patch = _parse_reference(json.dumps(raw), reflection=True, source="success", edit_budget=1)
    assert len(patch.edits) == 1
    assert patch.edits[0].support_count == 7
    raw["patch"]["edits"][0].pop("support_count")
    assert _parse_reference(json.dumps(raw), reflection=True, source="success").edits[0].support_count == 1


@pytest.mark.parametrize("stage", ["reflection", "merge"])
async def test_reference_model_output_fallback_never_swallows_provider_failure(tmp_path, stage):
    config = SkillForestTrainerConfig(optimizer={"model": "offline"})
    runtime = SimpleNamespace(
        config=config, env_adapter=SkillForestEnvAdapter.build("livemath", {}), chat=AsyncMock(return_value="not JSON")
    )
    patches = [Patch(edits=[{"op": "append", "content": text}]) for text in ("A", "B")]

    async def invoke():
        if stage == "reflection":
            return await _reflect_batch((), success=False, index=0, text="parent", runtime=runtime, directory=tmp_path)
        return await _merge_group(
            patches, name="failure", level=1, index=0, text="parent", runtime=runtime, directory=tmp_path
        )

    result = await invoke()
    assert [edit.content for edit in result.edits] == ([] if stage == "reflection" else ["A", "B"])
    if stage == "merge":
        assert all(edit.merge_level == 1 for edit in result.edits)
    runtime.chat.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await invoke()


@pytest.mark.parametrize(
    "response, expected", [("not JSON", ["0", "1"]), ('{"selected_indices":["bad",2,2,0,99]}', ["2", "0"])]
)
async def test_reference_ranking_fallback_and_mixed_indices(tmp_path, response, expected):
    from mindmemos_skill.algos.train.skill_forest.generation import _rank_patch

    config = SkillForestTrainerConfig(optimizer={"model": "offline"}, ranking_edit_budgets=(2,))
    runtime = SimpleNamespace(
        config=config, env_adapter=SkillForestEnvAdapter.build("livemath", {}), chat=AsyncMock(return_value=response)
    )
    patch = Patch(edits=[{"op": "append", "content": str(index)} for index in range(3)])
    result = await _rank_patch(patch, text="parent", depth=0, runtime=runtime, directory=tmp_path)
    assert [edit.content for edit in result.edits] == expected
    runtime.chat.side_effect = RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await _rank_patch(patch, text="parent", depth=0, runtime=runtime, directory=tmp_path)


async def test_final_merge_invalid_response_keeps_both_valid_groups(monkeypatch, tmp_path):
    from mindmemos_skill.algos.train.skill_forest import generation
    from mindmemos_skill.skill import Skill, SkillContent

    async def reduce(batches, *, success, **kwargs):
        return Patch(edits=[{"op": "append", "content": "success" if success else "failure"}])

    monkeypatch.setattr(generation, "_reduce_evidence", reduce)
    config = SkillForestTrainerConfig(optimizer={"model": "offline"}, ranking_enabled=False)
    runtime = SimpleNamespace(
        config=config, env_adapter=SkillForestEnvAdapter.build("livemath", {}), chat=AsyncMock(return_value="not JSON")
    )
    patch = await generation.generate_patch(
        skill=Skill(SkillContent(content="Parent", name="test", description="test", standard_format=False)),
        samples=[],
        results=[],
        runtime=runtime,
        directory=tmp_path,
        seed=42,
    )
    assert [edit.content for edit in patch.edits] == ["failure", "success"]
    assert patch.reasoning == "fallback: failure first, then success"
