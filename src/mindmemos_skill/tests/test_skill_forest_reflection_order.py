"""Verify optimizer-visible reflection ordering and reference ranking instructions."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from mindmemos_skill.algos.train.skill_forest import SkillForestEnvAdapter, SkillForestTrainerConfig, generation
from mindmemos_skill.algos.train.skill_forest.runtime import parallel
from mindmemos_skill.algos.train.skill_forest.typing import Edit, Patch


async def test_reflection_submission_order_reaches_merge_despite_reverse_completion(monkeypatch, tmp_path):
    second_finished = asyncio.Event()
    observed = []

    async def reflect(batch, *, index, **kwargs):
        if index == 0:
            await second_finished.wait()
        else:
            second_finished.set()
        return Patch(edits=[Edit(op="append", content="first" if index == 0 else "second")])

    async def merge(patches, **kwargs):
        observed.extend(patch.edits[0].content for patch in patches)
        return patches[0]

    monkeypatch.setattr(generation, "_reflect_batch", reflect)
    monkeypatch.setattr(generation, "_merge_group", merge)
    runtime = SimpleNamespace(config=SkillForestTrainerConfig(optimizer={"model": "offline"}))
    await generation._reduce_evidence([(), ()], success=False, text="parent", runtime=runtime, directory=tmp_path)
    assert observed == ["first", "second"]


async def test_parallel_failure_drains_sibling():
    started = asyncio.Event()
    drained = asyncio.Event()

    async def pending():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            drained.set()

    async def fail():
        await started.wait()
        raise ValueError("invalid response")

    with pytest.raises(ExceptionGroup):
        await parallel([pending(), fail()])
    assert drained.is_set()


def test_reference_ranking_prompt_matches_original_text():
    reference = (Path(__file__).parent / "fixtures/skill_forest/reference_ranking_prompt.txt").read_text()
    adapter = SkillForestEnvAdapter.build("livemath", {})
    assert adapter.prompts.ranking == reference
    custom = SkillForestEnvAdapter.build("livemath", {"ranking": "custom ranking"})
    assert custom.prompts.ranking == "custom ranking"
