"""Verify independently readable and editable collections at the public algorithm boundary."""

import pytest
from mindmemos_skill.algos.evolve import SkillTrajectory
from mindmemos_skill.algos.evolve.offline_evolution.config import EvolutionOptions
from mindmemos_skill.algos.evolve.offline_evolution.evolver import OfflineSkillEvolver
from mindmemos_skill.algos.evolve.offline_evolution.typing import SkillEvolutionResult
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.envs.typing import Task
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import BaseModel

from .test_collection_publication import skill


class Options(BaseModel):
    """Model-free test algorithm settings."""


class ProposalEvolver(OfflineSkillEvolver):
    """Return an explicitly selected proposal through the real boundary checks."""

    config_type = Options
    """Empty model-free configuration."""

    async def _evolve(self, request):
        self.calls += 1
        assert set(request.skills) == {"a", "b"}
        return self.proposal


def make_evolver(proposal):
    """Configure a test algorithm without adding a global registry entry."""
    value = ProposalEvolver(Options())
    value.proposal = proposal
    value.calls = 0
    return value


def evidence():
    """Supply one public, detached historical execution."""
    return [SkillTrajectory(task=Task(id="task", instruction="Use both Skills"), skills={}, trajectory=[])]


async def test_readonly_inputs_can_be_analyzed_while_new_skills_are_allowed(tmp_path):
    inputs = {"a": skill("alpha"), "b": skill("beta")}
    proposal = SkillEvolutionResult(skills=inputs, created_skills={"new/delta": skill("delta")})
    evolver = make_evolver(proposal)
    result = await evolver.evolve(
        skills=inputs,
        trajectories=evidence(),
        work_dir=tmp_path,
        options=EvolutionOptions(editable_skills=[]),
    )
    assert list(result.created_skills) == ["new/delta"]
    assert result.skills["b"].snapshot() == inputs["b"].snapshot()
    replay = await evolver.evolve(
        skills=inputs,
        trajectories=evidence(),
        work_dir=tmp_path,
        options=EvolutionOptions(editable_skills=[]),
    )
    assert evolver.calls == 2 and replay.snapshot() == result.snapshot()
    assert not list(tmp_path.iterdir())


async def test_readonly_member_cannot_be_rewritten(tmp_path):
    inputs = {"a": skill("alpha"), "b": skill("beta")}
    proposed = {"a": inputs["a"], "b": skill("beta", "Changed")}
    evolver = make_evolver(SkillEvolutionResult(skills=proposed))
    with pytest.raises(ValueError, match="read-only"):
        await evolver.evolve(
            skills=inputs,
            trajectories=evidence(),
            work_dir=tmp_path,
            options=EvolutionOptions(editable_skills=["a"]),
        )
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("editable_skills", [None, ["a", "b"], []])
async def test_existing_skills_cannot_be_deleted_under_any_edit_policy(tmp_path, editable_skills):
    inputs = {"a": skill("alpha"), "b": skill("beta")}
    evolver = make_evolver(SkillEvolutionResult(skills={"a": inputs["a"]}))
    with pytest.raises(ValueError, match="retain every input Skill"):
        await evolver.evolve(
            skills=inputs,
            trajectories=evidence(),
            work_dir=tmp_path,
            options=EvolutionOptions(editable_skills=editable_skills),
        )
    assert not (tmp_path / "result.json").exists()


async def test_creation_switch_is_independent_of_existing_edit_permissions(tmp_path):
    inputs = {"a": skill("alpha"), "b": skill("beta")}
    evolver = make_evolver(SkillEvolutionResult(skills=inputs, created_skills={"new/delta": skill("delta")}))
    with pytest.raises(ValueError, match="does not allow new"):
        await evolver.evolve(
            skills=inputs,
            trajectories=evidence(),
            work_dir=tmp_path,
            options=EvolutionOptions(allow_create_skills=False),
        )


@pytest.mark.parametrize("dynamic", [False, True])
async def test_shared_protocol_preserves_algorithm_owned_creation_formats(tmp_path, dynamic):
    original = skill("alpha")
    created = (
        TreeSkill(TreeSkillConfig(router={"model": "offline"}), source=skill("new-tree").source)
        if dynamic
        else Skill(
            SkillContent(
                content="# New source\r\n", name="new-source", description="New workflow", standard_format=False
            )
        )
    )
    evolver = make_evolver(
        SkillEvolutionResult(skills={"a": original, "b": original}, created_skills={"new/source": created})
    )
    result = await evolver.evolve(
        skills={"a": original, "b": original},
        trajectories=evidence(),
        work_dir=tmp_path,
        options=EvolutionOptions(editable_skills=[]),
    )
    assert result.created_skills["new/source"].snapshot() == created.snapshot()


async def test_permission_changes_are_checked_without_shared_disk_checkpoints(tmp_path):
    inputs = {"a": skill("alpha"), "b": skill("beta")}
    evolver = make_evolver(SkillEvolutionResult(skills=inputs))
    await evolver.evolve(skills=inputs, trajectories=evidence(), work_dir=tmp_path)
    await evolver.evolve(
        skills=inputs,
        trajectories=evidence(),
        work_dir=tmp_path,
        options=EvolutionOptions(editable_skills=[]),
    )
    assert evolver.calls == 2
    assert not list(tmp_path.iterdir())


async def test_an_algorithm_cannot_mutate_its_historical_evidence(tmp_path):
    class MutatingEvolver(ProposalEvolver):
        async def _evolve(self, request):
            request.trajectories[0].trajectory.append({"role": "assistant", "content": "Changed"})
            return self.proposal

    inputs = {"a": skill("alpha"), "b": skill("beta")}
    evolver = MutatingEvolver(Options())
    evolver.proposal = SkillEvolutionResult(skills=inputs)
    with pytest.raises(ValueError, match="historical evidence"):
        await evolver.evolve(skills=inputs, trajectories=evidence(), work_dir=tmp_path)


async def test_algorithm_cannot_replace_or_widen_the_frozen_permission_policy(tmp_path):
    class WideningEvolver(ProposalEvolver):
        async def _evolve(self, request):
            request.options = EvolutionOptions(editable_skills=None)
            return self.proposal

    inputs = {"a": skill("alpha"), "b": skill("beta")}
    evolver = WideningEvolver(Options())
    evolver.proposal = SkillEvolutionResult(skills=inputs)
    with pytest.raises(ValueError, match="frozen"):
        await evolver.evolve(
            skills=inputs,
            trajectories=evidence(),
            work_dir=tmp_path,
            options=EvolutionOptions(editable_skills=["a"]),
        )
    assert not (tmp_path / "result.json").exists()
