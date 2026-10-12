"""Content integrity across dynamic assembly, interruption, and training publication."""

import pytest
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.builtin_envs.common import export_skills
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillContent, SkillSet
from pydantic import BaseModel

from .test_treeskill import Router


async def test_runtime_restores_standard_crlf_source_without_loading_provider(tmp_path, monkeypatch):
    source = SkillContent(
        content="---\r\nname: test\r\ndescription: Rules\r\n---\r\n# First\r\nDo A.\r\n# Second\r\nDo B.\r\n",
        resources={"assets/raw.bin": b"\x00\xff\r\n"},
    )
    router = Router('{"selected_subtree_ids":["001"]}')
    skill = TreeSkill(TreeSkillConfig(router=router.config), source=source)
    calls = []

    def provider(config):
        calls.append(config)
        return router

    monkeypatch.setattr("mindmemos_skill.algos.train.treeskill.treeskill.OpenAIChatAgent", provider)
    snapshot = skill.snapshot()
    restored = load_runtime_skill(snapshot)
    restored.export(tmp_path / "source")
    assert calls == []
    assert (tmp_path / "source/SKILL.md").read_bytes() == source.content.encode()
    selected = await restored.assemble(Task(id="route", instruction="Do A"))
    assert len(calls) == 1
    assert "Do A" in selected.content and "Do B" not in selected.content
    assert selected.source.standard_format is False
    assert selected.resources == source.resources
    assert restored.snapshot() == snapshot
    assert load_runtime_skill(selected.snapshot()).snapshot() == selected.snapshot()
    with pytest.raises(NotImplementedError):
        load_runtime_skill(snapshot.model_copy(update={"implementation": "unknown"}))
    with pytest.raises(ValueError):
        load_runtime_skill(snapshot.model_copy(update={"resources": {"assets/raw.bin": "AA=="}}))


@pytest.mark.asyncio
async def test_interrupted_assembly_reuses_full_snapshot_and_rejects_resource_tampering(tmp_path):
    router = Router('{"selected_subtree_ids":[]}')
    source = TreeSkill(
        TreeSkillConfig(router=router.config),
        source=SkillContent(
            content="# First\nDo A",
            resources={"raw.bin": b"\xff"},
            standard_format=False,
            name="test",
            description="rules",
        ),
        router=router,
    )

    class Options(BaseModel):
        """No external settings."""

    class InterruptedEnvironment(Environment[Options]):
        config_type = Options
        attempts = 0

        async def _run(self, *, task, reward_data, skills, workspace):
            self.attempts += 1
            assert isinstance(skills[0], TreeSkill)
            assert skills[0].router is router
            assert skills[0].content == ""
            assert skills[0].resources == {"raw.bin": b"\xff"}
            if self.attempts == 1:
                raise RuntimeError("interrupted")
            return RolloutResult(output="ok", trajectory=[], reward=Reward(score=1, success=True, detail="ok"))

    env = InterruptedEnvironment(Options())
    args = dict(
        task=Task(id="task", instruction="Nothing relevant"),
        work_dir=tmp_path,
        reward_data=RewardData(),
        skill_set=SkillSet({"one": source}, max_concurrency=1),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        await env.rollout(**args)
    await env.rollout(**args)
    assert len(router.calls) == 1
    record = read_json(tmp_path / "audit/request.json")
    record["assembled_skills"][0]["resources"]["raw.bin"] = "AA=="
    write_json(tmp_path / "audit/request.json", record)
    with pytest.raises(ValueError):
        await env.rollout(**args)


def test_environment_install_preserves_task_inputs_and_rejects_symlinks(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "input.xlsx").write_bytes(b"task-input")
    skill = Skill(
        SkillContent(
            content="rules\r\n",
            resources={"assets/raw.bin": b"\x00\xff"},
            name="test",
            description="rules",
            standard_format=False,
        )
    )
    export_skills([skill], workspace)
    assert (workspace / "input.xlsx").read_bytes() == b"task-input"
    assert (workspace / "SKILL.md").read_bytes() == b"rules\r\n"
    assert (workspace / "assets/raw.bin").read_bytes() == b"\x00\xff"
    (workspace / "assets/raw.bin").unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(b"unchanged")
    (workspace / "assets/raw.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic"):
        export_skills([skill.with_content("changed")], workspace)
    assert outside.read_bytes() == b"unchanged"
    assert (workspace / "SKILL.md").read_bytes() == b"rules\r\n"


@pytest.mark.asyncio
async def test_forest_merge_keeps_standard_header_and_binary_resources(tmp_path):
    from mindmemos_skill.algos.train.skill_forest import SkillForestTrainer
    from mindmemos_skill.algos.train.skill_forest.tree import apply_patch
    from mindmemos_skill.algos.train.skill_forest.typing import Patch

    from .test_skill_forest import FakeEnvironment, FakeOptimizer, config, samples

    header = "---\r\nname: original\r\ndescription: Rules\r\nextra: keep\r\n---\r\n"
    source = Skill(SkillContent(content=header + "# Rules\r\nOriginal.\r\n", resources={"raw.bin": b"\x00\xff"}))
    snapshot = source.snapshot()
    assert apply_patch(source, Patch(edits=[])).snapshot() == snapshot
    trainer = SkillForestTrainer(config(ensemble_strategy="hierarchical_merge"), agent_factory=FakeOptimizer())
    selected = await trainer.train(
        skill=source, env=FakeEnvironment(), train_samples=samples(), validation_samples=[], work_dir=tmp_path
    )
    assert selected.content.startswith(header + "# Skill")
    assert "synthesized" in selected.content
    assert selected.source.standard_format is True
    assert selected.source.name == "original"
    assert selected.resources == {"raw.bin": b"\x00\xff"}
    assert source.snapshot() == snapshot
