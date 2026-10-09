"""Verify complete content and resource preservation during environment execution."""

import asyncio
import json

import pytest
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillContent, SkillSet
from pydantic import BaseModel, ValidationError


class EnvironmentOptions(BaseModel):
    """Settings for the deterministic test environment."""


class TaskSkill(Skill):
    """Append public task content, counting physical assemblies for tests."""

    calls = 0
    fail = False

    def assemble(self, task):
        self.calls += 1
        if self.fail:
            raise RuntimeError("Assembly failed")
        return self.with_content(self.content + task.instruction)


class ContentEnvironment(Environment[EnvironmentOptions]):
    """Consume identical assembled content as both text and files."""

    config_type = EnvironmentOptions
    fail = False
    calls = 0

    def _run(self, *, task, reward_data, skills, workspace):
        self.calls += 1
        skill = skills[0]
        assert isinstance(skill, Skill)
        text = skill.content
        skill.export(workspace / "skill", overwrite=True)
        assert (workspace / "skill/SKILL.md").read_text() == text
        if self.fail:
            raise RuntimeError("Execution failed")
        return RolloutResult(output=text, trajectory=[], reward=Reward(score=1, success=True, detail="Matched"))


def test_assembly_is_saved_once_reused_on_retry_and_checked_on_resume(tmp_path):
    task = Task(id="one", instruction="public task")
    skill = TaskSkill(
        source=SkillContent(content="Use: ", name="test-skill", description="Task instructions", standard_format=False)
    )
    env = ContentEnvironment(EnvironmentOptions())
    env.fail = True
    with pytest.raises(RuntimeError, match="Execution failed"):
        asyncio.run(
            env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
        )
    assert skill.calls == 1
    request_path = tmp_path / "audit/request.json"
    request = json.loads(request_path.read_text())
    assert request["skills"][0]["content_hash"] == skill.source.content_hash
    assert request["assembled_hashes"][0] != skill.source.skill_hash
    skill.fail = True
    env.fail = False
    result = asyncio.run(
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    )
    assert result.output == "Use: public task"
    assert skill.calls == 1
    assert (
        asyncio.run(
            env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
        )
        == result
    )
    assert env.calls == 2
    request["assembled_skills"][0]["content"] = "corrupt"
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValidationError, match="hash"):
        asyncio.run(
            env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
        )
    assert skill.calls == 1


def test_forced_rerun_invalidates_previous_assembly_before_failure(tmp_path):
    task = Task(id="one", instruction="task")
    skill = TaskSkill(
        source=SkillContent(content="Use: ", name="test-skill", description="Task instructions", standard_format=False)
    )
    env = ContentEnvironment(EnvironmentOptions())
    asyncio.run(
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    )
    skill.fail = True
    with pytest.raises(RuntimeError, match="Assembly failed"):
        asyncio.run(
            env.rollout(
                task,
                tmp_path,
                reward_data=RewardData(),
                skill_set=SkillSet({"0": skill}, max_concurrency=1),
                resume=False,
            )
        )
    assert not (tmp_path / "audit/result.json").exists()
    assert json.loads((tmp_path / "audit/request.json").read_text())["assembled_skills"] is None
    skill.fail = False
    asyncio.run(
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    )
    assert skill.calls == 3


@pytest.mark.parametrize("implementation", [Skill, TaskSkill])
def test_environment_receives_complete_skill_and_unchanged_references(tmp_path, implementation):
    root = tmp_path / "source"
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("---\nname: test\ndescription: Rules\n---\nRead references/guide.md. Task: ")
    (root / "references/guide.md").write_bytes(b"Use the reference.\r\n")
    (root / "asset.bin").write_bytes(b"\x00\xff\x01")
    skill = implementation.from_path(root)
    snapshot = skill.snapshot()
    restored = implementation.from_snapshot(snapshot)
    assert restored.resources == skill.resources
    assert restored.resources["asset.bin"] == b"\x00\xff\x01"
    task = Task(id="one", instruction="public task")

    class ReferenceEnvironment(ContentEnvironment):
        def _run(self, *, task, reward_data, skills, workspace):
            prepared = skills[0]
            assert isinstance(prepared, Skill)
            assert type(prepared) is implementation
            assert prepared is not skill
            assert prepared.resources == skill.resources
            result = super()._run(task=task, reward_data=reward_data, skills=skills, workspace=workspace)
            assert (workspace / "skill/references/guide.md").read_bytes() == b"Use the reference.\r\n"
            assert (workspace / "skill/asset.bin").read_bytes() == b"\x00\xff\x01"
            return result

    env = ReferenceEnvironment(EnvironmentOptions())
    work_dir = tmp_path / "rollout"
    asyncio.run(
        env.rollout(task, work_dir, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    )
    assert skill.snapshot() == snapshot
    assert (
        root / "SKILL.md"
    ).read_text() == "---\nname: test\ndescription: Rules\n---\nRead references/guide.md. Task: "
    (root / "references/guide.md").write_text("Changed reference")
    with pytest.raises(ValueError, match="inputs changed"):
        asyncio.run(
            env.rollout(
                task,
                work_dir,
                reward_data=RewardData(),
                skill_set=SkillSet({"0": implementation.from_path(root)}, max_concurrency=1),
            )
        )


def test_multiple_skills_export_raw_text_and_original_reference_paths(tmp_path):
    from mindmemos_skill.envs.builtin_envs.common import export_skills

    skills = [
        Skill(
            source=SkillContent(
                content=f"Read references/{name}.md",
                resources={f"references/{name}.md": name.encode()},
                name="test-skill",
                description="Task instructions",
                standard_format=False,
            )
        )
        for name in ("first", "second")
    ]
    text = export_skills(skills, tmp_path)
    assert (tmp_path / "references/first.md").read_bytes() == b"first"
    assert (tmp_path / "references/second.md").read_bytes() == b"second"
    assert text == "Read references/first.md\n\nRead references/second.md"
    assert (tmp_path / "SKILL.md").read_text() == text


@pytest.mark.parametrize("second", [b"first", b"different"])
def test_export_skills_rejects_conflicting_resources_before_writing(tmp_path, second):
    from mindmemos_skill.envs.builtin_envs.common import export_skills

    skills = [
        Skill(
            source=SkillContent(
                content="Read references/guide.md",
                resources={"references/guide.md": content},
                name="test-skill",
                description="Task instructions",
                standard_format=False,
            )
        )
        for content in (b"first", second)
    ]
    if second == b"first":
        export_skills(skills, tmp_path)
        assert (tmp_path / "references/guide.md").read_bytes() == b"first"
    else:
        with pytest.raises(ValueError, match="Conflicting Skill resource: references/guide.md"):
            export_skills(skills, tmp_path)
        assert not list(tmp_path.iterdir())
