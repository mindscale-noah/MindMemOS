"""Verify source bundles, task assembly, and durable execution content."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from mindmemos_skill.envs import Environment, Reward, RolloutResult, Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillBundle, SkillResources, SkillSet
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
        return self.with_bundle(SkillBundle(files={"SKILL.md": self.bundle.files["SKILL.md"] + task.instruction}))


class BundleEnvironment(Environment[EnvironmentOptions]):
    """Consume identical assembled content as both text and files."""

    config_type = EnvironmentOptions
    fail = False
    calls = 0

    def _run(self, *, task, reward_data, skills, workspace):
        self.calls += 1
        skill = skills[0]
        assert isinstance(skill, Skill)
        bundle = skill.bundle
        skill.export(workspace / "skill", overwrite=True)
        text = bundle.files["SKILL.md"]
        assert (workspace / "skill/SKILL.md").read_text() == text
        if self.fail:
            raise RuntimeError("Execution failed")
        return RolloutResult(output=text, trajectory=[], reward=Reward(score=1, success=True, detail="Matched"))


def test_bundle_sdk_compatibility_and_file_roundtrip(tmp_path):
    from mindmemos_sdk.skills.bundle import compute_content_hash, serialize_bundle

    raw = {"SKILL.md": "---\r\nname: 示例\r\n---\r\nUse checks.\r"}
    bundle = SkillBundle(files=raw)
    assert bundle.serialize() == serialize_bundle(raw)
    assert bundle.content_hash == compute_content_hash(raw)
    assert SkillBundle.deserialize(bundle.serialize()) == bundle
    resources = tmp_path / "references"
    resources.mkdir()
    (resources / "notes.txt").write_bytes(b"untouched\x00")
    (tmp_path / "SKILL.md").write_text("old instructions")
    assert bundle.export(tmp_path) == tmp_path.resolve()
    assert (resources / "notes.txt").read_bytes() == b"untouched\x00"
    assert SkillBundle.from_path(tmp_path) == bundle
    assert SkillBundle.from_path(tmp_path / "SKILL.md") == bundle


@pytest.mark.parametrize("files", [{}, {"../SKILL.md": "x"}, {"/SKILL.md": "x"}, {"SKILL.md": "x", "script.py": "y"}])
def test_bundle_rejects_paths_outside_its_scope(files):
    with pytest.raises(ValidationError):
        SkillBundle(files=files)


@pytest.mark.parametrize(
    "content",
    [
        "raw Markdown",
        "{}",
        '[{"path":"SKILL.md","content":1}]',
        '[{"path":"SKILL.md","content":"a"},{"path":"SKILL.md","content":"b"}]',
    ],
)
def test_malformed_serialized_content_is_not_silently_reinterpreted(content):
    with pytest.raises(ValueError):
        SkillBundle.deserialize(content)


def test_export_rejects_symlink_and_cleans_failed_write(tmp_path, monkeypatch):
    bundle = SkillBundle(files={"SKILL.md": "instructions"})
    outside = tmp_path / "outside.md"
    outside.write_text("outside")
    root = tmp_path / "skill"
    root.mkdir()
    (root / "SKILL.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        bundle.export(root)
    assert outside.read_text() == "outside"
    (root / "SKILL.md").unlink()
    (root / "SKILL.md").write_text("previous")

    def fail_replace(*args):
        raise OSError("replacement failed")

    monkeypatch.setattr("mindmemos_skill.skill.io.os.replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        bundle.export(root)
    assert (root / "SKILL.md").read_text() == "previous"
    assert list(root.iterdir()) == [root / "SKILL.md"]


def test_static_and_dynamic_isolation_and_snapshot_restore():
    source = SkillBundle(files={"SKILL.md": "Instructions: "})
    static = Skill(bundle=source)
    dynamic = TaskSkill(bundle=source)
    source.files["SKILL.md"] = "changed by caller"
    tasks = [Task(id=str(index), instruction=f"task {index}") for index in range(8)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(dynamic.assemble, tasks))
    assert [prepared.bundle.files["SKILL.md"] for prepared in outputs] == [
        "Instructions: " + task.instruction for task in tasks
    ]
    assert {static.assemble(task).bundle.content_hash for task in tasks} == {static.bundle.content_hash}
    outputs[0].bundle.files["SKILL.md"] = "modified output"
    static.assemble(tasks[0]).bundle.files["SKILL.md"] = "modified static output"
    assert static.bundle.files["SKILL.md"] == dynamic.bundle.files["SKILL.md"] == "Instructions: "
    candidate = dynamic.with_bundle(SkillBundle(files={"SKILL.md": "New: "}))
    assert candidate.assemble(tasks[0]).bundle.files["SKILL.md"] == "New: task 0"
    assert dynamic.bundle.files["SKILL.md"] == "Instructions: "
    restored = TaskSkill.from_snapshot(dynamic.snapshot())
    assert type(restored) is TaskSkill
    assert restored.assemble(tasks[1]).snapshot() == dynamic.assemble(tasks[1]).snapshot()
    assert Skill(bundle=source).assemble(tasks[0]).bundle == source
    with pytest.raises(ValueError, match="content hash"):
        Skill.from_snapshot(dynamic.snapshot().model_copy(update={"content_hash": "corrupted"}))


def test_assembly_is_saved_once_reused_on_retry_and_checked_on_resume(tmp_path):
    task = Task(id="one", instruction="public task")
    skill = TaskSkill(bundle=SkillBundle(files={"SKILL.md": "Use: "}))
    env = BundleEnvironment(EnvironmentOptions())
    env.fail = True
    with pytest.raises(RuntimeError, match="Execution failed"):
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    assert skill.calls == 1
    request_path = tmp_path / "audit/request.json"
    request = json.loads(request_path.read_text())
    assert request["skills"][0]["content_hash"] == skill.bundle.content_hash
    assert request["assembled_hashes"][0] != skill.bundle.content_hash
    skill.fail = True
    env.fail = False
    result = env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    assert result.output == "Use: public task"
    assert skill.calls == 1
    assert (
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
        == result
    )
    assert env.calls == 2
    request["assembled_bundles"][0]["files"]["SKILL.md"] = "corrupt"
    request_path.write_text(json.dumps(request))
    with pytest.raises(ValidationError, match="hashes"):
        env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    assert skill.calls == 1


def test_forced_rerun_invalidates_previous_assembly_before_failure(tmp_path):
    task = Task(id="one", instruction="task")
    skill = TaskSkill(bundle=SkillBundle(files={"SKILL.md": "Use: "}))
    env = BundleEnvironment(EnvironmentOptions())
    env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    skill.fail = True
    with pytest.raises(RuntimeError, match="Assembly failed"):
        env.rollout(
            task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1), resume=False
        )
    assert not (tmp_path / "audit/result.json").exists()
    assert json.loads((tmp_path / "audit/request.json").read_text())["assembled_bundles"] is None
    skill.fail = False
    env.rollout(task, tmp_path, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    assert skill.calls == 3


@pytest.mark.parametrize("implementation", [Skill, TaskSkill])
def test_environment_receives_complete_skill_and_unchanged_references(tmp_path, implementation):
    root = tmp_path / "source"
    (root / "references").mkdir(parents=True)
    (root / "SKILL.md").write_text("Read references/guide.md. Task: ")
    (root / "references/guide.md").write_bytes(b"Use the reference.\r\n")
    (root / "asset.bin").write_bytes(b"\x00\xff\x01")
    skill = implementation.from_path(root)
    snapshot = skill.snapshot()
    restored = implementation.from_snapshot(snapshot)
    assert restored.resources == skill.resources
    assert restored.resources.files["asset.bin"] == b"\x00\xff\x01"
    task = Task(id="one", instruction="public task")

    class ReferenceEnvironment(BundleEnvironment):
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
    env.rollout(task, work_dir, reward_data=RewardData(), skill_set=SkillSet({"0": skill}, max_concurrency=1))
    assert skill.snapshot() == snapshot
    assert (root / "SKILL.md").read_text() == "Read references/guide.md. Task: "
    (root / "references/guide.md").write_text("Changed reference")
    with pytest.raises(ValueError, match="inputs changed"):
        env.rollout(
            task,
            work_dir,
            reward_data=RewardData(),
            skill_set=SkillSet({"0": implementation.from_path(root)}, max_concurrency=1),
        )


@pytest.mark.parametrize("path", [".", "../secret", "/secret", "SKILL.md", "SKILL.md/other", "a\\b", "a/../b"])
def test_resource_paths_cannot_escape_or_replace_bundle(path):
    with pytest.raises(ValueError):
        SkillResources(files={path: b"content"})


def test_multiple_skills_export_raw_text_and_original_reference_paths(tmp_path):
    from mindmemos_skill.envs.builtin_envs.common import export_skills

    skills = [
        Skill(
            bundle=SkillBundle(files={"SKILL.md": f"Read references/{name}.md"}),
            resources=SkillResources(files={f"references/{name}.md": name.encode()}),
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
            bundle=SkillBundle(files={"SKILL.md": "Read references/guide.md"}),
            resources=SkillResources(files={"references/guide.md": content}),
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


def test_skill_export_without_runtime_dependencies(tmp_path):
    from mindmemos_skill.errors import SkillConflictError

    skill = Skill(
        bundle=SkillBundle(files={"SKILL.md": "stored instructions"}),
        resources=SkillResources(files={"references/image.bin": b"\0\xff"}),
    )
    snapshot = skill.snapshot()
    snapshot.implementation = "unavailable-runtime"
    restored = Skill.from_snapshot(snapshot)
    target = tmp_path / "exported"
    assert restored.export(target) == target.resolve()
    assert (target / "SKILL.md").read_text() == "stored instructions"
    assert (target / "references/image.bin").read_bytes() == b"\0\xff"
    (target / "stale.txt").write_text("old resource")
    with pytest.raises(SkillConflictError):
        restored.export(target)
    restored.export(target, overwrite=True)
    assert not (target / "stale.txt").exists()
    assert restored.snapshot() == snapshot
    assert snapshot == skill.snapshot().model_copy(update={"implementation": "unavailable-runtime"})


def test_restore_revalidates_before_export_touches_destination(tmp_path):
    snapshot = Skill(bundle=SkillBundle(files={"SKILL.md": "content"})).snapshot()
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    snapshot.content_hash = "corrupt"
    with pytest.raises(ValueError, match="hash"):
        Skill.from_snapshot(snapshot).export(target, overwrite=True)
    assert (target / "keep.txt").read_text() == "keep"
    assert not (target / "SKILL.md").exists()


def test_legacy_snapshot_metadata_survives_data_restore_and_content_updates():
    snapshot = Skill(bundle=SkillBundle(files={"SKILL.md": "original"})).snapshot()
    snapshot.implementation = "unavailable_algorithm"
    snapshot.config_schema_version = 9
    snapshot.config = {"router": {"model": "legacy-model"}}
    source = Skill.from_snapshot(snapshot)
    assert type(source) is Skill
    assert Skill.from_snapshot(source.snapshot()).snapshot() == snapshot
    snapshot.config["router"]["model"] = "changed by caller"
    candidate = source.with_bundle(SkillBundle(files={"SKILL.md": "updated"}))
    restored = Skill.from_snapshot(candidate.snapshot())
    assert restored.bundle.files["SKILL.md"] == "updated"
    assert restored.snapshot().implementation == "unavailable_algorithm"
    assert restored.snapshot().config_schema_version == 9
    assert restored.snapshot().config == {"router": {"model": "legacy-model"}}
    assert source.bundle.files["SKILL.md"] == "original"
