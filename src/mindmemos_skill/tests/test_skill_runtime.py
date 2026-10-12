"""Complete source ownership, task assembly, diff, and export rollback contracts."""

import asyncio

import pytest
from mindmemos_skill.skill import Skill, SkillContent, SkillSet, compare_snapshots, io


def source(text="body", resources=None):
    """Build a nonstandard source with explicit metadata."""
    return SkillContent(
        content=text, resources=resources or {}, name="Name", description="Description", standard_format=False
    )


class DynamicSkill(Skill):
    """Retain a runtime dependency that cannot be reconstructed from source alone."""

    def __init__(self, content, dependency):
        super().__init__(content)
        self.dependency = dependency

    async def assemble(self, task):
        return self.with_content(self.content + task)


def test_candidate_preserves_subclass_dependencies_and_nonstandard_metadata():
    dependency = object()
    original = DynamicSkill(source(resources={"asset": b"old"}), dependency)
    candidate = original.with_content("new")
    assert type(candidate) is DynamicSkill and candidate.dependency is dependency
    assert candidate.source.name == "Name" and candidate.source.description == "Description"
    assert original.content == "body" and candidate.content == "new"
    replaced = candidate.with_resources({"asset": b"new", "empty": b""})
    assert original.resources == candidate.resources == {"asset": b"old"}
    assert replaced.resources == {"asset": b"new", "empty": b""}
    assert candidate.snapshot().content_hash == replaced.snapshot().content_hash
    assert candidate.snapshot().skill_hash != replaced.snapshot().skill_hash


def test_mutated_models_are_revalidated_before_restore_and_export(tmp_path):
    snapshot = Skill(source()).snapshot()
    snapshot.resources["asset"] = "YQ=="
    with pytest.raises(ValueError, match="hash mismatch"):
        snapshot.to_content()
    altered = source()
    altered.resources["../escape"] = b"bad"
    with pytest.raises(ValueError):
        Skill(altered)
    with pytest.raises(ValueError):
        io.export_skill(altered, tmp_path / "export")
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "export").exists()


def test_diff_includes_empty_files_binary_resources_and_metadata():
    before = Skill(source(resources={"deleted": b"", "binary": b"\0old", "text": b"old\n"})).snapshot()
    after = Skill(source("changed", {"added": b"", "binary": b"\xff", "text": b"new\n"})).snapshot()
    after.name = "Renamed"
    result = compare_snapshots(before, after)
    assert [item.path for item in result.files] == ["SKILL.md", "added", "binary", "deleted", "text"]
    assert [item.change for item in result.files] == ["modified", "added", "modified", "deleted", "modified"]
    assert result.files[2].binary and result.files[2].diff == ""
    assert "-old" in result.files[4].diff and "+new" in result.files[4].diff
    assert result.metadata_changed and not result.implementation_changed


def test_export_failure_restores_previous_directory(tmp_path, monkeypatch):
    destination = tmp_path / "export"
    destination.mkdir()
    (destination / "keep").write_bytes(b"original")
    replace = io.os.replace

    def fail_publish(old, new):
        if old.is_dir() and "-export-" in old.name:
            raise OSError("publish failed")
        return replace(old, new)

    monkeypatch.setattr(io.os, "replace", fail_publish)
    with pytest.raises(OSError, match="publish failed"):
        Skill(source()).export(destination, overwrite=True)
    assert (destination / "keep").read_bytes() == b"original"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["export"]


@pytest.mark.parametrize("path", ["", ".", "./", "a/../b", "SKILL.md/a"])
def test_resource_paths_reject_invalid_roots(path):
    with pytest.raises(ValueError):
        source(resources={path: b"x"})


@pytest.mark.asyncio
async def test_skillset_assembly_preserves_selection_order_and_concurrency():
    active = peak = 0
    lock = asyncio.Lock()

    class MeasuredSkill(Skill):
        async def assemble(self, task):
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            try:
                await asyncio.sleep(0.03)
                return self.with_content(self.content + task)
            finally:
                async with lock:
                    active -= 1

    skills = SkillSet({str(i): MeasuredSkill(source(str(i))) for i in range(5)}, max_concurrency=2)
    assembled = await skills.assemble("!", selected_ids=["4", "1", "3", "0", "2"])
    assert [skill.content for skill in assembled] == ["4!", "1!", "3!", "0!", "2!"]
    assert peak == 2 and active == 0
    assert skills.identity().skill_ids == ["0", "1", "2", "3", "4"]
    with pytest.raises(ValueError):
        await skills.assemble("!", selected_ids=["0", "0"])


def test_export_overwrite_removes_stale_files_and_preserves_plain_text(tmp_path):
    destination = tmp_path / "export"
    destination.mkdir()
    (destination / "stale").write_bytes(b"stale")
    skill = Skill(source("plain\r\n", {"asset": b"\x00\xff"}))
    with pytest.raises(ValueError):
        skill.export(destination)
    skill.export(destination, overwrite=True)
    assert sorted(path.name for path in destination.iterdir()) == ["SKILL.md", "asset"]
    assert (destination / "SKILL.md").read_bytes() == b"plain\r\n"
    assert (destination / "asset").read_bytes() == b"\x00\xff"


def test_local_source_rejects_symlink_resources(tmp_path):
    (tmp_path / "SKILL.md").write_text("---\nname: A\ndescription: D\n---\n")
    (tmp_path / "link").symlink_to(tmp_path / "SKILL.md")
    with pytest.raises(ValueError, match="symbolic links"):
        Skill.from_path(tmp_path)


def test_snapshot_runtime_metadata_roundtrip_does_not_change_source_hashes():
    original = Skill(source()).snapshot()
    configured = original.model_copy(
        update={"implementation": "custom", "config_schema_version": 3, "config": {"options": {"temperature": 0.5}}}
    )
    restored = Skill.from_snapshot(configured)
    configured.config["options"]["temperature"] = 1.0
    result = restored.snapshot()
    assert result.implementation == "custom" and result.config_schema_version == 3
    assert result.config == {"options": {"temperature": 0.5}}
    assert (result.content_hash, result.skill_hash) == (original.content_hash, original.skill_hash)
