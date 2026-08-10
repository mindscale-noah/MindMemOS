"""Phase 3 contracts for standalone local Skill management."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mindmemos_skill.errors import SkillConflictError, SkillExportError
from mindmemos_skill.management import (
    DuplicateAction,
    ExportSkillRequest,
    LocalSkillManager,
    PublishSkillRequest,
    RegisterSkillRequest,
)
from mindmemos_skill.persistence import SkillVersionOrigin


def _source(tmp_path: Path, name: str = "source", *, version: str = "1.0.0", body: str = "Root\n") -> Path:
    source = tmp_path / name
    source.mkdir()
    (source / "SKILL.md").write_text(
        f'name: demo\nversion: "{version}"\ndescription: local demo\n\n{body}',
        encoding="utf-8",
    )
    references = source / "references"
    references.mkdir()
    (references / "guide.md").write_text("Private guide\n", encoding="utf-8")
    scripts = source / "scripts"
    scripts.mkdir()
    script = scripts / "check.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    script.chmod(0o755)
    return source


@pytest.mark.asyncio
async def test_register_query_duplicate_and_restart_are_database_backed(tmp_path: Path) -> None:
    database_path = tmp_path / "state.db"
    source = _source(tmp_path)
    manager = await LocalSkillManager.open(database_path)
    registered = await manager.register(
        RegisterSkillRequest(source_path=source, alias="demo-main", commit_message="  Initial import  ")
    )

    detail = await manager.get_skill("demo-main")
    version = await manager.get_version(registered.skill_id, registered.version_id)
    assert detail.skill.version_count == 1
    assert detail.skill.pending_count == 1
    assert version.alias == "demo-main"
    assert version.version_label == "1.0.0"
    assert version.commit_message == "Initial import"
    assert (await manager.repository.query_versions(alias="demo-main")) == [version]
    assert (await manager.repository.query_versions(version_id=version.version_id)) == [version]
    assert (await manager.repository.query_versions(content_hash=version.content_hash)) == [version]

    with pytest.raises(SkillConflictError, match="duplicate_action"):
        await manager.register(RegisterSkillRequest(source_path=source))
    reused = await manager.register(RegisterSkillRequest(source_path=source, duplicate_action=DuplicateAction.REUSE))
    assert (reused.action, reused.skill_id, reused.version_id) == (
        "reused",
        registered.skill_id,
        registered.version_id,
    )
    await manager.close()

    reopened = await LocalSkillManager.open(database_path)
    assert (await reopened.get_skill("demo-main")).latest_version.version_id == registered.version_id
    await reopened.close()


@pytest.mark.asyncio
async def test_repository_accepts_branches_and_multi_parent_merges(tmp_path: Path) -> None:
    manager = await LocalSkillManager.open(tmp_path / "state.db")
    root = await manager.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="dag"))
    first_branch = await manager.publish(
        PublishSkillRequest(
            skill_ref="dag",
            base_version_id=root.version_id,
            content='name: demo\nversion: "1.1.0"\n\nFirst branch\n',
        )
    )
    second_branch = await manager.publish(
        PublishSkillRequest(
            skill_ref="dag",
            base_version_id=root.version_id,
            content='name: demo\nversion: "1.2.0"\n\nSecond branch\n',
        )
    )
    branch_record = await manager.get_version("dag", second_branch.version_id)
    merged_record = branch_record.model_copy(
        update={
            "version_id": "merged-version",
            "parent_version_ids": [first_branch.version_id, second_branch.version_id],
            "version_label": "2.0.0",
            "origin": SkillVersionOrigin.MERGE,
            "created_at": branch_record.created_at + timedelta(seconds=1),
            "updated_at": branch_record.created_at + timedelta(seconds=1),
        }
    )

    state = await manager.repository.create_version(merged_record, now=datetime.now(UTC))
    stored_merge = await manager.get_version("dag", "merged-version")

    assert stored_merge.parent_version_ids == [first_branch.version_id, second_branch.version_id]
    assert state.skill_id == root.skill_id
    assert len(await manager.list_versions("dag")) == 4
    await manager.close()


@pytest.mark.asyncio
async def test_publish_builds_dag_and_derives_latest_version_without_pointer(tmp_path: Path) -> None:
    manager = await LocalSkillManager.open(tmp_path / "state.db")
    registered = await manager.register(RegisterSkillRequest(source_path=_source(tmp_path), alias="demo"))
    published = await manager.publish(
        PublishSkillRequest(
            skill_ref="demo",
            content='name: demo\nversion: "1.10.0"\n\nSecond\n',
            commit_message="candidate",
        )
    )

    versions = await manager.list_versions("demo")
    assert [item.version_id for item in versions] == [registered.version_id, published.version_id]
    assert versions[1].parent_version_ids == [registered.version_id]
    assert (await manager.get_skill("demo")).latest_version.version_id == published.version_id
    assert len(await manager.repository.list_operations(skill_id=registered.skill_id)) == 2

    older_label = await manager.publish(
        PublishSkillRequest(
            skill_ref="demo",
            content='name: demo\nversion: "1.2.0"\n\nNumerically older\n',
        )
    )
    assert (await manager.get_skill("demo")).latest_version.version_id == older_label.version_id
    assert len(await manager.list_versions("demo")) == 3
    await manager.close()


@pytest.mark.asyncio
async def test_repository_rejects_cross_family_parent_and_alias_conflicts_atomically(tmp_path: Path) -> None:
    manager = await LocalSkillManager.open(tmp_path / "state.db")
    first = await manager.register(RegisterSkillRequest(source_path=_source(tmp_path, "one"), alias="one"))
    second_source = _source(tmp_path, "two", body="Other family\n")
    second = await manager.register(RegisterSkillRequest(source_path=second_source, alias="two"))
    first_record = await manager.get_version(first.skill_id, first.version_id)

    invalid = first_record.model_copy(
        update={
            "version_id": "cross-family-child",
            "parent_version_ids": [second.version_id],
            "version_label": "2.0.0",
            "created_at": first_record.created_at + timedelta(seconds=1),
        }
    )
    with pytest.raises(SkillConflictError, match="same Skill family"):
        await manager.repository.create_version(invalid, now=datetime.now(UTC))
    assert len(await manager.list_versions(first.skill_id)) == 1

    third_source = _source(tmp_path, "three", body="Third family\n")
    with pytest.raises(SkillConflictError, match="alias already exists"):
        await manager.register(RegisterSkillRequest(source_path=third_source, alias="one"))
    assert len(await manager.list_skills()) == 2
    await manager.close()


@pytest.mark.asyncio
async def test_sync_state_has_no_head_pointers_and_outbox_is_flat(tmp_path: Path) -> None:
    manager = await LocalSkillManager.open(tmp_path / "state.db")
    registered = await manager.register(RegisterSkillRequest(source_path=_source(tmp_path)))
    published = await manager.publish(
        PublishSkillRequest(
            skill_ref=registered.skill_id,
            content='name: demo\nversion: "1.1.0"\n\nCandidate\n',
        )
    )
    detail = await manager.get_skill(registered.skill_id)
    state = detail.sync_state.model_dump()
    operations = await manager.repository.list_operations(skill_id=registered.skill_id)

    assert detail.latest_version.version_id == published.version_id
    assert not {"effective_version_id", "published_head_id", "pending_operations"}.intersection(state)
    assert [item.version_id for item in operations] == [registered.version_id, published.version_id]
    await manager.close()


@pytest.mark.asyncio
async def test_diff_and_export_restore_complete_snapshot_without_deleting_unmanaged_files(tmp_path: Path) -> None:
    manager = await LocalSkillManager.open(tmp_path / "managed" / "state.db")
    registered = await manager.register(RegisterSkillRequest(source_path=_source(tmp_path)))
    published = await manager.publish(
        PublishSkillRequest(
            skill_ref=registered.skill_id,
            content='name: demo\nversion: "1.1.0"\n\nEdited\n',
        )
    )
    difference = await manager.diff(
        registered.skill_id,
        from_version_id=registered.version_id,
        to_version_id=published.version_id,
    )
    assert difference.changed_files == ["SKILL.md"]
    assert "-Root" in difference.diff
    assert "+Edited" in difference.diff

    target = tmp_path / "exported"
    target.mkdir()
    (target / "unmanaged.txt").write_text("keep me\n", encoding="utf-8")
    result = await manager.export(
        ExportSkillRequest(
            skill_ref=registered.skill_id,
            version_id=published.version_id,
            target_path=target,
        )
    )
    assert result.exported_files == ["SKILL.md", "references/guide.md", "scripts/check.py"]
    assert (target / "SKILL.md").read_text(encoding="utf-8").endswith("Edited\n")
    assert (target / "references" / "guide.md").read_text(encoding="utf-8") == "Private guide\n"
    assert (target / "scripts" / "check.py").stat().st_mode & 0o777 == 0o755
    assert (target / "unmanaged.txt").read_text(encoding="utf-8") == "keep me\n"
    await manager.close()


@pytest.mark.asyncio
async def test_export_failure_restores_overwritten_files_and_keeps_unmanaged_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mindmemos_skill.management import installer as installer_module

    manager = await LocalSkillManager.open(tmp_path / "managed" / "state.db")
    registered = await manager.register(RegisterSkillRequest(source_path=_source(tmp_path)))
    target = tmp_path / "target"
    target.mkdir()
    (target / "SKILL.md").write_text("old managed content\n", encoding="utf-8")
    (target / "unmanaged.txt").write_text("keep me\n", encoding="utf-8")
    real_replace = os.replace
    staged_replacements = 0

    def fail_second_staged_replace(source, destination):
        nonlocal staged_replacements
        if "-export-" in str(source):
            staged_replacements += 1
            if staged_replacements == 2:
                raise OSError("simulated interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(installer_module.os, "replace", fail_second_staged_replace)
    with pytest.raises(SkillExportError, match="failed to export"):
        await manager.export(ExportSkillRequest(skill_ref=registered.skill_id, target_path=target))

    assert (target / "SKILL.md").read_text(encoding="utf-8") == "old managed content\n"
    assert (target / "unmanaged.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not (target / "references" / "guide.md").exists()
    await manager.close()
