"""Tests for the local UI Skill application service."""

from __future__ import annotations

from pathlib import Path

from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import (
    ExportSkillRequest,
    LocalSkillSyncState,
    PublishLocalRequest,
    RegisterLocalRequest,
    SkillManager,
)
from mindmemos_sdk.ui import LocalSkillUIService


class _UnusedCloud:
    pass


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "skill"
    source.mkdir()
    (source / "SKILL.md").write_text(
        'name: demo\ndescription: Demo description\nversion: "1.0.0"\n\nBody\n',
        encoding="utf-8",
    )
    (source / "references").mkdir()
    (source / "references" / "private.md").write_text("private\n", encoding="utf-8")
    return source


def test_ui_service_uses_immutable_versions_and_shared_manager(tmp_path):
    manager = SkillManager.from_config_manager(
        ConfigManager(config_dir=tmp_path / "config"),
        _UnusedCloud(),
    )
    service = LocalSkillUIService(manager)
    registered = service.register(RegisterLocalRequest(source_path=str(_source(tmp_path)), alias="demo-main"))

    items = service.list_skills()
    detail = service.detail("demo-main")
    content = service.content("demo-main")

    assert len(items) == 1
    assert items[0].skill_id == registered.skill_id
    assert items[0].description == "Demo description"
    assert items[0].latest_version_label == "1.0.0"
    assert items[0].pending_count == 1
    assert items[0].sync_state == "pending"
    assert items[0].model_dump().get("path") is None
    assert detail.skill.latest_version_id == registered.version_id
    assert detail.latest_version.is_latest is True
    assert detail.latest_version.has_linked_files is True
    assert content.version_id == registered.version_id
    assert content.content.endswith("Body\n")
    assert set(content.files) == {"SKILL.md", "references/private.md"}

    published, unchanged_detail = service.publish(
        PublishLocalRequest(
            skill_id=registered.skill_id,
            base_version_id=registered.version_id,
            content='name: demo\nversion: "1.1.0"\n\nEdited\n',
            commit_message="UI draft",
        )
    )

    assert published.latest_version_id == published.version_id
    assert unchanged_detail.skill.latest_version_label == "1.1.0"
    assert unchanged_detail.latest_version.version_id == published.version_id
    assert unchanged_detail.versions[-1].commit_message == "UI draft"
    assert unchanged_detail.versions[-1].is_latest is True
    assert service.content("demo-main").content.endswith("Edited\n")

    files = service.content("demo-main", published.version_id).files
    files["references/private.md"] = "edited in browser\n"
    files_published, _ = service.publish(
        PublishLocalRequest(
            skill_id=registered.skill_id,
            base_version_id=published.version_id,
            files=files,
            version_label="1.2.0",
        )
    )
    assert service.content("demo-main", files_published.version_id).files["references/private.md"] == (
        "edited in browser\n"
    )


def test_ui_service_compare_and_export_include_local_linked_file_changes(tmp_path):
    source = _source(tmp_path)
    manager = SkillManager.from_config_manager(
        ConfigManager(config_dir=tmp_path / "config"),
        _UnusedCloud(),
    )
    service = LocalSkillUIService(manager)
    registered = service.register(RegisterLocalRequest(source_path=str(source)))
    (source / "SKILL.md").write_text('name: demo\nversion: "2.0.0"\n\nSecond\n', encoding="utf-8")
    (source / "references" / "private.md").write_text("changed private\n", encoding="utf-8")
    published, _detail = service.publish(
        PublishLocalRequest(
            skill_id=registered.skill_id,
            source_path=str(source),
        )
    )

    comparison = service.compare(registered.skill_id, registered.version_id, published.version_id)
    exported = service.export(
        ExportSkillRequest(
            skill_id=registered.skill_id,
            target_path=str(tmp_path / "export"),
        )
    )

    assert "+Second" in comparison.content_diff
    assert comparison.linked_file_changes == ["references/private.md"]
    assert exported.version_id == published.version_id
    assert (tmp_path / "export" / "references" / "private.md").read_text(encoding="utf-8") == "changed private\n"


def test_ui_list_uses_application_pending_summary(tmp_path):
    manager = SkillManager.from_config_manager(
        ConfigManager(config_dir=tmp_path / "config"),
        _UnusedCloud(),
    )
    service = LocalSkillUIService(manager)
    registered = service.register(RegisterLocalRequest(source_path=str(_source(tmp_path))))
    [item] = service.list_skills()

    assert item.sync_state == LocalSkillSyncState.PENDING.value


def test_ui_service_unregister_reports_deleted_scope_and_preserves_source(tmp_path):
    source = _source(tmp_path)
    manager = SkillManager.from_config_manager(
        ConfigManager(config_dir=tmp_path / "config"),
        _UnusedCloud(),
    )
    service = LocalSkillUIService(manager)
    registered = service.register(RegisterLocalRequest(source_path=str(source), alias="demo-main"))
    service.publish(
        PublishLocalRequest(
            skill_id=registered.skill_id,
            base_version_id=registered.version_id,
            content='name: demo\nversion: "1.1.0"\n\nEdited\n',
        )
    )

    result = service.unregister(registered.skill_id)

    assert result.skill_id == registered.skill_id
    assert result.name == "demo"
    assert result.alias == "demo-main"
    assert result.deleted_version_count == 2
    assert result.deleted_pending_count == 2
    assert result.source_files_deleted is False
    assert result.cloud_skill_deleted is False
    assert source.is_dir()
    assert service.list_skills() == []
