"""Public SDK boundary checks for the pointer-free Skill protocol."""

from __future__ import annotations

from mindmemos_sdk.skills import LocalSkillManifest, LocalSkillVersionMetadata, SkillManager
from mindmemos_sdk.skills.models import SkillOrigin, SkillVersionStatus

from mindmemos_sdk import cli


def test_public_models_expose_latest_projection_and_multi_parent_dag() -> None:
    manifest_fields = set(LocalSkillManifest.model_fields)
    version_fields = set(LocalSkillVersionMetadata.model_fields)

    assert "latest_version_id" in manifest_fields
    assert "parent_version_ids" in version_fields
    assert not {"active_version_id", "effective_version_id", "published_head_id"}.intersection(manifest_fields)
    assert "parent_version_id" not in version_fields


def test_public_enums_match_shared_contract() -> None:
    assert {item.value for item in SkillVersionStatus} == {"draft", "rejected", "published", "archived"}
    assert {item.value for item in SkillOrigin} == {"local", "cloud", "evolution", "merge"}


def test_pointer_mutations_are_absent_from_manager_and_cli() -> None:
    assert not any(hasattr(SkillManager, name) for name in ("promote_local", "switch_local", "rollback_local"))

    skill_parser = next(
        action for action in cli.build_parser()._actions if getattr(action, "dest", None) == "command"
    ).choices["skill"]
    command_action = next(
        action for action in skill_parser._actions if getattr(action, "dest", None) == "skill_command"
    )
    assert not {"promote", "switch", "rollback"}.intersection(command_action.choices)
