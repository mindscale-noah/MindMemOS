"""Contracts for the synchronous SkillApplication compatibility facade."""

from __future__ import annotations

from pathlib import Path

import pytest
from mindmemos_sdk.config import ConfigManager, SDKPortalConfigV2
from mindmemos_sdk.errors import (
    SkillCapabilityUnavailableError,
    SkillRegistryError,
    SkillRemoteError,
    TransportError,
)
from mindmemos_sdk.skills import (
    PublishLocalRequest,
    PushVersionResult,
    RegisterLocalRequest,
    SkillManager,
)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text('name: demo\nversion: "1.0.0"\n\nBody\n', encoding="utf-8")
    return source


def test_sync_facade_persists_only_through_skill_application_state_db(tmp_path: Path) -> None:
    config_manager = ConfigManager(config_dir=tmp_path / "config")
    manager = SkillManager.from_config_manager(config_manager)

    registered = manager.register_local(RegisterLocalRequest(source_path=str(_source(tmp_path)), alias="demo"))
    published = manager.publish_local(
        PublishLocalRequest(
            skill_id="demo",
            content='name: demo\nversion: "1.1.0"\n\nUpdated\n',
        )
    )

    assert manager.show_local("demo").latest_version_id == published.version_id
    assert [item.version_id for item in manager.local_history("demo")] == [
        registered.version_id,
        published.version_id,
    ]
    assert (config_manager.config_dir / "skill" / "state.db").is_file()
    assert not (config_manager.config_dir / "skills").exists()
    assert not (config_manager.config_dir / "outbox.json").exists()
    manager.close()


def test_sync_facade_reopens_same_application_database(tmp_path: Path) -> None:
    config_manager = ConfigManager(config_dir=tmp_path / "config")
    first = SkillManager.from_config_manager(config_manager)
    registered = first.register_local(RegisterLocalRequest(source_path=str(_source(tmp_path)), alias="demo"))
    first.close()

    second = SkillManager.from_config_manager(config_manager)

    assert second.show_local("demo").latest_version_id == registered.version_id
    second.close()


def test_sync_facade_uses_portal_skill_application_and_named_connection(tmp_path: Path) -> None:
    config_manager = ConfigManager(config_dir=tmp_path / "config")
    root = tmp_path / "portal-skill"
    config_manager.save_portal(
        SDKPortalConfigV2.model_validate(
            {
                "version": 2,
                "profiles": {
                    "default": {
                        "connections": {
                            "mindmemos_main": {
                                "base_url": "http://127.0.0.1:8000",
                            }
                        },
                        "default_connection": "mindmemos_main",
                        "skill": {
                            "application": {
                                "local": {
                                    "root_dir": str(root),
                                    "database": {"path": str(root / "state.db")},
                                }
                            }
                        },
                    }
                },
            }
        )
    )

    manager = SkillManager.from_config_manager(config_manager)
    manager.register_local(RegisterLocalRequest(source_path=str(_source(tmp_path))))

    assert (root / "state.db").is_file()
    assert not (config_manager.config_dir / "skill" / "state.db").exists()
    manager.close()


def test_sync_facade_local_only_never_indexes_or_borrows_none_connection(tmp_path: Path) -> None:
    config_manager = ConfigManager(config_dir=tmp_path / "config")
    root = tmp_path / "local-only-skill"
    config_manager.save_portal(
        SDKPortalConfigV2.model_validate(
            {
                "version": 2,
                "profiles": {
                    "default": {
                        "connections": {
                            "mindmemos_main": {
                                "base_url": "http://127.0.0.1:8000",
                            }
                        },
                        "default_connection": "mindmemos_main",
                        "skill": {
                            "remote": {"connection": None},
                            "application": {
                                "local": {
                                    "root_dir": str(root),
                                    "database": {"path": str(root / "state.db")},
                                }
                            },
                        },
                    }
                },
            }
        )
    )

    manager = SkillManager.from_config_manager(config_manager, cloud=object(), shared_connection_name=None)
    registered = manager.register_local(RegisterLocalRequest(source_path=str(_source(tmp_path))))

    with pytest.raises(SkillCapabilityUnavailableError, match="remote push"):
        manager.push_local(registered.skill_id)
    assert manager.show_local(registered.skill_id).latest_version_id == registered.version_id
    manager.close()


def test_sync_facade_maps_remote_request_failure_to_structured_sdk_error(tmp_path: Path) -> None:
    class UnavailableCloud:
        def push_version(self, request):
            raise TransportError("provider details must not become a business code")

    manager = SkillManager.from_config_manager(
        ConfigManager(config_dir=tmp_path / "config"),
        UnavailableCloud(),
    )
    registered = manager.register_local(RegisterLocalRequest(source_path=str(_source(tmp_path))))

    with pytest.raises(SkillRemoteError) as failure:
        manager.push_local(registered.skill_id)

    assert failure.value.error_code == "remote_unavailable"
    assert failure.value.retryable is True
    assert manager.pending_local_operations(registered.skill_id)[0].last_error_code == "remote_unavailable"
    manager.close()


def test_sync_facade_projects_remote_port_results_without_owning_sync_rules(tmp_path: Path) -> None:
    captured = {}

    class Cloud:
        def push_version(self, request):
            captured["request"] = request
            return PushVersionResult(
                cloud_skill_id="cloud-demo",
                version_id=request.version_id,
                content_hash=request.expected_content_hash,
                status="draft",
                created_at=request.created_at,
                received_at=request.created_at,
            )

    manager = SkillManager.from_config_manager(ConfigManager(config_dir=tmp_path / "config"), Cloud())
    registered = manager.register_local(RegisterLocalRequest(source_path=str(_source(tmp_path))))

    pushed = manager.push_local(registered.skill_id)

    assert pushed.version_id == registered.version_id
    assert captured["request"].version_id == registered.version_id
    assert manager.pending_local_operations() == []
    manager.close()


def test_closed_sync_facade_rejects_calls(tmp_path: Path) -> None:
    manager = SkillManager.from_config_manager(ConfigManager(config_dir=tmp_path / "config"))
    manager.close()

    with pytest.raises(SkillRegistryError, match="closed"):
        manager.list_local()
