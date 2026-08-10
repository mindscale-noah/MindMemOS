"""Portal v2 configuration, v1 conversion, and CLI cutover contracts."""

from __future__ import annotations

import json

import pytest
from mindmemos_sdk.config import (
    AuthConfig,
    ConfigManager,
    HttpConnectionConfig,
    NetworkConfig,
    SDKConfig,
    SDKConfigV1,
    SDKPortalConfigV2,
    StorageConfig,
    mask_secret,
)
from mindmemos_sdk.errors import ConfigNotFoundError, ConfigValidationError

from mindmemos_sdk import MindMemOSClient, cli


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(config_dir=tmp_path)


def _write_legacy(manager: ConfigManager, config: SDKConfigV1 | None = None) -> SDKConfigV1:
    legacy = config or SDKConfigV1()
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text(
        json.dumps(legacy.model_dump(mode="json", exclude_none=True), indent=2) + "\n",
        encoding="utf-8",
    )
    return legacy


def test_load_missing_legacy_raises(manager):
    assert manager.legacy_exists() is False
    with pytest.raises(ConfigNotFoundError):
        manager.load_legacy()


def test_update_auth_creates_only_portal_v2(manager):
    updated = manager.update_auth(base_url="https://my.example.com", api_key="mk_secret", user_id="u_1")
    profile = updated.profiles[updated.active_profile]
    connection = profile.connections[profile.default_connection]

    assert manager.portal_exists()
    assert not manager.legacy_exists()
    assert connection.base_url == "https://my.example.com"
    assert connection.api_key == "mk_secret"
    assert profile.identity.user_id == "u_1"
    assert manager.compile_portal().version == 2


def test_update_auth_migrates_v1_then_updates_only_portal(manager):
    legacy = SDKConfigV1(
        base_url="https://old.example.com",
        auth={"api_key": "old"},
        defaults={
            "user_id": "old-user",
            "app_id": "legacy-app",
            "agent_id": "legacy-agent",
            "session_id": "legacy-session",
        },
    )
    _write_legacy(manager, legacy)
    original = manager.config_path.read_bytes()

    updated = manager.update_auth(base_url="https://new.example.com", api_key="new", user_id="new-user")
    profile = updated.profiles[updated.active_profile]

    assert profile.identity.model_dump() == {
        "user_id": "new-user",
        "app_id": None,
        "agent_id": None,
        "session_id": None,
    }
    assert manager.config_path.read_bytes() == original
    assert manager.portal_v1_backup_path.read_bytes() == original


def test_update_auth_preserves_existing_profile_routes(manager):
    portal = SDKPortalConfigV2.model_validate(
        {
            "profiles": {
                "default": {
                    "default_connection": "main",
                    "connections": {
                        "main": {"base_url": "https://old.example.com"},
                        "skill": {"base_url": "https://skill.example.com"},
                    },
                    "skill": {"remote": {"connection": "skill"}},
                }
            }
        }
    )
    manager.save_portal(portal)

    updated = manager.update_auth(base_url="https://new.example.com", api_key="new", user_id="u")
    profile = updated.profiles["default"]

    assert set(profile.connections) == {"main", "skill"}
    assert profile.connections["main"].base_url == "https://new.example.com"
    assert profile.connections["skill"].base_url == "https://skill.example.com"
    assert profile.skill.remote.connection == "skill"


def test_reset_removes_active_and_legacy_config_but_keeps_backup(manager):
    _write_legacy(manager)
    manager.load_portal()

    assert manager.reset() is True
    assert not manager.portal_exists()
    assert not manager.legacy_exists()
    assert manager.portal_v1_backup_path.exists()
    assert manager.reset() is False


def test_load_invalid_legacy_json_raises(manager):
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        manager.load_legacy()


def test_load_legacy_rejects_removed_in_memory_connection(manager):
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text(
        json.dumps(
            {
                "connections": {
                    "default": {
                        "type": "in_memory",
                        "runtime_config": "config/mindmemos_lite/dev.yaml",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigValidationError):
        manager.load_legacy()


def test_save_portal_is_atomic_no_leftover_temp(manager):
    manager.update_auth(base_url="https://a", api_key="k", user_id="u")
    assert list(manager.config_dir.glob(".config-*.tmp")) == []
    assert list(manager.config_dir.glob(".settings-*.tmp")) == []


def test_portal_migration_dry_run_is_deterministic_and_does_not_write(manager):
    legacy = SDKConfigV1(
        base_url="https://api.example.com",
        auth={"api_key": "secret"},
        defaults={"user_id": "u_1", "app_id": "app-1"},
        skills=[{"id": "ignored-local-state"}],
    )
    _write_legacy(manager, legacy)

    first = manager.migrate_portal(dry_run=True)
    second = manager.migrate_portal(dry_run=True)

    assert first == second
    assert first.ignored_local_skill_entries == 1
    profile = first.portal.profiles["default"]
    assert profile.default_connection == "mindmemos_main"
    assert profile.memory.connection == "mindmemos_main"
    assert profile.skill.remote.connection == "mindmemos_main"
    assert profile.identity.app_id == "app-1"
    assert not manager.portal_exists()
    assert not manager.portal_v1_backup_path.exists()


def test_portal_migration_applies_atomically_without_touching_local_state(manager):
    _write_legacy(
        manager,
        SDKConfigV1(base_url="https://api.example.com", auth={"api_key": "secret"}, defaults={"user_id": "u"}),
    )
    original = manager.config_path.read_bytes()
    legacy_state = manager.config_dir / "skills" / "legacy" / "manifest.json"
    legacy_state.parent.mkdir(parents=True)
    legacy_state.write_text("legacy-state", encoding="utf-8")

    plan = manager.migrate_portal(dry_run=False)

    assert plan.changes_required is True
    assert manager.config_path.read_bytes() == original
    assert manager.portal_v1_backup_path.read_bytes() == original
    assert legacy_state.read_text(encoding="utf-8") == "legacy-state"
    assert manager.compile_portal().profile.memory_connection == "mindmemos_main"
    assert manager.compile_portal().profile.skill_connection == "mindmemos_main"


def test_load_portal_transparently_migrates_legacy_config(manager):
    _write_legacy(
        manager,
        SDKConfigV1(base_url="https://api.example.com", auth={"api_key": "secret"}, defaults={"user_id": "u"}),
    )
    original = manager.config_path.read_bytes()

    portal = manager.load_portal()

    assert portal.version == 2
    assert manager.portal_v1_backup_path.read_bytes() == original
    assert manager.config_path.read_bytes() == original


def test_load_portal_can_disable_automatic_migration(manager):
    _write_legacy(manager)
    with pytest.raises(ConfigNotFoundError):
        manager.load_portal(auto_migrate=False)
    assert not manager.portal_exists()


def test_portal_migration_preserves_explicit_named_connection_routes(manager):
    legacy = SDKConfigV1(
        connections={
            "memory-route": HttpConnectionConfig(base_url="https://memory.example.com", api_key="memory-key"),
            "skill-route": HttpConnectionConfig(base_url="https://skill.example.com", api_key="skill-key"),
        },
        clients={
            "memory": {"connection": "memory-route"},
            "skills": {"connection": "skill-route"},
        },
    )
    _write_legacy(manager, legacy)

    profile = manager.plan_portal_migration().portal.profiles["default"]

    assert set(profile.connections) == {"memory-route", "skill-route"}
    assert profile.memory.connection == "memory-route"
    assert profile.skill.remote.connection == "skill-route"


def test_portal_migration_rejects_invalid_v1_routes_without_writing(manager):
    legacy = SDKConfigV1(
        connections={"available": HttpConnectionConfig(base_url="https://api.example.com")},
        clients={"memory": {"connection": "missing"}},
    )
    _write_legacy(manager, legacy)

    with pytest.raises(ConfigValidationError, match="connection does not exist"):
        manager.migrate_portal(dry_run=False)

    assert not manager.portal_exists()
    assert not manager.portal_v1_backup_path.exists()


def test_portal_migration_refuses_to_overwrite_different_v2(manager):
    _write_legacy(
        manager,
        SDKConfigV1(base_url="https://legacy.example.com", auth={"api_key": "legacy"}),
    )
    existing = SDKPortalConfigV2.model_validate(
        {"profiles": {"default": {"connections": {"default": {"base_url": "https://existing.example.com"}}}}}
    )
    manager.save_portal(existing)
    before = manager.portal_path.read_bytes()

    plan = manager.migrate_portal(dry_run=True)
    assert plan.target_exists is True
    assert plan.changes_required is True
    with pytest.raises(ConfigValidationError, match="already exists with different content"):
        manager.migrate_portal(dry_run=False)
    assert manager.portal_path.read_bytes() == before


def test_config_migrate_cli_remains_available_for_preview(manager, capsys):
    _write_legacy(
        manager,
        SDKConfigV1(base_url="https://api.example.com", auth={"api_key": "secret"}, defaults={"user_id": "u"}),
    )

    assert cli.main(["config", "migrate", "--config-dir", str(manager.config_dir)]) == 0
    output = capsys.readouterr().out
    assert "migration:    dry-run" in output
    assert "secret" not in output
    assert not manager.portal_exists()


def test_auth_cli_writes_portal_only(manager, monkeypatch, capsys):
    monkeypatch.setenv("MINDMEMOS_CONFIG_DIR", str(manager.config_dir))

    assert cli.main(["auth", "--base-url", "https://api.example.com", "--api-key", "secret", "--user-id", "u"]) == 0

    assert manager.portal_exists()
    assert not manager.legacy_exists()
    assert "Configuration saved" in capsys.readouterr().out


def test_sync_root_client_transparently_migrates_and_prefers_portal_profile(manager):
    _write_legacy(
        manager,
        SDKConfigV1(
            base_url="https://portal.example.com",
            auth={"api_key": "portal-key"},
            defaults={"user_id": "u_1", "app_id": "app-1"},
        ),
    )

    client = MindMemOSClient(config_manager=manager)
    try:
        assert manager.portal_exists()
        assert client._base_url == "https://portal.example.com"
        assert client.require_api_key() == "portal-key"
        assert client._user_id == "u_1"
        assert client._app_id == "app-1"
        assert client.skills._runner._connection is not None
    finally:
        client.close()


def test_legacy_public_config_api_remains_accepted_by_sync_root_client(manager):
    with pytest.warns(DeprecationWarning, match="SDKConfig is deprecated; use SDKPortalConfigV2"):
        config = SDKConfig(
            base_url="https://legacy.example.com",
            auth=AuthConfig(api_key="legacy-key"),
            defaults={"user_id": "legacy-user"},
            storage=StorageConfig(),
            network=NetworkConfig(timeout_seconds=17, max_retries=4),
        )

    client = MindMemOSClient(config=config, config_manager=manager)
    try:
        assert isinstance(config, SDKConfigV1)
        assert client._base_url == "https://legacy.example.com"
        assert client.require_api_key() == "legacy-key"
        assert client._user_id == "legacy-user"
        assert client._transport._timeout == 17
        assert client._transport._max_retries == 4
        assert not manager.portal_exists()
        assert not manager.legacy_exists()
    finally:
        client.close()


@pytest.mark.parametrize(
    ("secret", "expected"),
    [(None, "(not set)"), ("", "(not set)"), ("abc", "***"), ("mk_abcdef", "*****cdef")],
)
def test_mask_secret(secret, expected):
    assert mask_secret(secret) == expected
