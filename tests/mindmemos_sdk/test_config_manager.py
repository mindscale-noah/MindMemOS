"""Tests for the SDK ConfigManager and auth persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mindmemos_sdk.config import ConfigManager, mask_secret
from mindmemos_sdk.config.constants import CONFIG_DIR_ENV, MINDMEMOS_HOME_ENV
from mindmemos_sdk.config.models import StorageConfig
from mindmemos_sdk.errors import ConfigNotFoundError, ConfigValidationError


@pytest.fixture
def manager(tmp_path):
    return ConfigManager(config_dir=tmp_path)


@pytest.mark.parametrize("home", [None, "", "~/project-a/.mindmemos"])
def test_home_fallback_and_expansion(monkeypatch, tmp_path, home):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)
    monkeypatch.delenv(MINDMEMOS_HOME_ENV, raising=False)
    if home is not None:
        monkeypatch.setenv(MINDMEMOS_HOME_ENV, home)
    expected = Path(home).expanduser() if home else Path.home() / ".mindmemos"
    manager = ConfigManager()
    assert manager.config_path == expected / "settings.json"
    assert StorageConfig().skill_cache_dir == str(expected / "skills/cache")
    assert manager.load_or_default().storage.skill_backup_dir == str(expected / "skills/backups")


def test_home_precedence(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    home = tmp_path / "home"
    explicit = tmp_path / "explicit"
    monkeypatch.delenv(MINDMEMOS_HOME_ENV, raising=False)
    monkeypatch.setenv(CONFIG_DIR_ENV, str(legacy))
    assert ConfigManager().config_dir == legacy
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(home))
    assert ConfigManager().config_dir == home
    manager = ConfigManager(config_dir=explicit)
    assert manager.config_dir == explicit
    assert manager.load_or_default().storage.skill_cache_dir == str(explicit / "skills/cache")


def test_project_homes_remain_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path / "a"))
    first = ConfigManager()
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(tmp_path / "b"))
    second = ConfigManager()

    # Each manager keeps its home even after the process environment changes.
    first.update_auth(base_url="https://a", api_key="key-a", user_id="user-a")
    second.update_auth(base_url="https://b", api_key="key-b", user_id="user-b")
    for manager, suffix in [(first, "a"), (second, "b")]:
        config = manager.load()
        assert config.auth.api_key == f"key-{suffix}"
        assert config.defaults.user_id == f"user-{suffix}"
        assert config.storage.skill_cache_dir == str(tmp_path / suffix / "skills/cache")
        assert config.storage.skill_backup_dir == str(tmp_path / suffix / "skills/backups")
    first.reset()
    assert second.load().auth.api_key == "key-b"


def test_partial_storage_uses_manager_home_and_preserves_explicit_paths(manager):
    manager.config_path.write_text(json.dumps({"storage": {"skill_cache_dir": "/custom/cache"}}))
    config = manager.load()
    assert config.storage.skill_cache_dir == "/custom/cache"
    assert config.storage.skill_backup_dir == str(manager.config_dir / "skills/backups")
    config.storage.skill_backup_dir = "/custom/backups"
    manager.save(config)
    assert manager.load().storage.skill_backup_dir == "/custom/backups"


def test_load_missing_raises(manager):
    assert manager.exists() is False
    with pytest.raises(ConfigNotFoundError):
        manager.load()


def test_update_auth_persists_and_roundtrips(manager):
    manager.update_auth(
        base_url="https://my.example.com",
        api_key="mk_secret",
        user_id="u_1",
    )

    assert manager.exists() is True
    loaded = manager.load()
    assert loaded.base_url == "https://my.example.com"
    assert loaded.auth.api_key == "mk_secret"
    assert loaded.defaults.user_id == "u_1"
    assert loaded.version == 1
    assert loaded.metadata.created_at is not None
    assert loaded.metadata.updated_at is not None


def test_update_auth_drops_legacy_actor_defaults(manager):
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "base_url": "https://old.example.com",
                "auth": {"api_key": "old"},
                "defaults": {
                    "user_id": "old-user",
                    "app_id": "legacy-app",
                    "agent_id": "legacy-agent",
                    "session_id": "legacy-session",
                },
                "skills": [],
            }
        )
    )

    updated = manager.update_auth(base_url="https://new.example.com", api_key="new", user_id="new-user")
    raw = json.loads(manager.config_path.read_text())

    assert updated.defaults.user_id == "new-user"
    assert raw["defaults"] == {"user_id": "new-user"}


def test_update_auth_preserves_skills_and_created_at(manager):
    first = manager.update_auth(base_url="https://a", api_key="k1", user_id="u_1")
    created_at = first.metadata.created_at

    # Inject a skill record out-of-band to prove update_auth does not clobber it.
    raw = json.loads(manager.config_path.read_text())
    raw["skills"] = [{"id": "skill_1", "name": "demo"}]
    manager.config_path.write_text(json.dumps(raw))

    second = manager.update_auth(base_url="https://b", api_key="k2", user_id="u_2")
    assert second.base_url == "https://b"
    assert second.auth.api_key == "k2"
    assert second.skills == [{"id": "skill_1", "name": "demo"}]
    assert second.metadata.created_at == created_at


def test_reset_removes_file(manager):
    manager.update_auth(base_url="https://a", api_key="k", user_id="u")
    assert manager.reset() is True
    assert manager.exists() is False
    assert manager.reset() is False


def test_load_invalid_json_raises(manager):
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.config_path.write_text("{not json")
    with pytest.raises(ConfigValidationError):
        manager.load()


def test_save_is_atomic_no_leftover_temp(manager):
    manager.update_auth(base_url="https://a", api_key="k", user_id="u")
    leftovers = list(manager.config_dir.glob(".settings-*.tmp"))
    assert leftovers == []


@pytest.mark.parametrize(
    ("secret", "expected"),
    [
        (None, "(not set)"),
        ("", "(not set)"),
        ("abc", "***"),
        ("mk_abcdef", "*****cdef"),
    ],
)
def test_mask_secret(secret, expected):
    assert mask_secret(secret) == expected
