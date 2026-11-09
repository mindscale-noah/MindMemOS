"""Tests for the SDK ConfigManager and auth persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mindmemos_sdk.config import ConfigManager, mask_secret, paths
from mindmemos_sdk.config.constants import CONFIG_DIR_ENV, MINDMEMOS_HOME_ENV
from mindmemos_sdk.config.models import CONFIG_SCHEMA_VERSION, DEFAULT_BASE_URL, DEFAULT_SKILL_BASE_URL, SDKConfig
from mindmemos_sdk.errors import ConfigNotFoundError, ConfigValidationError
from pydantic import ValidationError


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
    assert manager.config_path == expected / "settings.v2.json"


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


def test_saved_home_is_persistent_and_environment_can_override(monkeypatch, tmp_path):
    preference = tmp_path / "preferences" / "home.json"
    monkeypatch.setattr(paths, "_home_preference_path", lambda: preference)
    monkeypatch.delenv(MINDMEMOS_HOME_ENV, raising=False)
    monkeypatch.delenv(CONFIG_DIR_ENV, raising=False)

    first = tmp_path / "first"
    second = tmp_path / "second"
    assert paths.set_default_mindmemos_home(first) == first
    assert ConfigManager().config_dir == first
    assert paths.set_default_mindmemos_home(second) == second
    assert ConfigManager().config_dir == second

    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(first))
    assert ConfigManager().config_dir == first
    assert ConfigManager(config_dir=second).config_dir == second


def test_saved_home_rejects_relative_directory(monkeypatch, tmp_path):
    preference = tmp_path / "preferences" / "home.json"
    monkeypatch.setattr(paths, "_home_preference_path", lambda: preference)

    with pytest.raises(ValueError, match="absolute path"):
        paths.set_default_mindmemos_home("relative/.mindmemos")
    assert not preference.exists()


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
        assert config.memory_service.auth.api_key == f"key-{suffix}"
        assert config.defaults.user_id == f"user-{suffix}"
    first.reset()
    assert second.load().memory_service.auth.api_key == "key-b"


def test_legacy_v2_storage_is_ignored_when_loaded(manager):
    manager.config_path.write_text(json.dumps({"version": 2, "storage": {"skill_cache_dir": "/custom/cache"}}))
    config = manager.load()
    assert config.version == 2
    assert "storage" in json.loads(manager.config_path.read_text())
    assert manager.load().version == 2


def test_load_missing_raises(manager):
    assert manager.exists() is False
    with pytest.raises(ConfigNotFoundError):
        manager.load()


def test_update_auth_persists_and_roundtrips(manager):
    manager.update_auth(
        base_url=" https://my.example.com/ ",
        api_key="mk_secret",
        user_id="u_1",
    )

    assert manager.exists() is True
    loaded = manager.load()
    assert loaded.service_type == "custom"
    assert loaded.memory_service.base_url == "https://my.example.com"
    assert loaded.memory_service.auth.api_key == "mk_secret"
    assert loaded.resolve_skill_connection() == (None, None)
    assert loaded.defaults.user_id == "u_1"
    assert loaded.version == CONFIG_SCHEMA_VERSION
    assert loaded.metadata.created_at is not None
    assert loaded.metadata.updated_at is not None


def test_official_skill_connection_never_reuses_a_custom_memory_credential(manager):
    """Existing settings use the official gateway without copied Skill secrets."""
    manager.update_auth(base_url=DEFAULT_BASE_URL, api_key="memory-key", user_id="user")
    loaded = manager.load()
    assert loaded.service_type == "official"
    assert loaded.memory_service.base_url is None
    assert loaded.skill_service.base_url is None
    assert loaded.skill_service.auth is None
    assert loaded.resolve_memory_connection() == (DEFAULT_BASE_URL, "memory-key")
    assert loaded.resolve_skill_connection() == (DEFAULT_SKILL_BASE_URL, "memory-key")
    assert loaded.resolve_skill_connection(
        memory_base_url="https://override.example", memory_api_key="override-key"
    ) == (None, None)
    assert SDKConfig.model_validate({"skill_service": {"base_url": ""}}).resolve_skill_connection(
        memory_base_url="https://override.example", memory_api_key="override-key"
    ) == (None, None)
    raw = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert raw["service_type"] == "official"
    assert "base_url" not in raw["memory_service"]
    assert "base_url" not in raw["skill_service"]
    assert "base_url" not in raw


def test_self_hosted_skill_connection_has_independent_address_and_key(manager):
    """A separate Skill deployment keeps Memory credentials out of its transport."""
    config = SDKConfig.model_validate(
        {
            "service_type": "custom",
            "memory_service": {"base_url": "https://memory.example", "auth": {"api_key": "memory-key"}},
            "skill_service": {
                "base_url": "https://skill.example/api/",
                "auth": {"api_key": "skill-key"},
            },
        }
    )
    manager.save(config)
    loaded = manager.load()
    assert loaded.memory_service.base_url == "https://memory.example"
    assert loaded.memory_service.auth.api_key == "memory-key"
    assert loaded.resolve_skill_connection() == ("https://skill.example/api", "skill-key")
    assert loaded.resolve_skill_connection(
        memory_base_url="https://override.example", memory_api_key="override-key"
    ) == ("https://skill.example/api", "skill-key")
    assert "skill-key" not in repr(loaded.skill_service)
    updated = manager.update_auth(base_url="https://new-memory.example", api_key="new-memory-key")
    assert updated.resolve_skill_connection() == ("https://skill.example/api", "skill-key")
    assert updated.memory_service.auth.api_key == "new-memory-key"


@pytest.mark.parametrize(
    "skill_service",
    [
        {"base_url": "https://skill.example"},
        {"base_url": "https://skill.example", "auth": {"api_key": ""}},
        {"base_url": "https://user:pass@skill.example", "auth": {"api_key": "key"}},
        {"mode": "official"},
    ],
)
def test_skill_connection_rejects_ambiguous_or_incomplete_configuration(skill_service):
    """Do not silently fall back to the Memory credential for a separate Skill API."""
    with pytest.raises(ValidationError):
        SDKConfig.model_validate(
            {
                "service_type": "custom",
                "memory_service": {"base_url": "https://memory.example"},
                "skill_service": skill_service,
            }
        )


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
    assert raw["version"] == 2
    assert raw["memory_service"]["base_url"] == "https://new.example.com"
    assert "skills" not in raw


def test_update_auth_preserves_created_at(manager):
    first = manager.update_auth(base_url="https://a", api_key="k1", user_id="u_1")
    created_at = first.metadata.created_at

    second = manager.update_auth(base_url="https://b", api_key="k2", user_id="u_2")
    assert second.memory_service.base_url == "https://b"
    assert second.memory_service.auth.api_key == "k2"
    assert "skills" not in second.model_dump()
    assert second.metadata.created_at == created_at


def test_reset_removes_file(manager):
    manager.update_auth(base_url="https://a", api_key="k", user_id="u")
    assert manager.reset() is True
    assert manager.exists() is False
    assert manager.reset() is False


@pytest.mark.parametrize(
    "config",
    [
        {"version": 3},
        {"service_type": "official", "memory_service": {"base_url": "https://custom.example"}},
        {"service_type": "custom"},
        {"service_type": "private"},
        {"base_url": "https://legacy.example"},
    ],
)
def test_v2_rejects_inconsistent_or_legacy_fields(config):
    with pytest.raises(ValidationError):
        SDKConfig.model_validate(config)


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
