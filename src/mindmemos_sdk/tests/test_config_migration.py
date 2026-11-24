"""Tests for resumable v1-to-v2 SDK settings migration."""

from __future__ import annotations

import json

import pytest
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.config import manager as manager_module
from mindmemos_sdk.config.models import DEFAULT_BASE_URL, DEFAULT_SKILL_BASE_URL
from mindmemos_sdk.errors import ConfigValidationError


def _write_v1(manager: ConfigManager, data: dict) -> None:
    """Write one legacy settings file without invoking the new serializer."""
    manager.config_dir.mkdir(parents=True, exist_ok=True)
    manager.legacy_config_path.write_text(json.dumps(data), encoding="utf-8")


def test_v1_custom_migration_preserves_connections(tmp_path):
    manager = ConfigManager(config_dir=tmp_path / "home")
    _write_v1(
        manager,
        {
            "version": 1,
            "base_url": "https://memory.example",
            "auth": {"api_key": "memory-key"},
            "skill_service": {
                "mode": "self_hosted",
                "base_url": "https://skill.example/",
                "auth": {"api_key": "skill-key"},
            },
            "defaults": {"user_id": "user-1"},
            "storage": {"skill_cache_dir": "/custom/cache"},
            "network": {"timeout_seconds": 41, "max_retries": 3},
            "metadata": {"created_at": "2026-01-01T00:00:00Z"},
        },
    )

    legacy_bytes = manager.legacy_config_path.read_bytes()
    config = manager.load()
    assert not manager.config_path.exists()
    manager.save(config)
    assert manager.legacy_config_path.read_bytes() == legacy_bytes
    saved = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert config.version == 2
    assert config.service_type == "custom"
    assert config.resolve_memory_connection() == ("https://memory.example", "memory-key")
    assert config.resolve_skill_connection() == ("https://skill.example/v1/skills", "skill-key")
    assert config.defaults.user_id == "user-1"
    assert config.network.timeout_seconds == 41
    assert config.network.max_retries == 3
    assert config.metadata.created_at == "2026-01-01T00:00:00Z"
    assert set(saved) == {
        "version",
        "service_type",
        "memory_service",
        "skill_service",
        "defaults",
        "network",
        "metadata",
    }
    assert "mode" not in saved["skill_service"]
    assert manager.load().model_dump() == config.model_dump()


@pytest.mark.parametrize("old_url", ["https://mindmemos.cn/", "https://api.mindmemos.example.com"])
def test_v1_official_migration_uses_implicit_service_urls(tmp_path, old_url):
    manager = ConfigManager(config_dir=tmp_path)
    _write_v1(manager, {"version": 1, "base_url": old_url, "auth": {"api_key": "cloud-key"}})

    config = manager.load()
    assert config.service_type == "official"
    assert config.memory_service.base_url is None
    assert config.skill_service.base_url is None
    assert config.resolve_memory_connection() == (DEFAULT_BASE_URL, "cloud-key")
    assert config.resolve_skill_connection() == (DEFAULT_SKILL_BASE_URL, "cloud-key")


def test_v1_mixed_deployment_preserves_official_memory_and_separate_skill(tmp_path):
    manager = ConfigManager(config_dir=tmp_path)
    _write_v1(
        manager,
        {
            "version": 1,
            "base_url": DEFAULT_BASE_URL,
            "auth": {"api_key": "memory-key"},
            "skill_service": {"base_url": "https://skill.example", "auth": {"api_key": "skill-key"}},
        },
    )

    config = manager.load()
    assert config.service_type == "custom"
    assert config.memory_service.base_url == DEFAULT_BASE_URL
    assert config.resolve_skill_connection() == ("https://skill.example/v1/skills", "skill-key")


def test_v1_migration_retries_after_interrupted_settings_write(tmp_path, monkeypatch):
    manager = ConfigManager(config_dir=tmp_path)
    _write_v1(manager, {"version": 1, "base_url": "https://memory.example", "auth": {"api_key": "key"}})
    original_write = manager_module._write_atomic

    def interrupted_write(path, payload, *, prefix):
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(manager_module, "_write_atomic", interrupted_write)
    with pytest.raises(OSError, match="simulated interrupted write"):
        manager.save(manager.load())
    assert not manager.config_path.exists()
    assert json.loads(manager.legacy_config_path.read_text(encoding="utf-8"))["version"] == 1

    monkeypatch.setattr(manager_module, "_write_atomic", original_write)
    manager.save(manager.load())
    assert manager.load().version == 2
    assert manager.reset()
    assert not manager.exists()
    assert manager.load_or_default().resolve_memory_connection() == (DEFAULT_BASE_URL, None)
    assert json.loads(manager.legacy_config_path.read_text(encoding="utf-8"))["auth"]["api_key"] == "key"


def test_unknown_config_version_is_not_rewritten(tmp_path):
    manager = ConfigManager(config_dir=tmp_path)
    _write_v1(manager, {"version": 3, "service_type": "official"})
    before = manager.legacy_config_path.read_bytes()

    with pytest.raises(ConfigValidationError, match="unsupported SDK config version"):
        manager.load()
    assert manager.legacy_config_path.read_bytes() == before
