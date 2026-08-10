"""Tests for the top-level ``mindmemos ui`` command."""

from __future__ import annotations

import threading

import pytest
from mindmemos_sdk.config import ConfigManager, SDKPortalConfigV2
from mindmemos_sdk.ui import server

from mindmemos_sdk import cli


def test_ui_is_a_top_level_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(server, "run_ui", lambda **kwargs: calls.append(kwargs))

    result = cli.main(["ui", "--port", "8765", "--no-open", "--config-dir", "/tmp/mindmemos"])

    assert result == 0
    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 8765,
            "open_browser": False,
            "config_dir": "/tmp/mindmemos",
        }
    ]


def test_ui_is_not_a_skill_subcommand() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(["skill", "ui"])

    assert exc_info.value.code == 2


def test_ui_config_reads_and_writes_portal_v2_only(tmp_path) -> None:
    manager = ConfigManager(config_dir=tmp_path / "config")
    manager.update_auth(base_url="https://old.example.com", api_key="secret", user_id="old-user")
    config = manager.load_portal()

    server._apply_config_update(
        config,
        {
            "base_url": "https://new.example.com",
            "user_id": "new-user",
            "timeout_seconds": 45,
            "memory": {"search_top_k": 25},
        },
    )
    manager.save_portal(config)
    payload = server._config_payload(manager)

    assert payload["config_path"] == str(manager.portal_path)
    assert payload["base_url"] == "https://new.example.com"
    assert payload["defaults"]["user_id"] == "new-user"
    assert payload["memory"]["search_top_k"] == 25
    assert payload["network"]["timeout_seconds"] == 45
    assert not manager.legacy_exists()


def test_cli_closes_application_owner_loop_after_skill_command(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("MINDMEMOS_CONFIG_DIR", str(tmp_path / "config"))

    assert cli.main(["skill", "list"]) == 0

    assert "No SDK-registered skills" in capsys.readouterr().out
    assert not any(thread.name == "mindmemos-skill-application" for thread in threading.enumerate())


def test_cli_displays_disabled_skill_route_and_cloud_command_returns_capability_error(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    config_dir = tmp_path / "config"
    manager = ConfigManager(config_dir=config_dir)
    skill_root = tmp_path / "skill"
    manager.save_portal(
        SDKPortalConfigV2.model_validate(
            {
                "version": 2,
                "profiles": {
                    "default": {
                        "connections": {
                            "mindmemos_main": {
                                "base_url": "https://api.test",
                                "api_key": "memory-key",
                            }
                        },
                        "default_connection": "mindmemos_main",
                        "skill": {
                            "remote": {"connection": None},
                            "application": {
                                "local": {
                                    "root_dir": str(skill_root),
                                    "database": {"path": str(skill_root / "state.db")},
                                }
                            },
                        },
                    }
                },
            }
        )
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text('name: demo\nversion: "1.0.0"\n\nBody\n', encoding="utf-8")
    monkeypatch.setenv("MINDMEMOS_CONFIG_DIR", str(config_dir))

    assert cli.main(["config", "show"]) == 0
    assert "skill route: (disabled)" in capsys.readouterr().out
    assert cli.main(["doctor"]) == 0
    assert "skill route: (disabled)" in capsys.readouterr().out
    assert cli.main(["skill", "register", str(source), "--alias", "demo"]) == 0
    capsys.readouterr()

    assert cli.main(["skill", "push", "demo"]) == 1
    output = capsys.readouterr().out
    assert "remote push is not configured" in output
    assert "No api_key configured" not in output
