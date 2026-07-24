"""Tests for the top-level ``mindmemos ui`` command."""

from __future__ import annotations

import pytest
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
