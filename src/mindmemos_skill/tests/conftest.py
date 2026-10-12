"""Make the standalone Skill package importable from the repository checkout."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/mindmemos_skill"))


@pytest.fixture(autouse=True)
def existing_spreadsheet_environment(monkeypatch):
    """Keep protocol tests offline; managed installation has separate process tests."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import execution
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode import environment

    monkeypatch.setattr(execution, "prepare_python_environment", AsyncMock(return_value=Path(sys.executable)))
    monkeypatch.setattr(environment, "execute", AsyncMock(return_value=SimpleNamespace(returncode=0, stderr="")))
