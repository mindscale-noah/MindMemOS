"""Packaging contracts for the lightweight ``mindmemos-skill`` core."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from mindmemos_skill.errors import SkillCapabilityUnavailableError
from mindmemos_skill.llm import router as router_module

REPOSITORY_ROOT = Path(__file__).parents[2]
SKILL_PYPROJECT = REPOSITORY_ROOT / "src/mindmemos_skill/pyproject.toml"
SDK_PYPROJECT = REPOSITORY_ROOT / "src/mindmemos_sdk/pyproject.toml"


def test_skill_core_and_optional_dependency_metadata_are_separate() -> None:
    skill_project = tomllib.loads(SKILL_PYPROJECT.read_text(encoding="utf-8"))["project"]
    sdk_project = tomllib.loads(SDK_PYPROJECT.read_text(encoding="utf-8"))["project"]

    assert skill_project["dependencies"] == ["pydantic>=2.0"]
    assert skill_project["optional-dependencies"] == {
        "llm": ["litellm>=1.85.0"],
        "pgvector": ["psycopg[binary,pool]>=3.2"],
        "claude-sdk": ["claude-agent-sdk>=0.1.0"],
        "alfworld": ["alfworld>=0.4.0", "pyyaml>=6.0"],
        "dataset-download": ["alfworld>=0.4.0", "huggingface-hub>=0.34.0"],
        "spreadsheetbench": ["openpyxl>=3.1.5"],
    }
    assert "mindmemos-skill>=0.1.0" in sdk_project["dependencies"]


def test_skill_core_imports_without_optional_dependencies() -> None:
    script = """
import importlib.abc
import sys

blocked = {"claude_agent_sdk", "litellm", "openpyxl", "psycopg", "psycopg_pool"}

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in blocked:
            error = ModuleNotFoundError(f"blocked optional dependency: {fullname}")
            error.name = root
            raise error
        return None

sys.meta_path.insert(0, BlockOptionalDependencies())

import mindmemos_skill
from mindmemos_skill.infra.database import DatabaseConfig, create_database
from mindmemos_skill.infra.vector_store import BackendRegistry, register_builtin_vector_stores
from mindmemos_skill.errors import SkillCapabilityUnavailableError
from mindmemos_skill.persistence import SkillRecord

assert mindmemos_skill.SkillAlgorithms
assert DatabaseConfig
assert create_database
assert SkillRecord
assert blocked.isdisjoint(sys.modules)

try:
    register_builtin_vector_stores(BackendRegistry())
except SkillCapabilityUnavailableError as error:
    assert "mindmemos-skill[pgvector]" in str(error)
else:
    raise AssertionError("pgvector backend loaded without its optional dependency")

assert blocked.isdisjoint(sys.modules)
"""
    environment = dict(os.environ)
    source_root = str(REPOSITORY_ROOT / "src/mindmemos_skill")
    environment["PYTHONPATH"] = os.pathsep.join(part for part in (source_root, environment.get("PYTHONPATH")) if part)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_llm_extra_has_an_actionable_missing_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_litellm(name: str):
        error = ModuleNotFoundError(f"No module named {name!r}")
        error.name = name
        raise error

    monkeypatch.setattr(router_module, "import_module", missing_litellm)

    with pytest.raises(SkillCapabilityUnavailableError, match=r"mindmemos-skill\[llm\]"):
        router_module.build_router({"endpoints": []}, "chat")
