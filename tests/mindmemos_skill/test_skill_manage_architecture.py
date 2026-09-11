"""AC-P2-004: the ``algos/skill_manage`` module is backend-neutral."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "mindmemos_skill" / "mindmemos_skill"
MODULE_ROOT = PACKAGE_ROOT / "algos" / "skill_manage"


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_skill_manage_package_has_no_concrete_driver_imports() -> None:
    forbidden = {"fast_graphrag", "psycopg", "psycopg_pool", "sqlite3", "litellm"}
    for path in MODULE_ROOT.glob("*.py"):
        imported = _imports_of(path)
        leaked = {name.split(".")[0] for name in imported} & forbidden
        assert not leaked, f"{path.name} imports a concrete driver: {sorted(leaked)}"


def test_skill_manage_package_has_no_application_or_storage_imports() -> None:
    forbidden_roots = {"mindmemos_skill.application", "mindmemos_skill.persistence", "mindmemos_skill.infra"}
    for path in MODULE_ROOT.glob("*.py"):
        imported = _imports_of(path)
        leaked = {name for name in imported if any(name.startswith(root) for root in forbidden_roots)}
        assert not leaked, f"{path.name} imports application/storage internals: {sorted(leaked)}"


@pytest.mark.parametrize("module", ["application/skill_manage/models.py", "typing/skill_manage.py"])
def test_public_skill_manage_models_have_no_storage_imports(module: str) -> None:
    forbidden_roots = {
        "fast_graphrag",
        "psycopg",
        "psycopg_pool",
        "pgvector",
        "igraph",
        "hnswlib",
        "sqlite3",
        "mindmemos_skill.infra",
        "mindmemos_skill.persistence",
    }
    imports = _imports_of(PACKAGE_ROOT / module)
    leaked = {name for name in imports if any(name == root or name.startswith(root + ".") for root in forbidden_roots)}
    # Relative imports must not bypass the same boundary.
    leaked.update(name for name in imports if name.split(".")[0] in {"infra", "persistence"})
    assert not leaked, f"{module} imports storage internals: {sorted(leaked)}"


def test_workspace_does_not_depend_on_the_reference_graph_engine() -> None:
    for path in (REPOSITORY_ROOT / "pyproject.toml", PACKAGE_ROOT.parent / "pyproject.toml"):
        metadata = path.read_text(encoding="utf-8")
        assert "fast-graphrag" not in metadata
        assert "skill-graph" not in metadata
