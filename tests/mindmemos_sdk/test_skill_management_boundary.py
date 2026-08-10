"""Final guards for the SDK-to-Application Skill-management boundary."""

from __future__ import annotations

import ast
from pathlib import Path

SDK_ROOT = Path("src/mindmemos_sdk/mindmemos_sdk")

DIRECT_REPOSITORY_ACCESS_ALLOWLIST: set[Path] = set()


def test_no_new_sdk_module_reaches_through_skill_manager_to_local_repository() -> None:
    violations: list[str] = []
    for path in SDK_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "local_repository":
                if path not in DIRECT_REPOSITORY_ACCESS_ALLOWLIST:
                    violations.append(f"{path}:{node.lineno}")
    assert violations == []


def test_sync_skill_manager_depends_on_application_not_sdk_local_stores() -> None:
    path = SDK_ROOT / "skills/manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "mindmemos_skill" in imported_modules
    assert not imported_modules.intersection(
        {
            "history",
            "local_repository",
            "pending",
            "registry",
            "mindmemos_sdk.skills.history",
            "mindmemos_sdk.skills.local_repository",
            "mindmemos_sdk.skills.pending",
            "mindmemos_sdk.skills.registry",
        }
    )


def test_legacy_sdk_local_skill_stores_are_deleted() -> None:
    for module in ("history.py", "local_repository.py", "pending.py", "registry.py"):
        assert not (SDK_ROOT / "skills" / module).exists()


def test_legacy_config_is_referenced_only_by_the_migration_boundary() -> None:
    allowed = {
        SDK_ROOT / "config" / "__init__.py",
        SDK_ROOT / "config" / "manager.py",
        SDK_ROOT / "config" / "models.py",
        SDK_ROOT / "cli.py",
    }
    violations = []
    for path in SDK_ROOT.rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        if path not in allowed and ("SDKConfigV1" in content or "settings.json" in content):
            violations.append(str(path))
    assert violations == []
