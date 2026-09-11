"""Unified hard-gate validation for candidate Skill trees."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.diff import scope_tree_diff
from mindmemos_skill.algos.trace2skill.offline_evolution.impl.pipeline.editor.skill_format import skill_format_errors


def _python_check(path: Path) -> tuple[str, str]:
    if path.stem in sys.stdlib_module_names:
        return (
            "failed",
            f"Script filename {path.name!r} shadows the Python standard-library module of the same name.",
        )
    try:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    except (OSError, UnicodeError, SyntaxError) as exc:
        return "failed", f"{type(exc).__name__}: {exc}"
    return "passed", "Python source compiled without executing it."


def _shell_check(path: Path) -> tuple[str, str]:
    bash = shutil.which("bash")
    if bash is None:
        return "not_checked", "bash is unavailable; shell syntax was not checked."
    try:
        result = subprocess.run(
            [bash, "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "not_checked", f"{type(exc).__name__}: {exc}"
    if result.returncode == 0:
        return "passed", "bash -n accepted the script without executing it."
    detail = (result.stderr or result.stdout or "bash -n failed").strip()
    return "failed", detail[:1000]


def validate_skill_tree(baseline_root: str | Path, candidate_root: str | Path) -> dict[str, Any]:
    """Validate each candidate Skill and its changed Python/Bash scripts."""
    baseline = Path(baseline_root)
    candidate = Path(candidate_root)
    diff = scope_tree_diff(baseline, candidate)
    skills: list[dict[str, Any]] = []

    for skill_diff in diff.get("skills", []):
        skill_name = str(skill_diff.get("skill_name") or "")
        skill_dir = candidate / skill_name
        errors = skill_format_errors(skill_dir) if skill_dir.is_dir() else ["evolved Skill directory is missing"]
        scripts: list[dict[str, str]] = []

        for changed in skill_diff.get("changed_files", []) or []:
            relative = str(changed.get("path") or "")
            parts = Path(relative).parts
            if not parts or parts[0] != "scripts" or changed.get("status") == "removed_file":
                continue
            path = skill_dir / relative
            suffix = path.suffix.lower()
            if suffix == ".py":
                check = "python_compile"
                status, message = _python_check(path)
            elif suffix in {".sh", ".bash"}:
                check = "bash_syntax"
                status, message = _shell_check(path)
            else:
                check = "none"
                status = "not_checked"
                message = "No static check is configured for this script type."
            scripts.append(
                {
                    "path": f"skills/{skill_name}/{relative}",
                    "check": check,
                    "status": status,
                    "message": message,
                }
            )
            if status == "failed":
                errors.append(f"{relative} ({check}): {message}")

        skills.append(
            {
                "skill_name": skill_name,
                "errors": errors,
                "scripts": scripts,
            }
        )

    script_records = [script for skill in skills for script in skill["scripts"]]
    return {
        "version": 1,
        "skills": skills,
        "summary": {
            "skills_checked": len(skills),
            "skills_failed": sum(bool(skill["errors"]) for skill in skills),
            "scripts_checked": len(script_records),
            "scripts_passed": sum(script["status"] == "passed" for script in script_records),
            "scripts_failed": sum(script["status"] == "failed" for script in script_records),
            "scripts_not_checked": sum(script["status"] == "not_checked" for script in script_records),
        },
    }


def failed_skill_errors(validation: dict[str, Any]) -> dict[str, list[str]]:
    """Return only Skill names that failed and their concrete errors."""
    return {
        str(skill.get("skill_name") or ""): [str(error) for error in skill.get("errors", [])]
        for skill in validation.get("skills", []) or []
        if skill.get("errors")
    }


__all__ = ["failed_skill_errors", "validate_skill_tree"]
