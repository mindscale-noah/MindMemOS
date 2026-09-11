"""Authoritative filesystem diffs for scope-local Skill trees."""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any

_IGNORED_NAMES = {"__pycache__", ".DS_Store"}


def _iter_files(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_NAMES for part in relative.parts):
            continue
        if path.name.endswith(".pyc"):
            continue
        files[relative.as_posix()] = path
    return files


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_hash(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for relative, path in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path.read_bytes())))
        digest.update(b"\0")
    return digest.hexdigest()


def _text_lines(data: bytes) -> list[str] | None:
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def directory_diff(original_dir: Path, edited_dir: Path) -> dict[str, Any]:
    """Compare two directories byte-for-byte and return a reviewable diff."""
    original_files = _iter_files(original_dir)
    edited_files = _iter_files(edited_dir)
    changed_files: list[dict[str, Any]] = []
    lines_added = 0
    lines_removed = 0
    binary_changes = 0

    for relative in sorted(original_files.keys() | edited_files.keys()):
        original_path = original_files.get(relative)
        edited_path = edited_files.get(relative)
        original_data = original_path.read_bytes() if original_path else None
        edited_data = edited_path.read_bytes() if edited_path else None
        if original_data == edited_data:
            continue

        original_lines = _text_lines(original_data) if original_data is not None else None
        edited_lines = _text_lines(edited_data) if edited_data is not None else None
        is_binary = (original_data is not None and original_lines is None) or (
            edited_data is not None and edited_lines is None
        )
        if original_data is None:
            status = "added_file"
        elif edited_data is None:
            status = "removed_file"
        else:
            status = "modified"

        added = 0
        removed = 0
        unified = ""
        if is_binary:
            binary_changes += 1
        else:
            before = original_lines or []
            after = edited_lines or []
            unified_lines = list(difflib.unified_diff(before, after, lineterm="", n=2))
            added = sum(line.startswith("+") and not line.startswith("+++") for line in unified_lines)
            removed = sum(line.startswith("-") and not line.startswith("---") for line in unified_lines)
            unified = "\n".join(unified_lines)
        lines_added += added
        lines_removed += removed
        changed_files.append(
            {
                "path": relative,
                "status": status,
                "binary": is_binary,
                "before_sha256": _sha256(original_data) if original_data is not None else None,
                "after_sha256": _sha256(edited_data) if edited_data is not None else None,
                "lines_added": added,
                "lines_removed": removed,
                "unified_diff": unified,
            }
        )

    return {
        "before_tree_hash": _tree_hash(original_files),
        "after_tree_hash": _tree_hash(edited_files),
        "changed_files": changed_files,
        "summary": {
            "n_changed_files": len(changed_files),
            "files_added": sum(item["status"] == "added_file" for item in changed_files),
            "files_removed": sum(item["status"] == "removed_file" for item in changed_files),
            "files_modified": sum(item["status"] == "modified" for item in changed_files),
            "binary_changes": binary_changes,
            "lines_added": lines_added,
            "lines_removed": lines_removed,
        },
    }


def scope_tree_diff(original_root: Path, edited_root: Path) -> dict[str, Any]:
    """Compare complete scope Skill trees and classify every Skill directory."""
    original_skills = (
        {path.name: path for path in original_root.iterdir() if path.is_dir() and not path.name.startswith(".")}
        if original_root.is_dir()
        else {}
    )
    edited_skills = (
        {path.name: path for path in edited_root.iterdir() if path.is_dir() and not path.name.startswith(".")}
        if edited_root.is_dir()
        else {}
    )

    skills: list[dict[str, Any]] = []
    for skill_name in sorted(original_skills.keys() | edited_skills.keys()):
        original = original_skills.get(skill_name)
        edited = edited_skills.get(skill_name)
        if original is None:
            status = "created"
        elif edited is None:
            status = "removed"
        else:
            status = "unchanged"
        diff = directory_diff(
            original or original_root / ".missing-skill",
            edited or edited_root / ".missing-skill",
        )
        if status == "unchanged" and diff["changed_files"]:
            status = "updated"
        skills.append(
            {
                "skill_name": skill_name,
                "status": status,
                **diff,
            }
        )

    return {
        "original_root": str(original_root),
        "edited_root": str(edited_root),
        "skills": skills,
        "summary": {
            "skills_total": len(skills),
            "skills_created": sum(item["status"] == "created" for item in skills),
            "skills_updated": sum(item["status"] == "updated" for item in skills),
            "skills_unchanged": sum(item["status"] == "unchanged" for item in skills),
            "skills_removed": sum(item["status"] == "removed" for item in skills),
            "changed_files": sum(item["summary"]["n_changed_files"] for item in skills),
            "lines_added": sum(item["summary"]["lines_added"] for item in skills),
            "lines_removed": sum(item["summary"]["lines_removed"] for item in skills),
        },
    }


__all__ = ["directory_diff", "scope_tree_diff"]
