"""Shared utilities for pipeline prepare stage."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def _sha256(identity: str) -> str:
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _remove_tree_under_run_dir(path: Path, run_dir: Path) -> None:
    resolved_path = path.resolve()
    resolved_run_dir = run_dir.resolve()
    if not _is_relative_to(resolved_path, resolved_run_dir):
        raise ValueError(f"refuse to remove path outside run workspace: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


__all__ = ["_sha256", "_is_relative_to", "_remove_tree_under_run_dir"]
