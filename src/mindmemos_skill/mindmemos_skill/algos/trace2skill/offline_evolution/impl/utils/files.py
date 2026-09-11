"""Shared file helpers for CLI entrypoints and pipeline modules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for ``path`` when it has one."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_json_file(path: str) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text_file(path: str | Path, content: str) -> None:
    """Write UTF-8 text, creating the parent directory when needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def write_json_file(path: str, payload: Any, *, indent: int | None = 2) -> None:
    """Write JSON with UTF-8, ensure_ascii=False, and a trailing newline."""
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
        f.write("\n")
