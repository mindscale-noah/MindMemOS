"""Repository paths used by the source checkout StateBench runner."""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    configured = os.environ.get("MINDMEMOS_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    start = Path(__file__).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data" / "STATE-Bench").exists() and (candidate / "config").exists():
            return candidate
    return Path.cwd().resolve()


REPO_ROOT = _find_repo_root()
STATE_BENCH_ROOT = REPO_ROOT / "data" / "STATE-Bench"
CONFIG_ROOT = REPO_ROOT / "config" / "mindmemos_eval"
