"""Make local workspace packages available without installing the monorepo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in ("mindmemos_ui", "mindmemos_sdk", "mindmemos_skill"):
    sys.path.insert(0, str(ROOT / "src" / package))
