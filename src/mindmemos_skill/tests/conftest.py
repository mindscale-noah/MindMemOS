"""Make the standalone Skill package importable from the repository checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src/mindmemos_skill"))
