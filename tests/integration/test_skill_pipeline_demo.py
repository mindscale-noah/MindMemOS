"""Run the offline Skill demo through actual SDK CLI and HTTP processes."""

import json
import subprocess
import sys
from pathlib import Path


def test_complete_offline_cli_pipeline_matrix(tmp_path):
    """Verify training, evolution, progress cursors, and graph publication."""
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "demo"
    result = subprocess.run(
        [sys.executable, str(root / "scripts/skill/demo/verify.py"), "--output", str(output), "--delay-seconds", "0.1"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["passed"]
    assert summary["llm_calls"] == 0
    assert summary["skill_versions"] == 5
    assert {(run["kind"], run["auto_accept"]) for run in summary["runs"]} == {
        (kind, active) for kind in ("train", "evolve") for active in (False, True)
    }
    assert summary["completed_main_trajectories"] == 7
    assert summary["graph_nodes"] == 2 and summary["graph_edges"] == 1
