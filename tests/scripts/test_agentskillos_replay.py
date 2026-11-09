"""Verify offline legacy ranking comparison without network calls."""

import json
import subprocess
import sys

import pytest


def test_cli_baseline_and_no_overwrite(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    relevance = tmp_path / "relevance.json"
    tasks.write_text(json.dumps({"task_id": "q1", "instruction_text": "Find a suitable Skill"}) + "\n")
    relevance.write_text(json.dumps({"q1": {"core_gt_ids": ["alpha", "beta"]}}))
    rankings = tmp_path / "rankings.json"
    rankings.write_text('{"q1":["alpha","beta"]}')
    output = tmp_path / "report.json"
    command = [
        sys.executable,
        "-m",
        "scripts.skill_eval.agentskillos.replay",
        "--tasks",
        str(tasks),
        "--relevance",
        str(relevance),
        "--rankings",
        str(rankings),
        "--baseline",
        str(rankings),
        "--protocol",
        "legacy_replay",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["task_count"] == 1
    assert all(value == 0 for value in report["delta"].values())
    assert report["legacy_top_k_overlap"]["@3"] == pytest.approx(2 / 3)
    before = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == before
    rankings.write_text('{"wrong":["alpha"]}')
    output.unlink()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not output.exists()
