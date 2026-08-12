"""Tests for STATE-Bench runner report persistence."""

from __future__ import annotations

from scripts.statebench.run_feedback_evo_loop import _write_round_report


def test_write_round_report_persists_each_round(tmp_path):
    report_dir = tmp_path / "reports"

    _write_round_report(report_dir, {"round_index": 1, "signals": 2})
    _write_round_report(report_dir, {"round_index": 2, "signals": 0})

    assert (report_dir / "round_01.json").exists()
    assert (report_dir / "round_02.json").exists()
    lines = (report_dir / "rounds.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"round_index": 1' in lines[0]
    assert '"round_index": 2' in lines[1]
