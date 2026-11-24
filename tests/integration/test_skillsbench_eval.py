"""Run SkillsBench through the existing evaluation CLI and a substituted Harbor executable."""

import json

import pytest
import yaml
from mindmemos_eval.cli import main

from src.mindmemos_skill.tests.skillsbench import conftest as skillsbench_fixtures

dataset = skillsbench_fixtures.dataset
"""Reuse the environment's checked synthetic task source."""

harbor = skillsbench_fixtures.harbor
"""Replace the external CLI without invoking Docker or a model."""


@pytest.mark.parametrize("with_skill", [False, True])
def test_existing_evaluation_cli_runs_skillsbench(tmp_path, dataset, harbor, with_skill):
    """Preserve scoring, public trajectories and complete static Skill injection."""
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "environment": {
                    "class_name": "skillsbench",
                    "config": {"harbor_command": harbor, "agent": "claude-code", "model": "test-model"},
                },
                "data": {"class_name": "skillsbench", "config": {"benchmark_root": str(dataset)}},
            }
        )
    )
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "SKILL.md").write_text("---\nname: candidate\ndescription: Candidate workflow\n---\n\nInspect inputs.\n")
    (source / "input.bin").write_bytes(b"\x00\xff")
    output = tmp_path / "output"
    options = ["--skills", str(source)] if with_skill else ["--no-skill"]
    assert main(["skill", "test", "--config", str(config), "--output-dir", str(output), *options]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert (summary["completed"], summary["errors"], summary["success_rate"]) == (1, 0, 1.0)
    assert summary["total_tokens"] == 14
    trace = (output / "tasks/trajectories.jsonl").read_text()
    assert "PUBLIC OBSERVATION" in trace
    assert "PRIVATE REFERENCE" not in trace and "HIDDEN REASONING" not in trace
    exported = next((output / "tasks").glob("task-*/audit/harbor-*/tasks/demo/environment/skills"))
    if with_skill:
        assert (exported / "candidate/input.bin").read_bytes() == b"\x00\xff"
    else:
        assert not list(exported.iterdir())
