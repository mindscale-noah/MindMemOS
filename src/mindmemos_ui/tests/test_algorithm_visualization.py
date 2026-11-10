"""Check read-only visualization against partial and untrusted artifacts."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mindmemos_sdk.config import ConfigManager
from mindmemos_ui.algorithm_visualization import read_forest_report
from mindmemos_ui.app import create_app


def save_node(root: Path, node_id: str, **values) -> Path:
    """Write one small checkpoint fixture."""
    directory = root / "trees" / "tree-0" / f"node-{node_id}"
    directory.mkdir(parents=True)
    data = dict(id=node_id, tree_id=0, status="root", scores={"a": 1, "b": 0})
    data.update(values)
    (directory / "node.json").write_text(json.dumps(data))
    return directory


def test_partial_rejection_does_not_display_sentinel_gain(tmp_path):
    save_node(tmp_path, "root")
    save_node(tmp_path, "child", status="rejected", parent_id="root", scores={"a": 0}, gain=0)
    report = read_forest_report(tmp_path)
    child = next(n for n in report.runs[0].nodes if n.checkpoint.id == "child")
    assert (child.completed, child.total, child.gain) == (1, 2, None)
    assert child.score == 0


def test_incremental_results_deduplicate_and_exclude_invalid_records(tmp_path):
    save_node(tmp_path, "root")
    directory = save_node(tmp_path, "child", status="generated", parent_id="root", scores={})
    record = {"task_id": "a", "result": {"reward": {"success": True}}}
    (directory / "evaluation.jsonl").write_text(json.dumps(record) + "\n" + json.dumps(record) + '\n{"partial":')
    child = next(n for n in read_forest_report(tmp_path).runs[0].nodes if n.checkpoint.id == "child")
    assert (child.completed, child.total, child.errors, child.gain) == (1, 2, 1, None)


def test_nested_runs_secrets_and_external_symlink(tmp_path):
    root = tmp_path / "output"
    training = root / "training"
    save_node(training, "root", skill={"name": "name", "content": "text", "config": {"api_key": "SECRET"}})
    (training / "config.json").write_text(
        json.dumps({"trainer": {"api_key": "SECRET"}, "splits": [{"validation_ids": ["a", "b"]}]})
    )
    external = tmp_path / "external.json"
    external.write_text(json.dumps({"id": "external", "tree_id": 0, "status": "root"}))
    bad = training / "trees" / "tree-0" / "node-external"
    bad.mkdir()
    (bad / "node.json").symlink_to(external)
    report = read_forest_report(root)
    assert len(report.runs) == 1
    assert len(report.runs[0].nodes) == 1
    assert "SECRET" not in report.model_dump_json()
    assert report.warnings


def test_final_gain_and_independent_runs(tmp_path):
    for name in ("one", "two"):
        save_node(tmp_path / name, "root")
        save_node(tmp_path / name, "child", status="accepted", parent_id="root", scores={"a": 1, "b": 1}, gain=0.5)
    report = read_forest_report(tmp_path)
    assert len(report.runs) == 2
    assert all(next(n for n in run.nodes if n.checkpoint.id == "child").gain == 0.5 for run in report.runs)


@pytest.mark.asyncio
async def test_route_requires_session_and_reports_missing_directory(tmp_path):
    with TestClient(await create_app(ConfigManager(tmp_path / "sdk"))) as client:
        url = "/api/algorithm-visualization/report"
        assert client.post(url, json={"path": str(tmp_path)}).status_code == 403
        client.headers["x-workbench-token"] = client.get("/api/session").json()["token"]
        assert client.post(url, json={"path": str(tmp_path / "missing")}).status_code == 400
        assert client.post(url, json={"path": str(tmp_path)}).status_code == 200


def test_winners_only_come_from_completed_official_result(tmp_path):
    save_node(tmp_path, "root")
    save_node(tmp_path, "child", status="accepted", parent_id="root", scores={"a": 1, "b": 1}, gain=0.5)
    assert all(node.winner is None for node in read_forest_report(tmp_path).runs[0].nodes)
    result = {
        "status": "running",
        "training": {"winners": [{"id": "child", "tree_id": 0}], "algorithm_report": {"selected_node_id": "child"}},
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    assert all(node.winner is None for node in read_forest_report(tmp_path).runs[0].nodes)
    result["status"] = "completed"
    (tmp_path / "result.json").write_text(json.dumps(result))
    nodes = {node.checkpoint.id: node for node in read_forest_report(tmp_path).runs[0].nodes}
    assert nodes["child"].winner == "final"
    assert nodes["root"].winner is None
    result["training"]["algorithm_report"]["selected_node_id"] = None
    (tmp_path / "result.json").write_text(json.dumps(result))
    assert next(n for n in read_forest_report(tmp_path).runs[0].nodes if n.checkpoint.id == "child").winner == "tree"


def test_missing_or_corrupt_winner_evidence_does_not_guess(tmp_path):
    save_node(tmp_path, "root")
    (tmp_path / "result.json").write_text('{"status":')
    report = read_forest_report(tmp_path)
    assert report.runs[0].nodes[0].winner is None
    assert any("Winner evidence" in warning for warning in report.warnings)
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "training": {
                    "winners": [{"id": "missing", "tree_id": 0}],
                    "algorithm_report": {"selected_node_id": "missing"},
                },
            }
        )
    )
    report = read_forest_report(tmp_path)
    assert report.runs[0].nodes[0].winner is None
    assert any("Winner checkpoint" in warning for warning in report.warnings)


def test_training_stages_follow_persisted_milestones(tmp_path):
    directory = save_node(tmp_path, "root", expanded=False)
    (tmp_path / "config.json").write_text(
        json.dumps({"trainer": {"expansion_budget": 10}, "splits": [{"validation_ids": ["a", "b"]}]})
    )
    progress = read_forest_report(tmp_path).runs[0].progress
    assert (progress.stage, progress.roots_completed, progress.roots_total) == ("root", 1, 1)
    candidate = tmp_path / "trees" / "tree-0" / "node-child" / "generation"
    candidate.mkdir(parents=True)
    assert read_forest_report(tmp_path).runs[0].progress.stage == "search"
    node = json.loads((directory / "node.json").read_text())
    node["expanded"] = True
    (directory / "node.json").write_text(json.dumps(node))
    progress = read_forest_report(tmp_path).runs[0].progress
    assert (progress.expansions_completed, progress.expansion_budget) == (1, 10)
    (tmp_path / "skills").mkdir()
    (tmp_path / "skills" / "skill-tree-0.md").write_text("frozen winner")
    assert read_forest_report(tmp_path).runs[0].progress.stage == "select"
    (tmp_path / "result.json").write_text(
        json.dumps({"status": "completed", "training": {"winners": [], "algorithm_report": {"selected_node_id": None}}})
    )
    assert read_forest_report(tmp_path).runs[0].progress.stage == "completed"


def test_budget_exhaustion_does_not_invent_selection(tmp_path):
    save_node(tmp_path, "root", expanded=True)
    (tmp_path / "config.json").write_text(
        json.dumps({"trainer": {"expansion_budget": 1}, "splits": [{"validation_ids": ["a", "b"]}]})
    )
    assert read_forest_report(tmp_path).runs[0].progress.stage == "search"


def test_root_stage_before_first_tree_directory(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"trainer": {"expansion_budget": 30}, "splits": [{"validation_ids": ["a"]}]})
    )
    run = read_forest_report(tmp_path).runs[0]
    assert run.nodes == []
    assert run.progress.stage == "root"
    assert run.progress.roots_completed == 0
    assert run.progress.roots_total == 1
