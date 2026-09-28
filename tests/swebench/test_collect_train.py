"""Offline regression checks for batch export completeness and frozen selection."""

# ruff: noqa: E402 -- Source checkout imports require path initialization.

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/swebench/collect_train.py"
spec = importlib.util.spec_from_file_location("collect_train", SCRIPT)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

from mindmemos_eval.swebench.config import load_config
from mindmemos_eval.swebench.data import create_split, write_json


@pytest.fixture
def prepared(tmp_path):
    config = load_config(SCRIPT.parents[2] / "config/eval/swebench_verified.example.yaml")
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(
                {
                    "instance_id": f"example__repo-{i}",
                    "repo": "example/repo",
                    "base_commit": "a" * 40,
                    "problem_statement": "Fix the issue",
                    "patch": "GOLD MUST NOT LEAK",
                }
            )
            for i in range(100)
        )
    )
    config = config.model_copy(update={"dataset_path": dataset, "output_dir": tmp_path / "run"})
    split = create_split(config)
    return config, split


def test_missing_is_not_collected_and_gold_is_excluded(prepared):
    config, split = prepared
    report = collector.export_training(config)
    assert report["states"] == {"missing": 50}
    assert report["unique_tasks"] == 50
    assert not report["collection_complete"]
    assert "GOLD MUST NOT LEAK" not in json.dumps(report)
    assert not {t.instance_id for t in split.train} & {t.instance_id for t in split.test}


def test_completed_marker_without_evidence_is_rejected(prepared):
    config, split = prepared
    for task in split.train:
        directory = config.output_dir / "train" / task.instance_id
        directory.mkdir(parents=True)
        write_json(directory / "result.json", {"state": "completed", "finished": True})
    report = collector.export_training(config)
    assert report["states"] == {"completed": 50}
    assert not report["collection_complete"]
    assert all(record["integrity_errors"] for record in report["records"])


def test_interruption_and_failure_remain_visible(prepared):
    config, split = prepared
    first = config.output_dir / "train" / split.train[0].instance_id
    first.mkdir(parents=True)
    second = config.output_dir / "train" / split.train[1].instance_id
    second.mkdir()
    write_json(second / "result.json", {"state": "failed", "error_type": "TimeoutError"})
    report = collector.export_training(config)
    assert report["states"] == {"interrupted": 1, "failed": 1, "missing": 48}
    assert not report["collection_complete"]


def test_complete_export_and_tool_pair_audit(prepared):
    config, split = prepared
    for task in split.train:
        directory = config.output_dir / "train" / task.instance_id
        directory.mkdir(parents=True)
        write_json(directory / "result.json", {"state": "completed", "finished": True})
        write_json(
            directory / "parent.json",
            {
                "trajectory_id": "parent",
                "role": "parent",
                "task": task.problem_statement,
                "metadata": {"instance_id": task.instance_id},
                "finished": True,
                "messages": [{"role": "assistant", "content": "Done"}],
            },
        )
        (directory / "model.patch").write_text("")
        (directory / "calls.jsonl").write_text(json.dumps({"model": "actual-model", "total_tokens": 3}) + "\n")
    report = collector.export_training(config)
    assert report["collection_complete"]
    assert report["response_models"] == {"actual-model": 50}
    assert report["usage"]["total_tokens"] == 150
    path = config.output_dir / "train" / split.train[0].instance_id / "parent.json"
    trace = json.loads(path.read_text())
    trace["messages"] = [{"role": "assistant", "tool_calls": [{"id": "unreturned"}]}]
    write_json(path, trace)
    assert not collector.export_training(config)["collection_complete"]
