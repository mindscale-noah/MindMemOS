"""Verify held-out isolation, runtime restoration, and patch failure accounting."""

import argparse
from types import SimpleNamespace

import pytest
import yaml
from mindmemos_skill.common.json import read_json, write_json

from scripts.treeskill.report import build_comparison, collect_patch_report


@pytest.mark.asyncio
async def test_full_comparison_reuses_only_held_out_ids_for_both_scores(tmp_path, monkeypatch):
    """Baseline results never become training samples, and both tests use identical IDs."""
    from mindmemos_skill.agents import OpenAIChatConfig
    from mindmemos_skill.algos.train.treeskill import TreeSkillTrainerConfig

    from scripts.treeskill import compare

    config = TreeSkillTrainerConfig(
        optimizer=OpenAIChatConfig(model="offline"), router=OpenAIChatConfig(model="offline")
    )
    configuration = tmp_path / "service.yaml"
    configuration.write_text(
        yaml.safe_dump(
            {
                "service": {
                    "training_pipelines": {
                        "treeskill": {
                            "config": {
                                "trainer": {"config": config.model_dump(mode="json")},
                                "evaluation_concurrency": 16,
                            }
                        }
                    }
                }
            }
        )
    )
    splits = {
        name: [SimpleNamespace(task=SimpleNamespace(id=f"{name}-{i}")) for i in range(200)]
        for name in ("train", "test")
    }
    data = SimpleNamespace(load=lambda: [], split=lambda *args, **kwargs: splits)
    monkeypatch.setattr(compare.Data, "build", lambda *args: data)
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "SKILL.md").write_text("---\nname: workbook\ndescription: Edit workbooks.\n---\n# Work\nSave output.\n")
    calls = []

    async def initialize(self):
        pass

    async def sample_tasks(**kwargs):
        calls.append((kwargs["directory"].name, [sample.task.id for sample in kwargs["samples"]]))
        assert kwargs["concurrency"] == 16
        return [None] * 200, []

    async def train(self, *, skill, env, train_samples, work_dir):
        calls.append(("training", [sample.task.id for sample in train_samples]))
        write_json(work_dir / "summary.json", {"train": {"complete": True}})
        return skill

    monkeypatch.setattr(compare.SpreadsheetBenchTextActionEnvironment, "initialize", initialize)
    monkeypatch.setattr(compare.TreeSkillTrainer, "train", train)
    monkeypatch.setattr(compare, "sample_tasks", sample_tasks)
    monkeypatch.setattr(compare, "score_report", lambda *args: {"success_rate": 0.5, "complete": True})
    output = tmp_path / "comparison"
    await compare.run(
        argparse.Namespace(
            output=output,
            service_config=configuration,
            data_root=tmp_path,
            split_file=tmp_path / "split.json",
            initial_skill=initial,
            preflight=False,
        )
    )
    assert [name for name, _ in calls] == ["baseline", "training", "evolved"]
    assert calls[0][1] == calls[2][1] == [sample.task.id for sample in splits["test"]]
    assert calls[1][1] == [sample.task.id for sample in splits["train"]]
    assert read_json(output / "summary.json")["status"] == "completed"


def test_patch_report_distinguishes_reject_noop_and_pre_patch_failure(tmp_path):
    for index, (operation, accepted, changed) in enumerate(
        (
            ("update_node", True, True),
            ("update_node", True, False),
            ("reject", True, False),
            ("create_child", False, False),
        )
    ):
        directory = tmp_path / "patches" / str(index)
        write_json(
            directory / "patch.json",
            {
                "result": {
                    "operation": operation,
                    "accepted": accepted,
                    "target_node_id": "1",
                    "message": "reason",
                }
            },
        )
        (directory / "before.md").write_text("Before\n")
        (directory / "after.md").write_text("After\n" if changed else "Before\n")
    write_json(
        tmp_path / "stages/fusion.json",
        {
            "edits": [],
            "failures": [
                {"target_node_id": "2", "error": "Invalid JSON"},
            ],
        },
    )
    report = collect_patch_report(tmp_path)
    assert report["applied"] == 1
    assert report["accepted_no_change"] == 1
    assert report["model_declined"] == 1
    assert report["validation_rejected"] == 1
    assert report["stage_failures"] == [{"stage": "fusion", "target_node_id": "2", "error": "Invalid JSON"}]


def test_comparison_pairs_only_matching_completed_tasks(tmp_path):
    write_json(tmp_path / "manifest.json", {"status": "partial", "protocol": {}, "test_ids": ["a", "b", "c"]})
    write_json(
        tmp_path / "baseline/summary.json",
        {
            "items": [
                {"task_id": "a", "score": 0, "success": False},
                {"task_id": "b", "score": 1, "success": True},
            ]
        },
    )
    write_json(
        tmp_path / "evolved/summary.json",
        {
            "items": [
                {"task_id": "b", "score": 0, "success": False},
                {"task_id": "c", "score": 1, "success": True},
            ]
        },
    )
    report = build_comparison(tmp_path)
    assert report["paired"] == {
        "expected": 3,
        "completed": 1,
        "mean_score_delta": -1,
        "gained_successes": 0,
        "lost_successes": 1,
    }


def test_comparison_reads_early_local_service_split_without_changing_manifest(tmp_path):
    manifest = {"status": "failed", "protocol": {"test_count": 2}}
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "split.json", {"train": ["train"], "val": [], "test": ["a", "b"]})
    report = build_comparison(tmp_path)
    assert report["status"] == "failed"
    assert report["paired"]["expected"] == 2
    assert report["paired"]["completed"] == 0
    assert read_json(tmp_path / "manifest.json") == manifest


def test_analysis_failure_ids_are_reported_without_requiring_patch_records(tmp_path):
    write_json(tmp_path / "stages/analysis.json", {"records": [], "failures": ["trajectory-1"]})
    report = collect_patch_report(tmp_path)
    assert report["stage_failures"] == [{"stage": "analysis", "trajectory_id": "trajectory-1"}]
    assert report["patch_records"] == 0


def test_comparison_reads_service_held_out_results(tmp_path):
    write_json(
        tmp_path / "manifest.json",
        {"status": "completed", "protocol": {"baseline_skipped": True}, "test_ids": ["test"]},
    )
    write_json(tmp_path / "training/test/summary.json", {"count": 1, "mean_score": 0, "success_rate": 0})
    write_json(tmp_path / "training/test/results.json", {"tasks": [{"task_id": "test", "score": 0, "success": False}]})
    result = build_comparison(tmp_path)
    assert result["phases"]["evolved"]["count"] == 1
    assert result["phases"]["evolved"]["items"][0]["score"] == 0
    assert result["phases"]["baseline"] == {"status": "skipped"}
    assert result["paired"]["completed"] == 0


def test_report_counts_trainer_outcomes_and_exposes_missing_usage(tmp_path):
    write_json(
        tmp_path / "items/analysis/item-a/attempt-001/outcome.json",
        {
            "responses": [{"output": "done", "usage": {"total_tokens": 12}}],
            "failed_calls": 1,
        },
    )
    write_json(
        tmp_path / "items/analysis/item-a/attempt-001/workspace/outcome.json",
        {
            "responses": [],
            "failed_calls": 999,
        },
    )
    report = collect_patch_report(tmp_path)
    assert report["model_calls"] == 2
    assert report["failed_model_calls"] == 1
    assert report["model_usage_complete"] is False
