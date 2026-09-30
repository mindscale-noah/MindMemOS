import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from mindmemos_skill.envs.builtin_envs.agentskillos import (
    AgentSkillOSRetrievalData,
    AgentSkillOSRetrievalDataConfig,
    aggregate_metrics,
    compute_metrics,
)


def sources(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    relevance = tmp_path / "relevance.json"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "q1",
                "instruction_text": "Find a suitable Skill",
                "core_gt_ids": ["LEAK"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    relevance.write_text(json.dumps({"q1": {"core_gt_ids": ["alpha", "beta"]}}), encoding="utf-8")
    return AgentSkillOSRetrievalDataConfig(tasks_path=tasks, relevance_path=relevance)


def test_metrics_hand_calculated():
    metrics = compute_metrics(["other", "alpha", "beta"], ["alpha", "beta"])
    assert metrics["nDCG@1"] == metrics["Hit@1"] == 0
    expected = (1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3))
    assert metrics["nDCG@3"] == pytest.approx(expected)
    assert metrics["nDCG@10"] == pytest.approx(expected)
    assert metrics["Precision@3"] == pytest.approx(2 / 3)
    assert metrics["MRR@10"] == 0.5
    assert metrics["Recall@50"] == metrics["FullCoverage@3"] == 1
    assert len(metrics) == 12
    assert compute_metrics(["alpha"], ["alpha"])["Precision@3"] == pytest.approx(1 / 3)
    assert all(value == 0 for value in compute_metrics([], ["alpha"]).values())
    assert aggregate_metrics([metrics, metrics]) == pytest.approx(metrics)


@pytest.mark.parametrize("ranking,labels", [(["a", "a"], ["a"]), (["a"], []), (["a"], ["a", "a"]), ([""], ["a"])])
def test_invalid_metrics_fail(ranking, labels):
    with pytest.raises(ValueError):
        compute_metrics(ranking, labels)


def test_cutoffs_and_exact_ids():
    ranking = [f"miss{i}" for i in range(50)] + ["alpha"]
    assert all(value == 0 for value in compute_metrics(ranking, ["alpha"]).values())
    assert compute_metrics(["Alpha", "skill:alpha"], ["alpha"])["Recall@50"] == 0
    ranking[9] = "alpha"
    assert compute_metrics(ranking[:50], ["alpha"])["MRR@10"] == 0.1


@pytest.mark.parametrize("kind", ["duplicate", "missing", "extra", "empty_labels", "empty_tasks"])
def test_invalid_datasets_fail(tmp_path, kind):
    config = sources(tmp_path)
    if kind == "duplicate":
        config.tasks_path.write_text(config.tasks_path.read_text() * 2)
    elif kind == "empty_tasks":
        config.tasks_path.write_text("")
    else:
        contents = {"q1": {"core_gt_ids": ["a"]}}
        if kind == "missing":
            contents = {}
        elif kind == "extra":
            contents["q2"] = {"core_gt_ids": ["a"]}
        else:
            contents["q1"]["core_gt_ids"] = []
        config.relevance_path.write_text(json.dumps(contents))
    with pytest.raises(ValueError):
        AgentSkillOSRetrievalData(config).load()


def test_cli_baseline_and_no_overwrite(tmp_path):
    config = sources(tmp_path)
    rankings = tmp_path / "rankings.json"
    rankings.write_text('{"q1":["alpha","beta"]}')
    output = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[2] / "scripts" / "agentskillos" / "evaluate.py"
    command = [
        sys.executable,
        str(script),
        "--tasks",
        str(config.tasks_path),
        "--relevance",
        str(config.relevance_path),
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
