"""Unit coverage for the single-domain experiment orchestration."""

from __future__ import annotations

import json

from mindmemos_eval.memory.envs.statebench.runner import (
    _batch_observations,
    _bootstrap_task_stratified,
    _collect,
    _collect_round,
    _evolve,
    _official_trajectory,
    _repair_checkpoint_zero_after_resume_probe,
    _replay_add,
    _summarize_eval,
    _validate_online_resume_state,
    _write_paired_results,
)


def _summary(values: dict[tuple[str, int], bool]) -> dict:
    rows = [
        {"task_id": task_id, "run_index": run_index, "task_completion_pass": passed}
        for (task_id, run_index), passed in values.items()
    ]
    return {"checkpoints": {"10": {"per_task": rows}}}


def test_task_stratified_bootstrap_is_deterministic():
    rows = [
        {"task_id": "a", "pass_at_1_delta": 1},
        {"task_id": "a", "pass_at_1_delta": 0},
        {"task_id": "b", "pass_at_1_delta": -1},
    ]
    first = _bootstrap_task_stratified(rows, reps=100, seed=7)
    second = _bootstrap_task_stratified(rows, reps=100, seed=7)
    assert first == second
    assert first["estimate"] == 0
    assert first["reps"] == 100


def test_paired_results(tmp_path):
    baseline = _summary({("a", 1): False, ("a", 2): False, ("b", 1): True})
    evolution = _summary({("a", 1): True, ("a", 2): False, ("b", 1): True})
    report = _write_paired_results(
        tmp_path / "paired_results.jsonl",
        baseline,
        evolution,
        domain="customer_support",
        seed=42,
    )
    assert report["rows"] == 3
    assert len((tmp_path / "paired_results.jsonl").read_text().splitlines()) == 3


def test_official_trajectory_gets_task_id_and_replay_uses_payload(monkeypatch, tmp_path):
    path = tmp_path / "official-task.json"
    path.write_text(json.dumps({"conversation": [{"role": "user", "content": "hello"}]}))
    trajectory = _official_trajectory(path, "official-task")
    assert trajectory["task_id"] == "official-task"

    captured = {}

    def fake_post(api_base, api_key, endpoint, payload):
        captured.update({"api_base": api_base, "api_key": api_key, "endpoint": endpoint, "payload": payload})
        return {"data": {"accepted": True}}

    monkeypatch.setattr("mindmemos_eval.memory.envs.statebench.runner._post", fake_post)
    result = _replay_add("http://api", "key", trajectory)
    assert result["message_count"] == 1
    assert result["payload_sha256"]
    assert captured["endpoint"] == "/v1/memory/add"


def test_api_failures_are_recorded_without_raising(monkeypatch):
    def failed_post(*args, **kwargs):
        raise RuntimeError("temporary API outage")

    monkeypatch.setattr("mindmemos_eval.memory.envs.statebench.runner._post", failed_post)
    trajectory = {
        "task_id": "task-a",
        "conversation": [{"role": "user", "content": "hello"}],
    }
    collect = _collect("http://api", "key", trajectory)
    assert collect["status"] == "failed"
    assert collect["signal_count"] == 0
    assert "temporary API outage" in collect["error"]
    evolve = _evolve("http://api", "key")
    assert evolve["status"] == "failed"
    assert evolve["evolved"] is False

    replay = _replay_add("http://api", "key", trajectory)
    assert replay["status"] == "failed"
    assert "temporary API outage" in replay["error"]


def test_eval_summary_records_missing_and_unscored_tasks(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "task-a.json").write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "task_completion_pass": True,
            }
        )
    )
    summary = _summarize_eval(tmp_path, ("task-a", "task-b"), 1)
    assert summary["complete"] is False
    assert summary["scored"] == 1
    assert any(item["task_id"] == "task-b" for item in summary["failures"])


def test_feedback_round_records_missing_trajectory(tmp_path):
    result = _collect_round(tmp_path, ("task-a",), "http://api", "key")
    assert result["signals_this_round"] == 0
    assert result["collected_events"][0]["status"] == "failed"
    assert result["failures"][0]["task_id"] == "task-a"


def test_feedback_round_retries_failed_cached_event(monkeypatch, tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    trajectory = {
        "task_id": "task-a",
        "conversation": [{"role": "user", "content": "hello"}],
    }
    (run_dir / "task-a.json").write_text(json.dumps(trajectory))
    artifact = tmp_path / "feedback.json"
    digest = __import__(
        "mindmemos_eval.memory.envs.statebench.runner",
        fromlist=["_trajectory_digest"],
    )._trajectory_digest(trajectory)
    artifact.write_text(
        json.dumps(
            {
                "collected_events": [
                    {
                        "task_id": "task-a",
                        "status": "failed",
                        "trajectory_sha256": digest,
                        "signal_count": 0,
                        "signals": [],
                    }
                ],
            }
        )
    )

    calls = []

    def successful_collect(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "status": "ok",
            "event_id": "event-a",
            "signal_count": 1,
            "signals": [{"evolvable_path": "add"}],
        }

    monkeypatch.setattr(
        "mindmemos_eval.memory.envs.statebench.runner._collect",
        successful_collect,
    )
    result = _collect_round(
        tmp_path,
        ("task-a",),
        "http://api",
        "key",
        artifact,
        retry_failed=True,
    )
    assert len(calls) == 1
    assert result["collected_event_count"] == 1
    assert result["failures"] == []


def test_batch_observations_summarizes_search_provenance(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    (run_dir / "task-a.json").write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "task_completion_pass": True,
                "mindmemos": {
                    "searches": [
                        {
                            "provenance_version": "statebench-search-provenance-v1",
                            "status": "ok",
                            "source": "auto_prepare",
                            "result_count": 1,
                            "results": [
                                {
                                    "rank": 1,
                                    "memory_id": "memory-1",
                                    "memory": "Ask before acting.",
                                    "memory_sha256": "digest",
                                    "score": None,
                                }
                            ],
                        },
                        {
                            "provenance_version": "statebench-search-provenance-v1",
                            "status": "ok",
                            "source": "agent_tool",
                            "result_count": 0,
                            "results": [],
                        },
                    ]
                },
            }
        )
    )

    report = _batch_observations(tmp_path, ["task-a"], 1, role="eval")

    assert report["search_calls"] == 2
    assert report["search_provenance_calls"] == 2
    assert report["search_provenance_coverage"] == 1.0
    assert report["search_source_counts"] == {"agent_tool": 1, "auto_prepare": 1}
    assert report["retrieved_memory_refs"] == 1
    assert report["retrieved_memory_refs_with_id"] == 1
    assert report["retrieved_memory_refs_with_score"] == 0


def test_resume_validation_reconstructs_evolved_config(monkeypatch, tmp_path):
    reports = tmp_path / "evolution" / "reports"
    reports.mkdir(parents=True)
    (reports / "round_01.json").write_text(json.dumps({"round_index": 1}))
    (reports / "evolution_round01.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "evolved": True,
                "version": 2,
                "changes": [
                    {"path": "search_config.score_threshold", "after": 0.35},
                    {"path": "add_config.entity_types.warranty", "after": 1.05},
                ],
            }
        )
    )
    initial = {
        "add_config": {"entity_types": {"warranty": 1.0}},
        "search_config": {},
    }
    evolved = {
        "add_config": {"entity_types": {"warranty": 1.05}},
        "search_config": {"score_threshold": 0.35},
    }

    def fake_current(*, project_id, checkpoint=0):
        config = initial if project_id == "baseline" else evolved
        return {
            "available": True,
            "version": 1 if project_id == "baseline" else 2,
            "normalized_config": config,
            "config_hash": __import__(
                "mindmemos_eval.memory.envs.statebench.runner",
                fromlist=["_digest"],
            )._digest(config),
        }

    monkeypatch.setattr(
        "mindmemos_eval.memory.envs.statebench.runner._current_config_state",
        fake_current,
    )
    result = _validate_online_resume_state(
        root=tmp_path,
        baseline_project_id="baseline",
        evolution_project_id="evolution",
    )
    assert result is not None
    assert result["completed_evolution_rounds"] == [1]
    assert result["expected_evolution_version"] == 2


def test_resume_probe_repair_restores_checkpoint_zero(tmp_path):
    initial = {"add_config": {"prompt": "initial"}, "search_config": {}}
    digest = __import__(
        "mindmemos_eval.memory.envs.statebench.runner",
        fromlist=["_digest"],
    )._digest(initial)
    config_path = tmp_path / "evolution" / "config_snapshots" / "checkpoint00.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"config_hash": "wrong"}))
    memory_path = tmp_path / "evolution" / "memory_snapshots" / "checkpoint00.json"
    memory_path.parent.mkdir(parents=True)
    memory_path.write_text(json.dumps({"memory_fingerprint": "current"}))
    original_path = tmp_path / "evolution" / "eval_state_before" / "memory_snapshots" / "checkpoint00.json"
    original_path.parent.mkdir(parents=True)
    original_path.write_text(json.dumps({"memory_fingerprint": "empty", "memory_count": 0}))

    _repair_checkpoint_zero_after_resume_probe(
        root=tmp_path,
        evolution_project_id="evolution",
        initial_config=initial,
        initial_hash=digest,
    )

    repaired_config = json.loads(config_path.read_text())
    repaired_memory = json.loads(memory_path.read_text())
    assert repaired_config["config_hash"] == digest
    assert repaired_config["version"] == 1
    assert repaired_memory["memory_fingerprint"] == "empty"
    assert repaired_memory["restored_from_resume_probe"] is True
