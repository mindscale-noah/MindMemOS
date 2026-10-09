"""Verify held-out isolation, runtime restoration, and patch failure accounting."""

import subprocess
import sys

import pytest
from mindmemos_skill.algos.train.treeskill.config import TreeSkillConfig
from mindmemos_skill.algos.train.treeskill.skill import TreeSkill
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs import Reward, RolloutResult, Task
from mindmemos_skill.envs.data import DataSample, RewardData
from mindmemos_skill.skill import Skill, SkillContent

from scripts.treeskill.report import build_comparison, collect_patch_report
from scripts.treeskill.run import _check_recalculation, _evaluate, _train


def _skill():
    return Skill(SkillContent(content="---\nname: example\ndescription: Example\n---\n# Save\nSave output.\n"))


def _samples():
    return [
        DataSample(task=Task(id=str(index), instruction="Save output"), reward_data=RewardData()) for index in range(3)
    ]


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


class _Environment:
    def __init__(self, failed_id):
        self.failed_id = failed_id
        self.ids = []

    async def rollout(self, task, work_dir, **kwargs):
        self.ids.append(task.id)
        if task.id == self.failed_id:
            raise RuntimeError("Model unavailable")
        return RolloutResult(
            output="saved",
            trajectory=[{"role": "assistant", "content": "Done"}],
            reward=Reward(score=0 if task.id == "1" else 1, success=task.id != "1", detail="checked"),
        )


@pytest.mark.asyncio
async def test_evaluation_preserves_zero_rewards_and_excludes_execution_errors(tmp_path):
    report = await _evaluate(
        env=_Environment("2"),
        skill=_skill(),
        samples=_samples(),
        root=tmp_path,
        phase="baseline",
        concurrency=2,
    )
    assert report["expected"] == 3
    assert report["completed"] == 2
    assert report["failed"] == 1
    assert report["mean_score"] == 0.5
    assert report["success_rate"] == 0.5
    assert not report["complete"]


@pytest.mark.asyncio
async def test_first_rollout_error_prevents_batch_calls(tmp_path):
    env = _Environment("0")
    with pytest.raises(RuntimeError, match="batch was not started"):
        await _evaluate(env=env, skill=_skill(), samples=_samples(), root=tmp_path, phase="baseline", concurrency=32)
    assert env.ids == ["0"]


@pytest.mark.asyncio
async def test_resume_does_not_repeat_recorded_execution_errors(tmp_path):
    write_json(tmp_path / "baseline/0/error.json", {"task_id": "0", "error_type": "RuntimeError"})
    env = _Environment(None)
    with pytest.raises(RuntimeError, match="batch was not started"):
        await _evaluate(env=env, skill=_skill(), samples=_samples(), root=tmp_path, phase="baseline", concurrency=32)
    assert env.ids == []


def test_recalculation_timeout_terminates_only_its_probe(tmp_path, monkeypatch):
    launch = subprocess.Popen
    processes = []

    def sleeper(command, **options):
        process = launch([sys.executable, "-c", "import time; time.sleep(30)"], **options)
        processes.append(process)
        return process

    monkeypatch.setattr("scripts.treeskill.run.subprocess.Popen", sleeper)
    with pytest.raises(RuntimeError, match="preflight exceeded"):
        _check_recalculation(tmp_path, timeout=0.1)
    assert len(processes) == 1
    assert processes[0].poll() is not None


def test_analysis_failure_ids_are_reported_without_requiring_patch_records(tmp_path):
    write_json(tmp_path / "stages/analysis.json", {"records": [], "failures": ["trajectory-1"]})
    report = collect_patch_report(tmp_path)
    assert report["stage_failures"] == [{"stage": "analysis", "trajectory_id": "trajectory-1"}]
    assert report["patch_records"] == 0


@pytest.mark.asyncio
async def test_service_training_uses_only_train_and_restores_dynamic_skill(tmp_path):
    snapshot = TreeSkill(TreeSkillConfig(router={"model": "offline"}), source=_skill().source).snapshot()
    write_json(tmp_path / "manifest.json", {"status": "training", "protocol": {}, "test_ids": ["test-only"]})

    class Client:
        request = None

        async def register(self, request):
            return {"skill_id": "family", "skill_version_id": "base"}

        async def train(self, request):
            self.request = request
            return {"task_id": "job"}

        async def task_progress(self, request):
            return [{"sequence": 1, "stage": "fusion", "message": "Finished"}]

        async def show_task(self, request):
            return {"status": "succeeded", "result": {"version_id": "evolved"}}

        async def show(self, request):
            return {"version": snapshot.model_dump(mode="json")}

    client = Client()
    final = await _train(client, root=tmp_path, skill=_skill(), samples=_samples(), env_config={})
    assert isinstance(final, TreeSkill)
    assert client.request.init_skill_version_id == "base"
    assert client.request.split.train == ["0", "1", "2"]
    assert client.request.split.test == []
    assert client.request.split.val == []
    assert (tmp_path / "evolved_skill/SKILL.md").is_file()
    assert read_json(tmp_path / "evolved_skill.json")["implementation"] == "treeskill"
