from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml
from experiments import initial_skill_evaluation as SCRIPT
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (
    RolloutAttempt,
    RolloutOutcome,
    RolloutPhase,
    RolloutSpec,
)
from mindmemos_skill.typing import ExecutionInfo, Reward, Rollout, Task, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_outcome(task_id: str, sample_index: int, reward: float) -> RolloutOutcome:
    task = Task(task_id=task_id, instruction="test")
    spec = RolloutSpec(
        sequence_no=sample_index,
        rollout_id=f"{task_id}-{sample_index}",
        phase=RolloutPhase.TEST,
        task=task,
        skills=[],
        sample_index=sample_index,
        agent_ref="react",
        env_ref="alfworld",
    )
    now = datetime.now(UTC)
    trajectory = Trajectory(
        trajectory_id=f"trajectory-{task_id}-{sample_index}",
        task=task,
        rollout=Rollout(rollout_id=spec.rollout_id),
        reward=Reward(score=reward),
        execution=ExecutionInfo(started_at=now, finished_at=now),
    )
    return RolloutOutcome(
        spec=spec,
        attempts=[RolloutAttempt(attempt_no=0, trajectory=trajectory, started_at=now, finished_at=now)],
        trajectory=trajectory,
        succeeded=True,
    )


def test_summary_matches_skill_grpo_alfworld_metrics() -> None:
    outcomes = [
        make_outcome("task-1", 0, 1.0),
        make_outcome("task-1", 1, 0.0),
        make_outcome("task-1", 2, 0.0),
        make_outcome("task-2", 0, 0.0),
        make_outcome("task-2", 1, 0.0),
        make_outcome("task-2", 2, 0.0),
    ]

    summary = SCRIPT.summarize(outcomes, rollouts_per_task=3)

    assert summary["total"] == 6
    assert summary["correct"] == 1
    assert summary["accuracy"] == 1 / 6
    assert summary["tasks"] == 2
    assert summary["pass_at_k"] == 0.5
    assert summary["mean_per_task"] == 1 / 6
    assert summary["reward_histogram"] == {"0": 5, "1": 1}
    assert summary["task_mean_reward_histogram"] == {"0": 1, "0.333333": 1}


def test_result_record_keeps_skill_grpo_evaluation_shape() -> None:
    record = SCRIPT.result_record(make_outcome("task-1", 0, 1.0))

    assert record["split"] == "test"
    assert record["rollout"]["task"]["id"] == "task-1"
    assert record["rollout"]["reward"] == 1.0
    assert record["rollout"]["trajectory"] == []


def test_bundled_initial_skill_matches_reference_experiment() -> None:
    bundled = REPO_ROOT / "resources" / "mindmemos_skill" / "skills" / "alfworld" / "SKILL.md"
    reference = (REPO_ROOT / "../../../skill-grpo-opensource/skill-grpo/resources/skills/initial/alfworld.md").resolve()

    assert bundled.read_bytes() == reference.read_bytes()


def test_yaml_config_keeps_reference_protocol_defaults() -> None:
    config_path = (
        REPO_ROOT / "config" / "mindmemos_skill" / "initial_skill_evaluation" / "alfworld" / "default.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["parameters"]["evaluation"]["rollouts"] == 3
    assert config["parameters"]["evaluation"]["max_concurrent_rollouts"] == 16
    assert config["parameters"]["environment_options"]["max_turns"] == 50
    assert config["parameters"]["models"]["reasoning_effort"] == "medium"
