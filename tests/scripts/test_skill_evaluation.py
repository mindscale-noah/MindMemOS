from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest
from experiments import skill_evaluation as evaluation
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (
    RolloutAttempt,
    RolloutOutcome,
    RolloutPhase,
    RolloutSpec,
)
from mindmemos_skill.typing import ExecutionInfo, Reward, Rollout, Skill, Task, Trajectory


def _outcome(task_id: str, sample_index: int, score: float, *, completed: bool = True) -> RolloutOutcome:
    now = datetime.now(UTC)
    task = Task(task_id=task_id, instruction=task_id)
    spec = RolloutSpec(
        sequence_no=sample_index,
        rollout_id=f"{task_id}-{sample_index}",
        phase=RolloutPhase.TEST,
        task=task,
        skills=[],
        sample_index=sample_index,
        agent_ref="react",
        env_ref="fake",
    )
    trajectory = (
        Trajectory(
            trajectory_id=f"trajectory-{task_id}-{sample_index}",
            task=task,
            rollout=Rollout(rollout_id=spec.rollout_id),
            reward=Reward(score=score),
            execution=ExecutionInfo(started_at=now, finished_at=now),
        )
        if completed
        else None
    )
    return RolloutOutcome(
        spec=spec,
        attempts=[
            RolloutAttempt(
                attempt_no=0,
                trajectory=trajectory,
                error_type=None if completed else "RuntimeError",
                error=None if completed else "failed",
                started_at=now,
                finished_at=now,
            )
        ],
        trajectory=trajectory,
        succeeded=completed,
    )


def test_no_skill_summary_counts_requested_rollouts_and_failures() -> None:
    summary = evaluation.summarize(
        [_outcome("task-1", 0, 1.0), _outcome("task-1", 1, 0.0, completed=False)],
        rollouts_per_task=2,
        skill=None,
    )

    assert summary["mode"] == "no_skill"
    assert summary["skill_content_hash"] is None
    assert summary["total"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["pass_at_k"] == 1.0


@pytest.mark.asyncio
async def test_progress_reports_completed_correct_errors_and_mean_reward() -> None:
    stream = StringIO()
    progress = evaluation.EvaluationProgress(2, stream=stream, width=10)

    progress.start()
    await progress.on_outcome(_outcome("task-1", 0, 1.0))
    await progress.on_outcome(_outcome("task-2", 0, 0.0, completed=False))
    progress.close()

    assert stream.getvalue().splitlines()[-1] == ("test [##########] 2/2 correct=1 errors=1 mean_reward=0.5000")


def test_parser_requires_exactly_one_skill_mode(tmp_path: Path) -> None:
    common = [
        "--benchmark",
        "alfworld",
        "--data-root",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "output"),
        "--run-id",
        "run",
        "--max-turns",
        "50",
    ]

    assert evaluation.parse_args([*common, "--no-skill"]).no_skill is True
    skill_args = evaluation.parse_args([*common, "--skill", str(tmp_path / "SKILL.md")])
    assert skill_args.skill == Path(tmp_path / "SKILL.md")


def test_parser_only_accepts_max_turns(tmp_path: Path) -> None:
    common = [
        "--benchmark",
        "alfworld",
        "--data-root",
        str(tmp_path / "data"),
        "--output-dir",
        str(tmp_path / "output"),
        "--run-id",
        "run",
        "--no-skill",
    ]

    assert evaluation.parse_args([*common, "--max-turns", "40"]).max_turns == 40
    with pytest.raises(SystemExit):
        evaluation.parse_args([*common, "--max-turns", "40", "--max-steps", "50"])


def test_build_skill_accepts_skill_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo\n", encoding="utf-8")

    skill = evaluation.build_skill(skill_dir, run_id="run", benchmark="alfworld")

    assert skill.content == "# Demo\n"


def test_build_skill_loads_virtual_runtime_and_disables_eager_selection(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "# Workbook editing\n\nChoose focused workbook guidance by task.\n",
        encoding="utf-8",
    )
    (skill_dir / "runtime_metadata.json").write_text(
        '{"components":[{"component_id":"lookup","name":"Lookup",'
        '"description":"Fill lookup results","content":"Use an indexed lookup."}],"max_initial_components":3}',
        encoding="utf-8",
    )

    skill = evaluation.build_skill(
        skill_dir,
        run_id="run",
        benchmark="spreadsheetbench",
        max_initial_components=0,
    )

    assert skill.name == "Workbook editing"
    assert skill.description == "Choose focused workbook guidance by task."
    assert skill.runtime_type == "virtual_components"
    assert skill.runtime_metadata["max_initial_components"] == 0


def test_skill_loading_summary_counts_static_and_virtual_tool_calls() -> None:
    now = datetime.now(UTC)
    skill = Skill(
        skill_id="skill",
        version_id="version:one",
        version_label="0.1.0",
        content_hash="sha256:skill",
        name="workbook",
        blob={"SKILL.md": "# Workbook\n"},
        runtime_type="virtual_components",
        runtime_metadata={
            "max_initial_components": 0,
            "components": [
                {
                    "component_id": "lookup",
                    "name": "Lookup",
                    "description": "Fill lookup results",
                    "content": "Use an indexed lookup.",
                }
            ],
        },
        created_at=now,
    )
    resource_id = "skill-resource:version:one:lookup"
    trajectory = Trajectory(
        trajectory_id="trajectory",
        task=Task(task_id="task", instruction="fill lookup"),
        rollout=Rollout(rollout_id="rollout"),
        reward=Reward(score=1),
        execution=ExecutionInfo(started_at=now, finished_at=now),
        events=[
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "skill", "arguments": '{"name":"workbook"}'}},
                    {
                        "function": {
                            "name": "load_skill_resource",
                            "arguments": f'{{"resource_id":"{resource_id}"}}',
                        }
                    },
                ],
            }
        ],
        metadata={
            "skill_runtime": {
                "skills": [{"version_id": "version:one", "loaded_resource_ids": [resource_id]}]
            }
        },
    )

    record = evaluation.skill_loading_record(trajectory, skill=skill)
    summary = evaluation.summarize_skill_loading([record, evaluation.skill_loading_record(None, skill=skill)])

    assert record["requested_virtual_skill_ids"] == ["lookup"]
    assert record["loaded_virtual_skill_ids"] == ["lookup"]
    assert summary["total_load_calls"] == 2
    assert summary["rollouts_with_load"] == 1
    assert summary["rollouts_without_load"] == 1
    assert summary["virtual_skill_load_calls_by_id"] == {"lookup": 1}
