from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments import skill_evaluation as evaluation
from experiments import treeskill as treeskill_experiment
from mindmemos_skill.algos.evolve.skill_grpo_with_replay_buffer.contracts import (
    RolloutAttempt,
    RolloutOutcome,
    RolloutPhase,
    RolloutSpec,
)
from mindmemos_skill.typing import ExecutionInfo, Reward, Rollout, Task, Trajectory, TrajectoryStatus


def _outcome(
    task_id: str,
    sample_index: int,
    score: float,
    *,
    completed: bool = True,
    execution_status: TrajectoryStatus = TrajectoryStatus.SUCCEEDED,
    execution_exception_type: str | None = None,
) -> RolloutOutcome:
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
            execution=ExecutionInfo(
                status=execution_status,
                started_at=now,
                finished_at=now,
                error_info="request failed" if execution_status is TrajectoryStatus.FAILED else None,
            ),
            metadata=(
                {"execution_exception_type": execution_exception_type} if execution_exception_type is not None else {}
            ),
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


def test_summary_treats_returned_failed_trajectory_as_failed_execution() -> None:
    outcome = _outcome(
        "task-1",
        0,
        0.0,
        execution_status=TrajectoryStatus.FAILED,
        execution_exception_type="RuntimeError",
    )

    summary = evaluation.summarize([outcome], rollouts_per_task=1, skill=None)
    record = evaluation.result_record(outcome)

    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["trajectories_returned"] == 1
    assert summary["trajectory_exceptions"] == 0
    assert summary["execution_exceptions"] == 1
    assert record["completed"] is False
    assert record["error_type"] == "TrajectoryExecutionFailed"
    assert record["error"] == "request failed"


def test_summary_does_not_treat_normal_task_failure_as_execution_exception() -> None:
    outcome = _outcome(
        "task-1",
        0,
        0.0,
        execution_status=TrajectoryStatus.FAILED,
    )

    summary = evaluation.summarize([outcome], rollouts_per_task=1, skill=None)

    assert summary["completed"] == 0
    assert summary["failed"] == 1
    assert summary["execution_exceptions"] == 0


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
    (skill_dir / "helper.py").write_text("print('helper')\n", encoding="utf-8")

    skill = evaluation.build_skill(skill_dir, run_id="run", benchmark="alfworld")

    assert skill.content == "# Demo\n"
    assert skill.resources == {}


def test_build_skill_can_include_package_resources_explicitly(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Demo\n", encoding="utf-8")
    (skill_dir / "helper.py").write_text("print('helper')\n", encoding="utf-8")

    skill = evaluation.build_skill(
        skill_dir,
        run_id="run",
        benchmark="spreadsheetbench",
        include_resources=True,
    )

    assert skill.resources == {"helper.py": "print('helper')\n"}


def test_build_agent_can_pin_reference_policy_decoding() -> None:
    agent = evaluation.build_agent(
        client=object(),
        model="fake",
        max_turns=100,
        reasoning_effort=None,
        max_completion_tokens=16384,
        temperature=0.7,
        stop=("Observation:",),
    )

    assert agent.config.temperature == 0.7
    assert agent.config.model_kwargs["stop"] == ["Observation:"]
    assert agent.config.model_kwargs["max_completion_tokens"] == 16384


def test_reference_configuration_requires_exact_local_skill_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir = tmp_path / "xlsx"
    skill_dir.mkdir()
    files = {
        "SKILL.md": "# Spreadsheet\n\n\n",
        "recalc.py": "print('recalc')\n",
        "LICENSE.txt": "local license\n",
    }
    for name, content in files.items():
        (skill_dir / name).write_text(content, encoding="utf-8")
    args = SimpleNamespace(
        analysis_adapter="spreadsheetbench_reference",
        trace2skill_reference_mode=True,
        benchmark="spreadsheetbench",
    )
    skill = evaluation.build_skill(
        skill_dir,
        run_id="run",
        benchmark="spreadsheetbench",
        include_resources=True,
    )
    canonical_package = {"SKILL.md": skill.content, **skill.resources}
    monkeypatch.setattr(
        treeskill_experiment,
        "_REFERENCE_SKILL_SHA256",
        {name: hashlib.sha256(content.encode()).hexdigest() for name, content in canonical_package.items()},
    )

    treeskill_experiment._validate_reference_configuration(args, skill)
    changed = skill.model_copy(update={"resources": {**skill.resources, "extra.txt": "unexpected"}})
    with pytest.raises(ValueError, match="unexpected text resources"):
        treeskill_experiment._validate_reference_configuration(args, changed)


def test_compiled_initial_treeskill_metadata_survives_unchanged_candidate(tmp_path: Path) -> None:
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text("# Workbook\n\nInspect first.\n", encoding="utf-8")
    base = evaluation.build_skill(skill_path, run_id="run", benchmark="spreadsheetbench")
    prepared = treeskill_experiment._with_compiled_treeskill_metadata(base)
    unchanged = treeskill_experiment._candidate_skill(prepared, None, run_id="run")

    metadata = unchanged.metadata["treeskill"]
    tree = treeskill_experiment.parse_tree_with_metadata(unchanged.content, metadata)
    assert tree.node_by_id["001"].heading == "Workbook"
