"""Summarize final test results already produced by the training pipeline."""

from math import isclose
from pathlib import Path

from mindmemos_skill.common.checkpoint import read_boundary
from mindmemos_skill.common.json import read_json
from mindmemos_skill.pipelines.builtin_pipelines.training.typing import HeldOutResultIndex

from .typing import LocalRunManifest, SkillEvalReport


def build_report(manifest: LocalRunManifest, training_dir: Path) -> SkillEvalReport:
    """Validate pipeline test artifacts and build a shareable run summary.

    Args:
        manifest: Durable project, version, and expected test identities.
        training_dir: Work directory passed to the training request.

    Returns:
        Report based only on completed pipeline test rollouts.

    Raises:
        ValueError: The test index, summary, or published identity is incomplete.
    """
    if manifest.skill_id is None or manifest.base_version_id is None or manifest.published_version_id is None:
        raise ValueError("A published Skill version is required before reporting")
    test_dir = training_dir / "test"
    index = HeldOutResultIndex.model_validate(read_json(test_dir / "results.json"))
    task_ids = [task.task_id for task in index.tasks]
    if task_ids != manifest.test_ids:
        raise ValueError("Pipeline test results do not match the requested task IDs")
    for task in index.tasks:
        result_path = (test_dir / task.result_path).resolve()
        if not result_path.is_relative_to(test_dir.resolve()) or not result_path.is_file():
            raise ValueError(f"Missing completed pipeline result for task {task.task_id!r}")
        completed = read_boundary(result_path).payload["result"]
        reward = completed["reward"]
        if reward["score"] != task.score or reward["success"] != task.success:
            raise ValueError(f"Pipeline test index disagrees with completed task {task.task_id!r}")
        recorded_usage = completed.get("usage") or {}
        expected_usage = {
            name: recorded_usage.get(name, 0) for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        if not any(expected_usage.values()):
            expected_usage = {}
        if task.usage != expected_usage or task.duration_s != completed.get("duration_s"):
            raise ValueError(f"Pipeline test metadata disagrees with completed task {task.task_id!r}")
    summary = read_json(test_dir / "summary.json")
    count = len(index.tasks)
    mean_score = sum(task.score for task in index.tasks) / count if count else None
    success_rate = sum(task.success for task in index.tasks) / count if count else None
    if summary["count"] != count:
        raise ValueError("Pipeline test count disagrees with the task index")
    for name, value in (("mean_score", mean_score), ("success_rate", success_rate)):
        if value is None or not isclose(summary[name], value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Pipeline test {name} disagrees with the task index")
    durations = [task.duration_s for task in index.tasks]
    usage = (
        {
            name: sum(task.usage.get(name, 0) for task in index.tasks)
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        if index.tasks and all(task.usage for task in index.tasks)
        else None
    )
    return SkillEvalReport(
        project_id=manifest.project_id,
        skill_id=manifest.skill_id,
        base_version_id=manifest.base_version_id,
        version_id=manifest.published_version_id,
        algorithm=manifest.algorithm,
        environment=manifest.environment,
        test_count=count,
        mean_score=mean_score,
        success_rate=success_rate,
        test_task_ids=task_ids,
        duration_s=sum(durations) if all(value is not None for value in durations) else None,
        usage=usage,
    )
