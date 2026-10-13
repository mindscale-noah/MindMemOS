"""Execute independent tests or adopt the training pipeline's completed first test."""

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from mindmemos_skill.common.asyncio import map_bounded
from mindmemos_skill.common.checkpoint import fingerprint, read_boundary
from mindmemos_skill.common.json import append_jsonl, read_json, write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import DataSample
from mindmemos_skill.envs.typing import RolloutResult
from mindmemos_skill.pipelines.builtin_pipelines.training.typing import HeldOutResultIndex
from mindmemos_skill.skill import SkillSet, SkillSnapshot

from .typing import CaseResult, TaskTrajectory


def _case(task_id: str, result: RolloutResult, path: Path) -> CaseResult:
    """Retain native scoring fields and distinguish unreported usage from zero."""
    usage = {
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "total_tokens": result.usage.total_tokens,
    }
    return CaseResult(
        task_id=task_id,
        score=result.reward.score,
        success=result.reward.success,
        metrics=result.reward.model_dump(mode="json", exclude={"score", "success", "detail"}),
        started_at=result.started_at,
        finished_at=result.finished_at,
        duration_s=result.duration_s,
        usage=usage if (result.usage.model_extra or {}).get("reported", any(usage.values())) else None,
        result_path=path,
    )


async def evaluate(
    env: Environment,
    samples: Sequence[DataSample],
    skill_set: SkillSet | None,
    output_dir: Path,
    concurrency: int,
) -> list[CaseResult]:
    """Run each task once in a fresh workspace with bounded concurrency.

    Args:
        env: Native environment with its final model and protocol settings.
        samples: Ordered held-out tasks with private scoring inputs.
        skill_set: Empty baseline or fixed Skill version collection.
        output_dir: Fresh repeat directory.
        concurrency: Maximum simultaneous environment calls.

    Returns:
        Ordered scored outcomes or explicit execution/scoring errors for every task.
    """
    output_dir.mkdir(parents=True, exist_ok=False)

    async def execute(item: tuple[int, DataSample]) -> CaseResult:
        position, sample = item
        directory = output_dir / f"task-{position:06d}"
        start = perf_counter()
        started_at = datetime.now(timezone.utc)
        result = None
        try:
            result = await env.rollout(
                task=sample.task,
                reward_data=sample.reward_data,
                skill_set=skill_set,
                work_dir=directory,
                resume=False,
            )
            case = _case(sample.task.id, result, directory / "audit/result.json")
        except Exception as exc:
            path = directory / "error.json"
            case = CaseResult(
                task_id=sample.task.id,
                error=f"{type(exc).__name__}: {exc}",
                duration_s=perf_counter() - start,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                result_path=path,
            )
            write_json(path, case)
        await append_jsonl(
            output_dir / "trajectories.jsonl",
            TaskTrajectory(position=position, task=sample.task, outcome=case, result=result),
        )
        return case

    results = await map_bounded(execute, enumerate(samples), concurrency=concurrency)
    write_json(output_dir / "results.json", results)
    return results


def read_pipeline_test(
    directory: Path,
    expected_task_ids: list[str],
    snapshot: SkillSnapshot,
    env: Environment,
    samples: Sequence[DataSample],
) -> list[CaseResult]:
    """Verify pipeline task order, raw outcomes, and the exact frozen Skill.

    Args:
        directory: Training pipeline test directory.
        expected_task_ids: Requested test IDs in fixed manifest order.
        snapshot: Exact published Skill snapshot used for trained repetitions.
        env: Native environment whose configuration must match the pipeline test.
        samples: Original ordered test inputs used to verify checkpoint identity.

    Returns:
        Outcomes adopted from the first pipeline test without calling any models.

    Raises:
        ValueError: Task order, completion, metadata, or frozen version disagrees.
        OSError: Required pipeline output is missing.
    """
    inputs = read_boundary(directory / "inputs.json")
    if read_json(directory.parent / "selected_skill.json") != snapshot.model_dump(mode="json"):
        raise ValueError("Pipeline test frozen snapshot disagrees with published version")
    identity = {
        "skill": snapshot.model_dump(mode="json"),
        "environment": f"{type(env).__module__}.{type(env).__qualname__}",
        "config": env.config.model_dump(mode="json"),
        "samples": [sample.model_dump(mode="json") for sample in samples],
    }
    if inputs.identity != fingerprint(identity):
        raise ValueError("Pipeline test inputs disagree with requested environment, tasks, or frozen version")
    index = HeldOutResultIndex.model_validate(read_json(directory / "results.json"))
    if [task.task_id for task in index.tasks] != expected_task_ids:
        raise ValueError("Pipeline test index disagrees with requested task order")
    cases = []
    for task in index.tasks:
        path = (directory / task.result_path).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Pipeline result path escapes its test directory")
        completed = read_boundary(path)
        if completed.identity != inputs.identity:
            raise ValueError("Pipeline completion belongs to different evaluation inputs")
        result = RolloutResult.model_validate(completed.payload["result"])
        case = _case(task.task_id, result, path)
        if (case.score, case.success, case.duration_s, case.usage or {}) != (
            task.score,
            task.success,
            task.duration_s,
            task.usage,
        ):
            raise ValueError(f"Pipeline index disagrees with completed task {task.task_id}")
        cases.append(case)
    summary = read_json(directory / "summary.json")
    if summary != {
        "count": len(cases),
        "mean_score": sum(case.score for case in cases) / len(cases),
        "success_rate": sum(case.success for case in cases) / len(cases),
    }:
        raise ValueError("Pipeline summary disagrees with completed outcomes")
    return cases
