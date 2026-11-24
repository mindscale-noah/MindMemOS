"""Execute selected tasks and record their native rollout outcomes."""

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from mindmemos_skill.common.asyncio import map_bounded
from mindmemos_skill.common.json import append_jsonl, write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import DataSample
from mindmemos_skill.envs.typing import RolloutResult
from mindmemos_skill.skill import SkillSet

from .typing import CaseResult, TaskTrajectory


def case_from_result(task_id: str, result: RolloutResult, path: Path) -> CaseResult:
    """Convert a native rollout to a recorded task result.

    Args:
        task_id: ID of the evaluated task.
        result: Environment outcome, including reward and available usage.
        path: Location of the original rollout.

    Returns:
        Task outcome with unreported usage kept separate from zero.
    """
    usage = {
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "total_tokens": result.usage.total_tokens,
    }
    extras = result.usage.model_extra or {}
    for key in ("cache_read_input_tokens", "cost_usd"):
        value = extras.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            usage[key] = value
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


async def run_tasks(
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
            case = case_from_result(sample.task.id, result, directory / "audit/result.json")
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
