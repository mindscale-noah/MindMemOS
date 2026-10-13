"""Evaluate a frozen Skill with reusable, independently audited task results."""

import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from mindmemos_skill.common.ids import generate_id

from ....common.asyncio import map_bounded
from ....common.checkpoint import boundary
from ....common.json import write_json
from ....envs import Environment
from ....envs.data.typing import DataSample
from ....logging import get_logger
from ....skill import Skill, SkillSet
from .typing import HeldOutEvaluation, HeldOutReport, HeldOutResultIndex, HeldOutSummary, HeldOutTaskResult


async def evaluate_skill(
    *,
    skill: Skill,
    env: Environment,
    samples: Sequence[DataSample],
    directory: Path,
    concurrency: int,
    stage_name: str | None = None,
) -> HeldOutEvaluation:
    """Score held-out tasks without feeding results into candidate selection.

    Args:
        skill: Frozen trainer output retaining its task-time implementation.
        env: Borrowed environment with independent workspaces per execution.
        samples: Ordered held-out samples, excluded from trainer optimization.
        directory: Dedicated evaluation directory bound to these inputs.
        concurrency: Maximum simultaneous environment calls.
        stage_name: Optional algorithm stage for static winners; dynamic TreeSkill is inferred.

    Returns:
        Scores, costs, and ordered task records; empty inputs have no score.

    Raises:
        Exception: Execution, scoring, or checkpoint validation fails; no zero is substituted.
    """
    started = perf_counter()
    identity = {
        "skill": skill.snapshot().model_dump(mode="json"),
        "environment": f"{type(env).__module__}.{type(env).__qualname__}",
        "config": env.config.model_dump(mode="json"),
        "samples": [sample.model_dump(mode="json") for sample in samples],
    }
    boundary(directory / "inputs.json", identity, payload={"protocol": 1})

    logger = get_logger(__name__)
    stage_name = stage_name or "skill.evaluation"
    logger.info(f"[{stage_name}] started | tasks={len(samples)} concurrency={concurrency}")

    async def evaluate(item):
        index, sample = item
        root = directory / f"task-{index:06d}"
        saved = boundary(root / "completed.json", identity)
        if saved is not None:
            logger.info(f"[{stage_name}] item.reused | sample_id={sample.task.id}")
            return env.result_type.model_validate(saved["result"])
        result = await env.rollout(
            task=sample.task,
            reward_data=sample.reward_data,
            skill_set=SkillSet({"selected": skill}),
            work_dir=root / "attempts" / generate_id(),
            resume=False,
        )
        boundary(root / "completed.json", identity, payload={"result": result.model_dump(mode="json")})
        logger.info(f"[{stage_name}] item.completed | sample_id={sample.task.id} success={result.reward.success}")
        return result

    results = await map_bounded(evaluate, enumerate(samples), concurrency=concurrency)
    summary = HeldOutSummary(
        count=len(results),
        success_rate=sum(result.reward.success for result in results) / len(results) if results else None,
        mean_score=sum(result.reward.score for result in results) / len(results) if results else None,
    )
    index = HeldOutResultIndex(
        tasks=[
            HeldOutTaskResult(
                task_id=sample.task.id,
                score=result.reward.score,
                success=result.reward.success,
                duration_s=result.duration_s,
                usage=(
                    {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                        "total_tokens": result.usage.total_tokens,
                    }
                    if result.usage.total_tokens or result.usage.input_tokens or result.usage.output_tokens
                    else {}
                ),
                result_path=f"task-{position:06d}/completed.json",
            )
            for position, (sample, result) in enumerate(zip(samples, results, strict=True))
        ]
    )
    (directory / "trajectories.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": sample.task.id,
                    "task": sample.task.model_dump(mode="json"),
                    "result": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )
            + "\n"
            for sample, result in zip(samples, results, strict=True)
        ),
        encoding="utf-8",
    )
    write_json(directory / "results.json", index)
    write_json(directory / "summary.json", summary)
    report = _build_report(summary, index.tasks, directory, started)
    write_json(directory / "report.json", report)
    logger.info(f"[{stage_name}] completed | count={summary.count} accuracy={summary.success_rate}")
    return HeldOutEvaluation(summary=summary, report=report, tasks=index.tasks)


def _build_report(
    summary: HeldOutSummary, tasks: list[HeldOutTaskResult], directory: Path, started: float
) -> HeldOutReport:
    """Summarize costs while preserving unknown timing and token usage."""
    durations = [task.duration_s for task in tasks]
    known_usage = all(task.usage for task in tasks)
    return HeldOutReport(
        **summary.model_dump(),
        wall_seconds=perf_counter() - started,
        task_seconds=sum(durations) if all(value is not None for value in durations) else None,
        input_tokens=sum(task.usage.get("input_tokens", 0) for task in tasks) if known_usage else None,
        output_tokens=sum(task.usage.get("output_tokens", 0) for task in tasks) if known_usage else None,
        total_tokens=sum(task.usage.get("total_tokens", 0) for task in tasks) if known_usage else None,
        trajectory_file=str(directory / "trajectories.jsonl"),
    )
