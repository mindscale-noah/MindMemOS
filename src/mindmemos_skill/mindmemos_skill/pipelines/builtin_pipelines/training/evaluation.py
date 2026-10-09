"""Evaluate a frozen Skill with reusable, independently audited task results."""

from pathlib import Path
from uuid import uuid4

from ....common.asyncio import map_bounded
from ....common.checkpoint import boundary
from ....common.json import write_json
from ....skill import SkillSet
from .typing import HeldOutResultIndex, HeldOutTaskResult


async def evaluate_skill(*, skill, env, samples, directory: Path, concurrency: int) -> dict:
    """Score held-out tasks without feeding results into candidate selection.

    Args:
        skill: Frozen trainer output retaining its task-time implementation.
        env: Borrowed environment with independent workspaces per execution.
        samples: Ordered held-out samples, excluded from trainer optimization.
        directory: Dedicated evaluation directory bound to these inputs.
        concurrency: Maximum simultaneous environment calls.

    Returns:
        Aggregate scores and counts; empty inputs produce zero counts and no score.

    Raises:
        Exception: Execution, scoring, or checkpoint validation fails; no zero is substituted.
    """
    identity = {
        "skill": skill.snapshot().model_dump(mode="json"),
        "environment": f"{type(env).__module__}.{type(env).__qualname__}",
        "config": env.config.model_dump(mode="json"),
        "samples": [sample.model_dump(mode="json") for sample in samples],
    }
    boundary(directory / "inputs.json", identity, payload={"protocol": 1})

    async def evaluate(item):
        index, sample = item
        root = directory / f"task-{index:06d}"
        saved = boundary(root / "completed.json", identity)
        if saved is not None:
            return env.result_type.model_validate(saved["result"])
        result = await env.rollout(
            task=sample.task,
            reward_data=sample.reward_data,
            skill_set=SkillSet({"selected": skill}),
            work_dir=root / "attempts" / uuid4().hex,
            resume=False,
        )
        boundary(root / "completed.json", identity, payload={"result": result.model_dump(mode="json")})
        return result

    results = await map_bounded(evaluate, enumerate(samples), concurrency=concurrency)
    summary = {
        "count": len(results),
        "success_rate": sum(result.reward.success for result in results) / len(results) if results else None,
        "mean_score": sum(result.reward.score for result in results) / len(results) if results else None,
    }
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
    write_json(directory / "results.json", index)
    write_json(directory / "summary.json", summary)
    return summary
