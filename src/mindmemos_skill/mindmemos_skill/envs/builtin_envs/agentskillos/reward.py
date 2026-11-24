"""Binary retrieval metrics matching the recorded GoS parity protocol."""

import json
from math import fsum, log2

from ...typing import RolloutResult
from .typing import Ranking, RelevanceRecord


def compute_metrics(ranking: list[str], core_gt_ids: list[str]) -> dict[str, float]:
    """Score an ordered retrieval without changing identities or the denominator.

    Args:
        ranking: Unique retrieved Skill names, in descending relevance order.
        core_gt_ids: Nonempty unique binary ground truth; never intersected with retrieval.

    Returns:
        The twelve metrics used by the September legacy parity report.
    """
    ranking = Ranking(skill_ids=ranking).skill_ids
    relevant = set(RelevanceRecord(core_gt_ids=core_gt_ids).core_gt_ids)
    metrics = {}
    for k in (1, 3, 10):
        dcg = fsum(1 / log2(i + 2) for i, name in enumerate(ranking[:k]) if name in relevant)
        ideal = fsum(1 / log2(i + 2) for i in range(min(k, len(relevant))))
        metrics[f"nDCG@{k}"] = dcg / ideal
    first = next((i for i, name in enumerate(ranking[:10], 1) if name in relevant), None)
    metrics.update(
        {
            "Hit@1": float(bool(ranking and ranking[0] in relevant)),
            "Precision@3": len(set(ranking[:3]) & relevant) / 3,
            "MRR@10": 1 / first if first else 0.0,
        }
    )
    for k in (10, 20, 50):
        metrics[f"Recall@{k}"] = len(set(ranking[:k]) & relevant) / len(relevant)
    for k in (3, 5, 10):
        metrics[f"FullCoverage@{k}"] = float(relevant <= set(ranking[:k]))
    return metrics


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Compute the unweighted macro average across evaluated tasks.

    Args:
        rows: Nonempty per-task metric dictionaries with matching keys.

    Returns:
        One mean for each metric.
    """
    if not rows or any(row.keys() != rows[0].keys() for row in rows):
        raise ValueError("Metric rows must be nonempty with identical keys")
    return {key: fsum(row[key] for row in rows) / len(rows) for key in rows[0]}


def summarize_results(results: dict[str, RolloutResult], errors: dict[str, str]) -> dict:
    """Summarize scored tasks without disguising infrastructure errors as zero reward.

    Args:
        results: Task IDs mapped to completed end-to-end rollout results.
        errors: Unscored task IDs mapped to execution or evaluator error descriptions.

    Returns:
        Task success rate among scored tasks, coverage, and separate retrieval means.
    """
    if results.keys() & errors.keys():
        raise ValueError("A task cannot be both scored and an execution error")
    details = [json.loads(result.reward.detail) for result in results.values()]
    for result, detail in zip(results.values(), details, strict=True):
        if result.reward.score != float(detail["task_evaluation"]["passed"]):
            raise ValueError("Only official end-to-end task rewards can be aggregated")
    successes = sum(result.reward.success for result in results.values())
    scored = len(results)
    attempted = scored + len(errors)
    retrieval = [detail["retrieval"]["metrics"] for detail in details if detail["retrieval"]["metrics"] is not None]
    return {
        "status": "complete" if attempted and not errors else "partial",
        "attempted_tasks": attempted,
        "scored_tasks": scored,
        "successful_tasks": successes,
        "task_success_rate": successes / scored if scored else None,
        "scoring_coverage": scored / attempted if attempted else 0.0,
        "scored_task_ids": list(results),
        "errors": errors,
        "retrieval_detail": aggregate_metrics(retrieval) if retrieval else None,
        "retrieval_scored_tasks": len(retrieval),
    }
