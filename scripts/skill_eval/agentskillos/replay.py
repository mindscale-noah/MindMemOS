"""Compare saved retrieval rankings under the binary-core parity protocol."""

import argparse
import json
from pathlib import Path

from mindmemos_skill.envs.builtin_envs.agentskillos import (
    AgentSkillOSRetrievalData,
    AgentSkillOSRetrievalDataConfig,
    aggregate_metrics,
    compute_metrics,
)
from mindmemos_skill.envs.builtin_envs.agentskillos.typing import Ranking
from pydantic import TypeAdapter


def _load_rankings(path: Path, task_ids: set[str]) -> dict[str, list[str]]:
    """Require exactly one valid ranking for every dataset task."""
    content = path.read_bytes()
    rankings = TypeAdapter(dict[str, list[str]]).validate_json(content)
    if set(rankings) != task_ids:
        raise ValueError(f"Ranking task IDs do not match the dataset: {path}")
    for ranking in rankings.values():
        Ranking(skill_ids=ranking)
    return rankings


def main() -> None:
    """Write a fresh, auditable report without making model or database calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--relevance", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--protocol", choices=("retrieval", "legacy_replay", "lexical"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = AgentSkillOSRetrievalData(
        AgentSkillOSRetrievalDataConfig(
            tasks_path=args.tasks,
            relevance_path=args.relevance,
        )
    ).load()
    task_ids = {sample.task.id for sample in samples}
    rankings = _load_rankings(args.rankings, task_ids)
    rows = {
        sample.task.id: compute_metrics(rankings[sample.task.id], sample.reward_data.core_gt_ids) for sample in samples
    }
    report = {
        "schema_version": 1,
        "evaluator_version": "agentskillos-core-binary-v1",
        "protocol": args.protocol,
        "task_count": len(samples),
        "task_ids": [sample.task.id for sample in samples],
        "metrics": aggregate_metrics(list(rows.values())),
        "per_task": rows,
        "ranking_lengths": {key: len(value) for key, value in rankings.items()},
        "limitations": [
            "Scores evaluate supplied rankings; their generation protocol is declared by the caller.",
            "legacy_replay and lexical results do not establish end-to-end embedding parity.",
        ],
    }
    if args.baseline:
        baseline = _load_rankings(args.baseline, task_ids)
        baseline_rows = {
            sample.task.id: compute_metrics(baseline[sample.task.id], sample.reward_data.core_gt_ids)
            for sample in samples
        }
        baseline_metrics = aggregate_metrics(list(baseline_rows.values()))
        report["baseline_metrics"] = baseline_metrics
        report["baseline_per_task"] = baseline_rows
        report["delta"] = {key: value - baseline_metrics[key] for key, value in report["metrics"].items()}
        report["legacy_top_k_overlap"] = {
            f"@{k}": sum(len(set(rankings[key][:k]) & set(baseline[key][:k])) / k for key in task_ids) / len(task_ids)
            for k in (3, 10, 50)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"task_count": len(samples), "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
