"""Reevaluate a frozen Skill on the exact tasks and settings of a previous run."""

import argparse
import asyncio
import json
from pathlib import Path

import yaml
from mindmemos_skill.agents import OpenAIChatAgent
from mindmemos_skill.agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from mindmemos_skill.common.checkpoint import boundary
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import SpreadsheetBenchRewardData, SpreadsheetBenchTask
from mindmemos_skill.envs.data.typing import DataSample
from mindmemos_skill.skill import SkillSet
from mindmemos_skill.skill.typing import SkillSnapshot


def summarize(root: Path, baseline: dict, task_ids: list[str]) -> dict:
    """Persist progress and paired outcomes without treating missing results as failures.

    Args:
        root: Independent output directory of the new evaluation.
        baseline: Completed baseline summary containing one item per task.
        task_ids: Frozen ordered task identities.

    Returns:
        Progress counts and a final comparison only when all tasks have results.
    """
    old = {row["task_id"]: row for row in baseline["items"]}
    rows = [
        read_json(root / "rollouts" / identifier / "summary.json")
        for identifier in task_ids
        if (root / "rollouts" / identifier / "summary.json").exists()
    ]
    current = {row["task_id"]: row for row in rows}
    successful = sum(row["success"] for row in rows)
    complete = len(rows) == len(task_ids)
    summary = {
        "status": "completed" if complete else "running",
        "completed": len(rows),
        "expected": len(task_ids),
        "successful": successful,
        "baseline_successful": sum(old[identity]["success"] for identity in task_ids),
        "baseline_success_rate": sum(old[identity]["success"] for identity in task_ids) / len(task_ids),
        "success_rate": successful / len(task_ids) if complete else None,
        "items": rows,
    }
    if complete:
        summary["delta_percentage_points"] = 100 * (summary["success_rate"] - summary["baseline_success_rate"])
        summary["fixed_tasks"] = [
            identity for identity in task_ids if not old[identity]["success"] and current[identity]["success"]
        ]
        summary["regressed_tasks"] = [
            identity for identity in task_ids if old[identity]["success"] and not current[identity]["success"]
        ]
    write_json(root / "summary.json", summary)
    write_json(
        root / "comparison.json",
        [
            {
                "task_id": identity,
                "baseline_success": old[identity]["success"],
                "evolved_success": current[identity]["success"] if identity in current else None,
                "baseline_score": old[identity]["score"],
                "evolved_score": current[identity]["score"] if identity in current else None,
            }
            for identity in task_ids
        ],
    )
    return summary


async def main() -> None:
    """Validate frozen inputs, probe the model, and run gated evaluation phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--service-config", type=Path, default=Path("config/mindmemos_skill/dev.yaml"))
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--phase", choices=("preflight", "probe", "first", "remaining", "summary"), required=True)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("concurrency must be positive")
    root = args.output.resolve()
    baseline_root = args.baseline.resolve()
    if root == baseline_root:
        raise ValueError("Reevaluation requires an independent output directory")
    original = read_json(baseline_root / "manifest.json")
    baseline = read_json(baseline_root / "summary.json")
    task_ids = original["task_ids"]
    if len(set(task_ids)) != len(task_ids) or set(task_ids) != {row["task_id"] for row in baseline["items"]}:
        raise ValueError("Baseline task identities are incomplete or duplicated")
    samples = [
        DataSample(
            task=SpreadsheetBenchTask.model_validate(item["task"]),
            reward_data=SpreadsheetBenchRewardData.model_validate(item["reward_data"]),
        )
        for item in read_json(baseline_root / "samples.json")
    ]
    if [item.task.id for item in samples] != task_ids:
        raise ValueError("Saved sample order differs from the baseline manifest")
    snapshot = SkillSnapshot.model_validate(read_json(args.candidate))
    skill = load_runtime_skill(snapshot)
    if skill.snapshot().skill_hash != snapshot.skill_hash or snapshot.skill_hash == original["skill_hash"]:
        raise ValueError("Candidate is invalid or unchanged from the baseline")
    recorded = read_json(baseline_root / "rollouts" / task_ids[0] / "audit/request.json")["config"]
    for identity in task_ids:
        config = read_json(baseline_root / "rollouts" / identity / "audit/request.json")["config"]
        if config != recorded:
            raise ValueError("Baseline environment settings differ across tasks")
    root.mkdir(parents=True, exist_ok=True)
    source = yaml.safe_load(args.service_config.read_text())
    credentials = source["service"]["training_pipelines"]["treeskill"]["config"]["trainer"]["config"]["optimizer"]
    connection = OpenAIChatConfig.model_validate(
        {**recorded["agent"], "api_key": credentials["api_key"], "base_url": credentials["base_url"]}
    )
    connection.require_connection()
    agent = OpenAIChatAgent(connection)
    env = Environment.build("spreadsheetbench_text_action", recorded, agent=agent)
    await env.initialize()
    input_identity = {sample.task.id: env.input_identity(sample.task, sample.reward_data) for sample in samples}
    manifest = {
        "input_identity": input_identity,
        "baseline": str(baseline_root),
        "baseline_skill_hash": original["skill_hash"],
        "skill_version_id": args.version_id,
        "candidate_skill_hash": snapshot.skill_hash,
        "task_ids": task_ids,
        "environment_config": recorded,
        "python_runtime": {"python": env.execution.python_command[0]},
        "rollout_concurrency": args.concurrency,
        "repeats": 1,
        "execution_environment": "outside_sandbox",
        "runtime_limit": read_json(baseline_root / "validation.json").get("runtime_limit"),
        "evaluation_scope": "Replay of the same tasks used by evolution; not held-out generalization.",
    }
    boundary(root / "manifest.json", manifest, payload=manifest)
    write_json(root / "samples.json", [sample.model_dump(mode="json") for sample in samples])
    skill.export(root / "evolved_skill", overwrite=True)
    if args.phase == "preflight":
        write_json(
            root / "preflight.json",
            {
                "status": "passed",
                "input_identity": input_identity,
                "tasks": len(samples),
                "skill_hash": snapshot.skill_hash,
                "model": connection.model,
                "concurrency": args.concurrency,
            },
        )
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "tasks": len(samples),
                    "model": connection.model,
                    "concurrency": args.concurrency,
                }
            ),
            flush=True,
        )
        return
    if args.phase == "summary":
        print(json.dumps(summarize(root, baseline, task_ids), ensure_ascii=False), flush=True)
        return
    if args.phase == "probe":
        probe_config = connection.model_copy(update={"timeout_s": 45, "max_retries": 0, "max_completion_tokens": 64})
        result = await OpenAIChatAgent(probe_config).run(messages=[{"role": "user", "content": "Reply with OK only."}])
        if not isinstance(result.output, str) or not result.output.strip():
            raise ValueError("Model probe returned no text")
        write_json(
            root / "probe.json",
            {"status": "passed", "model": connection.model, "usage": result.usage},
        )
        print(json.dumps({"probe": "passed", "model": connection.model}), flush=True)
        return
    skills = SkillSet({"initial": skill}, max_concurrency=1)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def execute(sample):
        async with semaphore:
            directory = root / "rollouts" / sample.task.id
            summary_path = directory / "summary.json"
            if summary_path.exists():
                return
            print(json.dumps({"event": "started", "task_id": sample.task.id}), flush=True)
            result = await env.rollout(sample.task, directory, reward_data=sample.reward_data, skill_set=skills)
            if not any(event.get("role") == "assistant" for event in result.trajectory):
                raise ValueError("Formal evaluation produced no assistant messages")
            row = {
                "task_id": sample.task.id,
                "score": result.reward.score,
                "success": result.reward.success,
                "messages": len(result.trajectory),
                "duration_s": result.duration_s,
                "usage": result.usage.model_dump(mode="json"),
            }
            write_json(summary_path, row)
            summary = summarize(root, baseline, task_ids)
            print(json.dumps({"event": "completed", **row, "completed": summary["completed"]}), flush=True)

    if args.phase == "first":
        await execute(samples[0])
    else:
        if not (root / "rollouts" / task_ids[0] / "summary.json").exists():
            raise ValueError("The gated first task must complete before starting the remaining batch")
        try:
            async with asyncio.TaskGroup() as group:
                for sample in samples[1:]:
                    group.create_task(execute(sample))
        except Exception:
            summary = summarize(root, baseline, task_ids)
            write_json(
                root / "failure.json",
                {
                    "status": "failed",
                    "completed": summary["completed"],
                    "message": "Evaluation interrupted; inspect the run log before recovery.",
                },
            )
            raise


if __name__ == "__main__":
    asyncio.run(main())
