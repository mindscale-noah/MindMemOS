"""Sample persisted SpreadsheetBench rollouts with a gated first task."""

import argparse
import asyncio
import json
import random
from pathlib import Path

import yaml
from mindmemos_skill.agents import OpenAIChatAgent
from mindmemos_skill.agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill, SkillSet


def write_json(path: Path, value: object) -> None:
    """Save an audit artifact.

    Args:
        path: Destination file inside the run directory.
        value: JSON-compatible artifact.
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def main() -> None:
    """Run one task first, then the remaining selected tasks concurrently."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", type=Path, default=Path("config/mindmemos_skill/dev.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=("first", "remaining"), required=True)
    args = parser.parse_args()
    if args.count < 1 or args.concurrency < 1:
        parser.error("count and concurrency must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_load(args.service_config.read_text())
    settings = document["service"]["training_pipelines"]["treeskill"]["config"]["trainer"]["config"]["optimizer"]
    connection = OpenAIChatConfig.model_validate(settings)
    connection.require_connection()
    agent = OpenAIChatAgent(connection)
    # The injected agent holds credentials; serialized environment inputs do not.
    recorded = connection.model_dump(mode="json")
    recorded["api_key"] = None
    recorded["base_url"] = None
    env = Environment.build(
        "spreadsheetbench_text_action",
        {
            "agent": recorded,
            "max_turns": 30,
            "transactional_recalculation": True,
        },
        agent=agent,
    )
    data = Data.build(
        "spreadsheetbench",
        {
            "data_path": args.data_root / "dataset.json",
            "data_root": args.data_root,
        },
    )
    samples = data.load()
    split_path = Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json")
    train = data.split(samples, split_file=split_path)["train"]
    selected = random.Random(args.seed).sample(train, args.count)
    skill = Skill.from_path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/initial_skill/SKILL.md")
    skills = SkillSet({"initial": skill}, max_concurrency=1)
    manifest = {
        "input_identity": {item.task.id: env.input_identity(item.task, item.reward_data) for item in selected},
        "task_ids": [item.task.id for item in selected],
        "seed": args.seed,
        "model": connection.model,
        "reasoning_effort": connection.reasoning_effort,
        "max_completion_tokens": connection.max_completion_tokens,
        "max_turns_per_case": 30,
        "rollout_concurrency": args.concurrency,
        "service_config": str(args.service_config.resolve()),
        "data_root": str(args.data_root.resolve()),
        "skill_hash": skill.snapshot().skill_hash,
        "python_runtime": {"python": env.execution.python_command[0]},
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Run inputs changed; choose a new output directory")
    write_json(manifest_path, manifest)
    write_json(root / "samples.json", [item.model_dump(mode="json") for item in selected])
    write_json(root / "split.json", {"train": manifest["task_ids"], "val": [], "test": []})
    skill.export(root / "initial_skill", overwrite=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def execute(sample):
        """Persist one independent rollout and its summary."""
        async with semaphore:
            print(json.dumps({"event": "started", "task_id": sample.task.id}), flush=True)
            result = await env.rollout(
                sample.task, root / "rollouts" / sample.task.id, reward_data=sample.reward_data, skill_set=skills
            )
            if not result.trajectory or not any(item.get("role") == "assistant" for item in result.trajectory):
                raise ValueError("Rollout has no assistant trajectory")
            row = {
                "task_id": sample.task.id,
                "score": result.reward.score,
                "success": result.reward.success,
                "messages": len(result.trajectory),
                "usage": result.usage.model_dump(mode="json"),
            }
            write_json(root / "rollouts" / sample.task.id / "summary.json", row)
            print(json.dumps({"event": "completed", **row}), flush=True)
            return row

    if args.phase == "first":
        await execute(selected[0])
        return
    first = root / "rollouts" / selected[0].task.id / "summary.json"
    if not first.exists():
        raise ValueError("First-task validation has not completed")
    results = await asyncio.gather(*(execute(sample) for sample in selected[1:]), return_exceptions=True)
    failures = [
        {"task_id": sample.task.id, "error": type(result).__name__, "message": str(result)}
        for sample, result in zip(selected[1:], results, strict=True)
        if isinstance(result, Exception)
    ]
    write_json(root / "failures.json", failures)
    rows = [json.loads(first.read_text())] + [row for row in results if not isinstance(row, Exception)]
    write_json(
        root / "summary.json",
        {
            "completed": len(rows),
            "expected": args.count,
            "successful": sum(row["success"] for row in rows),
            "items": rows,
        },
    )
    with (root / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            result_path = root / "rollouts" / row["task_id"] / "audit" / "result.json"
            stream.write(
                json.dumps({"task_id": row["task_id"], **json.loads(result_path.read_text())}, ensure_ascii=False)
                + "\n"
            )
    if failures:
        raise RuntimeError(f"{len(failures)} tasks failed; inspect failures.json")


if __name__ == "__main__":
    asyncio.run(main())
