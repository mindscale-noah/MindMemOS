"""Collect the frozen 50-task training split and export a single audited JSON file."""

# ruff: noqa: E402 -- Source checkout imports require path initialization.

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for package in ("mindmemos_eval", "mindmemos_sdk"):
    sys.path.insert(0, str(ROOT / "src" / package))

from mindmemos_eval.swebench.config import load_config
from mindmemos_eval.swebench.constants import (
    CHILD_API_KEY_ENV,
    CHILD_BASE_URL_ENV,
    PARENT_API_KEY_ENV,
    PARENT_BASE_URL_ENV,
)
from mindmemos_eval.swebench.data import create_split, load_split, write_json
from mindmemos_eval.swebench.environment import TaskContainer, docker
from mindmemos_eval.swebench.typing import Trajectory


def export_training(config, phase: str = "train") -> dict:
    """Export raw training records, preserving incomplete and failed attempts.

    Args:
        config: Frozen experiment configuration.
        phase: Train or baseline phase to aggregate.

    Returns:
        Aggregate document with exactly fifty unique task records and file hashes.
    """
    split = load_split(config)
    records = []
    models = Counter()
    usage = Counter()
    hashes = {}
    tasks = split.train if phase == "train" else split.test
    for task in tasks:
        directory = config.output_dir / phase / task.instance_id
        result_path = directory / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {
                "state": "interrupted" if directory.exists() else "missing",
                "finished": False,
            }
        )
        traces = []
        errors = []
        paths = [directory / "parent.json", *sorted(directory.glob("child-*.json"))]
        for path in paths:
            if not path.exists():
                continue
            trace = Trajectory.model_validate_json(path.read_text())
            if trace.metadata.get("instance_id") != task.instance_id:
                errors.append(f"Wrong instance identity: {path.name}")
            if trace.error:
                errors.append(f"{path.name}: {trace.error}")
            pending = set()
            seen = set()
            for message in trace.messages:
                for call in message.get("tool_calls", []):
                    if call["id"] in seen:
                        errors.append(f"Duplicate tool call: {call['id']}")
                    seen.add(call["id"])
                    pending.add(call["id"])
                if message["role"] == "tool":
                    identifier = message["tool_call_id"]
                    if identifier not in pending:
                        errors.append(f"Unmatched tool return: {identifier}")
                    pending.discard(identifier)
            if pending:
                errors.append(f"Missing tool returns: {sorted(pending)}")
            traces.append(trace.model_dump(mode="json"))
        calls_path = directory / "calls.jsonl"
        calls = [json.loads(line) for line in calls_path.read_text().splitlines()] if calls_path.exists() else []
        for call in calls:
            models[str(call.get("model", "unknown"))] += 1
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[name] += call.get(name, 0) or 0
        if result["state"] == "completed" and (not (directory / "parent.json").exists() or not calls):
            errors.append("Completed result lacks parent trajectory or model calls")
        if result["state"] == "completed" and not (directory / "model.patch").exists():
            errors.append("Completed result lacks exported patch")
        for path in sorted(directory.glob("*")):
            if path.is_file():
                hashes[str(path.relative_to(config.output_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "task": task.model_dump(),
                "result": result,
                "trajectories": traces,
                "model_calls": calls,
                "integrity_errors": errors,
                "model_patch": (directory / "model.patch").read_text()
                if (directory / "model.patch").exists()
                else None,
            }
        )
    states = Counter(record["result"]["state"] for record in records)
    document = {
        "schema_version": 1,
        "dataset_sha256": split.dataset_sha256,
        "split_sha256": hashlib.sha256((config.output_dir / "split.json").read_bytes()).hexdigest(),
        "config": config.model_dump(mode="json"),
        "expected": 50,
        "unique_tasks": len({record["task"]["instance_id"] for record in records}),
        "states": dict(states),
        "collection_complete": states["completed"] == 50 and not any(r["integrity_errors"] for r in records),
        "officially_graded": False,
        "response_models": dict(models),
        "usage": dict(usage),
        "source_sha256": hashes,
        "records": records,
    }
    target = config.output_dir / f"{phase}-trajectories.json"
    write_json(target, document)
    target.with_suffix(".json.sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest() + "\n")
    return document


async def _preflight(config, *, check_containers: bool) -> list[str]:
    """Check all required inputs before any paid model request."""
    errors = []
    split = load_split(config)
    for name in (PARENT_API_KEY_ENV, PARENT_BASE_URL_ENV, CHILD_API_KEY_ENV, CHILD_BASE_URL_ENV):
        if not os.environ.get(name):
            errors.append(f"Missing environment variable: {name}")
    for settings in (config.parent, config.child):
        if settings.model.startswith("REPLACE_"):
            errors.append("Configure a real model identifier")
    if not config.image_map_path.exists():
        errors.append(f"Missing image map: {config.image_map_path}")
        return errors
    images = json.loads(config.image_map_path.read_text())
    for task in split.train:
        image = images.get(task.instance_id)
        if not isinstance(image, str) or not image:
            errors.append(f"Missing image mapping: {task.instance_id}")
            continue
        code, _ = await docker("image", "inspect", image)
        if code:
            errors.append(f"Local image unavailable: {task.instance_id}")
        elif check_containers:
            try:
                async with TaskContainer(task, image, config.docker):
                    pass
            except Exception as exc:
                errors.append(f"Container check failed: {task.instance_id}: {type(exc).__name__}")
    return errors


def main() -> int:
    """Prepare, validate, collect, or export without extracting any memories.

    Returns:
        Zero only when the selected operation succeeds; collection requires all fifty records.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "check", "run", "export"), default="check")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "prepare":
        split = create_split(config)
        print(json.dumps({"train": len(split.train), "test": len(split.test), "output": str(config.output_dir)}))
        return 0
    load_split(config)
    with (config.output_dir / ".collection.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode in {"check", "run"}:
            errors = asyncio.run(_preflight(config, check_containers=args.mode == "run"))
            write_json(config.output_dir / "preflight.json", {"ready": not errors, "errors": errors})
            if errors:
                print(json.dumps({"ready": False, "errors": errors}, indent=2))
                return 2
            if args.mode == "check":
                print(json.dumps({"ready": True, "container_execution_checked": False}))
                return 0
        if args.mode == "run":
            from mindmemos_eval.swebench.runner import run_rollouts

            try:
                asyncio.run(run_rollouts(config, "train"))
            finally:
                export_training(config)
        report = export_training(config)
        print(json.dumps({key: report[key] for key in ("expected", "unique_tasks", "states", "collection_complete")}))
        return 0 if report["collection_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
