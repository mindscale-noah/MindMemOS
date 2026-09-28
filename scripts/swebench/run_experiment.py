"""Run training collection followed by the no-memory baseline with configured provider defaults."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from collect_train import export_training
from dotenv import dotenv_values
from mindmemos_eval.swebench.config import load_config
from mindmemos_eval.swebench.constants import (
    AGENT_API_KEY_ENV,
    AGENT_BASE_URL_ENV,
    CHILD_API_KEY_ENV,
    CHILD_BASE_URL_ENV,
    PARENT_API_KEY_ENV,
    PARENT_BASE_URL_ENV,
)
from mindmemos_eval.swebench.data import create_split, load_split, write_json
from mindmemos_eval.swebench.runner import run_rollouts
from openai import AsyncOpenAI


def _credentials(path: Path) -> None:
    """Load the explicitly selected environment file without persisting secrets."""
    values = {**dotenv_values(path), **os.environ}
    for destination, fallback in (
        (PARENT_API_KEY_ENV, AGENT_API_KEY_ENV),
        (CHILD_API_KEY_ENV, AGENT_API_KEY_ENV),
        (PARENT_BASE_URL_ENV, AGENT_BASE_URL_ENV),
        (CHILD_BASE_URL_ENV, AGENT_BASE_URL_ENV),
    ):
        value = values.get(destination) or values.get(fallback)
        if not value:
            raise ValueError(f"Missing credential setting: {destination}")
        os.environ[destination] = value


async def _probe(config) -> None:
    """Verify the actual endpoint with the same thinking settings as the rollouts."""
    async with AsyncOpenAI(
        api_key=os.environ[PARENT_API_KEY_ENV],
        base_url=os.environ[PARENT_BASE_URL_ENV],
        timeout=180,
        max_retries=0,
    ) as client:
        response = await client.chat.completions.create(
            model=config.parent.model,
            messages=[{"role": "user", "content": "Reply with OK only."}],
            temperature=0,
            **({"extra_body": {"thinking": {"type": config.parent.thinking}}} if config.parent.thinking else {}),
        )
    message = response.choices[0].message.model_dump(exclude_none=True)
    if not message.get("content"):
        raise RuntimeError("Probe must return a nonempty answer")
    if config.parent.thinking == "disabled" and message.get("reasoning_content"):
        raise RuntimeError("Probe returned reasoning despite disabled thinking")
    write_json(
        config.output_dir / "probe.json",
        {
            "requested_model": config.parent.model,
            "response_model": response.model,
            "thinking": config.parent.thinking,
            "message": message,
            "usage": response.usage.model_dump() if response.usage else None,
            "endpoint_host": urlsplit(os.environ[PARENT_BASE_URL_ENV]).hostname,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"PROBE_OK requested={config.parent.model} returned={response.model} thinking={config.parent.thinking or 'provider_default'}",
        flush=True,
    )


async def _run(config, *, train_only: bool = False) -> None:
    """Run each frozen phase once and persist progress after every task."""
    await _probe(config)
    for phase in ("train",) if train_only else ("train", "baseline"):

        def progress():
            report = export_training(config, phase)
            write_json(
                config.output_dir / "progress.json",
                {
                    "phase": phase,
                    "states": report["states"],
                    "collection_complete": report["collection_complete"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        try:
            await run_rollouts(config, phase, on_result=progress)
        finally:
            progress()
        if not export_training(config, phase)["collection_complete"]:
            raise RuntimeError(f"Incomplete phase: {phase}")
    write_json(
        config.output_dir / ("train-complete.json" if train_only else "rollouts-complete.json"),
        {"train": 50} if train_only else {"train": 50, "baseline": 50},
    )


def main() -> int:
    """Freeze inputs, probe the provider, or execute the authorized serial queue.

    Returns:
        Zero for a successful operation, one for a recorded failure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--harness-python", type=Path)
    parser.add_argument(
        "--train-only", action="store_true", help="Collect training trajectories only; skip baseline and grading"
    )
    parser.add_argument("--mode", choices=("prepare", "probe", "run"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.parent.thinking != config.child.thinking:
        raise ValueError("Both roles must use the same thinking policy")
    if config.parent != config.child and config.parent.model != config.child.model:
        raise ValueError("Both roles must use the same model")
    if args.mode == "prepare":
        split = create_split(config)
        images = {
            task.instance_id: f"swebench/sweb.eval.x86_64.{task.instance_id.lower().replace('__', '_1776_')}:latest"
            for task in split.train + split.test
        }
        write_json(config.image_map_path, images)
        raw = [json.loads(line) for line in config.dataset_path.read_text().splitlines() if line.strip()]
        ids = {task.instance_id for task in split.test}
        write_json(config.output_dir / "grading-dataset.json", [row for row in raw if row["instance_id"] in ids])
        source_paths = list(Path("src/mindmemos_eval/mindmemos_eval/swebench").glob("*.py"))
        source_paths += list(Path("scripts/swebench").glob("*.py"))
        write_json(
            config.output_dir / "source-code-sha256.json",
            {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        )
        print(f"PREPARED {config.output_dir}")
        return 0
    load_split(config)
    _credentials(args.env_file)
    if os.environ[PARENT_BASE_URL_ENV] != os.environ[CHILD_BASE_URL_ENV]:
        raise ValueError("Both roles must use the same endpoint for this experiment")
    with (config.output_dir / ".collection.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            asyncio.run(_probe(config) if args.mode == "probe" else _run(config, train_only=args.train_only))
        except Exception as exc:
            error = str(exc)
            for name in (PARENT_API_KEY_ENV, CHILD_API_KEY_ENV, PARENT_BASE_URL_ENV, CHILD_BASE_URL_ENV):
                error = error.replace(os.environ[name], "[REDACTED]")
            write_json(
                config.output_dir / "failure.json",
                {
                    "error_type": type(exc).__name__,
                    "error": error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(f"STOPPED {type(exc).__name__}; inspect saved task records", flush=True)
            return 1
        if args.mode == "run" and args.harness_python and not args.train_only:
            from grade_baseline import grade

            report = grade(args.config, args.harness_python)
            print(json.dumps(report), flush=True)
            return 0 if report["grading_complete"] else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
