"""Evaluate one environment with a fixed Skill collection or no Skills."""

import asyncio
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data, DataSample
from mindmemos_skill.envs.data.typing import SplitManifest
from mindmemos_skill.skill import Skill, SkillContent, SkillSet

from .config import EVAL_CONFIGS, ROOT, EnvTestConfig, read_environment_yaml
from .rollout import run_tasks
from .typing import EnvTestSummary


def evaluate_environment(args: Namespace) -> int:
    """Run a direct evaluation with a fixed set of Skills or no Skills.

    Args:
        args: Environment, task selection, Skill inputs, and execution options.

    Returns:
        Zero on success, or one when a task reports an error.
    """
    payload = read_environment_yaml(
        args.config or EVAL_CONFIGS / "skill_test_direct/dev.yaml", env=args.env, output_dir=args.output_dir
    )
    if args.no_skill:
        payload["skills"] = []
    elif args.skills is not None:
        payload["skills"] = args.skills
    if args.samples is not None:
        payload.update(samples_file=args.samples, data=None)
    overrides = {
        "split_file": args.split_file,
        "split": args.split,
        "limit": args.limit,
        "concurrency": args.concurrency,
    }
    payload.update({name: value for name, value in overrides.items() if value is not None})

    summary = asyncio.run(run_test(EnvTestConfig.model_validate(payload)))
    print(summary.model_dump_json(indent=2), flush=True)
    print(f"Report: {summary.trajectory_file.parent.parent / 'report.md'}", flush=True)
    return int(summary.errors > 0)


def _load_samples(config: EnvTestConfig, env: Environment) -> list[DataSample]:
    """Validate native task and reward types before selecting an ordered partition."""
    if config.data is not None:
        loader = Data.build(config.data.class_name, config.data.config)
        samples = loader.load()
    else:
        rows = read_json(config.samples_file.expanduser())
        if not isinstance(rows, list) or any(
            not isinstance(row, dict) or set(row) != {"task", "reward_data"} for row in rows
        ):
            raise ValueError("samples_file must be an array of objects with task and reward_data fields")
        samples = [
            DataSample(
                task=env.task_type.model_validate(row["task"]),
                reward_data=env.reward_data_type.model_validate(row["reward_data"]),
            )
            for row in rows
        ]
    index = {}
    for sample in samples:
        if not isinstance(sample.task, env.task_type) or not isinstance(sample.reward_data, env.reward_data_type):
            raise ValueError("Dataset task or reward_data types do not match the selected environment")
        if sample.task.id in index:
            raise ValueError(f"Duplicate task ID: {sample.task.id}")
        index[sample.task.id] = sample
    if config.split_file is not None:
        manifest = SplitManifest.model_validate(read_json(config.split_file.expanduser()))
        if config.split not in manifest.root:
            raise ValueError(f"Unknown split: {config.split}")
        unknown = {identifier for ids in manifest.root.values() for identifier in ids} - index.keys()
        if unknown:
            raise ValueError(f"Unknown task IDs in split file: {sorted(unknown)[:5]}")
        samples = [index[identifier] for identifier in manifest.root[config.split]]
    samples = samples[: config.limit]
    if not samples:
        raise ValueError("No evaluation tasks selected")
    return samples


async def run_test(config: EnvTestConfig) -> EnvTestSummary:
    """Execute one independent mode and save the six required report categories.

    Args:
        config: Native environment and data settings, static paths, and run options.

    Returns:
        Summary with task counts, elapsed time, token coverage, success rate, and JSONL path.

    Raises:
        ValueError: Inputs are invalid, empty, or output is not a fresh directory.
        OSError: Source or output files cannot be accessed.
    """
    start = perf_counter()
    directory = config.output_dir or ROOT / "output/mindmemos_eval/test" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    directory = directory.expanduser().resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"Output directory must be empty: {directory}")
    env = Environment.build(config.environment.class_name, config.environment.config)
    samples = _load_samples(config, env)
    skills = {
        f"skill-{position:03d}": Skill.from_path(path.expanduser()) for position, path in enumerate(config.skills)
    }
    if config.skill_body_only:
        skills = {
            name: Skill(
                SkillContent(
                    content=skill.content.split("---", 2)[2].lstrip("\r\n"),
                    resources=skill.resources,
                    name=skill.source.name,
                    description=skill.source.description,
                    standard_format=False,
                )
            )
            for name, skill in skills.items()
        }
    skill_set = SkillSet(skills) if skills else None
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", redact_credentials(config.model_dump(mode="json")))
    write_json(directory / "task_ids.json", [sample.task.id for sample in samples])
    write_json(directory / "skills.json", {name: skill.snapshot() for name, skill in skills.items()})
    cases = await run_tasks(env, samples, skill_set, directory / "tasks", config.concurrency)
    completed = [case for case in cases if case.error is None]
    known = [case.usage for case in cases if case.usage is not None]
    usage = {key: sum(item[key] for item in known) for key in ("input_tokens", "output_tokens", "total_tokens")}
    missing = len(cases) - len(known)
    successful = sum(case.success is True for case in completed)
    summary = EnvTestSummary(
        environment=config.environment.class_name,
        mode="static_skill" if skills else "no_skill",
        planned=len(samples),
        attempted=len(cases),
        completed=len(completed),
        errors=len(cases) - len(completed),
        successful=successful,
        success_rate=successful / len(completed) if completed else None,
        wall_seconds=perf_counter() - start,
        input_tokens=usage["input_tokens"] if not missing else None,
        output_tokens=usage["output_tokens"] if not missing else None,
        total_tokens=usage["total_tokens"] if not missing else None,
        reported_usage=usage,
        usage_missing=missing,
        trajectory_file=directory / "tasks/trajectories.jsonl",
    )
    write_json(directory / "summary.json", summary)
    _write_report(summary, directory)
    return summary


def _write_report(summary: EnvTestSummary, directory: Path) -> None:
    """Write the direct evaluation report with task and token coverage.

    Args:
        summary: Counts, time, token coverage, and trajectory location.
        directory: Result directory containing summary.json.
    """
    missing = summary.usage_missing
    usage = summary.reported_usage
    elapsed = f"{summary.wall_seconds:.3f}" if summary.wall_seconds is not None else "unknown"
    successful = summary.successful
    rate = f"{summary.success_rate:.2%}" if summary.success_rate is not None else "N/A"
    tokens = str(summary.total_tokens) if not missing else f"unknown; reported {usage['total_tokens']}"
    (directory / "report.md").write_text(
        f"Environment: {summary.environment}\n\nMode: {summary.mode}\n\n"
        "| Metric | Value |\n| --- | --- |\n"
        f"| Tasks attempted / planned | {summary.attempted} / {summary.planned} |\n"
        f"| Tasks not started | {summary.not_started} |\n"
        f"| Elapsed seconds | {elapsed} |\n"
        f"| Tokens | {tokens} |\n"
        f"| Tasks with unknown token usage | {missing} |\n"
        f"| Execution errors | {summary.errors} |\n"
        f"| Successful / completed | {successful} / {summary.completed} |\n"
        f"| Success rate (completed tasks) | {rate} |\n\n"
        f"Trajectories: [{summary.trajectory_file.name}]({summary.trajectory_file.relative_to(directory)})\n",
        encoding="utf-8",
    )
