"""Resolve training evaluation options and select the requested algorithm."""

import asyncio
from argparse import Namespace
from pathlib import Path

from mindmemos_skill.common.json import read_json

from ..algos import skill_forest, treeskill
from ..config import (
    ENVIRONMENT_NAMES,
    EVAL_CONFIGS,
    RunOptions,
    TreeSkillEvalConfig,
    load_config,
    read_environment_yaml,
)


def evaluate(args: Namespace) -> int:
    """Select the trainer evaluation for the requested algorithm.

    Args:
        args: Training algorithm and evaluation options.

    Returns:
        The selected evaluator's exit status.
    """
    if args.algo == "skill-forest":
        return _evaluate_forest(args)

    forest_options = _forest_overrides(args)
    if forest_options:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in forest_options)
        raise ValueError(f"{flags} require --algo skill-forest")
    return _evaluate_treeskill(args)


def rebuild_report(args: Namespace) -> int:
    """Rebuild a report from an existing Skill-Forest run.

    Args:
        args: Directory containing the saved run.

    Returns:
        Zero for complete results, or one for incomplete results.
    """
    report = skill_forest.rebuild_report(args.run_dir)
    return _forest_report_status(report.output_dir)


def _evaluate_forest(args: Namespace) -> int:
    """Run the requested Skill-Forest conditions.

    Args:
        args: Dataset or environment selection, configuration, and experiment options.

    Returns:
        Zero for complete results, or one for incomplete results.
    """
    if args.check:
        raise ValueError("--check requires --algo tree-skill")
    overrides = _forest_overrides(args)
    if args.env is not None:
        if args.benchmark is not None:
            raise ValueError("Choose either --env or --benchmark")
        environments = {env: benchmark for benchmark, env in ENVIRONMENT_NAMES.items()}
        if args.env not in environments:
            raise ValueError(
                f"Skill-Forest does not support environment {args.env!r}. Choose from: {', '.join(environments)}"
            )
        overrides["benchmark"] = environments[args.env]

    options = RunOptions.model_validate(overrides)
    config = load_config(args.config or EVAL_CONFIGS / "skill_train_skill_forest/dev.yaml", output_dir=args.output_dir)
    options.output_dir = config.output_dir
    report = asyncio.run(skill_forest.run(config, options))
    return _forest_report_status(report.output_dir)


def _evaluate_treeskill(args: Namespace) -> int:
    """Evaluate TreeSkill training and held-out performance.

    Args:
        args: Config, environment, output directory, and preflight options.

    Returns:
        Zero on success, or one when a task reports an error.
    """
    config = TreeSkillEvalConfig.model_validate(
        read_environment_yaml(
            args.config or EVAL_CONFIGS / "skill_train_treeskill/dev.yaml", env=args.env, output_dir=args.output_dir
        )
    )
    if args.check:
        _, pools, _, _ = treeskill.prepare(config)
        print(f"Prepared {config.environment.class_name}: {len(pools['train'])} train / {len(pools['test'])} test.")
        return 0

    summary = asyncio.run(treeskill.run(config))
    print(summary.model_dump_json(indent=2), flush=True)
    return int(summary.errors > 0)


def _forest_overrides(args: Namespace) -> dict[str, object]:
    """Return supplied Skill-Forest options in their configuration form."""
    options = {
        "benchmark": args.benchmark,
        "smoke": args.smoke,
        "test_repeats": args.test_repeats,
        "arms": args.arms.split(",") if args.arms is not None else None,
        "data_root": args.data_root,
    }
    return {name: value for name, value in options.items() if value is not None}


def _forest_report_status(output_dir: Path) -> int:
    """Print the Skill-Forest report path and return its completion status."""
    print(f"Report: {output_dir / 'report.md'}", flush=True)
    return int(read_json(output_dir / "summary.json")["incomplete"])
