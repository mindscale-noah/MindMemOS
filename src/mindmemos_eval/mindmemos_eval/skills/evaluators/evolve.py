"""Resolve and run the evolution pipeline evaluation."""

import asyncio
from argparse import Namespace

from ..algos import offline_evolution
from ..config import EVAL_CONFIGS, SkillEvolveConfig, read_environment_yaml


def evaluate(args: Namespace) -> int:
    """Evaluate offline evolution on matched tasks.

    Args:
        args: Config, environment, output directory, and preflight options.

    Returns:
        Zero on success, or one when a task reports an error.
    """
    config = SkillEvolveConfig.model_validate(
        read_environment_yaml(
            args.config or EVAL_CONFIGS / "skill_evolve_offline/dev.yaml", env=args.env, output_dir=args.output_dir
        )
    )
    if args.check:
        _, samples, skills = offline_evolution.prepare(config)
        skill_count = sum(len(items) for items in skills.values())
        print(f"Prepared {len(samples)} cases; {skill_count} initial Skills.", flush=True)
        return 0

    summary = asyncio.run(offline_evolution.run(config))
    print(f"Report: {summary['output_dir']}/report.md", flush=True)
    return int(summary["errors"] > 0)
