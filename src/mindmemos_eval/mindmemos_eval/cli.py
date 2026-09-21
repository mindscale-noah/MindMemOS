"""MindMemOS evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio

try:
    from .memory import add_memory_args, run_benchmark_matrix
    from .memory.db_reset import ProjectResetError
    from .skills import add_skill_args, run_skill_benchmark
except ImportError:  # pragma: no cover - used when running this file directly
    from mindmemos_eval.memory import add_memory_args, run_benchmark_matrix
    from mindmemos_eval.memory.db_reset import ProjectResetError
    from mindmemos_eval.skills import add_skill_args, run_skill_benchmark


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MindMemOS evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Run MindMemOS evaluation benchmarks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    memory = subparsers.add_parser("memory", help="Run memory benchmark matrix add/search evaluations.")
    add_memory_args(memory)

    skill = subparsers.add_parser("skill", help="Run skill benchmark self-evolution evaluation.")
    add_skill_args(skill)

    statebench = subparsers.add_parser("statebench", help="Run STATE-Bench memory self-evolution evaluations.")
    statebench_commands = statebench.add_subparsers(dest="statebench_command", required=True)

    from .memory.envs.statebench.config_runner import add_run_args
    from .memory.envs.statebench.official_reference import add_reference_args
    from .memory.envs.statebench.repair_runner import add_repair_args

    run = statebench_commands.add_parser("run", help="Run one configured online-controlled seed.")
    add_run_args(run)

    repair = statebench_commands.add_parser(
        "repair", help="Resume a later bulk failure into a separate repair output tree."
    )
    add_repair_args(repair)

    aggregate = statebench_commands.add_parser("aggregate", help="Aggregate completed seed reports.")
    aggregate.add_argument("--root", type=str, required=True)
    aggregate.add_argument("--seeds", default="42,43,44")
    aggregate.add_argument("--bootstrap-reps", type=int, default=10_000)

    official = statebench_commands.add_parser(
        "official-reference", help="Run the fixed-trajectory external reference baseline."
    )
    add_reference_args(official)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    if args.command == "memory":
        try:
            asyncio.run(run_benchmark_matrix(args))
        except ProjectResetError:
            return 1
        return 0
    if args.command == "skill":
        return run_skill_benchmark(args)
    if args.command == "statebench":
        if args.statebench_command == "run":
            from .memory.envs.statebench.config_runner import run

            return run(args.config, args.seed, dry_run=args.dry_run)
        if args.statebench_command == "repair":
            from .memory.envs.statebench.repair_runner import run

            return run(
                args.config,
                args.seed,
                source_output_dir=args.source_output_dir,
                output_dir=args.output_dir,
                from_round=args.from_round,
                separate_output=args.separate_output,
                resume=args.resume,
                dry_run=args.dry_run,
            )
        if args.statebench_command == "aggregate":
            from .memory.envs.statebench import aggregate

            return aggregate.main(
                [
                    "--root",
                    args.root,
                    "--seeds",
                    args.seeds,
                    "--bootstrap-reps",
                    str(args.bootstrap_reps),
                ]
            )
        if args.statebench_command == "official-reference":
            from .memory.envs.statebench.official_reference import run

            return run(
                args.config,
                args.seed,
                args.api_key,
                output_dir=args.output_dir,
                trajectory_dir=args.trajectory_dir,
                final_runs=args.final_runs,
            )
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
