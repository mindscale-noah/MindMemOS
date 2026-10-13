"""Expose only fresh Skill runs and saved-result reports."""

import argparse
import asyncio
from pathlib import Path

import yaml
from mindmemos_skill.envs import Environment

from .config import ROOT, EnvTestConfig, RunOptions, TreeSkillEvalConfig, load_config
from .report import read_report, write_report
from .runner import run
from .testing import run_test


def add_skill_args(parser: argparse.ArgumentParser) -> None:
    """Add the supported Skill commands to the top-level parser.

    Args:
        parser: Skill subparser owned by the evaluation CLI.
    """
    commands = parser.add_subparsers(dest="skill_command", required=True)
    treeskill = commands.add_parser(
        "treeskill", help="Run the reference TreeSkill SpreadsheetBench protocol through the UI service."
    )
    treeskill.add_argument("--config", type=Path, default=ROOT / "config/mindmemos_eval/treeskill/dev.yaml")
    treeskill.add_argument(
        "--check", action="store_true", help="Verify reference inputs and service configuration without model calls."
    )
    treeskill.add_argument(
        "--env", help="Registered environment; other environments have no reference score guarantee."
    )
    treeskill.add_argument("--output-dir", type=Path)
    commands.add_parser("envs", help="List registered environments without running tasks.")
    test = commands.add_parser("test", help="Test one environment with no Skills or a fixed static Skill collection.")
    test.add_argument("--config", type=Path, default=ROOT / "config/mindmemos_eval/env_test/dev.yaml")
    test.add_argument("--env", help="Override the registered environment name in the configuration.")
    mode = test.add_mutually_exclusive_group()
    mode.add_argument("--no-skill", action="store_true", help="Run without any Skills.")
    mode.add_argument("--skills", type=Path, nargs="+", help="Inject these static Skill paths together.")
    test.add_argument("--samples", type=Path, help="Override the data loader with a task/reward_data JSON array.")
    test.add_argument("--split-file", type=Path)
    test.add_argument("--split")
    test.add_argument("--limit", type=int)
    test.add_argument("--concurrency", type=int)
    test.add_argument("--output-dir", type=Path)
    execute = commands.add_parser("run", help="Run fresh Skill-Forest control and trained tests.")
    execute.add_argument("--config", type=Path, default=ROOT / "config/mindmemos_eval/dev.yaml")
    execute.add_argument("--algorithm", choices=["skill_forest"], default="skill_forest")
    execute.add_argument(
        "--benchmark",
        choices=["livemath", "alfworld", "spreadsheetbench", "searchqa", "officeqa", "all"],
        default="livemath",
    )
    execute.add_argument("--smoke", action="store_true")
    execute.add_argument("--test-repeats", type=int, default=1)
    execute.add_argument("--arms", default="no_skill,initial_skill,trained_skill")
    execute.add_argument("--data-root", type=Path, default=ROOT / "data/mindmemos_skill")
    execute.add_argument("--output-dir", type=Path)
    report = commands.add_parser("report", help="Regenerate reports from saved outcomes without model calls.")
    report.add_argument("run_dir", type=Path)


def run_skill_benchmark(args: argparse.Namespace) -> int:
    """Dispatch a fresh run or a report and return an honest exit code.

    Args:
        args: Parsed arguments from add_skill_args.

    Returns:
        Zero for complete results; one for preflight, runtime, or incomplete outcomes.
    """
    try:
        if args.skill_command == "treeskill":
            from .treeskill import prepare
            from .treeskill import run as run_treeskill

            payload = yaml.safe_load(args.config.read_text())
            if args.env is not None:
                payload["environment"] = {**payload.get("environment", {}), "class_name": args.env}
            if args.output_dir is not None:
                payload["output_dir"] = args.output_dir
            config = TreeSkillEvalConfig.model_validate(payload)
            if args.check:
                _, pools, _ = prepare(config)
                print(
                    f"Prepared {config.environment.class_name}: {len(pools['train'])} train / {len(pools['test'])} test."
                )
                if config.environment.class_name == "spreadsheetbench_text_action":
                    print("Reference aligned: max_turns=100; recalculation=true; concurrency=32.")
                return 0
            summary = asyncio.run(run_treeskill(config))
            print(summary.model_dump_json(indent=2), flush=True)
            return int(summary.errors > 0)
        if args.skill_command == "envs":
            print("\n".join(Environment.registered_types()), flush=True)
            return 0
        if args.skill_command == "test":
            payload = yaml.safe_load(args.config.expanduser().read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Test configuration must be a YAML object")
            if args.env is not None:
                payload["environment"] = {**payload.get("environment", {}), "class_name": args.env}
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
                "output_dir": args.output_dir,
            }
            payload.update({key: value for key, value in overrides.items() if value is not None})
            summary = asyncio.run(run_test(EnvTestConfig.model_validate(payload)))
            print(summary.model_dump_json(indent=2), flush=True)
            print(f"Report: {summary.trajectory_file.parent.parent / 'report.md'}", flush=True)
            return int(summary.errors > 0)
        if args.skill_command == "report":
            report = read_report(args.run_dir)
            write_report(report, report.output_dir)
        else:
            options = RunOptions(
                benchmark=args.benchmark,
                algorithm=args.algorithm,
                smoke=args.smoke,
                test_repeats=args.test_repeats,
                arms=args.arms.split(","),
                data_root=args.data_root,
                output_dir=args.output_dir,
            )
            report = asyncio.run(run(load_config(args.config), options))
        from mindmemos_skill.common.json import read_json

        print(f"Report: {report.output_dir / 'report.md'}", flush=True)
        return int(read_json(report.output_dir / "summary.json")["incomplete"])
    except (ValueError, OSError, RuntimeError, TypeError, yaml.YAMLError) as exc:
        print(f"Evaluation failed: {exc}", flush=True)
        return 1
