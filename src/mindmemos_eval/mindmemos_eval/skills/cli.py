"""Define the Skill evaluation CLI and dispatch parsed commands."""

import argparse
from pathlib import Path

import yaml

from . import commands
from .typing import BENCHMARKS


def add_skill_args(parser: argparse.ArgumentParser) -> None:
    """Add Skill subcommands to the evaluation CLI.

    Args:
        parser: Parser for the top-level skill command.
    """
    subcommands = parser.add_subparsers(dest="skill_command", required=True)

    show = subcommands.add_parser("show", help="List registered environments or pipelines.")
    registries = show.add_subparsers(dest="registry", required=True)
    registries.add_parser("envs", help="List registered environments.").set_defaults(handler=commands.show_envs)
    registries.add_parser("pipelines", help="List registered pipelines and their evaluation names.").set_defaults(
        handler=commands.show_pipelines
    )

    evaluate = subcommands.add_parser("evaluate", help="Run a Skill evaluation or rebuild a saved report.")
    _add_evaluation_commands(evaluate)


def _add_evaluation_commands(parser: argparse.ArgumentParser) -> None:
    """Register pipeline evaluations and standalone evaluation operations."""
    evaluations = parser.add_subparsers(dest="evaluation", required=True)

    evolve = evaluations.add_parser("evolve", help="Evaluate offline Skill evolution.")
    evolve.add_argument("--algo", choices=("offline-evolv",), default="offline-evolv")
    _add_config_args(evolve)
    _add_env_arg(evolve)
    _add_check_arg(evolve)
    evolve.set_defaults(handler=commands.evaluate_evolution)

    manage = evaluations.add_parser("manage", help="Evaluate Skill Graph retrieval or end-to-end accuracy.")
    manage.add_argument("--algo", choices=("skill-graph",), default="skill-graph")
    manage.add_argument("--env", choices=("agentskillos",), help="AgentSkillOS environment.")
    _add_graph_args(manage)
    manage.set_defaults(handler=commands.evaluate_graph)

    train = evaluations.add_parser("train", help="Evaluate Skill-Forest or TreeSkill training.")
    train.add_argument("--algo", choices=("skill-forest", "tree-skill"), required=True)
    _add_config_args(train)
    _add_env_arg(train)
    _add_forest_args(train)
    _add_check_arg(train, help_text="Validate TreeSkill inputs without model calls.")
    train.set_defaults(handler=commands.evaluate_training)

    direct = evaluations.add_parser("direct", help="Test a single environment with no Skills or fixed Skills.")
    _add_direct_test_args(direct)
    direct.set_defaults(handler=commands.test_environment)

    report = evaluations.add_parser("rebuild-report", help="Rebuild a saved Skill-Forest report.")
    report.add_argument("run_dir", type=Path)
    report.set_defaults(handler=commands.rebuild_report)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add the YAML and output directory options shared by evaluation commands."""
    parser.add_argument("--config", type=Path, help="Evaluation YAML file.")
    parser.add_argument("--output-dir", type=Path, help="Output directory. Overrides output_dir in the YAML.")


def _add_env_arg(parser: argparse.ArgumentParser) -> None:
    """Add a registered environment override."""
    parser.add_argument("--env", help="Registered environment name. See skill show envs.")


def _add_check_arg(parser: argparse.ArgumentParser, help_text: str = "Validate inputs without model calls.") -> None:
    """Add the input validation mode."""
    parser.add_argument("--check", action="store_true", help=help_text)


def _add_forest_args(parser: argparse.ArgumentParser) -> None:
    """Add options for the Skill-Forest controls and trained runs."""
    parser.add_argument("--benchmark", choices=BENCHMARKS, help="Skill-Forest dataset.")
    parser.add_argument(
        "--smoke", action="store_true", default=None, help="Use small Skill-Forest dataset splits and search budget."
    )
    parser.add_argument("--test-repeats", type=int, help="Number of Skill-Forest test repetitions per arm.")
    parser.add_argument("--arms", help="Skill-Forest arms: no_skill,initial_skill,trained_skill.")
    parser.add_argument("--data-root", type=Path, help="Parent directory of the Skill-Forest datasets.")


def _add_graph_args(parser: argparse.ArgumentParser) -> None:
    """Add Skill Graph inputs and scoring mode."""
    _add_config_args(parser)
    parser.add_argument(
        "--eval-mode",
        choices=("retrieval", "accuracy"),
        help="Score retrieval rankings or execute and score AgentSkillOS tasks.",
    )
    parser.add_argument("--tasks-dir", type=Path, help="Directory of original task JSON files for retrieval scoring.")
    parser.add_argument("--skills-dir", type=Path, help="Directory of Skill packages to index and rank.")
    parser.add_argument("--service-config", type=Path, help="Local Skill service YAML file.")


def _add_direct_test_args(parser: argparse.ArgumentParser) -> None:
    """Add inputs for a direct environment test."""
    _add_config_args(parser)
    _add_env_arg(parser)
    skill_source = parser.add_mutually_exclusive_group()
    skill_source.add_argument("--no-skill", action="store_true", help="Run without any Skills.")
    skill_source.add_argument("--skills", type=Path, nargs="+", help="Inject these static Skill paths together.")
    parser.add_argument("--samples", type=Path, help="Override the data loader with a task/reward_data JSON array.")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int)


def run_skill_command(args: argparse.Namespace) -> int:
    """Execute the selected Skill command and report expected input or runtime failures.

    Args:
        args: Parsed Skill command and its options.

    Returns:
        The command's exit status, or one when execution raises an input or runtime error.
    """
    try:
        return args.handler(args)
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as exc:
        print(f"Evaluation failed: {exc}", flush=True)
        return 1
