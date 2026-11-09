"""Resolve Skill CLI options and call the existing evaluators."""

import asyncio
from argparse import Namespace
from pathlib import Path

from mindmemos_skill.common.json import read_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.pipelines import Pipeline

from . import evolve, runner, skill_graph, treeskill
from .config import (
    ENVIRONMENT_NAMES,
    ROOT,
    EnvTestConfig,
    RunOptions,
    SkillEvolveConfig,
    SkillGraphAccuracyConfig,
    SkillGraphRunOptions,
    TreeSkillEvalConfig,
    load_config,
    read_evaluation_yaml,
)
from .report import read_report, write_report
from .testing import run_test

EVAL_CONFIGS = ROOT / "config/mindmemos_eval"
EVALUATION_NAMES = {"offline_evolution": "evolve", "graph_manage": "manage", "training": "train"}


def show_envs(args: Namespace) -> int:
    """Print the registered environment names.

    Args:
        args: CLI handler arguments. The listing takes no options.

    Returns:
        Zero after listing the registered environments.
    """
    print("\n".join(Environment.registered_types()), flush=True)
    return 0


def show_pipelines(args: Namespace) -> int:
    """Print the registered pipelines and their evaluation commands.

    Args:
        args: CLI handler arguments. The listing takes no options.

    Returns:
        Zero after listing the registered pipelines.
    """
    for name in Pipeline.registered_types():
        evaluate_name = EVALUATION_NAMES.get(name)
        print(f"{name} (evaluate {evaluate_name})" if evaluate_name else name)
    return 0


def evaluate_evolution(args: Namespace) -> int:
    """Evaluate offline evolution on matched tasks.

    Args:
        args: Config, environment, output directory, and preflight options.

    Returns:
        Zero on success, or one when a task reports an error.
    """
    config = SkillEvolveConfig.model_validate(_environment_config(args, "skill_evolve_offline"))
    if args.check:
        _, samples, skills = evolve.prepare(config)
        skill_count = sum(len(items) for items in skills.values())
        print(f"Prepared {len(samples)} cases; {skill_count} initial Skills.", flush=True)
        return 0

    summary = asyncio.run(evolve.run(config))
    print(f"Report: {summary['output_dir']}/report.md", flush=True)
    return int(summary["errors"] > 0)


def evaluate_training(args: Namespace) -> int:
    """Select the trainer evaluation for the requested algorithm.

    Args:
        args: Training algorithm and evaluation options.

    Returns:
        The selected evaluator's exit status.
    """
    if args.algo == "skill-forest":
        return evaluate_forest(args)

    forest_options = _forest_overrides(args)
    if forest_options:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in forest_options)
        raise ValueError(f"{flags} require --algo skill-forest")
    return evaluate_treeskill(args)


def evaluate_forest(args: Namespace) -> int:
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
    report = asyncio.run(runner.run(config, options))
    return _forest_report_status(report.output_dir)


def evaluate_treeskill(args: Namespace) -> int:
    """Evaluate TreeSkill training and held-out performance.

    Args:
        args: Config, environment, output directory, and preflight options.

    Returns:
        Zero on success, or one when a task reports an error.
    """
    config = TreeSkillEvalConfig.model_validate(_environment_config(args, "skill_train_treeskill"))
    if args.check:
        _, pools, _, _ = treeskill.prepare(config)
        print(f"Prepared {config.environment.class_name}: {len(pools['train'])} train / {len(pools['test'])} test.")
        return 0

    summary = asyncio.run(treeskill.run(config))
    print(summary.model_dump_json(indent=2), flush=True)
    return int(summary.errors > 0)


def evaluate_graph(args: Namespace) -> int:
    """Evaluate AgentSkillOS retrieval rankings or task accuracy.

    Args:
        args: Evaluation mode, configuration, input paths, and output directory.

    Returns:
        Zero on success, or one when an accuracy task reports an error.
    """
    config_path = args.config
    if config_path is None and args.eval_mode == "accuracy":
        config_path = EVAL_CONFIGS / "skill_graph_setting_1/dev.yaml"
    payload = read_evaluation_yaml(config_path) if config_path else {}
    graph_overrides = {
        "env": args.env,
        "eval_mode": args.eval_mode,
        "tasks_dir": args.tasks_dir,
        "skills_dir": args.skills_dir,
        "service_config": args.service_config,
        "output_dir": args.output_dir,
    }
    graph_options = {name: payload.pop(name) for name in graph_overrides if name in payload}
    options = SkillGraphRunOptions.model_validate({**graph_options, **_provided(graph_overrides)})
    output = options.output_dir.expanduser().resolve()
    skills = options.skills_dir.expanduser().resolve()
    service_config = options.service_config.expanduser().resolve()
    if output.is_relative_to(skills):
        raise ValueError("Write evaluation outputs outside the Skill source directory")

    if options.eval_mode == "retrieval":
        if options.tasks_dir is None:
            raise ValueError("Retrieval requires --tasks-dir or tasks_dir in the YAML")
        tasks = options.tasks_dir.expanduser().resolve()
        if output.is_relative_to(tasks):
            raise ValueError("Write evaluation outputs outside the task directory")
        asyncio.run(skill_graph.run_retrieval(tasks, skills, service_config, output))
        exit_status = 0
    else:
        if args.tasks_dir is not None:
            raise ValueError("Set the accuracy benchmark root in the YAML instead of --tasks-dir")
        config = SkillGraphAccuracyConfig.model_validate(payload)
        if output.is_relative_to(config.benchmark_root.expanduser().resolve()):
            raise ValueError("Write evaluation outputs outside the benchmark root")
        result = asyncio.run(skill_graph.run_accuracy(config, skills, service_config, output))
        exit_status = int(bool(result.errors))

    print(f"Report: {output / 'summary.json'}", flush=True)
    return exit_status


def test_environment(args: Namespace) -> int:
    """Run a direct evaluation with a fixed set of Skills or no Skills.

    Args:
        args: Environment, task selection, Skill inputs, and execution options.

    Returns:
        Zero on success, or one when a task reports an error.
    """
    payload = _environment_config(args, "skill_test_direct")
    if args.no_skill:
        payload["skills"] = []
    elif args.skills is not None:
        payload["skills"] = args.skills
    if args.samples is not None:
        payload.update(samples_file=args.samples, data=None)
    payload.update(
        _provided(
            {"split_file": args.split_file, "split": args.split, "limit": args.limit, "concurrency": args.concurrency}
        )
    )

    summary = asyncio.run(run_test(EnvTestConfig.model_validate(payload)))
    print(summary.model_dump_json(indent=2), flush=True)
    print(f"Report: {summary.trajectory_file.parent.parent / 'report.md'}", flush=True)
    return int(summary.errors > 0)


def rebuild_report(args: Namespace) -> int:
    """Rebuild a report from an existing Skill-Forest run.

    Args:
        args: Directory containing the saved run.

    Returns:
        Zero for complete results, or one for incomplete results.
    """
    report = read_report(args.run_dir)
    write_report(report, report.output_dir)
    return _forest_report_status(report.output_dir)


def _forest_overrides(args: Namespace) -> dict[str, object]:
    """Return supplied Skill-Forest options in their configuration form."""
    return _provided(
        {
            "benchmark": args.benchmark,
            "smoke": args.smoke,
            "test_repeats": args.test_repeats,
            "arms": args.arms.split(",") if args.arms is not None else None,
            "data_root": args.data_root,
        }
    )


def _forest_report_status(output_dir: Path) -> int:
    """Print the Skill-Forest report path and return its completion status."""
    print(f"Report: {output_dir / 'report.md'}", flush=True)
    return int(read_json(output_dir / "summary.json")["incomplete"])


def _environment_config(args: Namespace, config_dir: str) -> dict[str, object]:
    """Load an environment-based evaluation configuration with CLI overrides."""
    payload = read_evaluation_yaml(args.config or EVAL_CONFIGS / config_dir / "dev.yaml")
    if args.env is not None:
        environment = payload.setdefault("environment", {})
        if not isinstance(environment, dict):
            raise ValueError("environment must be a YAML mapping")
        environment["class_name"] = args.env
    if args.output_dir is not None:
        payload["output_dir"] = args.output_dir
    return payload


def _provided(options: dict[str, object]) -> dict[str, object]:
    """Keep explicit overrides, including false values and empty collections."""
    return {name: value for name, value in options.items() if value is not None}
