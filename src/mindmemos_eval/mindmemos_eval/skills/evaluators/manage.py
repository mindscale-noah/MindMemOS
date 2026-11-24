"""Resolve and run the management pipeline evaluation."""

import asyncio
from argparse import Namespace

from ..algos import skill_graph
from ..config import EVAL_CONFIGS, SkillGraphAccuracyConfig, SkillGraphRunOptions, read_evaluation_yaml


def evaluate(args: Namespace) -> int:
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
    graph_options.update({name: value for name, value in graph_overrides.items() if value is not None})
    options = SkillGraphRunOptions.model_validate(graph_options)
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
