"""Run official AgentSkillOS tasks with a native agent and report task success."""

import argparse
import json
from pathlib import Path

from mindmemos_skill.envs.builtin_envs.agentskillos import (
    AgentSkillOSConfig,
    AgentSkillOSData,
    AgentSkillOSDataConfig,
    AgentSkillOSEnvironment,
)
from mindmemos_skill.envs.builtin_envs.agentskillos.reward import summarize_results
from mindmemos_skill.envs.builtin_envs.agentskillos.typing import Ranking
from mindmemos_skill.skill import Skill, SkillSet
from pydantic import TypeAdapter


def main() -> None:
    """Execute a serial benchmark with explicit selection and fresh per-task attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="AgentSkillOSConfig JSON; credentials come from environment"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", choices=("none", "all", "rankings"), required=True)
    parser.add_argument("--skills", type=Path)
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--relevance", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or (args.selection != "none" and args.skills is None):
        parser.error("A positive limit and a Skill directory are required for Skill selection")
    if args.selection == "rankings" and args.rankings is None:
        parser.error("rankings selection requires --rankings")
    if args.selection != "rankings" and args.rankings is not None:
        parser.error("--rankings is only valid with rankings selection")
    config = AgentSkillOSConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    data = AgentSkillOSData(AgentSkillOSDataConfig(benchmark_root=config.benchmark_root, relevance_path=args.relevance))
    samples = data.load()
    if args.split_file:
        samples = data.split(samples, split_file=args.split_file)[args.split]
    if not samples:
        parser.error("The selected split contains no tasks")
    skills = {}
    if args.selection != "none":
        skills = {path.parent.name: Skill.from_path(path) for path in sorted(args.skills.glob("*/SKILL.md"))}
        if not skills:
            parser.error("The Skill pool is empty")
    rankings = None
    if args.rankings:
        rankings = TypeAdapter(dict[str, list[str]]).validate_json(args.rankings.read_text(encoding="utf-8"))
        if set(rankings) != {sample.task.id for sample in samples}:
            parser.error("Rankings must exactly match the selected task IDs")
        for ranking in rankings.values():
            Ranking(skill_ids=ranking)
            if set(ranking) - skills.keys():
                parser.error("Ranking contains IDs outside the Skill pool")
    manifest = {
        "environment": config.model_dump(mode="json"),
        "selection": args.selection,
        "limit": args.limit,
        "tasks": [sample.model_dump(mode="json") for sample in samples],
        "rankings": rankings,
        "skills": {key: skill.snapshot().model_dump(mode="json") for key, skill in skills.items()},
    }
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = args.output / "run-inputs.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Benchmark inputs changed; use a new output directory")
    else:
        if args.resume and any(args.output.iterdir()):
            raise ValueError("Cannot resume a directory without its input manifest")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    env = AgentSkillOSEnvironment(config)
    results, errors = {}, {}
    for index, sample in enumerate(samples):
        ids = rankings[sample.task.id][: args.limit] if rankings is not None else list(skills)
        selection = SkillSet({key: skills[key] for key in ids}, max_concurrency=1)
        try:
            results[sample.task.id] = env.rollout(
                sample.task,
                args.output / f"task-{index:04d}",
                reward_data=sample.reward_data,
                skill_set=selection,
                resume=args.resume,
            )
        except Exception as exc:
            errors[sample.task.id] = f"{type(exc).__name__}: {exc}"
        summary = summarize_results(results, errors)
        if index + 1 < len(samples):
            summary["status"] = "running"
        summary["requested_tasks"] = len(samples)
        temporary = args.output / "summary.json.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(args.output / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
