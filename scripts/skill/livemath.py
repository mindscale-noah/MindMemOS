"""Load local LiveMath data, execute one real rollout, and verify resume."""

import argparse
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.livemath.config import LiveMathRunConfig
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill, SkillSet


def main() -> None:
    """Execute a configured task and save the loaded membership and audit summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--task-id", help="Select an ID within the configured split; default is its first sample")
    parser.add_argument("--skill-file", type=Path)
    args = parser.parse_args()
    if args.env_file is not None:
        if not args.env_file.is_file():
            raise FileNotFoundError(args.env_file)
        load_dotenv(args.env_file, override=False)
    config = LiveMathRunConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    data = Data.build("livemath", config.data.model_dump())
    samples = data.load()
    splits = data.split(samples, ratios=config.split.ratios, seed=config.split.seed, split_file=config.split.split_file)
    selected = splits[config.split_name]
    if args.task_id is not None:
        selected = [sample for sample in selected if sample.task.id == args.task_id]
    if not selected:
        raise ValueError("No matching sample in the configured split")
    sample = selected[0]
    skills = []
    if args.skill_file is not None:
        skills.append(Skill.from_path(args.skill_file))
    skill_set = SkillSet({str(index): skill for index, skill in enumerate(skills)}, max_concurrency=1)
    env = Environment.build("livemath", config.environment.model_dump())
    print(
        json.dumps(
            {
                "loaded": len(samples),
                "splits": {name: len(items) for name, items in splits.items()},
                "task_id": sample.task.id,
                "model": config.environment.agent.model,
                "concurrency": 1,
                "timeout_s": config.environment.agent.timeout_s,
                "max_completion_tokens": config.environment.agent.max_completion_tokens,
                "skills": len(skills),
            }
        ),
        flush=True,
    )
    result = env.rollout(
        sample.task,
        args.work_dir,
        reward_data=sample.reward_data,
        skill_set=skill_set,
    )
    resumed = env.rollout(
        sample.task,
        args.work_dir,
        reward_data=sample.reward_data,
        skill_set=skill_set,
    )
    if resumed != result:
        raise RuntimeError("Resume did not return the original result")
    audit_dir = args.work_dir / "audit"
    data.save_splits(splits, audit_dir / "splits.json")
    summary = {
        "task_id": sample.task.id,
        "loaded": len(samples),
        "splits": {name: len(items) for name, items in splits.items()},
        "score": result.reward.score,
        "success": result.reward.success,
        "usage": result.usage.model_dump(mode="json"),
        "duration_s": result.duration_s,
        "resume_equal": True,
        "data_config": config.data.model_dump(mode="json"),
        "split_config": config.split.model_dump(mode="json"),
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
