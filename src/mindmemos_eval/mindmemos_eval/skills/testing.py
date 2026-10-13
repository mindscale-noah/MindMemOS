"""Test a registered environment with no Skills or one fixed static collection."""

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.common.secrets import redact_credentials
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data, DataSample
from mindmemos_skill.envs.data.typing import SplitManifest
from mindmemos_skill.skill import Skill, SkillContent, SkillSet

from .config import ROOT, EnvTestConfig
from .evaluation import evaluate
from .report import write_test_report
from .typing import EnvTestSummary


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
    cases = await evaluate(env, samples, skill_set, directory / "tasks", config.concurrency)
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
    write_test_report(summary, directory)
    return summary
