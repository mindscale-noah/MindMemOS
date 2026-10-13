"""Evaluate empty Skill content on the comparison's fixed held-out split."""

import argparse
import asyncio
import logging
from pathlib import Path

import yaml
from mindmemos_skill.agents import AgentCatalog, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.recovery import sample_identity
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.artifacts import score_report
from mindmemos_skill.algos.train.treeskill.trainer import sample_tasks
from mindmemos_skill.common.checkpoint import boundary
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.config import SpreadsheetBenchTextActionConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.environment import (
    SpreadsheetBenchTextActionEnvironment,
)
from mindmemos_skill.envs.data import Data
from mindmemos_skill.logging import configure_logging
from mindmemos_skill.skill import Skill, SkillContent

from .compare import event


def load_policy(reference: Path, service_config: Path) -> OpenAIChatConfig:
    """Load recorded policy settings and supply credentials from local configuration.

    Args:
        reference: Completed comparison directory containing the training snapshot.
        service_config: Local YAML containing the TreeSkill trainer's model credentials.

    Returns:
        Recorded policy settings with the explicitly configured API key.
    """
    config = TreeSkillTrainerConfig.model_validate(read_json(reference / "training/config.json")["algorithm"])
    document = yaml.safe_load(service_config.read_text(encoding="utf-8"))
    local = TreeSkillTrainerConfig.model_validate(
        document["service"]["training_pipelines"]["treeskill"]["config"]["trainer"]["config"],
        context=AgentCatalog(agents=document["service"].get("agents", {})),
    )
    connection = local.failure_optimizer or local.optimizer
    connection.require_connection()
    return (config.failure_optimizer or config.optimizer).model_copy(
        update={"api_key": connection.api_key, "temperature": 0.7, "stop": ["Observation:"]}
    )


async def run(args: argparse.Namespace) -> None:
    """Evaluate without training, routing, Skill text, or Skill resources.

    Args:
        args: Reference comparison and explicit dataset and output paths.
    """
    reference = args.reference.resolve()
    root = args.output.resolve()
    policy = load_policy(reference, args.service_config)
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=policy, max_turns=100, transactional_recalculation=True)
    )
    data_root = args.data_root.resolve()
    data = Data.build("spreadsheetbench", {"data_path": data_root / "dataset.json", "data_root": data_root})
    samples = {sample.task.id: sample for sample in data.load()}
    split = read_json(reference / "split.json")
    if len(split["test"]) != 200 or len(set(split["test"])) != 200 or set(split["test"]) & set(split["train"]):
        raise ValueError("Expected 200 unique held-out tasks disjoint from training")
    test = [samples[identifier] for identifier in split["test"]]
    initial = Skill.from_path(args.initial_skill)
    for index, sample in enumerate(test):
        identity = {
            "task": sample_identity(sample, env),
            "skill": initial.snapshot().model_dump(mode="json"),
            "environment": env.config.model_dump(mode="json"),
        }
        saved = boundary(reference / f"baseline/task-{index:06d}-sample-000/completed.json", identity)
        if saved is None:
            raise ValueError("Reference baseline is incomplete")
    skill = Skill(
        SkillContent(content="", resources={}, name="no-skill", description="No Skill supplied.", standard_format=False)
    )
    boundary(
        root / "manifest.json",
        {
            "environment": env.config.model_dump(mode="json"),
            "skill": skill.snapshot().model_dump(mode="json"),
            "samples": [sample_identity(sample, env) for sample in test],
            "concurrency": 32,
        },
        payload={"protocol": 1},
    )
    write_json(root / "split.json", split)
    event("inputs_verified", test=200, model=policy.model, concurrency=32, skill_characters=0, skill_resources=0)
    await env.initialize()
    if args.preflight:
        response = await OpenAIChatAgent(policy).run(messages=[{"role": "user", "content": "Reply with OK."}])
        if not isinstance(response.output, str) or not response.output.strip():
            raise ValueError("Policy probe returned empty content")
        event("model_probe_passed", usage=response.usage)
        write_json(root / "preflight.json", {"status": "passed"})
        return
    summary = {"status": "running", "expected": 200, "reference": str(reference)}
    write_json(root / "summary.json", summary)
    try:
        records, failures = await sample_tasks(
            env=env,
            skill=skill,
            samples=test,
            directory=root / "no_skill",
            repeats=1,
            concurrency=32,
            stage_name="treeskill.no_skill",
        )
        summary.update(score_report(records, failures, 200))
        summary["status"] = "completed" if not failures and summary["complete"] else "failed"
        summary["failures"] = failures
        if failures:
            raise RuntimeError("No-skill evaluation has execution failures; inspect and resume")
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(root / "summary.json", summary)
    event("evaluation_completed", success_count=summary["success_count"], success_rate=summary["success_rate"])


def main() -> None:
    """Parse paths for a resumable no-skill evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--service-config", type=Path, default=Path("config/mindmemos_skill/dev.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--initial-skill",
        type=Path,
        default=Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench/initial_skill"),
    )
    parser.add_argument("--preflight", action="store_true")
    configure_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
