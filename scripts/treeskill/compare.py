"""Compare full initial Skill and trained TreeSkill on the same held-out split."""

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import yaml
from mindmemos_skill.agents import OpenAIChatAgent
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainer, TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.artifacts import score_report
from mindmemos_skill.algos.train.treeskill.trainer import sample_tasks
from mindmemos_skill.common.checkpoint import boundary
from mindmemos_skill.common.json import read_json, write_json
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.config import SpreadsheetBenchTextActionConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.environment import (
    SpreadsheetBenchTextActionEnvironment,
)
from mindmemos_skill.envs.data import Data
from mindmemos_skill.logging import configure_logging, get_logger
from mindmemos_skill.skill import Skill


def event(name: str, **details) -> None:
    """Print one timestamped experiment milestone.

    Args:
        name: Current stage or verification event.
        **details: Non-sensitive progress fields.
    """
    get_logger("treeskill.experiment").info(
        f"[treeskill.experiment] {name}",
        extra={"event_body": {"stage": "treeskill.experiment", "event": name, **details}},
    )


async def run(args: argparse.Namespace) -> None:
    """Run an input-bound baseline, training round and held-out evaluation.

    Args:
        args: Explicit dataset, service configuration and output paths.
    """
    root = args.output.resolve()
    document = yaml.safe_load(args.service_config.read_text())
    pipeline = document["service"]["training_pipelines"]["treeskill"]["config"]
    config = TreeSkillTrainerConfig.model_validate(pipeline["trainer"]["config"])
    # The upstream policy uses instruct decoding with its own temperature and stop.
    policy = (config.failure_optimizer or config.optimizer).model_copy(
        update={"temperature": 0.7, "stop": ["Observation:"]}
    )
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=policy, max_turns=100, transactional_recalculation=True)
    )
    data_root = args.data_root.resolve()
    data = Data.build("spreadsheetbench", {"data_path": data_root / "dataset.json", "data_root": data_root})
    splits = data.split(data.load(), split_file=args.split_file.resolve())
    train, test = splits["train"], splits["test"]
    if len(train) != 200 or len(test) != 200:
        raise ValueError("This comparison requires exactly 200 train and 200 test tasks")
    if {sample.task.id for sample in train} & {sample.task.id for sample in test}:
        raise ValueError("Train and held-out task IDs overlap")
    skill = Skill.from_path(args.initial_skill)
    identity = {
        "algorithm": config.model_dump(mode="json"),
        "environment": env.config.model_dump(mode="json"),
        "skill": skill.snapshot().model_dump(mode="json"),
        "data_root": str(data_root),
        "train_ids": [sample.task.id for sample in train],
        "test_ids": [sample.task.id for sample in test],
        "evaluation_concurrency": pipeline["evaluation_concurrency"],
    }
    boundary(root / "comparison_manifest.json", identity, payload={"protocol": 1})
    write_json(root / "split.json", {"train": identity["train_ids"], "test": identity["test_ids"]})
    event("inputs_verified", train=len(train), test=len(test), model=policy.model)
    await env.initialize()
    event("environment_ready")
    if args.preflight:
        for role, model in (("policy", policy), ("optimizer", config.optimizer), ("router", config.router)):
            response = await OpenAIChatAgent(model).run(messages=[{"role": "user", "content": "Reply with OK."}])
            if not isinstance(response.output, str) or not response.output.strip():
                raise ValueError(f"{role} probe returned no content")
            event("model_probe_passed", role=role, characters=len(response.output), usage=response.usage)
        write_json(root / "preflight.json", {"status": "passed", "time": datetime.now(UTC).isoformat()})
        return
    started = perf_counter()
    get_logger(__name__).info(f"[treeskill.experiment] started | tasks={len(test)}")
    summary = {"status": "running", "baseline": None, "training": None, "evolved": None}
    try:
        for phase in ("baseline", "training", "evolved"):
            summary["stage"] = phase
            write_json(root / "summary.json", summary)
            if phase == "training":
                skill = await TreeSkillTrainer(config).train(
                    skill=skill, env=env, train_samples=train, work_dir=root / "training"
                )
                summary[phase] = read_json(root / "training/summary.json")
                if not summary[phase]["train"]["complete"]:
                    raise RuntimeError("Training collection has execution failures; inspect and resume before testing")
            else:
                records, failures = await sample_tasks(
                    env=env,
                    skill=skill,
                    samples=test,
                    directory=root / phase,
                    stage_name=f"treeskill.{phase}",
                    repeats=1,
                    concurrency=pipeline["evaluation_concurrency"],
                )
                summary[phase] = score_report(records, failures, len(test))
                write_json(root / phase / "summary.json", summary[phase])
                if failures:
                    raise RuntimeError(f"{phase} has {len(failures)} execution failures; inspect and resume")
            write_json(root / "summary.json", summary)
        summary["status"] = "completed"
        summary["success_rate_delta"] = summary["evolved"]["success_rate"] - summary["baseline"]["success_rate"]
        get_logger(__name__).info(
            f"[treeskill.experiment] completed | baseline_accuracy={summary['baseline']['success_rate']:.2%}"
            f" evolved_accuracy={summary['evolved']['success_rate']:.2%}"
            f" | accuracy_delta_pp={summary['success_rate_delta'] * 100:+.2f} elapsed={perf_counter() - started:.1f}s"
        )
    except BaseException as exc:
        event_name = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        get_logger(__name__).log(
            logging.INFO if event_name == "cancelled" else logging.ERROR,
            f"[treeskill.experiment] {event_name} | elapsed={perf_counter() - started:.1f}s"
            f" | error={type(exc).__name__}: {exc}",
        )
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(root / "summary.json", summary)


def main() -> None:
    """Parse comparison paths and start the asynchronous experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-config", type=Path, default=Path("config/mindmemos_skill/dev.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    resources = Path("resources/mindmemos_skill_eval/treeskill-spreadsheetbench")
    parser.add_argument("--split-file", type=Path, default=resources / "data_splits/splits.json")
    parser.add_argument("--initial-skill", type=Path, default=resources / "initial_skill")
    parser.add_argument("--preflight", action="store_true")
    configure_logging()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
