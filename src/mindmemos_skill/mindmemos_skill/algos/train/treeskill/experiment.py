"""Experiment reporting around one training round and optional held-out testing."""

import asyncio
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from pydantic import JsonValue

from ....common.json import write_json
from ....envs import Environment
from ....envs.data import DataSample
from ....envs.data.typing import SplitManifest
from ....skill import Skill
from .artifacts import REFERENCE_COMMIT, experiment_config, model_usage
from .trainer import TreeSkillTrainer, sample_tasks, score_report


async def run_experiment(
    *,
    trainer: TreeSkillTrainer,
    skill: Skill,
    env: Environment,
    train_samples: Sequence[DataSample],
    work_dir: Path,
    validation_samples: Sequence[DataSample] = (),
    test_samples: Sequence[DataSample] = (),
    preflight: Callable[[Path], dict[str, JsonValue]] | None = None,
) -> Skill:
    """Train once and report final held-out scores without using them for selection.

    Args:
        trainer: Configured TreeSkill implementation, with no database dependencies.
        skill: Initial Skill and unchanged auxiliary resources.
        env: Caller-owned environment used for collection and final testing.
        train_samples: Tasks from which training evidence is actively sampled.
        work_dir: Fresh root for all experiment artifacts.
        validation_samples: Empty for the pinned one-round protocol.
        test_samples: Optional disjoint tasks, executed once after training.
        preflight: Optional local runtime check receiving the experiment preflight directory.

    Returns:
        The final trained Skill regardless of held-out score.

    Raises:
        ValueError: Splits overlap, validation selection is requested, or root exists.
    """
    start = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    if validation_samples:
        raise ValueError("The reference protocol has no validation selection; supply an empty val set")
    SplitManifest(
        {
            "train": [sample.task.id for sample in train_samples],
            "val": [],
            "test": [sample.task.id for sample in test_samples],
        }
    )
    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = experiment_config(trainer.config)
    write_json(root / "config.json", config)
    summary = {"status": "running", "reference_commit": REFERENCE_COMMIT, "train": None, "val": None, "test": None}
    write_json(root / "summary.json", summary)
    try:
        if preflight is not None:
            preflight_start = perf_counter()
            write_json(root / "preflight/result.json", await asyncio.to_thread(preflight, root / "preflight"))
            summary["preflight_duration_s"] = perf_counter() - preflight_start
        final = await trainer.train(
            skill=skill, env=env, train_samples=train_samples, validation_samples=(), work_dir=root / "training"
        )
        summary.update(json.loads((root / "training/summary.json").read_text()))
        summary["status"] = "testing" if test_samples else "completed"
        write_json(root / "summary.json", summary)
        if test_samples:
            testing_start = perf_counter()
            records, failures = await sample_tasks(
                env=env,
                skill=final,
                samples=test_samples,
                directory=root / "test",
                repeats=1,
                concurrency=trainer.config.rollout_concurrency,
            )
            summary["test"] = score_report(records, failures, len(test_samples))
            summary["test"]["skill_role"] = "final_routed_skill"
            summary["stages"]["test_s"] = perf_counter() - testing_start
        summary["status"] = (
            "completed"
            if summary["train"]["complete"]
            and not summary["training"]["has_failures"]
            and (summary["test"] is None or summary["test"]["complete"])
            else "partial"
        )
        summary["final_skill"] = "training/final_skill"
        return final
    except BaseException as exc:
        if (root / "training/summary.json").exists():
            summary.update(json.loads((root / "training/summary.json").read_text()))
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["duration_s"] = perf_counter() - start
        summary["started_at"] = started_at
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary["model_usage"] = model_usage(root / "training/model_calls")
        write_json(root / "summary.json", summary)
