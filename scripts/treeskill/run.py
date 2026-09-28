#!/usr/bin/env python3
"""Run one TreeSkill experiment from YAML, without mindmemos_eval or a database."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import yaml
from mindmemos_skill.algos.train.treeskill import TreeSkillTrainer
from mindmemos_skill.algos.train.treeskill.artifacts import experiment_config
from mindmemos_skill.algos.train.treeskill.config import TreeSkillExperimentConfig
from mindmemos_skill.algos.train.treeskill.experiment import run_experiment
from mindmemos_skill.common.json import write_json
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.data import Data
from mindmemos_skill.skill import Skill


async def _main():
    """Load explicit task splits, construct the environment, and report final test scores."""
    script_start = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = TreeSkillExperimentConfig.model_validate(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    source = Skill.from_path(config.initial_skill)
    loader = Data.build(config.data, config.data_options)
    splits = loader.split(loader.load(), split_file=config.split_file)
    unknown = set(splits) - {"train", "val", "test"}
    if unknown or not splits.get("train") or splits.get("val"):
        raise ValueError("Provide train and optional test splits, with val empty and no unknown split names")
    splits["train"] = splits["train"][: config.train_limit]
    splits["test"] = splits.get("test", [])[: config.test_limit]
    preflight = None
    if config.environment == "spreadsheetbench_reference":
        from mindmemos_skill.algos.train.treeskill.env_specific.config import SpreadsheetBenchReferenceConfig
        from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench import (
            SpreadsheetBenchReferenceEnvironment,
            validate_reference_skill,
        )
        from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import (
            preflight_recalculation_runtime,
        )

        env = SpreadsheetBenchReferenceEnvironment(
            SpreadsheetBenchReferenceConfig.model_validate(config.environment_options)
        )
        if config.strict_reference_skill:
            validate_reference_skill(source)
        if env.config.transactional_recalculation:
            preflight = preflight_recalculation_runtime
    else:
        if config.algorithm.env_adapter.class_name == "spreadsheetbench_reference":
            raise ValueError("spreadsheetbench_reference analysis requires its matching reference environment")
        env = Environment.build(config.environment, config.environment_options)
    if args.preflight_only:
        start = perf_counter()
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            args.output_dir / "config.json",
            experiment_config(config.algorithm),
        )
        report = {"status": "preflight", "train": len(splits["train"]), "test": len(splits.get("test", []))}
        try:
            report["runtime"] = preflight(args.output_dir / "preflight") if preflight else {}
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            report["duration_s"] = perf_counter() - start
            write_json(args.output_dir / "summary.json", report)
        print(json.dumps(report, indent=2))
        return
    preparation_s = perf_counter() - script_start
    trainer = TreeSkillTrainer(config.algorithm)
    await run_experiment(
        trainer=trainer,
        skill=source,
        env=env,
        train_samples=splits["train"],
        validation_samples=splits.get("val", []),
        test_samples=splits.get("test", []),
        work_dir=args.output_dir,
        preflight=preflight,
    )
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    summary["stages"]["input_preparation_s"] = preparation_s
    summary["duration_s"] = perf_counter() - script_start
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    """Own the process event loop for the asynchronous entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
