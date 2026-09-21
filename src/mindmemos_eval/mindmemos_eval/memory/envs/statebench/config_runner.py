from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from . import runner
from .paths import CONFIG_ROOT, REPO_ROOT

DEFAULT_CONFIG = CONFIG_ROOT / "memory_evaluation_statebench.yaml"


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _format(template: str, seed: int) -> str:
    return template.format(seed=seed)


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"unsupported StateBench config: {path}")
    return data


def _load_keys(path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("api_keys") or []
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and entry.get("key_id"):
            result[str(entry["key_id"])] = entry
    return result


def _checkpoint_text(values: Any, rounds: int, name: str) -> str:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    checkpoints = [int(value) for value in values]
    if any(value < 0 or value > rounds for value in checkpoints):
        raise ValueError(f"{name} contains a checkpoint outside 0..{rounds}")
    return ",".join(str(value) for value in checkpoints)


def build_argv(
    config: dict[str, Any],
    seed: int,
    *,
    require_credentials: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    experiment = config["experiment"]
    training = config["training"]
    evaluation = config["evaluation"]
    isolation = config["isolation"]

    allowed_seeds = experiment.get("seeds")
    if allowed_seeds is not None and seed not in {int(value) for value in allowed_seeds}:
        raise ValueError(f"seed {seed} is not listed in experiment.seeds")

    rounds = int(training["rounds"])
    experiment_id = _format(str(experiment["group_id_template"]), seed)
    output_dir = _resolve(str(experiment["output_dir"])) / f"seed{seed}"
    state_bench_dir = _resolve(str(experiment["state_bench_dir"]))
    keys = _load_keys(_resolve(str(isolation["api_keys_file"])))

    branches: dict[str, dict[str, str]] = {}
    for branch in ("baseline", "evolution"):
        branch_config = isolation[branch]
        project_id = _format(str(branch_config["project_id_template"]), seed)
        key_id = _format(str(branch_config["key_id_template"]), seed)
        entry = keys.get(key_id)
        if not entry:
            if require_credentials:
                raise ValueError(f"key_id {key_id!r} is missing from api_keys.yaml")
            branches[branch] = {
                "project_id": project_id,
                "key_id": key_id,
                "api_key": "",
                "credential_status": "missing",
            }
            continue
        if not entry.get("enabled", True):
            raise ValueError(f"key_id {key_id!r} is disabled")
        if entry.get("project_id") != project_id:
            raise ValueError(f"{key_id} points to {entry.get('project_id')!r}, expected {project_id!r}")
        branches[branch] = {
            "project_id": project_id,
            "key_id": key_id,
            "api_key": str(entry["api_key"]),
            "credential_status": "available",
        }

    argv = [
        "--protocol",
        str(experiment["protocol"]),
        "--domain",
        str(experiment["domain"]),
        "--state-bench-dir",
        str(state_bench_dir),
        "--output-dir",
        str(output_dir),
        "--experiment-id",
        experiment_id,
        "--rounds",
        str(rounds),
        "--train-per-round",
        str(int(training["tasks_per_round"])),
        "--eval-count",
        str(int(evaluation["test_task_count"])),
        "--seed",
        str(seed),
        "--eval-checkpoints",
        _checkpoint_text(evaluation["schedule_checkpoints"], rounds, "schedule_checkpoints"),
        "--baseline-eval-checkpoints",
        _checkpoint_text(evaluation["baseline_checkpoints"], rounds, "baseline_checkpoints"),
        "--evolution-eval-checkpoints",
        _checkpoint_text(evaluation["evolution_checkpoints"], rounds, "evolution_checkpoints"),
        "--checkpoint-runs",
        str(int(evaluation["checkpoint_runs"])),
        "--final-runs",
        str(int(evaluation["final_runs"])),
        "--api-base",
        str(experiment["api_base"]),
        "--baseline-api-key",
        branches["baseline"]["api_key"],
        "--evolution-api-key",
        branches["evolution"]["api_key"],
        "--baseline-project-id",
        branches["baseline"]["project_id"],
        "--evolution-project-id",
        branches["evolution"]["project_id"],
        "--baseline-key-id",
        branches["baseline"]["key_id"],
        "--evolution-key-id",
        branches["evolution"]["key_id"],
        "--agent-model-name",
        str(experiment["agent_model_name"]),
        "--train-workers",
        str(int(training["train_workers"])),
        "--eval-workers",
        str(int(evaluation["eval_workers"])),
        "--branch-order",
        str(experiment.get("branch_order", "baseline-first")),
    ]
    if bool(training.get("no_score", False)):
        argv.append("--no-score")
    metadata = {
        "config": str(DEFAULT_CONFIG),
        "seed": seed,
        "experiment_id": experiment_id,
        "output_dir": str(output_dir),
        "protocol": experiment["protocol"],
        "domain": experiment["domain"],
        "branch_order": str(experiment.get("branch_order", "baseline-first")),
        "branches": {
            branch: {
                "project_id": values["project_id"],
                "key_id": values["key_id"],
                "credential_status": values["credential_status"],
            }
            for branch, values in branches.items()
        },
        "schedule_checkpoints": evaluation["schedule_checkpoints"],
        "baseline_checkpoints": evaluation["baseline_checkpoints"],
        "evolution_checkpoints": evaluation["evolution_checkpoints"],
    }
    return argv, metadata


def add_run_args(parser: argparse.ArgumentParser) -> None:
    """Add the centralized-config runner arguments to an existing parser."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print redacted metadata without starting STATE-Bench.",
    )


def run(config_path: Path, seed: int, *, dry_run: bool = False) -> int:
    """Validate and run one configured StateBench seed."""
    config_path = _resolve(config_path)
    config = _load_config(config_path)
    argv, metadata = build_argv(config, seed, require_credentials=not dry_run)
    metadata["config"] = str(config_path)
    if dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return runner.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one StateBench formal seed from the centralized YAML config.")
    add_run_args(parser)
    args = parser.parse_args(argv)
    return run(args.config, args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
