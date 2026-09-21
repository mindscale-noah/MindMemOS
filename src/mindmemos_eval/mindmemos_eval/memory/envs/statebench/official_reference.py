"""Run the external STATE-Bench fixed-trajectory reference baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import runner
from .config_runner import _load_config, _resolve
from .paths import CONFIG_ROOT


def add_reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=CONFIG_ROOT / "memory_evaluation_statebench.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--api-key", required=True, help="API key for the isolated official-reference project.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--final-runs", type=int, default=None)


def run(
    config_path: Path,
    seed: int,
    api_key: str,
    *,
    output_dir: Path | None = None,
    trajectory_dir: Path | None = None,
    final_runs: int | None = None,
) -> int:
    config = _load_config(_resolve(config_path))
    experiment = config["experiment"]
    training = config["training"]
    evaluation = config["evaluation"]
    state_bench_dir = _resolve(str(experiment["state_bench_dir"]))
    output_root = output_dir or (
        _resolve(str(experiment["output_dir"])).parent
        / "official-reference"
        / str(experiment["domain"])
        / f"seed{seed}"
    )
    official_dir = trajectory_dir or (
        state_bench_dir / "datasets" / "train_task_trajectories" / str(experiment["domain"])
    )
    rounds = int(training["rounds"])
    argv = [
        "--protocol",
        "official-baseline",
        "--domain",
        str(experiment["domain"]),
        "--state-bench-dir",
        str(state_bench_dir),
        "--output-dir",
        str(_resolve(output_root)),
        "--rounds",
        str(rounds),
        "--train-per-round",
        str(int(training["tasks_per_round"])),
        "--eval-count",
        str(int(evaluation["test_task_count"])),
        "--seed",
        str(seed),
        "--api-base",
        str(experiment["api_base"]),
        "--api-key",
        api_key,
        "--official-trajectory-dir",
        str(_resolve(official_dir)),
        "--agent-model-name",
        str(experiment["agent_model_name"]),
        "--eval-workers",
        str(int(training.get("eval_workers", evaluation.get("eval_workers", 4)))),
        "--final-runs",
        str(final_runs or evaluation.get("final_runs", 1)),
    ]
    return runner.main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_reference_args(parser)
    args = parser.parse_args(argv)
    return run(
        args.config,
        args.seed,
        args.api_key,
        output_dir=args.output_dir,
        trajectory_dir=args.trajectory_dir,
        final_runs=args.final_runs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
