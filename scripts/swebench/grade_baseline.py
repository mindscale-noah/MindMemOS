"""Grade all fifty held-out patches with the pinned official SWE-bench harness."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from collect_train import export_training
from mindmemos_eval.swebench.config import load_config
from mindmemos_eval.swebench.data import load_split, write_json


def grade(config_path: Path, harness_python: Path) -> dict:
    """Run official tests serially and require complete denominator accounting.

    Args:
        config_path: Frozen experiment YAML path.
        harness_python: Python executable with swebench==4.1.0 installed.

    Returns:
        Audited official accuracy, or a report explicitly marked incomplete.
    """
    config = load_config(config_path)
    split = load_split(config)
    if not export_training(config, "baseline")["collection_complete"]:
        raise RuntimeError("All fifty baseline rollouts must be complete before grading")
    predictions = (config.output_dir / "baseline" / "predictions.jsonl").resolve()
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    expected = {task.instance_id for task in split.test}
    if len(rows) != 50 or {row["instance_id"] for row in rows} != expected:
        raise ValueError("Predictions must cover exactly the frozen test split")
    directory = (config.output_dir / "grading").resolve()
    directory.mkdir(exist_ok=True)
    run_id = (
        "baseline-provider-default" if config.parent.thinking is None else f"baseline-thinking-{config.parent.thinking}"
    )
    command = [
        str(harness_python.resolve()),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str((config.output_dir / "grading-dataset.json").resolve()),
        "--predictions_path",
        str(predictions),
        "--max_workers",
        "1",
        "--run_id",
        run_id,
        "--namespace",
        "swebench",
        "--cache_level",
        "none",
        "--clean",
        "False",
        "--timeout",
        "1800",
        "--report_dir",
        str(directory),
    ]
    write_json(
        directory / "invocation.json",
        {
            "command": command,
            "swebench_version": "4.1.0",
            "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        },
    )
    with (directory / "harness.log").open("a") as log:
        result = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
    report_path = directory / f"{rows[0]['model_name_or_path'].replace('/', '__')}.{run_id}.json"
    if result.returncode or not report_path.exists():
        raise RuntimeError("Official harness failed; inspect grading/harness.log")
    report = json.loads(report_path.read_text())
    completed = set(report["completed_ids"])
    empty = set(report["empty_patch_ids"])
    errors = set(report["error_ids"])
    resolved = set(report["resolved_ids"])
    complete = not errors and completed | empty == expected and resolved <= completed
    summary = {
        "expected": 50,
        "resolved": len(resolved),
        "officially_graded": True,
        "grading_complete": complete,
        "accuracy": len(resolved) / 50 if complete else None,
        "completed_ids": sorted(completed),
        "empty_patch_ids": sorted(empty),
        "error_ids": sorted(errors),
        "missing_ids": sorted(expected - completed - empty),
        "resolved_ids": sorted(resolved),
        "official_report": str(report_path),
        "swebench_version": "4.1.0",
    }
    write_json(config.output_dir / "baseline-accuracy.json", summary)
    return summary


def main() -> int:
    """Grade the frozen baseline and report whether all tasks were accounted for.

    Returns:
        Zero for complete grading, otherwise two.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--harness-python", type=Path, required=True)
    args = parser.parse_args()
    report = grade(args.config, args.harness_python)
    print(json.dumps(report, indent=2))
    return 0 if report["grading_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
