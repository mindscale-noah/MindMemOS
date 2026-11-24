"""Atomic experiment artifacts and reference text normalization."""

import difflib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, JsonValue

from ....common.json import read_json, write_json
from ....common.secrets import redact_credentials
from .config import TreeSkillTrainerConfig
from .typing import TrainingRollout


def record_patch(directory: Path, *, before: str, after: str, payload: dict) -> None:
    """Save a proposed edit, its disposition, full snapshots, and a unified diff.

    Args:
        directory: Unique directory for one node edit.
        before: Complete source before this edit.
        after: Complete source after acceptance or rejection.
        payload: Proposal, evidence and validation result.
    """
    write_json(directory / "patch.json", payload)
    (directory / "before.md").write_text(before, encoding="utf-8")
    (directory / "after.md").write_text(after, encoding="utf-8")
    (directory / "change.diff").write_text(
        "".join(
            difflib.unified_diff(
                before.splitlines(True), after.splitlines(True), fromfile="before.md", tofile="after.md"
            )
        ),
        encoding="utf-8",
    )


def experiment_config(trainer_config: TreeSkillTrainerConfig) -> dict[str, JsonValue]:
    """Serialize algorithm settings with credentials masked for experiment records.

    Args:
        trainer_config: Validated algorithm and provider settings.

    Returns:
        JSON-compatible algorithm configuration with redacted credentials.
    """
    return {
        "algorithm": redact_credentials(trainer_config.model_dump(mode="json")),
    }


def model_usage(directory: Path) -> dict:
    """Aggregate trainer-owned item records without counting checkpoint copies.

    Args:
        directory: Run root containing item and routing outcomes.

    Returns:
        Logical call counts, reported tokens and explicit usage coverage.
    """
    training = directory / "training" if (directory / "training").is_dir() else directory
    paths = [
        *training.glob("items/*/*/attempt-*/outcome.json"),
        *training.glob("rollouts/train/*/attempt-*/routing/outcome.json"),
        *directory.glob("test/*/attempt-*/routing/outcome.json"),
    ]
    outcomes = [read_json(path) for path in sorted(paths)]
    responses = [response for outcome in outcomes for response in outcome.get("responses", [])]
    failed = sum(outcome.get("failed_calls", 0) for outcome in outcomes)
    interrupted = sum(outcome.get("interrupted", False) for outcome in outcomes)
    reported = sum(bool(response.get("usage")) for response in responses)
    return {
        "calls": len(responses) + failed,
        "failed_calls": failed,
        "usage_reported_calls": reported,
        "usage_complete": failed == 0 and interrupted == 0 and reported == len(responses),
        "interrupted_items": interrupted,
        "scope": "logical_agent_calls",
        **{
            key: sum(int(response.get("usage", {}).get(key, 0)) for response in responses)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
    }


def score_report(
    records: Sequence[TrainingRollout], failures: Sequence[dict[str, JsonValue]], expected: int
) -> dict[str, JsonValue]:
    """Report environment scores with explicit execution coverage.

    Args:
        records: Completed rollouts with actual rewards.
        failures: Failed execution slots, never substituted with zero rewards.
        expected: Requested number of physical sampling slots.

    Returns:
        Mean score and success fraction over completed slots, with denominators.
    """
    count = len(records)
    return {
        "expected": expected,
        "completed": count,
        "failed": len(failures),
        "skipped": len(failures),
        "score_sum": sum(record.result.reward.score for record in records),
        "mean_score": sum(record.result.reward.score for record in records) / count if count else None,
        "success_count": sum(record.result.reward.success for record in records),
        "success_rate": sum(record.result.reward.success for record in records) / count if count else None,
        "complete": count == expected,
        "rollout_duration_s": sum(record.result.duration_s for record in records),
        "usage": {
            "input_tokens": sum(record.result.usage.input_tokens for record in records),
            "output_tokens": sum(record.result.usage.output_tokens for record in records),
            "total_tokens": sum(record.result.usage.total_tokens for record in records),
        },
    }


def record_training_stage(
    root: Path,
    name: Literal["evidence", "analysis", "localization", "fusion"],
    records: Sequence[BaseModel],
    failures: Sequence[BaseModel | str] = (),
) -> None:
    """Save one stage's validated records and independent item failures.

    Args:
        root: Experiment training directory.
        name: Stage owning this artifact's fixed JSON format.
        records: Evidence, analyses, localized lessons or applied edits for this stage.
        failures: Structured item failures or failed analysis trajectory IDs.
    """
    values = [item.model_dump(mode="json") for item in records]
    errors = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in failures]
    keys = {"analysis": "records", "localization": "evidence", "fusion": "edits"}
    write_json(
        root / "stages" / f"{name}.json", values if name == "evidence" else {keys[name]: values, "failures": errors}
    )


def record_experiment_report(directory: Path, summary: dict[str, JsonValue]) -> None:
    """Save the current training or held-out experiment report atomically.

    Args:
        directory: Run root receiving summary.json.
        summary: Report with status, stage timings, coverage denominators, training
            metrics and optional held-out scores; unavailable scores remain None.
    """
    write_json(directory / "summary.json", summary)
