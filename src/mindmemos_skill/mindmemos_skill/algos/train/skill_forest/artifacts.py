"""Validated JSON artifact storage and summaries of physical execution evidence."""

from collections import Counter
from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter

from ....common.json import read_json, write_json
from ....skill.typing import SkillSnapshot
from .config import SkillForestTrainerConfig
from .typing import (
    DecisionRecord,
    ForestRecord,
    MergePoolRecord,
    ModelCall,
    Patch,
    RolloutCall,
    RunStatus,
    RunSummary,
    ScoresRecord,
    SelectionRecord,
    SharedTrainRecord,
    TreeSplit,
    WinnerRecord,
)

ARTIFACT_SCHEMAS = {
    "config.json": TypeAdapter(SkillForestTrainerConfig),
    "splits.json": TypeAdapter(list[TreeSplit]),
    "initial_skill.json": TypeAdapter(SkillSnapshot),
    "selected_skill.json": TypeAdapter(SkillSnapshot),
    "skill.json": TypeAdapter(SkillSnapshot),
    "status.json": TypeAdapter(RunStatus),
    "forest.json": TypeAdapter(ForestRecord),
    "shared_train.json": TypeAdapter(SharedTrainRecord),
    "decision.json": TypeAdapter(DecisionRecord),
    "scores.json": TypeAdapter(ScoresRecord),
    "winners.json": TypeAdapter(list[WinnerRecord]),
    "selection.json": TypeAdapter(SelectionRecord),
    "merge_pool.json": TypeAdapter(MergePoolRecord),
    "parents.json": TypeAdapter(Annotated[list[str], Field(min_length=2, max_length=2)]),
    "patch.json": TypeAdapter(Patch),
    "pre_ranking_patch.json": TypeAdapter(Patch),
    "call.json": TypeAdapter(RolloutCall | ModelCall),
    "summary.json": TypeAdapter(RunSummary),
}
"""Filename-to-schema catalog for every JSON file owned by Skill-Forest."""


def record(path: Path, payload: object) -> None:
    """Validate an artifact and atomically replace its JSON file.

    Args:
        path: Destination whose filename identifies a supported artifact schema.
        payload: JSON-compatible data matching that schema.

    Raises:
        ValueError: The filename is unknown or the payload fails validation.
        OSError: The atomic write fails.
    """
    path = Path(path)
    try:
        schema = ARTIFACT_SCHEMAS[path.name]
    except KeyError as exc:
        raise ValueError(f"Unknown Skill-Forest artifact: {path.name}") from exc
    validated = schema.validate_python(payload)
    if path.name == "config.json":
        payload = validated.model_dump(mode="json", exclude={"optimizer": {"api_key"}})
    else:
        payload = schema.dump_python(validated, mode="json", exclude_unset=True)
    # Preserve omitted defaults and environment extensions in the validated payload.
    write_json(path, payload)


def summarize_run(root: Path) -> dict:
    """Aggregate actual rollout/model counts, usage and candidate decisions.

    Args:
        root: Dedicated run directory containing immutable per-call audit records.

    Returns:
        Counts and durations with explicit missing-usage coverage.
    """
    totals = {}
    for kind in ("rollout", "model"):
        calls = []
        for path in root.rglob("call.json"):
            call = read_json(path)
            ARTIFACT_SCHEMAS["call.json"].validate_python(call)
            if ("task_id" in call) == (kind == "rollout"):
                calls.append(call)
        responses = [call.get("result" if kind == "rollout" else "response", {}) for call in calls]
        totals[kind] = {
            "calls": len(calls),
            "statuses": dict(Counter(call["status"] for call in calls)),
            "summed_call_duration_s": sum(call.get("duration_s", 0) for call in calls),
            "usage_present_calls": sum(bool(response.get("usage")) for response in responses),
            "usage": {
                name: sum(response.get("usage", {}).get(name, 0) for response in responses)
                for name in ("input_tokens", "output_tokens", "total_tokens")
            },
        }
    totals["decisions"] = dict(Counter(read_json(path)["status"] for path in root.glob("nodes/*/decision.json")))
    return totals
