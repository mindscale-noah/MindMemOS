"""Atomic evidence localization against one immutable initial Skill tree."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any

from pydantic import ValidationError

from .....agents import OpenAIChatAgent
from .....common.json import extract_json_object, strict_json_schema_response_format
from ..typing import (
    LocalizationOutcome,
    LocatedEvidence,
    MarkdownSkillTree,
    TrajectoryAnalysisRecord,
    _RawEvidence,
)
from .fusion import analysis_artifact_reason
from .markdown import tree_prompt_payload

_LOCALIZATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["instance_id", "evidence"],
    "properties": {
        "instance_id": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "reusable_lesson", "target_node_id", "rationale"],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "reusable_lesson": {"type": "string"},
                    "target_node_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


async def locate_record(
    tree: MarkdownSkillTree,
    record: TrajectoryAnalysisRecord,
    *,
    agents: tuple[OpenAIChatAgent, OpenAIChatAgent],
) -> LocalizationOutcome:
    """Localize one record, retaining valid items from mixed responses.

    Args:
        tree: Immutable initial tree with stable node identities.
        record: One validated trajectory analysis.
        agents: Direct agents for the initial and increased-budget attempts.

    Returns:
        Localized lessons, rejected-item details and completed model responses.
    """
    outcome = LocalizationOutcome()
    if not record.items:
        return outcome
    messages = [
        {"role": "system", "content": LOCALIZATION_SYSTEM_PROMPT},
        {"role": "user", "content": localization_user_prompt(tree, record)},
    ]
    for agent in agents:
        try:
            response = await agent.run(
                messages=messages,
                response_format=strict_json_schema_response_format("tree_fusion_locator", _LOCALIZATION_SCHEMA),
            )
        except Exception as exc:
            outcome.failed_calls += 1
            outcome.error = f"{type(exc).__name__}: {exc}"
            continue
        outcome.responses.append(response)
        try:
            outcome.evidence, rejected = _parse_localization_items(
                response.output, record=record, known_ids=set(tree.node_by_id)
            )
            outcome.error = "; ".join(rejected) if rejected else None
            return outcome
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


def _parse_localization_items(
    text: str,
    *,
    record: TrajectoryAnalysisRecord,
    known_ids: set[str],
) -> tuple[list[LocatedEvidence], list[str]]:
    payload = extract_json_object(text, repair=False)
    if payload.get("instance_id") != record.instance_id:
        raise ValueError("localization instance_id does not match the analysis record")
    raw_items = payload.get("evidence")
    if not isinstance(raw_items, list):
        raise ValueError("localization evidence must be a list")

    located: list[LocatedEvidence] = []
    rejected: list[str] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_items, start=1):
        try:
            item = _RawEvidence.model_validate(raw)
        except ValidationError as exc:
            rejected.append(f"evidence[{index}] is invalid: {exc.errors()[0]['msg']}")
            continue
        if item.evidence_id in seen_ids:
            rejected.append(f"evidence[{index}] duplicates evidence_id {item.evidence_id!r}")
            continue
        seen_ids.add(item.evidence_id)
        if item.target_node_id not in known_ids:
            rejected.append(f"{item.evidence_id}: unknown target_node_id {item.target_node_id!r}")
            continue
        artifact_reason = analysis_artifact_reason(item.reusable_lesson)
        if artifact_reason:
            rejected.append(f"{item.evidence_id}: {artifact_reason}")
            continue
        located.append(
            LocatedEvidence(
                instance_id=record.instance_id,
                evidence_id=item.evidence_id,
                record_source=record.record_source,
                reusable_lesson=item.reusable_lesson,
                target_node_id=item.target_node_id,
                rationale=item.rationale,
            )
        )
    return located, rejected


__all__ = ["locate_record"]


LOCALIZATION_SYSTEM_PROMPT = (
    files(f"{__package__}.prompts").joinpath("locating_system_prompt.txt").read_text(encoding="utf-8")
)


def localization_user_prompt(tree: MarkdownSkillTree, record: TrajectoryAnalysisRecord) -> str:
    """Localization user prompt.

    Args:
        tree: Parsed Markdown tree with stable heading identities.
        record: Validated analysis record to render for localization.
    """
    payload = {
        "analysis_record": _reference_analysis_record(record),
        "skill_tree": tree_prompt_payload(tree),
    }
    return (
        "Locate trajectory-derived evidence into the Markdown skill tree.\n"
        "Use the recursive skill tree to choose existing fusion target nodes.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _reference_analysis_record(record: TrajectoryAnalysisRecord) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for index, item in enumerate(record.items, start=1):
        number = item.number
        if number is None:
            match = re.search(r"(\d+)$", item.item_id)
            number = int(match.group(1)) if match else index
        rendered: dict[str, Any] = {
            "type": item.kind,
            "number": number,
            "title": item.title,
            "description": item.description,
            "content": item.content,
        }
        if item.kind == "failure_cause":
            rendered["relation_to_skill"] = item.relation_to_skill
        elif item.kind == "failure_memory":
            rendered["skill_reflection"] = item.skill_reflection
        items.append(rendered)
    return {
        "record_source": record.record_source,
        "instance_id": record.instance_id,
        "source_file": record.source_file,
        "items": items,
    }
