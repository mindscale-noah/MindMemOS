"""Versioned prompts and renderers used by TreeSkill."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

from ..contracts import TraceEvidence
from .models import LocatedEvidence, TrajectoryAnalysisRecord
from .tree import MarkdownSkillTree, tree_prompt_payload


def _load_prompt_template(name: str, *, strip_trailing: bool = False) -> str:
    text = files(f"{__package__}.prompt_templates").joinpath(name).read_text(encoding="utf-8")
    return text.rstrip() if strip_trailing else text


ANALYSIS_SYSTEM_PROMPT = """You analyze one completed agent trajectory for reusable skill evidence.

Use the task, transcript, execution outcome, and evaluator feedback provided by the user. Extract only lessons supported by the trajectory. Separate distinct lessons into separate items. A successful trajectory may contain failed intermediate attempts; preserve only reusable behavior that contributed to recovery or success. A failed trajectory may still contain useful partial behavior, but do not present an unverified action as reliable guidance.

Do not include task-specific values, filenames, exact answers, private paths, ground-truth content, evaluator-only instructions, or unsupported conclusions. Return an empty items list when the trajectory contains no reusable evidence.

Return JSON only:
{
  "instance_id": "<trajectory id>",
  "task_id": "<task id>",
  "record_source": "error|success|unlabeled",
  "items": [
    {
      "item_id": "i1",
      "kind": "failure_cause|failure_memory|success_memory|unlabeled_memory",
      "title": "<short title>",
      "description": "<evidence-grounded explanation>",
      "content": "<concise reusable lesson>"
    }
  ]
}"""

LOCALIZATION_SYSTEM_PROMPT = _load_prompt_template("locating_system_prompt.txt")
NODE_FUSION_SYSTEM_PROMPT = _load_prompt_template("node_fusion_system_prompt.txt")
ROUTING_SYSTEM_PROMPT = _load_prompt_template(
    "tree_only_skill_routing_system_prompt.txt",
    strip_trailing=True,
)


def analysis_user_prompt(evidence: TraceEvidence, *, source: str) -> str:
    payload = {
        "instance_id": evidence.trajectory_id,
        "task_id": evidence.task_id,
        "record_source": source,
        "score": evidence.score,
        "evaluator_feedback": evidence.annotation_detail,
        "annotation_metadata": evidence.annotation_metadata,
        "trajectory": evidence.transcript,
    }
    return "Analyze this trajectory.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def localization_user_prompt(tree: MarkdownSkillTree, record: TrajectoryAnalysisRecord) -> str:
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


def fusion_user_prompt(
    tree: MarkdownSkillTree,
    target_node_id: str,
    evidence: list[LocatedEvidence],
) -> str:
    payload = {
        "target_node_id": target_node_id,
        "skill_tree": tree_prompt_payload(tree),
        "located_evidence": [
            {
                "instance_id": item.instance_id,
                "evidence_id": item.evidence_id,
                "record_source": item.record_source,
                "reusable_lesson": item.reusable_lesson,
                "target_node_id": item.target_node_id,
            }
            for item in evidence
        ],
    }
    return (
        "Fuse the located evidence for this target Markdown node.\n"
        "Use the complete recursive skill tree to avoid redundant or misplaced edits.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def routing_user_prompt(
    tree: MarkdownSkillTree,
    task: Any,
    routing_context: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "task": {
            "instance_id": _routing_value(routing_context, task, "instance_id", "task_id"),
            "instruction_type": _routing_value(routing_context, task, "instruction_type"),
            "answer_position": _routing_value(routing_context, task, "answer_position"),
            "instruction": _routing_value(routing_context, task, "instruction", "instruction"),
            "spreadsheet_content": _truncate(
                str(_routing_value(routing_context, task, "spreadsheet_content") or ""),
                5000,
            ),
        },
        "skill_tree": tree_prompt_payload(tree),
    }
    return f"Route skill subtrees for this spreadsheet task.\n\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def _routing_value(
    routing_context: Mapping[str, Any] | None,
    task: Any,
    key: str,
    task_attribute: str | None = None,
) -> Any:
    if routing_context is not None and key in routing_context:
        return routing_context[key]
    metadata = getattr(task, "metadata", {})
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    return getattr(task, task_attribute, "") if task_attribute else ""


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


__all__ = [
    "ANALYSIS_SYSTEM_PROMPT",
    "LOCALIZATION_SYSTEM_PROMPT",
    "NODE_FUSION_SYSTEM_PROMPT",
    "ROUTING_SYSTEM_PROMPT",
    "analysis_user_prompt",
    "fusion_user_prompt",
    "localization_user_prompt",
    "routing_user_prompt",
]
