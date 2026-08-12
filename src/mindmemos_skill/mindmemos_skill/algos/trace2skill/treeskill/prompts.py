"""Versioned prompts and renderers used by TreeSkill."""

from __future__ import annotations

import json
import re
from typing import Any

from ..contracts import TraceEvidence
from .models import LocatedEvidence, TrajectoryAnalysisRecord
from .tree import MarkdownSkillTree, tree_prompt_payload

_SENSITIVE_METADATA_KEYS = {
    "api_key",
    "cookie",
    "credentials",
    "data_root",
    "dataset_path",
    "model_path",
    "output_dir",
    "password",
    "running_dir",
    "secret",
    "src_dir",
    "token",
    "workspace",
    "workspace_root",
}
_ABSOLUTE_PATH_RE = re.compile(r"^(?:/|~[/\\]|[A-Za-z]:[/\\])")

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

LOCALIZATION_SYSTEM_PROMPT = """You are a tree-only skill-fusion locator.

Extract reusable lessons from one trajectory-analysis record and assign each lesson to exactly one existing Markdown skill-tree node. Do not invent node ids, edit skill content, or choose edit operations.

Markdown headings define the tree. Each node supplies a stable id, numeric heading level, plain heading, complete local_content directly under that heading, and recursive children. The supplied skill_tree is the complete initial tree and is the source of truth for placement and redundancy checks.

The later fusion step may update only the selected target node's local content, create a direct child/subtree under it, or reject the lesson. Select the most specific existing node whose semantic scope should own the lesson. If no child fits and the lesson should become a new child under a known node, select that prospective parent. Omit lessons already covered by the tree and omit task-specific, noisy, unsupported, ground-truth, evaluator-only, or run-specific material.

One analysis record may yield zero, one, or multiple atomic evidence items. Assign every item exactly one target_node_id and use unique evidence_id values within the record.

Return JSON only:
{
  "instance_id": "<trajectory id>",
  "evidence": [
    {
      "evidence_id": "e1",
      "reusable_lesson": "<one atomic reusable lesson>",
      "target_node_id": "<one existing node id>",
      "rationale": "<why this node is the canonical owner>"
    }
  ]
}"""

NODE_FUSION_SYSTEM_PROMPT = """You are a conservative tree-only skill editor.

You receive one target_node_id, the complete current Markdown skill tree, and reusable evidence already assigned to that target. Similar lessons from different trajectories are independent support; consolidate semantic repetition into one clear rule rather than copying it. Resolve conflicts using the complete current tree and the supplied evidence.

Downstream routing injects a selected node's entire subtree, with ancestors used only for path context. Put generally applicable target-subtree guidance in the target's local content and narrower guidance in a new child/subtree.

Allowed operations:
- update_node: replace only the target node's local content; preserve its heading, id, children, and every other node.
- create_child: add one new direct child node or direct child subtree under the target node.
- reject: make no edit because the evidence is noisy, task-specific, unsafe, unsupported, redundant, or belongs elsewhere.

Use the complete tree to check scope, consistency, and redundancy. Do not edit parents, siblings, descendants, or unrelated nodes. update_node content is replacement local content and must not contain Markdown headings outside fenced code. A target with children may have empty local content; a leaf may not. New headings are plain single-line text without # markers. New content is local content only; represent nested sections through children. Every generated node must be supported by supplied evidence. Do not output ids for new nodes or target_node_id; the runner owns them.

Return at most one update_node and at most one create_child, either alone or together. Return reject alone.

Return JSON only. Normal edit shape:
{
  "rationale": "<overall reason>",
  "edits": [
    {
      "operation": "update_node",
      "content": "<replacement local content>",
      "rationale": "<why this update is safe>"
    },
    {
      "operation": "create_child",
      "new_child": {
        "heading": "<plain heading>",
        "content": "<local content>",
        "children": []
      },
      "rationale": "<why this child/subtree is needed>"
    }
  ]
}

Reject shape:
{
  "rationale": "<overall reason>",
  "edits": [{"operation": "reject", "rationale": "<reason>"}]
}"""

ROUTING_SYSTEM_PROMPT = """You are a tree-only skill retrieval router.

Select every and only existing Markdown skill-tree subtree needed for a complete execution pathway for the current task. Operational completeness takes priority over context reduction. Do not select irrelevant or parent-child redundant nodes and do not invent node ids.

Each node provides its id, numeric heading level, plain heading, complete local_content directly under that heading, and recursive children. Selecting a node selects its whole subtree. Ancestors are added later only as path context; siblings are not automatically included.

The selected roots must collectively cover direct task guidance and supporting inspection, execution, error handling, saving, and verification steps that the task actually needs. Select the most specific useful roots. Select a broad parent only when the whole subtree is needed. Do not select nodes for weak topical similarity. If no node is useful, return an empty list.

Return JSON only:
{
  "selected_subtree_ids": ["<existing node id>"],
  "rationale": "<short reason>"
}"""

SPREADSHEET_ROUTING_GUIDANCE = """Spreadsheet-specific completeness rules:
- Existing-workbook tasks need available loading, modification, and saving guidance.
- Lookup, copying, filling, aggregation, or mapping tasks need available inspection or data-analysis guidance.
- Decide whether the output requires formulas or computed literal values. Do not select formula guidance merely because the task uses a spreadsheet.
- Formula outputs require available formula construction, workbook editing/saving, recalculation, and verification guidance. Verification alone is not executable recalculation.
- Literal-value outputs require inspection, computation, write-back, saving, and verification guidance; do not add formula/recalculation guidance unless formulas are required or must be preserved.
- Select financial-model guidance only for explicitly financial tasks or when existing workbook conventions require it."""


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
        "analysis_record": record.model_dump(mode="json"),
        "skill_tree": tree_prompt_payload(tree),
    }
    return "Locate trajectory-derived evidence into the initial Markdown skill tree.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


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
    return "Fuse the located evidence for this target Markdown node.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def routing_user_prompt(tree: MarkdownSkillTree, task: Any) -> str:
    metadata = _safe_routing_metadata(getattr(task, "metadata", {}))
    payload = {
        "task": {
            "task_id": getattr(task, "task_id", ""),
            "instruction": getattr(task, "instruction", ""),
            "system_prompt": getattr(task, "system_prompt", None),
            "tags": getattr(task, "tags", []),
            "metadata": metadata,
        },
        "skill_tree": tree_prompt_payload(tree),
    }
    return "Route the required skill subtrees for this task.\n\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def _safe_routing_metadata(value: Any, *, key: str | None = None) -> Any:
    """Remove operational paths and credentials from metadata sent to the router."""

    if (key or "").lower() in _SENSITIVE_METADATA_KEYS:
        return None
    if isinstance(value, dict):
        return {
            str(item_key): sanitized
            for item_key, item_value in value.items()
            if (sanitized := _safe_routing_metadata(item_value, key=str(item_key))) is not None
        }
    if isinstance(value, (list, tuple)):
        return [sanitized for item in value if (sanitized := _safe_routing_metadata(item)) is not None]
    if isinstance(value, str) and _ABSOLUTE_PATH_RE.match(value.strip()):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "ANALYSIS_SYSTEM_PROMPT",
    "LOCALIZATION_SYSTEM_PROMPT",
    "NODE_FUSION_SYSTEM_PROMPT",
    "ROUTING_SYSTEM_PROMPT",
    "SPREADSHEET_ROUTING_GUIDANCE",
    "analysis_user_prompt",
    "fusion_user_prompt",
    "localization_user_prompt",
    "routing_user_prompt",
]
