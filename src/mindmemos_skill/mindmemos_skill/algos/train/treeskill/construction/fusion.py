"""Bottom-up node-local fusion over a mutable in-memory Skill tree."""

from __future__ import annotations

import json
import re
from importlib.resources import files

from .....agents import OpenAIChatAgent
from .....common.json import extract_json_object, strict_json_schema_response_format
from ..typing import (
    AppliedEditRecord,
    FusionOutcome,
    LocatedEvidence,
    MarkdownSkillTree,
    NewChildSpec,
    NewTreeNode,
    NodeFusionDecision,
    PatchRecord,
)
from .markdown import (
    compile_tree_metadata,
    create_child_subtree,
    tree_prompt_payload,
    update_node_content,
)

_NEW_CHILD_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["heading", "content", "children"],
    "properties": {
        "heading": {"type": "string"},
        "content": {"type": "string"},
        "children": {"type": "array", "items": {"$ref": "#/$defs/new_child"}},
    },
}

_NODE_FUSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rationale", "edits"],
    "$defs": {"new_child": _NEW_CHILD_SCHEMA},
    "properties": {
        "rationale": {"type": "string"},
        "edits": {
            "type": "array",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "rationale"],
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["update_node", "create_child", "reject"],
                    },
                    "content": {"type": "string"},
                    "new_child": {"$ref": "#/$defs/new_child"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


async def fuse_node(
    tree: MarkdownSkillTree,
    target_id: str,
    evidence: list[LocatedEvidence],
    *,
    agent: OpenAIChatAgent,
) -> FusionOutcome:
    """Apply one node's proposals in memory without persisting artifacts.

    Args:
        tree: Current tree after all preceding node edits.
        target_id: Stable target selected by the trainer's bottom-up ordering.
        evidence: Lessons assigned to this node.
        agent: Direct agent with the fusion generation settings.

    Returns:
        Updated source, stable identities, edit snapshots and model evidence.
    """
    outcome = FusionOutcome(content=tree.full_content, tree_metadata=compile_tree_metadata(tree))
    if target_id not in tree.node_by_id:
        outcome.error = "target node disappeared before fusion"
        return outcome
    messages = [
        {"role": "system", "content": NODE_FUSION_SYSTEM_PROMPT},
        {"role": "user", "content": fusion_user_prompt(tree, target_id, evidence)},
    ]
    for attempt in range(2):
        try:
            response = await agent.run(
                messages=messages,
                response_format=strict_json_schema_response_format("tree_fusion_node_edit", _NODE_FUSION_SCHEMA),
            )
        except Exception as exc:
            outcome.failed_calls += 1
            outcome.error = f"{type(exc).__name__}: {exc}"
            return outcome
        outcome.responses.append(response)
        try:
            decision = NodeFusionDecision.model_validate(extract_json_object(response.output, repair=False))
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            if attempt == 1:
                return outcome
        else:
            outcome.error = ""
            break
    current = tree
    ordered = sorted(
        decision.edits, key=lambda edit: {"update_node": 0, "create_child": 1, "reject": 2}[edit.operation]
    )
    for edit in ordered:
        before = current.full_content
        accepted, message = True, edit.rationale
        if edit.operation != "reject":
            try:
                if edit.operation == "update_node":
                    assert edit.content is not None
                    reason = analysis_artifact_reason(edit.content)
                    if reason:
                        raise ValueError(reason)
                    current = update_node_content(current, target_id, edit.content)
                else:
                    assert edit.new_child is not None
                    reason = _new_child_artifact_reason(edit.new_child)
                    if reason:
                        raise ValueError(reason)
                    current = create_child_subtree(current, target_id, _new_tree_node(edit.new_child))
            except Exception as exc:
                accepted, message = False, f"{type(exc).__name__}: {exc}"
        record = AppliedEditRecord(
            target_node_id=target_id, operation=edit.operation, accepted=accepted, message=message
        )
        outcome.edits.append(record)
        outcome.patches.append(
            PatchRecord(
                before=before,
                after=current.full_content,
                payload={
                    "target_node_id": target_id,
                    "proposal": edit.model_dump(mode="json"),
                    "decision_rationale": decision.rationale,
                    "result": record.model_dump(mode="json"),
                    "evidence": [item.model_dump(mode="json") for item in evidence],
                },
            )
        )
    outcome.content = current.full_content
    outcome.tree_metadata = compile_tree_metadata(current)
    return outcome


def _new_tree_node(spec: NewChildSpec) -> NewTreeNode:
    """Convert a validated proposal to the Markdown insertion structure."""
    return NewTreeNode(
        heading=spec.heading, content=spec.content, children=tuple(_new_tree_node(child) for child in spec.children)
    )


def _new_child_artifact_reason(spec: NewChildSpec) -> str:
    """Reject private answer or run-specific text anywhere in a new subtree."""
    reason = analysis_artifact_reason(spec.heading) or analysis_artifact_reason(spec.content)
    if reason:
        return reason
    for child in spec.children:
        reason = _new_child_artifact_reason(child)
        if reason:
            return reason
    return ""


NODE_FUSION_SYSTEM_PROMPT = (
    files(f"{__package__}.prompts").joinpath("node_fusion_system_prompt.txt").read_text(encoding="utf-8")
)


def fusion_user_prompt(
    tree: MarkdownSkillTree,
    target_node_id: str,
    evidence: list[LocatedEvidence],
) -> str:
    """Fusion user prompt.

    Args:
        tree: Parsed Markdown tree with stable heading identities.
        target_node_id: Existing stable node ID that owns this fusion request.
        evidence: Ordered trajectory-derived evidence available to this stage.
    """
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


_ANALYSIS_ARTIFACT_RULES = (
    (
        "contains a ground-truth or gold-answer reference",
        r"\b(?:ground[\s-]?truth|gold(?:en)?(?:\s+(?:answer|file|output|workbook)|\.xlsx))\b",
    ),
    (
        "contains evaluator-only guidance",
        r"\b(?:evaluation system|evaluator feedback|official evaluator|reference answer)\b",
    ),
    (
        "contains a run-specific absolute path",
        r"(?:^|[\s`\"'])/(?:mnt|home)/[^\s`\"']*/(?:outputs?|runs?)/",
    ),
)


def analysis_artifact_reason(text: str) -> str:
    """Return why analysis-only text must not enter a reusable Skill.

    Args:
        text: Raw model text to parse without changing its contents.
    """

    if not text:
        return ""
    for reason, pattern in _ANALYSIS_ARTIFACT_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return reason
    return ""


__all__ = ["analysis_artifact_reason"]
