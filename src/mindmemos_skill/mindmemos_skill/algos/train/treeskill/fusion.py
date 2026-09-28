"""Bottom-up node-local fusion over a mutable in-memory Skill tree."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ....common.json import extract_json_object, strict_json_schema_response_format
from .artifacts import record_patch
from .fusion_tree import (
    create_child_subtree,
    update_node_content,
)
from .prompts import NODE_FUSION_SYSTEM_PROMPT, fusion_user_prompt
from .typing import (
    AppliedEditRecord,
    ChatModel,
    FusionFailure,
    LocatedEvidence,
    MarkdownSkillTree,
    NewTreeNode,
)
from .validation import analysis_artifact_reason

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


class TreeSkillNodeFuser:
    """Aggregate evidence by node and apply node edits from leaves upward."""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        task: str,
        temperature: float,
        max_tokens: int,
        audit_dir: Path | None = None,
    ) -> None:
        """init  .

        Args:
            chat_model: Model client implementing the recorded chat request protocol.
            task: Stage audit label or public task supplied by the caller.
            temperature: Sampling temperature passed unchanged to the model.
            max_tokens: Completion-token budget passed to the model.
            audit_dir: Optional experiment directory for individual patch records.
        """
        self._chat_model = chat_model
        self._task = task
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._audit_dir = audit_dir

    async def fuse(
        self,
        initial_tree: MarkdownSkillTree,
        evidence: list[LocatedEvidence],
    ) -> tuple[MarkdownSkillTree, list[AppliedEditRecord], list[FusionFailure]]:
        """Fuse.

        Args:
            initial_tree: Immutable tree at the start of this fusion transaction.
            evidence: Ordered trajectory-derived evidence available to this stage.
        """
        grouped: dict[str, list[LocatedEvidence]] = defaultdict(list)
        for item in evidence:
            grouped[item.target_node_id].append(item)

        initial_by_id = initial_tree.node_by_id
        target_ids = sorted(
            grouped,
            key=lambda node_id: (
                -len(initial_by_id[node_id].heading_path),
                initial_by_id[node_id].source_line,
            ),
        )
        current = initial_tree
        records: list[AppliedEditRecord] = []
        failures: list[FusionFailure] = []

        for target_id in target_ids:
            if target_id not in current.node_by_id:
                failures.append(FusionFailure(target_node_id=target_id, error="target node disappeared before fusion"))
                continue
            messages = [
                {"role": "system", "content": NODE_FUSION_SYSTEM_PROMPT},
                {"role": "user", "content": fusion_user_prompt(current, target_id, grouped[target_id])},
            ]
            try:
                response = await self._chat_model.chat(
                    task=self._task,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format=strict_json_schema_response_format(
                        "tree_fusion_node_edit",
                        _NODE_FUSION_SCHEMA,
                    ),
                )
            except Exception as exc:
                failures.append(FusionFailure(target_node_id=target_id, error=f"{type(exc).__name__}: {exc}"))
                continue
            try:
                payload = extract_json_object(response.content or "", repair=False)
                ordered_edits = normalize_node_fusion_edits(payload, target_id)
            except Exception as exc:
                failures.append(FusionFailure(target_node_id=target_id, error=f"{type(exc).__name__}: {exc}"))
                continue

            for edit in ordered_edits:
                before = current.full_content
                try:
                    if edit["operation"] == "reject":
                        records.append(
                            AppliedEditRecord(
                                target_node_id=target_id,
                                operation="reject",
                                accepted=True,
                                message=str(edit.get("rationale", "")),
                            )
                        )
                        continue
                    try:
                        if edit["operation"] == "update_node":
                            artifact_reason = analysis_artifact_reason(edit["content"])
                            if artifact_reason:
                                records.append(
                                    AppliedEditRecord(
                                        target_node_id=target_id,
                                        operation=edit["operation"],
                                        accepted=False,
                                        message=artifact_reason,
                                    )
                                )
                                continue
                            current = update_node_content(current, target_id, edit["content"])
                        else:
                            if str(edit.get("parent_node_id") or target_id).strip() != target_id:
                                raise ValueError("parent_node_id must be the target node for create_child")
                            child = _new_tree_node(edit["new_child"])
                            artifact_reason = _new_child_artifact_reason(child)
                            if artifact_reason:
                                records.append(
                                    AppliedEditRecord(
                                        target_node_id=target_id,
                                        operation=edit["operation"],
                                        accepted=False,
                                        message=artifact_reason,
                                    )
                                )
                                continue
                            current = create_child_subtree(current, target_id, child)
                    except Exception as exc:
                        records.append(
                            AppliedEditRecord(
                                target_node_id=target_id,
                                operation=edit["operation"],
                                accepted=False,
                                message=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue
                    records.append(
                        AppliedEditRecord(
                            target_node_id=target_id,
                            operation=edit["operation"],
                            accepted=True,
                            message=str(edit.get("rationale", "")),
                        )
                    )
                finally:
                    if self._audit_dir is not None:
                        record_patch(
                            self._audit_dir / f"{len(records):04d}-{target_id}-{edit['operation']}",
                            before=before,
                            after=current.full_content,
                            payload={
                                "target_node_id": target_id,
                                "proposal": edit,
                                "decision_rationale": str(payload.get("rationale", "")),
                                "result": records[-1].model_dump(mode="json"),
                                "evidence": [item.model_dump(mode="json") for item in grouped[target_id]],
                            },
                        )
        return current, records, failures


def normalize_node_fusion_edits(payload: dict[str, Any], target_node_id: str) -> list[dict[str, Any]]:
    """Normalize model edits using the upstream tree-only fusion checks.

    Args:
        payload: Decoded model JSON, including optional audit-only fields.
        target_node_id: Runner-owned target that overrides any model-supplied ID.

    Returns:
        Ordered edits, or one rejection explaining an invalid operation set.
    """

    # Ported from TreeSkill-colleague-share bc3b0a9, tree_only_fusion.py.
    def reject(reason: str) -> list[dict[str, Any]]:
        return [{"operation": "reject", "target_node_id": target_node_id, "rationale": reason}]

    raw_edits = payload.get("edits") if isinstance(payload.get("edits"), list) else []
    if not raw_edits:
        return reject("node fusion returned no edits")
    if len(raw_edits) > 2:
        return reject("node fusion returned more than two edits")
    normalized = []
    seen = set()
    for raw_edit in raw_edits:
        if not isinstance(raw_edit, dict):
            return reject("node fusion edit is not an object")
        operation = str(raw_edit.get("operation") or "reject").strip().lower()
        if operation == "update":
            operation = "update_node"
        if operation not in {"update_node", "create_child", "reject"}:
            return reject(f"unsupported node fusion operation: {operation}")
        edit = dict(raw_edit, operation=operation, target_node_id=target_node_id)
        if operation == "reject":
            if len(raw_edits) != 1:
                return reject("reject cannot be mixed with other edits")
            return [edit]
        if operation in seen:
            return reject(f"duplicate {operation} edit")
        seen.add(operation)
        if operation == "update_node":
            if "content" not in edit or not isinstance(edit.get("content"), str):
                return reject("update_node requires string content")
            if "new_child" in edit:
                return reject("update_node must not include new_child")
        else:
            if not isinstance(edit.get("new_child"), dict):
                return reject("create_child requires new_child")
            if "content" in edit:
                return reject("create_child must not include top-level content")
        normalized.append(edit)
    return sorted(normalized, key=lambda edit: {"update_node": 0, "create_child": 1}[edit["operation"]])


def _new_tree_node(spec: dict[str, Any]) -> NewTreeNode:
    """Read upstream subtree fields, retaining extra fields only in the audit."""
    children = spec.get("children") or []
    if not isinstance(children, list):
        raise ValueError("children must be a list")
    if any(not isinstance(child, dict) for child in children):
        raise ValueError("children must contain objects")
    return NewTreeNode(
        heading=str(spec.get("heading") or ""),
        content=str(spec.get("content") or ""),
        children=tuple(_new_tree_node(child) for child in children),
    )


def _new_child_artifact_reason(spec: NewTreeNode) -> str:
    reason = analysis_artifact_reason(spec.heading) or analysis_artifact_reason(spec.content)
    if reason:
        return reason
    for child in spec.children:
        reason = _new_child_artifact_reason(child)
        if reason:
            return reason
    return ""


__all__ = ["TreeSkillNodeFuser"]
