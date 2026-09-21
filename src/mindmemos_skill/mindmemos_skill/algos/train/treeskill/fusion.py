"""Bottom-up node-local fusion over a mutable in-memory Skill tree."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ....common.json import parse_model, strict_json_schema_response_format
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
    NewChildSpec,
    NewTreeNode,
    NodeFusionDecision,
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
                decision = parse_model(response.content or "", NodeFusionDecision)
            except Exception as exc:
                failures.append(FusionFailure(target_node_id=target_id, error=f"{type(exc).__name__}: {exc}"))
                continue

            ordered_edits = sorted(
                decision.edits,
                key=lambda edit: {"update_node": 0, "create_child": 1, "reject": 2}[edit.operation],
            )
            for edit in ordered_edits:
                before = current.full_content
                try:
                    if edit.operation == "reject":
                        records.append(
                            AppliedEditRecord(
                                target_node_id=target_id,
                                operation="reject",
                                accepted=True,
                                message=edit.rationale,
                            )
                        )
                        continue
                    try:
                        if edit.operation == "update_node":
                            assert edit.content is not None
                            artifact_reason = analysis_artifact_reason(edit.content)
                            if artifact_reason:
                                records.append(
                                    AppliedEditRecord(
                                        target_node_id=target_id,
                                        operation=edit.operation,
                                        accepted=False,
                                        message=artifact_reason,
                                    )
                                )
                                continue
                            current = update_node_content(current, target_id, edit.content)
                        else:
                            assert edit.new_child is not None
                            artifact_reason = _new_child_artifact_reason(edit.new_child)
                            if artifact_reason:
                                records.append(
                                    AppliedEditRecord(
                                        target_node_id=target_id,
                                        operation=edit.operation,
                                        accepted=False,
                                        message=artifact_reason,
                                    )
                                )
                                continue
                            current = create_child_subtree(current, target_id, _new_tree_node(edit.new_child))
                    except Exception as exc:
                        records.append(
                            AppliedEditRecord(
                                target_node_id=target_id,
                                operation=edit.operation,
                                accepted=False,
                                message=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue
                    records.append(
                        AppliedEditRecord(
                            target_node_id=target_id,
                            operation=edit.operation,
                            accepted=True,
                            message=edit.rationale,
                        )
                    )
                finally:
                    if self._audit_dir is not None:
                        record_patch(
                            self._audit_dir / f"{len(records):04d}-{target_id}-{edit.operation}",
                            before=before,
                            after=current.full_content,
                            payload={
                                "target_node_id": target_id,
                                "proposal": edit.model_dump(mode="json"),
                                "decision_rationale": decision.rationale,
                                "result": records[-1].model_dump(mode="json"),
                                "evidence": [item.model_dump(mode="json") for item in grouped[target_id]],
                            },
                        )
        return current, records, failures


def _new_tree_node(spec: NewChildSpec) -> NewTreeNode:
    return NewTreeNode(
        heading=spec.heading,
        content=spec.content,
        children=tuple(_new_tree_node(child) for child in spec.children),
    )


def _new_child_artifact_reason(spec: NewChildSpec) -> str:
    reason = analysis_artifact_reason(spec.heading) or analysis_artifact_reason(spec.content)
    if reason:
        return reason
    for child in spec.children:
        reason = _new_child_artifact_reason(child)
        if reason:
            return reason
    return ""


__all__ = ["TreeSkillNodeFuser"]
