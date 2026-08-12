"""Bottom-up node-local fusion over a mutable in-memory Skill tree."""

from __future__ import annotations

from collections import defaultdict

from .analysis import ChatModel
from .json_utils import parse_model, strict_json_schema_response_format
from .models import (
    AppliedEditRecord,
    FusionFailure,
    LocatedEvidence,
    NewChildSpec,
    NodeFusionDecision,
)
from .prompts import NODE_FUSION_SYSTEM_PROMPT, fusion_user_prompt
from .tree import (
    MarkdownSkillTree,
    NewTreeNode,
    create_child_subtree,
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


class TreeSkillNodeFuser:
    """Aggregate evidence by node and apply node edits from leaves upward."""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        task: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._chat_model = chat_model
        self._task = task
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def fuse(
        self,
        initial_tree: MarkdownSkillTree,
        evidence: list[LocatedEvidence],
    ) -> tuple[MarkdownSkillTree, list[AppliedEditRecord], list[FusionFailure]]:
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
                decision = parse_model(response.content or "", NodeFusionDecision)
            except Exception as exc:
                failures.append(FusionFailure(target_node_id=target_id, error=f"{type(exc).__name__}: {exc}"))
                continue

            ordered_edits = sorted(
                decision.edits,
                key=lambda edit: {"update_node": 0, "create_child": 1, "reject": 2}[edit.operation],
            )
            for edit in ordered_edits:
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
                        current = update_node_content(current, target_id, edit.content)
                    else:
                        assert edit.new_child is not None
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
        return current, records, failures


def _new_tree_node(spec: NewChildSpec) -> NewTreeNode:
    return NewTreeNode(
        heading=spec.heading,
        content=spec.content,
        children=tuple(_new_tree_node(child) for child in spec.children),
    )


__all__ = ["TreeSkillNodeFuser"]
