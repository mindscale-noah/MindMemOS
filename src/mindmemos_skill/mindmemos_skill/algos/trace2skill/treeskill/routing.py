"""One-call LLM subtree routing over persisted TreeSkill metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from ....typing import Skill, Task
from .analysis import ChatModel
from .json_utils import parse_model
from .models import TreeRoutingResult
from .prompts import (
    ROUTING_SYSTEM_PROMPT,
    SPREADSHEET_ROUTING_GUIDANCE,
    routing_user_prompt,
)
from .tree import (
    TreeMetadataError,
    ancestor_closure,
    collapse_ancestor_descendant_selections,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
)


class _RoutePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_subtree_ids: tuple[str, ...] = ()
    rationale: str = ""

    @model_validator(mode="after")
    def validate_unique_ids(self) -> _RoutePayload:
        if len(self.selected_subtree_ids) != len(set(self.selected_subtree_ids)):
            raise ValueError("selected_subtree_ids must not contain duplicates")
        return self


class TreeSkillRouter:
    """Route one persisted TreeSkill once for one physical task."""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        task: str = "treeskill_subtree_routing",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._chat_model = chat_model
        self._task = task
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def route(self, *, skill: Skill, task: Task, env_ref: str = "unknown") -> TreeRoutingResult:
        metadata = skill.metadata.get("treeskill")
        try:
            tree = parse_tree_with_metadata(skill.content, metadata)
        except TreeMetadataError as exc:
            return _full_result(skill.content, reason=f"metadata_validation_failed: {exc}")
        if not tree.nodes:
            return _full_result(skill.content, reason="skill_tree_has_no_routable_nodes")

        known_ids = set(tree.node_by_id)

        def parse(text: str) -> _RoutePayload:
            payload = parse_model(text, _RoutePayload)
            unknown = [node_id for node_id in payload.selected_subtree_ids if node_id not in known_ids]
            if unknown:
                raise ValueError(f"router returned unknown node ids: {sorted(set(unknown))}")
            return payload

        system_prompt = ROUTING_SYSTEM_PROMPT
        if env_ref == "spreadsheetbench" or task.metadata.get("benchmark") == "SpreadsheetBench":
            system_prompt = f"{system_prompt}\n\n{SPREADSHEET_ROUTING_GUIDANCE}"
        try:
            response = await self._chat_model.chat(
                task=self._task,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": routing_user_prompt(tree, task)},
                ],
                format_parser=parse,
                feedback_on_parse_error=True,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            payload = getattr(response, "parsed", None) or parse(response.content or "")
        except Exception as exc:
            return _full_result(skill.content, reason=f"router_failed: {type(exc).__name__}: {exc}")

        content_ids = collapse_ancestor_descendant_selections(tree, payload.selected_subtree_ids)
        if not content_ids:
            return TreeRoutingResult(
                selected_node_ids=(),
                content_node_ids=(),
                ancestor_node_ids=(),
                skill_content="",
                fallback_used=False,
                fallback_reason="router_selected_no_skill",
                full_char_count=len(tree.full_content),
                routed_char_count=0,
            )
        ancestors = ancestor_closure(tree, content_ids)
        selected = tuple(node.node_id for node in tree.nodes if node.node_id in {*ancestors, *content_ids})
        content = render_selected_subtrees(tree, content_ids)
        return TreeRoutingResult(
            selected_node_ids=selected,
            content_node_ids=content_ids,
            ancestor_node_ids=ancestors,
            skill_content=content,
            fallback_used=False,
            fallback_reason="",
            full_char_count=len(tree.full_content),
            routed_char_count=len(content),
        )


def _full_result(content: str, *, reason: str) -> TreeRoutingResult:
    all_ids = tuple(node.node_id for node in parse_skill_markdown(content).nodes)
    return TreeRoutingResult(
        selected_node_ids=all_ids,
        content_node_ids=all_ids,
        ancestor_node_ids=(),
        skill_content=content,
        fallback_used=True,
        fallback_reason=reason,
        full_char_count=len(content),
        routed_char_count=len(content),
    )


__all__ = ["TreeSkillRouter"]
