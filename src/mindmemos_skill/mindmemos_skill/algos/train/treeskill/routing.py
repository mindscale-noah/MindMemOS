"""One-call LLM subtree routing over persisted TreeSkill metadata."""

from __future__ import annotations

from collections.abc import Mapping

from ....common.json import parse_model
from ....envs.typing import Task
from ....skill import Skill
from .env_specific.spreadsheetbench import routing_user_prompt
from .fusion_tree import (
    TreeMetadataError,
    ancestor_closure,
    collapse_ancestor_descendant_selections,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
)
from .prompts import ROUTING_SYSTEM_PROMPT
from .typing import ChatModel, TreeRoutingResult, _RoutePayload


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
        """init  .

        Args:
            chat_model: Model client implementing the recorded chat request protocol.
            task: Stage audit label or public task supplied by the caller.
            temperature: Sampling temperature passed unchanged to the model.
            max_tokens: Completion-token budget passed to the model.
        """
        self._chat_model = chat_model
        self._task = task
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def route(
        self,
        *,
        skill: Skill,
        task: Task,
        env_ref: str = "unknown",
        routing_context: Mapping[str, object] | None = None,
        tree_metadata: dict | None = None,
    ) -> TreeRoutingResult:
        """Route.

        Args:
            skill: Immutable source Skill being routed for this task.
            task: Stage audit label or public task supplied by the caller.
            env_ref: Environment name retained by the reference router interface.
            routing_context: Optional public spreadsheet preview and task-field overrides.
            tree_metadata: Persisted stable-ID metadata for the complete source Skill.
        """
        metadata = tree_metadata
        try:
            tree = parse_tree_with_metadata(skill.bundle.files["SKILL.md"], metadata)
        except TreeMetadataError as exc:
            return _full_result(skill.bundle.files["SKILL.md"], reason=f"metadata_validation_failed: {exc}")
        if not tree.nodes:
            return _full_result(skill.bundle.files["SKILL.md"], reason="skill_tree_has_no_routable_nodes")

        known_ids = set(tree.node_by_id)

        def parse(text: str) -> _RoutePayload:
            """Parse.

            Args:
                text: Raw model text to parse without changing its contents.
            """
            payload = parse_model(text, _RoutePayload)
            unknown = [node_id for node_id in payload.selected_subtree_ids if node_id not in known_ids]
            if unknown:
                raise ValueError(f"router returned unknown node ids: {sorted(set(unknown))}")
            return payload

        response = await self._chat_model.chat(
            task=self._task,
            messages=[
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": routing_user_prompt(tree, task, routing_context),
                },
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            payload = parse(response.content or "")
        except Exception as exc:
            return _full_result(skill.bundle.files["SKILL.md"], reason=f"router_failed: {type(exc).__name__}: {exc}")

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
