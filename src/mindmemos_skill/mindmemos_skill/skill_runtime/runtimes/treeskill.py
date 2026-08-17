"""Task-scoped subtree routing for TreeSkill versions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ...typing import Skill, Task
from ..contracts import SkillResourcePayload, SkillRuntime, SkillRuntimeRequest, SkillRuntimeSession

if TYPE_CHECKING:
    from ...algos.trace2skill.treeskill.models import TreeRoutingResult
    from ...algos.trace2skill.treeskill.tree import MarkdownSkillTree


class TreeSkillNodeMetadata(BaseModel):
    """Persisted identity and structure for one Markdown heading node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    level: int = Field(ge=1, le=6)
    heading: str = Field(min_length=1)
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    ordinal: int = Field(ge=0)
    local_content_hash: str = Field(min_length=1)


class TreeSkillRuntimeMetadata(BaseModel):
    """Versioned tree topology stored with an evolved Skill version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Literal[True] = True
    schema_version: Literal[1] = 1
    router: str = Field(default="llm_subtree_v1", min_length=1)
    skill_content_hash: str = Field(min_length=1)
    root_ids: list[str] = Field(default_factory=list)
    nodes: list[TreeSkillNodeMetadata] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_topology(self) -> TreeSkillRuntimeMetadata:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("TreeSkill node IDs must be unique")
        if len(self.root_ids) != len(set(self.root_ids)):
            raise ValueError("TreeSkill root IDs must be unique")

        known = set(node_ids)
        if unknown_roots := set(self.root_ids) - known:
            raise ValueError(f"TreeSkill root IDs are unknown: {sorted(unknown_roots)}")
        expected_roots = [node.id for node in self.nodes if node.parent_id is None]
        if self.root_ids != expected_roots:
            raise ValueError("TreeSkill root IDs must match root nodes in source order")

        by_id = {node.id: node for node in self.nodes}
        for node in self.nodes:
            if len(node.child_ids) != len(set(node.child_ids)):
                raise ValueError(f"TreeSkill node {node.id!r} has duplicate child IDs")
            if node.parent_id is not None and node.parent_id not in known:
                raise ValueError(f"TreeSkill node {node.id!r} has an unknown parent ID")
            if unknown_children := set(node.child_ids) - known:
                raise ValueError(f"TreeSkill node {node.id!r} has unknown child IDs: {sorted(unknown_children)}")
            for child_id in node.child_ids:
                if by_id[child_id].parent_id != node.id:
                    raise ValueError(f"TreeSkill child {child_id!r} does not point back to parent {node.id!r}")
        return self


class TreeSkillRouteResolver(Protocol):
    """Router callback invoked by the generic Skill runtime for each task."""

    async def route(
        self,
        *,
        skill: Skill,
        task: Task,
        env_ref: str = "unknown",
        routing_context: Mapping[str, object] | None = None,
    ) -> TreeRoutingResult: ...


class TreeSkillRuntimeSession(SkillRuntimeSession):
    """One immutable routed TreeSkill projection for one physical attempt."""

    async def _load(self, resource_id: str) -> SkillResourcePayload:
        raise KeyError(f"TreeSkill runtime exposes no lazy resource: {resource_id}")


class TreeSkillRuntime(SkillRuntime):
    """Call a TreeSkill router and project selected subtrees into normal injection."""

    runtime_type = "treeskill"
    metadata_models = {1: TreeSkillRuntimeMetadata}

    def __init__(self, *, router: TreeSkillRouteResolver) -> None:
        self._router = router

    def validate_skill(self, skill: Skill) -> None:
        from ...algos.trace2skill.treeskill.tree import parse_tree_with_metadata

        metadata = TreeSkillRuntimeMetadata.model_validate(
            self.parse_metadata(skill.runtime_schema_version, skill.runtime_metadata)
        )
        parse_tree_with_metadata(skill.content, metadata.model_dump(mode="json"))

    async def on_task(self, request: SkillRuntimeRequest) -> SkillRuntimeSession:
        from ...algos.trace2skill.treeskill.tree import (
            ancestor_closure,
            parse_tree_with_metadata,
            render_selected_subtrees,
        )

        metadata = TreeSkillRuntimeMetadata.model_validate(
            self.parse_metadata(request.skill.runtime_schema_version, request.skill.runtime_metadata)
        )
        tree = parse_tree_with_metadata(request.skill.content, metadata.model_dump(mode="json"))
        routing_context = dict(request.context)
        env_ref = str(routing_context.get("env_ref") or request.task.metadata.get("env_ref") or "unknown")
        result = await self._router.route(
            skill=request.skill,
            task=request.task,
            env_ref=env_ref,
            routing_context=routing_context,
        )
        selected_ids = _validated_content_ids(tree, result)
        content = render_selected_subtrees(tree, selected_ids)
        ancestors = ancestor_closure(tree, selected_ids)
        full_chars = len(tree.full_content)
        trace_metadata: dict[str, JsonValue] = {
            "selected_node_ids": list(result.selected_node_ids),
            "content_node_ids": list(selected_ids),
            "ancestor_node_ids": list(ancestors),
            "fallback_used": result.fallback_used,
            "fallback_reason": result.fallback_reason,
            "full_char_count": full_chars,
            "routed_char_count": len(content),
            "context_saving_ratio": 0.0 if full_chars == 0 else 1.0 - (len(content) / full_chars),
        }
        return TreeSkillRuntimeSession(
            skill=request.skill,
            initial_content=content,
            trace_metadata=trace_metadata,
        )


def _validated_content_ids(tree: MarkdownSkillTree, result: TreeRoutingResult) -> tuple[str, ...]:
    from ...algos.trace2skill.treeskill.tree import collapse_ancestor_descendant_selections

    known = set(tree.node_by_id)
    unknown = set(result.content_node_ids) - known
    if unknown:
        raise ValueError(f"TreeSkill router returned unknown node IDs: {sorted(unknown)}")
    return collapse_ancestor_descendant_selections(tree, result.content_node_ids)


__all__ = [
    "TreeSkillNodeMetadata",
    "TreeSkillRouteResolver",
    "TreeSkillRuntime",
    "TreeSkillRuntimeMetadata",
    "TreeSkillRuntimeSession",
]
