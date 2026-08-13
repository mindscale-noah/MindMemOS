"""Data structures for the Markdown Skill tree."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_HEADING_PATTERN = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*(?:\r?\n)?$")
_FENCE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_NODE_ID_NAMESPACE = UUID("bb97c637-342e-49a9-8fc0-938b29af6818")


class NewSkillTreeNode(BaseModel):
    """One new node or recursive subtree proposed by the optimizer model."""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(min_length=1)
    content: str = ""
    children: list[NewSkillTreeNode] = Field(default_factory=list)

    @field_validator("heading")
    @classmethod
    def validate_heading(cls, value: str) -> str:
        heading = value.strip()
        if not heading or "\n" in heading or "\r" in heading:
            raise ValueError("heading must be one non-empty line")
        return heading


class SkillTreeEdit(BaseModel):
    """Parsed JSON edit emitted by the optimizer model."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["edit_node", "append_node", "create_child", "reject"]
    target_node_id: str | None = None
    find: str | None = None
    replace: str | None = None
    content: str | None = None
    child: NewSkillTreeNode | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_operation_fields(self) -> SkillTreeEdit:
        if self.operation == "edit_node":
            if self.target_node_id is None or not self.find or self.replace is None:
                raise ValueError("edit_node requires target_node_id, non-empty find, and replace")
            if self.content is not None or self.child is not None:
                raise ValueError("edit_node must not include content or child")
        elif self.operation == "append_node":
            if self.target_node_id is None or self.content is None or not self.content.strip():
                raise ValueError("append_node requires target_node_id and non-empty content")
            if self.find is not None or self.replace is not None or self.child is not None:
                raise ValueError("append_node must not include find, replace, or child")
        elif self.operation == "create_child":
            if self.target_node_id is None or self.child is None:
                raise ValueError("create_child requires target_node_id and child")
            if self.find is not None or self.replace is not None or self.content is not None:
                raise ValueError("create_child must not include find, replace, or content")
        elif any(
            value is not None for value in (self.target_node_id, self.find, self.replace, self.content, self.child)
        ):
            raise ValueError("reject must not include edit fields")
        return self


class TreeEditProposal(BaseModel):
    """One task-local edit proposal produced against a frozen batch tree."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    base_tree_hash: str = Field(min_length=1)
    outcome_pattern: Literal["failure_to_success", "repeated_failure"]
    edit: SkillTreeEdit


class TreeFusionResult(BaseModel):
    """Complete canonical direct content and new children for one target node."""

    model_config = ConfigDict(extra="forbid")

    content: str
    children: list[NewSkillTreeNode] = Field(default_factory=list)


@dataclass(slots=True)
class SkillTreeNode:
    """One Markdown heading and the local content directly under it."""

    node_id: str
    level: int
    heading: str
    content: str = ""
    children: list[SkillTreeNode] = field(default_factory=list)


@dataclass(slots=True)
class SkillTree:
    """A Skill represented by its Markdown heading hierarchy."""

    preamble: str
    roots: list[SkillTreeNode]

    @classmethod
    def from_markdown(cls, markdown: str) -> SkillTree:
        """Build a tree from ATX headings outside fenced code blocks."""

        roots: list[SkillTreeNode] = []
        stack: list[SkillTreeNode] = []
        preamble: list[str] = []
        local_content: dict[str, list[str]] = {}
        current: SkillTreeNode | None = None
        fence_marker: str | None = None
        used_node_ids: set[str] = set()

        for line in markdown.splitlines(keepends=True):
            fence = _FENCE_PATTERN.match(line)
            if fence:
                marker = fence.group(1)
                if fence_marker is None:
                    fence_marker = marker
                elif marker[0] == fence_marker[0] and len(marker) >= len(fence_marker):
                    fence_marker = None

            heading_match = None if fence_marker is not None or fence else _HEADING_PATTERN.match(line)
            if heading_match is None:
                target = preamble if current is None else local_content[current.node_id]
                target.append(line)
                continue

            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            while stack and stack[-1].level >= level:
                stack.pop()
            siblings = stack[-1].children if stack else roots
            occurrence = sum(item.heading == heading for item in siblings) + 1
            parent_node_id = stack[-1].node_id if stack else "root"
            node = SkillTreeNode(
                node_id=cls._allocate_node_id(
                    parent_node_id=parent_node_id,
                    heading=heading,
                    occurrence=occurrence,
                    used_node_ids=used_node_ids,
                ),
                level=level,
                heading=heading,
            )
            local_content[node.node_id] = []

            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
            current = node

        def assign_content(nodes: list[SkillTreeNode]) -> None:
            for node in nodes:
                node.content = "".join(local_content[node.node_id]).strip()
                assign_content(node.children)

        assign_content(roots)
        return cls(preamble="".join(preamble), roots=roots)

    def to_markdown(self) -> str:
        """Render the tree with blank lines between structural sections."""

        sections: list[str] = []
        if preamble := self.preamble.strip():
            sections.append(preamble)
        sections.extend(self._render_node(root) for root in self.roots)
        return "\n\n".join(sections).rstrip() + "\n"

    def apply(self, edit: SkillTreeEdit) -> bool:
        """Apply one parsed optimizer edit and report whether the tree changed."""

        if edit.operation == "reject":
            return False
        if edit.target_node_id is None:
            raise ValueError(f"{edit.operation} requires target_node_id")
        target = self.node(edit.target_node_id)
        if edit.operation == "edit_node":
            find = edit.find or ""
            if target.content.count(find) != 1:
                raise ValueError(f"edit_node find must match exactly once in {target.node_id}")
            replacement = target.content.replace(find, edit.replace or "", 1).strip()
            if replacement == target.content:
                return False
            target.content = replacement
            return True
        if edit.operation == "append_node":
            addition = (edit.content or "").strip()
            replacement = f"{target.content.rstrip()}\n\n{addition}" if target.content else addition
            if replacement == target.content:
                return False
            target.content = replacement
            return True
        if edit.child is None:
            raise ValueError("create_child requires child")
        occurrence = sum(item.heading == edit.child.heading for item in target.children) + 1
        target.children.append(
            self._build_subtree(
                edit.child,
                level=target.level + 1,
                parent_node_id=target.node_id,
                occurrence=occurrence,
                used_node_ids=self._node_ids(),
            )
        )
        return True

    def node(self, node_id: str) -> SkillTreeNode:
        """Return one node by ID."""

        pending = list(reversed(self.roots))
        while pending:
            node = pending.pop()
            if node.node_id == node_id:
                return node
            pending.extend(reversed(node.children))
        raise ValueError(f"unknown Skill tree node {node_id!r}")

    def _node_ids(self) -> set[str]:
        node_ids: set[str] = set()
        pending = list(self.roots)
        while pending:
            node = pending.pop()
            node_ids.add(node.node_id)
            pending.extend(node.children)
        return node_ids

    @classmethod
    def _build_subtree(
        cls,
        proposed: NewSkillTreeNode,
        *,
        level: int,
        parent_node_id: str,
        occurrence: int,
        used_node_ids: set[str],
    ) -> SkillTreeNode:
        if level > 6:
            raise ValueError("create_child would exceed Markdown heading level 6")
        node = SkillTreeNode(
            node_id=cls._allocate_node_id(
                parent_node_id=parent_node_id,
                heading=proposed.heading,
                occurrence=occurrence,
                used_node_ids=used_node_ids,
            ),
            level=level,
            heading=proposed.heading,
            content=proposed.content.strip(),
        )
        child_occurrences: dict[str, int] = {}
        for child in proposed.children:
            child_occurrences[child.heading] = child_occurrences.get(child.heading, 0) + 1
            node.children.append(
                cls._build_subtree(
                    child,
                    level=level + 1,
                    parent_node_id=node.node_id,
                    occurrence=child_occurrences[child.heading],
                    used_node_ids=used_node_ids,
                )
            )
        return node

    @staticmethod
    def _allocate_node_id(
        *,
        parent_node_id: str,
        heading: str,
        occurrence: int,
        used_node_ids: set[str],
    ) -> str:
        salt = 0
        while True:
            seed = f"{parent_node_id}\x1f{heading}\x1f{occurrence}\x1f{salt}"
            node_id = f"node-{uuid5(_NODE_ID_NAMESPACE, seed).hex[:12]}"
            if node_id not in used_node_ids:
                used_node_ids.add(node_id)
                return node_id
            salt += 1

    @classmethod
    def _render_node(cls, node: SkillTreeNode) -> str:
        sections = [f"{'#' * node.level} {node.heading}"]
        if content := node.content.strip():
            sections.append(content)
        sections.extend(cls._render_node(child) for child in node.children)
        return "\n\n".join(sections)


__all__ = [
    "NewSkillTreeNode",
    "SkillTree",
    "SkillTreeEdit",
    "SkillTreeNode",
    "TreeEditProposal",
    "TreeFusionResult",
]
