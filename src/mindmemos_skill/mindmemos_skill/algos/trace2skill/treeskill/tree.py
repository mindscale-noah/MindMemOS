"""Pure Markdown tree parsing, mutation, and subtree rendering for TreeSkill."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, TypeAlias

from ....typing import compute_skill_content_hash, normalize_skill_text

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)

NodeKey: TypeAlias = tuple[tuple[str, ...], int]


class TreeMetadataError(ValueError):
    """Raised when persisted TreeSkill metadata does not match ``SKILL.md``."""


@dataclass(frozen=True, slots=True)
class MarkdownTreeNode:
    """One content-bearing Markdown heading and its source spans."""

    node_id: str
    node_key: NodeKey
    heading: str
    heading_path: tuple[str, ...]
    level: int
    parent_id: str | None
    child_ids: tuple[str, ...]
    ordinal: int
    source_line: int
    section_start: int
    section_end: int
    intro_end: int
    local_content: str
    intro_markdown: str
    subtree_markdown: str


@dataclass(frozen=True, slots=True)
class MarkdownSkillTree:
    """Parsed Skill Markdown plus the stable ID ledger used during fusion."""

    full_content: str
    frontmatter: str
    markdown_content: str
    pre_heading_content: str
    nodes: tuple[MarkdownTreeNode, ...]
    ids_by_key: dict[NodeKey, str]

    @property
    def node_by_id(self) -> dict[str, MarkdownTreeNode]:
        return {node.node_id: node for node in self.nodes}

    @property
    def node_by_key(self) -> dict[NodeKey, MarkdownTreeNode]:
        return {node.node_key: node for node in self.nodes}

    @property
    def root_nodes(self) -> tuple[MarkdownTreeNode, ...]:
        return tuple(node for node in self.nodes if node.parent_id is None)


@dataclass(frozen=True, slots=True)
class NewTreeNode:
    """Structured child/subtree requested by the node-fusion model."""

    heading: str
    content: str
    children: tuple[NewTreeNode, ...] = ()


def parse_skill_markdown(
    content: str,
    *,
    previous_ids_by_key: dict[NodeKey, str] | None = None,
) -> MarkdownSkillTree:
    """Parse content-bearing headings and preserve IDs from a prior parse."""

    normalized = normalize_skill_text(content)
    frontmatter, markdown = _split_frontmatter(normalized)
    markdown = markdown.strip("\n")
    lines = markdown.splitlines()
    headings = _collect_headings(lines)
    if not headings:
        return MarkdownSkillTree(
            full_content=normalized,
            frontmatter=frontmatter,
            markdown_content=markdown,
            pre_heading_content=markdown.strip(),
            nodes=(),
            ids_by_key=dict(previous_ids_by_key or {}),
        )

    kept_indices = [index for index in range(len(headings)) if _has_section_content(lines, headings, index)]
    keys_by_index = _node_keys(headings, kept_indices)
    ids_by_key = _stable_ids(list(keys_by_index.values()), previous_ids_by_key or {})
    ids_by_index = {index: ids_by_key[key] for index, key in keys_by_index.items()}

    raw_parents = _heading_parent_indices(headings)
    parent_id_by_index: dict[int, str | None] = {}
    children_by_id = {ids_by_index[index]: [] for index in kept_indices}
    for index in kept_indices:
        parent_index = raw_parents.get(index)
        while parent_index is not None and parent_index not in ids_by_index:
            parent_index = raw_parents.get(parent_index)
        parent_id = ids_by_index.get(parent_index) if parent_index is not None else None
        parent_id_by_index[index] = parent_id
        if parent_id is not None:
            children_by_id[parent_id].append(ids_by_index[index])

    nodes: list[MarkdownTreeNode] = []
    for ordinal, index in enumerate(kept_indices):
        heading = headings[index]
        section_end = _section_end(headings, index, len(lines))
        intro_end = _intro_end(headings, index, section_end)
        node_id = ids_by_index[index]
        nodes.append(
            MarkdownTreeNode(
                node_id=node_id,
                node_key=keys_by_index[index],
                heading=heading["title"],
                heading_path=tuple(_heading_path(headings, index)),
                level=heading["level"],
                parent_id=parent_id_by_index[index],
                child_ids=tuple(children_by_id[node_id]),
                ordinal=ordinal,
                source_line=heading["line"] + 1,
                section_start=heading["line"],
                section_end=section_end,
                intro_end=intro_end,
                local_content="\n".join(lines[heading["line"] + 1 : intro_end]).strip("\n"),
                intro_markdown="\n".join(lines[heading["line"] : intro_end]).rstrip(),
                subtree_markdown="\n".join(lines[heading["line"] : section_end]).rstrip(),
            )
        )

    first_heading_line = headings[0]["line"]
    return MarkdownSkillTree(
        full_content=normalized,
        frontmatter=frontmatter,
        markdown_content=markdown,
        pre_heading_content="\n".join(lines[:first_heading_line]).strip(),
        nodes=tuple(nodes),
        ids_by_key=ids_by_key,
    )


def update_node_content(tree: MarkdownSkillTree, node_id: str, content: str) -> MarkdownSkillTree:
    """Replace one node's local content and immediately reparse the tree."""

    node = tree.node_by_id.get(node_id)
    if node is None:
        raise ValueError(f"unknown TreeSkill node: {node_id!r}")
    replacement = content.strip("\n")
    if not replacement and not node.child_ids:
        raise ValueError("a leaf node cannot be updated to empty content")
    _validate_local_content(replacement)

    lines = tree.markdown_content.splitlines()
    updated_lines = lines[: node.section_start + 1] + replacement.splitlines() + lines[node.intro_end :]
    updated = _join_skill(tree.frontmatter, updated_lines)
    reparsed = parse_skill_markdown(updated, previous_ids_by_key=tree.ids_by_key)
    refreshed = reparsed.node_by_key.get(node.node_key)
    if refreshed is None or refreshed.node_id != node_id:
        raise ValueError("updated node lost its stable identity after reparsing")
    return reparsed


def create_child_subtree(
    tree: MarkdownSkillTree,
    parent_id: str,
    new_child: NewTreeNode,
) -> MarkdownSkillTree:
    """Append one structured direct child/subtree and immediately reparse."""

    parent = tree.node_by_id.get(parent_id)
    if parent is None:
        raise ValueError(f"unknown TreeSkill parent node: {parent_id!r}")
    if parent.level >= 6:
        raise ValueError("cannot create a child below Markdown heading level 6")

    heading = _normalize_generated_heading(new_child.heading)
    for child_id in parent.child_ids:
        if tree.node_by_id[child_id].heading.casefold() == heading.casefold():
            raise ValueError(f"duplicate child heading under {parent_id}: {heading!r}")

    rendered, expected_nodes = _render_new_subtree(replace(new_child, heading=heading), parent.level + 1)
    old_keys = set(tree.node_by_key)
    lines = tree.markdown_content.splitlines()
    updated_lines = lines[: parent.section_end] + ["", *rendered] + lines[parent.section_end :]
    updated = _join_skill(tree.frontmatter, updated_lines)
    reparsed = parse_skill_markdown(updated, previous_ids_by_key=tree.ids_by_key)
    new_keys = set(reparsed.node_by_key) - old_keys
    if len(new_keys) != expected_nodes:
        raise ValueError(f"expected {expected_nodes} new nodes but parsed {len(new_keys)}")
    return reparsed


def collapse_ancestor_descendant_selections(
    tree: MarkdownSkillTree,
    node_ids: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Remove selected descendants already covered by a selected ancestor."""

    node_by_id = tree.node_by_id
    selected = [node_id for node_id in _ordered_unique(node_ids) if node_id in node_by_id]
    selected_set = set(selected)
    result: list[str] = []
    for node_id in selected:
        parent_id = node_by_id[node_id].parent_id
        while parent_id is not None and parent_id not in selected_set:
            parent = node_by_id.get(parent_id)
            parent_id = parent.parent_id if parent is not None else None
        if parent_id is None:
            result.append(node_id)
    return tuple(result)


def ancestor_closure(tree: MarkdownSkillTree, node_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Return non-selected ancestors in deterministic source order."""

    node_by_id = tree.node_by_id
    selected = set(node_ids)
    found: set[str] = set()
    for node_id in node_ids:
        node = node_by_id.get(node_id)
        while node is not None and node.parent_id is not None:
            if node.parent_id not in selected:
                found.add(node.parent_id)
            node = node_by_id.get(node.parent_id)
    return tuple(node.node_id for node in tree.nodes if node.node_id in found)


def render_selected_subtrees(tree: MarkdownSkillTree, node_ids: tuple[str, ...]) -> str:
    """Render selected subtrees plus ancestor intro context in source order."""

    selected = set(collapse_ancestor_descendant_selections(tree, node_ids))
    if not selected:
        return ""
    ancestors = set(ancestor_closure(tree, tuple(selected)))
    node_by_id = tree.node_by_id
    parts: list[str] = []
    if tree.pre_heading_content:
        parts.append(tree.pre_heading_content)

    def intersects(node: MarkdownTreeNode) -> bool:
        return node.node_id in selected or any(intersects(node_by_id[child_id]) for child_id in node.child_ids)

    def render(node: MarkdownTreeNode) -> None:
        if not intersects(node):
            return
        if node.node_id in selected:
            parts.append(node.subtree_markdown)
            return
        if node.node_id in ancestors:
            parts.append(node.intro_markdown)
        for child_id in node.child_ids:
            render(node_by_id[child_id])

    for root in tree.root_nodes:
        render(root)
    rendered = "\n\n".join(part.strip("\n") for part in parts if part.strip())
    return rendered.strip() + ("\n" if rendered.strip() else "")


def tree_prompt_payload(tree: MarkdownSkillTree) -> list[dict[str, Any]]:
    """Return the recursive complete-content tree used by TreeSkill LLM calls."""

    node_by_id = tree.node_by_id

    def build(node: MarkdownTreeNode) -> dict[str, Any]:
        return {
            "id": node.node_id,
            "level": node.level,
            "heading": node.heading,
            "local_content": node.local_content,
            "children": [build(node_by_id[child_id]) for child_id in node.child_ids],
        }

    return [build(root) for root in tree.root_nodes]


def compile_tree_metadata(tree: MarkdownSkillTree, *, router: str = "llm_subtree_v1") -> dict[str, Any]:
    """Build the namespaced persisted metadata for one final Skill tree."""

    return {
        "enabled": True,
        "schema_version": 1,
        "router": router,
        "skill_content_hash": compute_skill_content_hash({"SKILL.md": tree.full_content}),
        "root_ids": [node.node_id for node in tree.root_nodes],
        "nodes": [
            {
                "id": node.node_id,
                "level": node.level,
                "heading": node.heading,
                "parent_id": node.parent_id,
                "child_ids": list(node.child_ids),
                "ordinal": node.ordinal,
                "local_content_hash": _text_hash(node.local_content),
            }
            for node in tree.nodes
        ],
    }


def parse_tree_with_metadata(content: str, metadata: Any) -> MarkdownSkillTree:
    """Validate metadata against content and rebuild its persisted node IDs."""

    if not isinstance(metadata, dict) or metadata.get("enabled") is not True:
        raise TreeMetadataError("TreeSkill metadata is missing or disabled")
    if metadata.get("schema_version") != 1:
        raise TreeMetadataError("unsupported TreeSkill metadata schema")
    normalized = normalize_skill_text(content)
    expected_hash = compute_skill_content_hash({"SKILL.md": normalized})
    if metadata.get("skill_content_hash") != expected_hash:
        raise TreeMetadataError("TreeSkill metadata content hash does not match SKILL.md")

    raw_nodes = metadata.get("nodes")
    if not isinstance(raw_nodes, list):
        raise TreeMetadataError("TreeSkill metadata nodes must be a list")
    source_tree = parse_skill_markdown(normalized)
    if len(raw_nodes) != len(source_tree.nodes):
        raise TreeMetadataError("TreeSkill metadata node count does not match SKILL.md")

    ids_by_key: dict[NodeKey, str] = {}
    for ordinal, (node, raw) in enumerate(zip(source_tree.nodes, raw_nodes, strict=True)):
        if not isinstance(raw, dict) or raw.get("ordinal") != ordinal:
            raise TreeMetadataError("TreeSkill metadata node ordinals are invalid")
        if raw.get("heading") != node.heading or raw.get("level") != node.level:
            raise TreeMetadataError("TreeSkill metadata heading structure does not match SKILL.md")
        if raw.get("local_content_hash") != _text_hash(node.local_content):
            raise TreeMetadataError("TreeSkill metadata local content does not match SKILL.md")
        node_id = raw.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise TreeMetadataError("TreeSkill metadata contains an invalid node id")
        ids_by_key[node.node_key] = node_id

    tree = parse_skill_markdown(normalized, previous_ids_by_key=ids_by_key)
    expected = compile_tree_metadata(tree, router=str(metadata.get("router") or "llm_subtree_v1"))
    for field in ("root_ids", "nodes"):
        if metadata.get(field) != expected[field]:
            raise TreeMetadataError(f"TreeSkill metadata {field} does not match SKILL.md")
    return tree


def _split_frontmatter(content: str) -> tuple[str, str]:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return "", content.strip("\n")
    return content[: match.end()], content[match.end() :].strip("\n")


def _collect_headings(lines: list[str]) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    fence_char = ""
    fence_size = 0
    for line_number, line in enumerate(lines):
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = "", 0
            continue
        if fence_char:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append({"line": line_number, "level": len(match.group(1)), "title": match.group(2).strip()})
    return headings


def _heading_parent_indices(headings: list[dict[str, Any]]) -> dict[int, int | None]:
    stack: list[int] = []
    parents: dict[int, int | None] = {}
    for index, heading in enumerate(headings):
        while stack and headings[stack[-1]]["level"] >= heading["level"]:
            stack.pop()
        parents[index] = stack[-1] if stack else None
        stack.append(index)
    return parents


def _has_section_content(lines: list[str], headings: list[dict[str, Any]], index: int) -> bool:
    start = headings[index]["line"] + 1
    end = _section_end(headings, index, len(lines))
    fence_char = ""
    fence_size = 0
    for line in lines[start:end]:
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = "", 0
            continue
        if fence_char or (line.strip() and not _HEADING_RE.match(line)):
            return True
    return False


def _section_end(headings: list[dict[str, Any]], index: int, default: int) -> int:
    level = headings[index]["level"]
    for candidate in headings[index + 1 :]:
        if candidate["level"] <= level:
            return min(candidate["line"], default)
    return default


def _intro_end(headings: list[dict[str, Any]], index: int, section_end: int) -> int:
    for candidate in headings[index + 1 :]:
        if candidate["line"] >= section_end:
            break
        if candidate["level"] > headings[index]["level"]:
            return candidate["line"]
    return section_end


def _heading_path(headings: list[dict[str, Any]], index: int) -> list[str]:
    stack: list[dict[str, Any]] = []
    for heading in headings[: index + 1]:
        while stack and stack[-1]["level"] >= heading["level"]:
            stack.pop()
        stack.append(heading)
    return [heading["title"] for heading in stack]


def _node_keys(headings: list[dict[str, Any]], kept_indices: list[int]) -> dict[int, NodeKey]:
    occurrences: dict[tuple[str, ...], int] = {}
    result: dict[int, NodeKey] = {}
    for index in kept_indices:
        path = tuple(_heading_path(headings, index))
        occurrences[path] = occurrences.get(path, 0) + 1
        result[index] = (path, occurrences[path])
    return result


def _stable_ids(keys: list[NodeKey], previous: dict[NodeKey, str]) -> dict[NodeKey, str]:
    result = dict(previous)
    used = set(previous.values())
    numeric = [int(value) for value in used if value.isdigit()]
    next_id = max(numeric, default=0) + 1
    for key in keys:
        if key in result:
            continue
        while f"{next_id:03d}" in used:
            next_id += 1
        result[key] = f"{next_id:03d}"
        used.add(result[key])
        next_id += 1
    return result


def _validate_local_content(content: str) -> None:
    fence_char = ""
    fence_size = 0
    for line in content.splitlines():
        fence = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = "", 0
            continue
        if not fence_char and _HEADING_RE.match(line):
            raise ValueError("local node content must not contain Markdown headings")
    if fence_char:
        raise ValueError("local node content contains an unclosed Markdown fence")


def _normalize_generated_heading(heading: str) -> str:
    value = str(heading or "").strip()
    match = re.match(r"^#{1,6}\s+(.+)$", value)
    if match:
        value = match.group(1).strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("new node heading must be non-empty single-line text")
    return value


def _render_new_subtree(node: NewTreeNode, level: int) -> tuple[list[str], int]:
    if level > 6:
        raise ValueError("new subtree exceeds Markdown heading depth 6")
    heading = _normalize_generated_heading(node.heading)
    content = node.content.strip("\n")
    _validate_local_content(content)
    if not content and not node.children:
        raise ValueError("a new leaf node must contain local content")

    lines = [f"{'#' * level} {heading}"]
    if content:
        lines.extend(["", *content.splitlines()])
    child_names: set[str] = set()
    node_count = 1
    for child in node.children:
        child_heading = _normalize_generated_heading(child.heading)
        key = child_heading.casefold()
        if key in child_names:
            raise ValueError(f"duplicate generated sibling heading: {child_heading!r}")
        child_names.add(key)
        child_lines, child_count = _render_new_subtree(replace(child, heading=child_heading), level + 1)
        lines.extend(["", *child_lines])
        node_count += child_count
    return lines, node_count


def _join_skill(frontmatter: str, markdown_lines: list[str]) -> str:
    markdown = "\n".join(markdown_lines).rstrip()
    return normalize_skill_text(f"{frontmatter}{markdown}")


def _ordered_unique(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "MarkdownSkillTree",
    "MarkdownTreeNode",
    "NewTreeNode",
    "NodeKey",
    "TreeMetadataError",
    "ancestor_closure",
    "collapse_ancestor_descendant_selections",
    "compile_tree_metadata",
    "create_child_subtree",
    "parse_skill_markdown",
    "parse_tree_with_metadata",
    "render_selected_subtrees",
    "tree_prompt_payload",
    "update_node_content",
]
