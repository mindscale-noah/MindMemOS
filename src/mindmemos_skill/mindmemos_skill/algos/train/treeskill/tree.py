"""Parse and render the tree-only Markdown routing contract."""

import re

from .typing import SkillNode, SkillTree

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
"""ATX headings used by the reference TreeSkill implementation."""

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
"""Fenced code delimiters, including tilde and extended backtick fences."""


def parse_tree(content: str) -> SkillTree:
    """Parse headings without treating YAML or fenced code as tree structure.

    Args:
        content: Source Markdown from the mutable bundle.

    Returns:
        A tree with stable document-order IDs and original subtree text.
    """
    content = re.sub(r"\A---\s*\n.*?\n---\s*\n?", "", content, count=1, flags=re.DOTALL)
    lines = content.strip("\n").splitlines()
    headings = []
    fence = ""
    for index, line in enumerate(lines):
        delimiter = _FENCE.match(line)
        if fence:
            if (
                delimiter
                and delimiter[1][0] == fence[0]
                and len(delimiter[1]) >= len(fence)
                and not delimiter[2].strip()
            ):
                fence = ""
            continue
        if delimiter:
            fence = delimiter[1]
            continue
        match = _HEADING.match(line)
        if match:
            headings.append((index, len(match[1]), match[2].strip()))

    roots = []
    stack = []
    count = 0
    for index, (start, level, heading) in enumerate(headings):
        end = next((line for line, depth, _ in headings[index + 1 :] if depth <= level), len(lines))
        # Keep ancestors with nonempty descendants, but skip empty leaf sections.
        if not any(line.strip() and not _HEADING.match(line) for line in lines[start + 1 : end]):
            continue
        intro_end = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
        count += 1
        node = SkillNode(
            id=f"{count:03d}",
            level=level,
            heading=heading,
            local_content="\n".join(lines[start + 1 : intro_end]).strip("\n"),
            intro="\n".join(lines[start:intro_end]).rstrip(),
            body="\n".join(lines[start:end]).rstrip(),
        )
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    first = headings[0][0] if headings else len(lines)
    return SkillTree(preamble="\n".join(lines[:first]).strip(), roots=roots)


def render_selection(tree: SkillTree, selected_ids: list[str]) -> tuple[str, list[str], list[str]]:
    """Keep selected subtrees and only the introductions of their ancestors.

    Args:
        tree: Parsed source content.
        selected_ids: Valid IDs selected by the router, possibly duplicated.

    Returns:
        Rendered Markdown, nonredundant selected roots, and ancestor IDs.
    """
    selected = set(selected_ids)
    valid = {node.id for node in tree.nodes()}
    if not selected <= valid:
        raise ValueError("Selected subtree IDs are not present in the Skill tree")
    if not selected:
        return "", [], []
    chosen, ancestors = [], []

    def render(node):
        if node.id in selected:
            chosen.append(node.id)
            return [node.body]
        descendants = [part for child in node.children for part in render(child)]
        if descendants:
            ancestors.append(node.id)
            return [node.intro, *descendants]
        return []

    parts = [tree.preamble, *(part for root in tree.roots for part in render(root))]
    text = "\n\n".join(part.strip("\n") for part in parts if part.strip()).strip()
    ordered_ancestors = [node.id for node in tree.nodes() if node.id in ancestors]
    return text + ("\n" if text else ""), chosen, ordered_ancestors
