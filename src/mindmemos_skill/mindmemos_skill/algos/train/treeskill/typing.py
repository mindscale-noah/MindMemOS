"""Markdown tree structure and observable task-time routing results."""

from pydantic import BaseModel, ConfigDict, Field

from ....agents.typing import AgentResult
from ....skill.typing import SkillBundle


class SkillNode(BaseModel):
    """Describe one nonempty Markdown heading section."""

    model_config = ConfigDict(extra="forbid")
    """Reject undeclared parser fields."""

    id: str
    """One-based, zero-padded node identifier in document order."""

    level: int
    """Markdown heading depth from one through six."""

    heading: str
    """Heading title without Markdown markers."""

    local_content: str
    """Content immediately below the heading, excluding child sections."""

    intro: str
    """Heading and local content retained when this node is an ancestor."""

    body: str
    """Entire original subtree, including all descendant sections."""

    children: list["SkillNode"] = Field(default_factory=list)
    """Direct children in source order."""


class SkillTree(BaseModel):
    """Represent a heading tree parsed from one source SKILL.md."""

    preamble: str
    """Text preceding the first heading, excluding YAML frontmatter."""

    roots: list[SkillNode]
    """Top-level heading nodes in source order."""

    def nodes(self) -> list[SkillNode]:
        """Return all nodes in document order."""
        nodes = []

        def visit(node):
            nodes.append(node)
            for child in node.children:
                visit(child)

        for root in self.roots:
            visit(root)
        return nodes


class RoutingResult(BaseModel):
    """Expose routed content and selection evidence without mutable Skill state."""

    bundle: SkillBundle
    """Task-time bundle containing the selected subtree content."""

    selected_subtree_ids: list[str] = Field(default_factory=list)
    """Selected roots after removing duplicates and covered descendants."""

    ancestor_ids: list[str] = Field(default_factory=list)
    """Ancestor nodes whose introductions provide context."""

    fallback_reason: str | None = None
    """Reason full source content was used, or None for a valid selection."""

    response: AgentResult | None = None
    """Router output, trajectory, and usage; absent when no call was needed."""
