"""Markdown tree structure and observable task-time routing results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ....agents.typing import AgentResult
from ....envs.data import DataSample
from ....envs.typing import RolloutResult, Task
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
            """Visit.

            Args:
                node: Parsed heading node or proposed subtree.
            """
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


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnalysisItem(_StrictModel):
    """One structured finding extracted from a physical trajectory."""

    item_id: str = Field(min_length=1)
    """Identifier of one reusable finding within its trajectory analysis."""

    kind: Literal["failure_cause", "failure_memory", "success_memory", "unlabeled_memory"]
    """Evidence category distinguishing failure causes and reusable memories."""

    number: int | None = Field(default=None, ge=1)
    """Original report item number, or None when derived from the item ID."""

    title: str = ""
    """Short descriptive label for one analysis finding."""

    description: str = ""
    """Evidence-grounded explanation of the observed behavior."""

    content: str = ""
    """Proposed Markdown instruction text, excluding child sections when node-local."""

    relation_to_skill: str = ""
    """Failure-cause explanation relating the observation to existing instructions."""

    skill_reflection: str = ""
    """Failure-memory reflection on how the source Skill should change."""


class TrajectoryAnalysisRecord(_StrictModel):
    """Outcome-aware analysis record consumed by evidence localization."""

    instance_id: str = Field(min_length=1)
    """Identifier of the physical trajectory that supplied this evidence."""

    task_id: str = Field(min_length=1)
    """Stable public task identifier attached to the sampled trajectory."""

    record_source: Literal["error", "success", "unlabeled"]
    """Outcome category determined from the recorded environment score."""

    source_file: str = ""
    """Relative analysis report name retained by the reference prompt payload."""

    items: tuple[AnalysisItem, ...] = ()
    """Ordered independent findings extracted from the trajectory."""


class LocatedEvidence(_StrictModel):
    """One atomic reusable lesson assigned to one initial tree node."""

    instance_id: str = Field(min_length=1)
    """Identifier of the physical trajectory that supplied this evidence."""

    evidence_id: str = Field(min_length=1)
    """Unique atomic lesson identifier within one localization response."""

    record_source: Literal["error", "success", "unlabeled"]
    """Outcome category determined from the recorded environment score."""

    reusable_lesson: str = Field(min_length=1)
    """Generalizable instruction supported by this trajectory."""

    target_node_id: str = Field(min_length=1)
    """Stable initial-tree node ID receiving this evidence or edit."""

    rationale: str = Field(min_length=1)
    """Model-provided justification for placement or editing."""


class LocalizationResult(_StrictModel):
    """Validated localization result for one trajectory analysis record."""

    instance_id: str = Field(min_length=1)
    """Identifier of the physical trajectory that supplied this evidence."""

    evidence: tuple[LocatedEvidence, ...] = ()
    """Validated localized lessons in their original response order."""


class AppliedEditRecord(_StrictModel):
    """Auditable result of validating and applying one fusion edit."""

    target_node_id: str
    """Stable initial-tree node ID receiving this evidence or edit."""

    operation: str
    """Requested update_node, create_child or reject operation."""

    accepted: bool
    """Whether the edit passed validation; reject records represent deliberate no-ops."""

    message: str
    """Validation result or model rationale explaining the edit disposition."""


class LocalizationFailure(_StrictModel):
    instance_id: str
    """Identifier of the physical trajectory that supplied this evidence."""

    error: str
    """Exception type and message for this failed item."""


class FusionFailure(_StrictModel):
    target_node_id: str
    """Stable initial-tree node ID receiving this evidence or edit."""

    error: str
    """Exception type and message for this failed item."""


class TreeSkillReport(_StrictModel):
    """Compact audit report for one TreeSkill evolution transaction."""

    algorithm_name: str = "treeskill"
    """Stable algorithm identifier used in experiment reports."""

    run_id: str = Field(min_length=1)
    """Identifier of the isolated experiment producing this report."""

    algorithm_version: str
    """Recorded implementation behavior revision."""

    prompt_version: str
    """Recorded prompt protocol revision."""

    input_trajectory_ids: tuple[str, ...]
    """Ordered IDs of successfully collected evidence trajectories."""

    duplicate_trajectory_ids: tuple[str, ...] = ()
    """Repeated input IDs excluded from the evidence set."""

    failed_analysis_trajectory_ids: tuple[str, ...] = ()
    """IDs whose analyses failed validation or execution."""

    localization_failures: tuple[LocalizationFailure, ...] = ()
    """Rejected or failed localization items with explanations."""

    fusion_failures: tuple[FusionFailure, ...] = ()
    """Node fusion requests that failed before producing a usable edit."""

    analysis_records: tuple[TrajectoryAnalysisRecord, ...] = ()
    """Structured reusable evidence before localization."""

    located_evidence: tuple[LocatedEvidence, ...] = ()
    """Atomic lessons assigned to initial-tree nodes."""

    applied_edits: tuple[AppliedEditRecord, ...] = ()
    """Ordered results of proposed node edits."""

    initial_node_count: int = Field(ge=0)
    """Number of content-bearing headings before training."""

    final_node_count: int = Field(ge=0)
    """Number of content-bearing headings in the resulting Skill."""

    changed: bool = False
    """Whether normalized final Markdown differs from the initial Skill."""

    reason: str | None = None
    """Explanation for an unchanged outcome, or None for effective updates."""


class TreeRoutingResult(_StrictModel):
    """One ephemeral routing decision used by a Skill runtime."""

    selected_node_ids: tuple[str, ...]
    """Chosen subtree roots plus ancestor context nodes in source order."""

    content_node_ids: tuple[str, ...]
    """Nonredundant subtree roots containing task-relevant content."""

    ancestor_node_ids: tuple[str, ...]
    """Ancestor nodes retained only as path and local context."""

    skill_content: str
    """Task-specific rendered Markdown after routing."""

    fallback_used: bool = False
    """Whether invalid routing metadata or output required the full Skill."""

    fallback_reason: str = ""
    """Explanation for fallback or an explicit empty selection."""

    full_char_count: int = Field(ge=0)
    """Character count of the full source Markdown."""

    routed_char_count: int = Field(ge=0)
    """Character count of the rendered task-specific Markdown."""


class AnnotationMode(StrEnum):
    """How an algorithm consumes trajectory reward annotations."""

    AUTO = "auto"
    REQUIRED = "required"
    IGNORE = "ignore"


class TraceEvidence(_StrictModel):
    """Normalized evidence extracted from one pre-collected trajectory."""

    trajectory_id: str = Field(min_length=1)
    """Stable physical rollout ID used to join evidence and artifacts."""

    task_id: str = Field(min_length=1)
    """Stable public task identifier attached to the sampled trajectory."""

    transcript: str
    """Reference-rendered event text with per-event truncation."""

    score: float | None = None
    """Environment score, or None when annotations are explicitly ignored."""

    annotation_detail: str | None = None
    """Environment feedback text made available to trajectory analysis."""

    annotation_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Additional recorded reward annotations for this execution."""


class EvidenceSelection(_StrictModel):
    """Deterministic evidence selection plus audit information."""

    evidence: list[TraceEvidence]
    """Validated localized lessons in their original response order."""

    duplicate_trajectory_ids: list[str] = Field(default_factory=list)
    """Repeated input IDs excluded from the evidence set."""


class TrainingRollout(BaseModel):
    """Physical environment execution and its explicitly supplied private sample."""

    trajectory_id: str
    """Run-local stable identifier, independent of unsafe task path characters."""

    sample: DataSample
    """Public task and private scoring data; only reference analysis may access training gold."""

    result: RolloutResult
    """Completed rollout with environment reward and original events."""

    work_dir: Path
    """Directory containing this rollout's audit and workspace."""


class ChatResponse(BaseModel):
    """Recorded raw response and optional validated algorithm output."""

    content: str
    """Original provider response, before parsing."""

    parsed: BaseModel | str | None = None
    """Validated parser result, if a parser was supplied."""


NodeKey: TypeAlias = tuple[tuple[str, ...], int]


@dataclass(frozen=True, slots=True)
class MarkdownTreeNode:
    """One content-bearing Markdown heading and its source spans."""

    node_id: str
    """Stable zero-padded heading ID maintained across fusion edits."""

    node_key: NodeKey
    """Heading-path and occurrence key used to preserve node identity."""

    heading: str
    """Plain Markdown heading text without heading markers."""

    heading_path: tuple[str, ...]
    """Ordered ancestor heading titles ending at this node."""

    level: int
    """Markdown heading level from one through six."""

    parent_id: str | None
    """Parent heading ID, or None for a root node."""

    child_ids: tuple[str, ...]
    """Ordered IDs of direct child headings."""

    ordinal: int
    """Occurrence index disambiguating repeated heading paths."""

    source_line: int
    """Zero-based heading line in the parsed Markdown body."""

    section_start: int
    """First line index of this heading section."""

    section_end: int
    """Exclusive end line of this complete subtree."""

    intro_end: int
    """Exclusive end of local node text before child headings."""

    local_content: str
    """Instruction text directly under this heading, excluding descendants."""

    intro_markdown: str
    """Heading and local text retained as ancestor context."""

    subtree_markdown: str
    """Complete source text for this heading and descendants."""


@dataclass(frozen=True, slots=True)
class MarkdownSkillTree:
    """Parsed Skill Markdown plus the stable ID ledger used during fusion."""

    full_content: str
    """Normalized full source Markdown including frontmatter."""

    frontmatter: str
    """Original YAML frontmatter retained during node edits."""

    markdown_content: str
    """Normalized body without YAML frontmatter."""

    pre_heading_content: str
    """Body text preceding the first heading."""

    nodes: tuple[MarkdownTreeNode, ...]
    """Content-bearing heading records in document order."""

    ids_by_key: dict[NodeKey, str]
    """Stable heading identity ledger carried across reparsing."""

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
    """Plain Markdown heading text without heading markers."""

    content: str
    """Proposed Markdown instruction text, excluding child sections when node-local."""

    children: tuple[NewTreeNode, ...] = ()
    """Ordered direct child nodes or recursively proposed child subtrees."""


class _RawEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    """Unique atomic lesson identifier within one localization response."""

    reusable_lesson: str = Field(min_length=1)
    """Generalizable instruction supported by this trajectory."""

    target_node_id: str = Field(min_length=1)
    """Stable initial-tree node ID receiving this evidence or edit."""

    rationale: str = Field(min_length=1)
    """Model-provided justification for placement or editing."""


class _RoutePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_subtree_ids: tuple[str, ...] = ()
    """Quoted existing subtree IDs selected by the router."""

    rationale: str = ""
    """Model-provided justification for placement or editing."""

    @model_validator(mode="after")
    def validate_unique_ids(self) -> _RoutePayload:
        if len(self.selected_subtree_ids) != len(set(self.selected_subtree_ids)):
            raise ValueError("selected_subtree_ids must not contain duplicates")
        return self


class ChatModel(Protocol):
    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


class TrajectoryAnalyzer(Protocol):
    async def analyze(
        self,
        evidence: list[TraceEvidence],
        *,
        trajectories_by_id: Mapping[str, TrainingRollout],
    ) -> tuple[list[TrajectoryAnalysisRecord], list[str]]: ...


if TYPE_CHECKING:
    from .config import TreeSkillTrainerConfig


class TreeSkillAnalysisContext(BaseModel):
    """Inputs available to generic training analysis, excluding private samples."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent reassignment; nested evidence is borrowed and must not be mutated."""

    evidence: TraceEvidence
    """One normalized training trajectory and its allowed feedback."""

    source: Literal["error", "success", "unlabeled"]
    """Outcome category chosen using the trainer's existing score threshold."""


class TreeSkillRoutingContext(BaseModel):
    """Public inputs for either stable-ID or legacy subtree routing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent reassignment of routing inputs."""

    task: Task
    """Public task retaining its concrete environment-specific type."""

    tree: list[dict[str, Any]]
    """Public recursive Skill tree payload with identifiers and node content."""

    public_context: dict[str, Any] = Field(default_factory=dict)
    """Explicit public preview overrides; never private reward inputs."""


@dataclass(frozen=True)
class TreeSkillAnalysisRuntime:
    """Borrow live model dependencies without serializing or owning their lifecycle."""

    config: TreeSkillTrainerConfig
    """Fixed training-stage settings shared with the trainer."""

    chat_model: ChatModel
    """Recorded optimizer model transport."""

    failure_chat_model: ChatModel
    """Recorded transport for optional tool-using failure analysis."""

    output_root: Path
    """Run-local directory for analyzer artifacts."""
