"""Markdown tree structure and observable task-time routing results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from ....agents import AgentResult, OpenAIChatAgent
from ....envs.data import DataSample
from ....envs.typing import RolloutResult, Task


class RoutingResult(BaseModel):
    """Expose routed content and selection evidence without mutable Skill state."""

    content: str
    """Task-selected Markdown, which may be empty for an empty selection."""

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


class AnalysisInput(BaseModel):
    """One trainer-scheduled analysis and its optional tool inputs."""

    evidence: TraceEvidence
    """Normalized trajectory evidence and allowed outcome feedback."""

    trajectory: TrainingRollout
    """Completed training execution, including permitted private analysis references."""

    workspace: Path
    """Fresh trainer-owned directory for tool execution; not an artifact destination."""


class TrajectoryAnalyzer(Protocol):
    async def __call__(self, inputs: AnalysisInput) -> AnalysisOutcome: ...


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
    """Public inputs for stable-ID subtree routing."""

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

    agent: OpenAIChatAgent
    """Direct agent for success or generic analysis."""

    failure_agent: OpenAIChatAgent
    """Direct agent for tool-using failure analysis."""


class NodeFusionEdit(_StrictModel):
    """One strictly validated model-proposed node-local edit."""

    operation: Literal["update_node", "create_child", "reject"]
    """Node edit requested by the model."""

    rationale: str = Field(min_length=1)
    """Explanation for this edit or explicit rejection."""

    content: str | None = None
    """Replacement local text for update_node; None for other operations."""

    new_child: NewChildSpec | None = None
    """Recursive new subtree for create_child; None for other operations."""

    @model_validator(mode="after")
    def validate_operation_payload(self) -> NodeFusionEdit:
        if self.operation == "update_node":
            if self.content is None or self.new_child is not None:
                raise ValueError("update_node requires content and must not include new_child")
        elif self.operation == "create_child":
            if self.new_child is None or self.content is not None:
                raise ValueError("create_child requires new_child and must not include content")
        elif self.content is not None or self.new_child is not None:
            raise ValueError("reject must not include content or new_child")
        return self


class NewChildSpec(_StrictModel):
    """Strict recursive subtree payload returned by the fusion model."""

    heading: str = Field(min_length=1)
    """New heading title, validated before Markdown insertion."""

    content: str = ""
    """Local instruction text; empty only when nonempty descendants exist."""

    children: tuple[NewChildSpec, ...] = ()
    """Ordered direct child subtrees."""


NodeFusionEdit.model_rebuild()


class NodeFusionDecision(_StrictModel):
    """Complete validated response for one target-node fusion request."""

    rationale: str = Field(min_length=1)
    """Overall justification for this node's proposed edits."""

    edits: tuple[NodeFusionEdit, ...] = Field(min_length=1, max_length=2)
    """One rejection or at most one update and one child insertion."""

    @model_validator(mode="after")
    def validate_edit_set(self) -> NodeFusionDecision:
        operations = [edit.operation for edit in self.edits]
        if "reject" in operations and len(operations) != 1:
            raise ValueError("reject cannot be mixed with another operation")
        if len(operations) != len(set(operations)):
            raise ValueError("node fusion may return each operation at most once")
        return self


class StageOutcome(BaseModel):
    """In-memory execution evidence returned to the trainer for persistence."""

    responses: list[AgentResult] = Field(default_factory=list)
    """Completed logical calls, including their trajectories and reported usage."""

    failed_calls: int = 0
    """Calls that raised without a result; their token consumption is unknown."""

    interrupted: bool = False
    """Execution ended before returning an outcome; usage coverage is incomplete."""

    error: str | None = None
    """Item failure or partial-result rejection; None means complete success."""

    artifacts: dict[str, str] = Field(default_factory=dict)
    """Relative report filenames mapped to UTF-8 contents for trainer-owned writes."""


class AnalysisOutcome(StageOutcome):
    """One trajectory analysis and its execution evidence."""

    record: TrajectoryAnalysisRecord | None = None
    """Validated analysis; None when the item failed."""

    @field_validator("responses", mode="before")
    @classmethod
    def restore_analysis_responses(cls, value):
        """Restore the concrete parser output when reading analysis checkpoints."""
        return [
            {**item, "output": TrajectoryAnalysisRecord.model_validate(item["output"])}
            if isinstance(item, dict) and isinstance(item.get("output"), dict)
            else item
            for item in value
        ]


class LocalizationOutcome(StageOutcome):
    """One analysis record's localized lessons and rejected-item details."""

    evidence: list[LocatedEvidence] = Field(default_factory=list)
    """Valid lessons, including valid entries from a partially rejected response."""


class PatchRecord(BaseModel):
    """One proposed edit and the exact source snapshots surrounding it."""

    before: str
    """Complete source before attempting this edit."""

    after: str
    """Complete source after accepting or rejecting this edit."""

    payload: dict[str, JsonValue]
    """Proposal, decision rationale, result and supporting evidence."""


class FusionOutcome(StageOutcome):
    """One node's edits and a recoverable stable-ID tree snapshot."""

    content: str
    """Complete resulting Markdown source."""

    tree_metadata: dict[str, Any]
    """Stable node identity metadata required to restore the resulting tree."""

    edits: list[AppliedEditRecord] = Field(default_factory=list)
    """Ordered accepted and rejected edits."""

    patches: list[PatchRecord] = Field(default_factory=list)
    """Exact per-edit snapshots and proposals, saved by the trainer."""
