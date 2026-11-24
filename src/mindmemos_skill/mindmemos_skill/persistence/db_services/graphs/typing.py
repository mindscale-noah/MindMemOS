"""Validated graph records shared by query and atomic publication services."""

from enum import StrEnum
from typing import Annotated, Literal, NotRequired, TypedDict
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from ....common.time_utils import utcnow
from ....infra.database import Vector
from ...typing import RecordModel

Nonempty = Annotated[str, Field(min_length=1)]
"""A nonempty external or logical identity."""
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
"""A lowercase SHA-256 hexadecimal digest."""


class GraphNodeType(StrEnum):
    """Supported categories of nodes in a Skill graph."""

    SKILL = "skill"
    """A node bound to one immutable Skill version."""

    SCENE = "scene"
    """A scene owned by a Skill node in the same snapshot."""


class GraphWriteRequest(RecordModel):
    """Bind one publication to exact members and a captured graph head."""

    graph_id: Nonempty
    """Target graph family within the active project."""

    actor_id: Nonempty
    """Actor recorded on newly published snapshots."""

    version_ids: dict[Nonempty, Nonempty]
    """Collection member IDs mapped to their exact immutable Skill version IDs."""

    expected_snapshot: Nonempty | None
    """Captured graph head; None requires a graph with no published head."""

    alias: Nonempty | None = None
    """Alias to assign when creating a graph; None uses its graph ID."""

    message: str | None = None
    """Optional publication explanation."""


class GraphResolution(TypedDict):
    """Public graph reference resolved inside one authorized project."""

    graph_id: str
    """Exact graph family identity."""

    graph_version_id: str | None
    """Prefixed selected snapshot identity, or None before publication."""

    alias: str
    """Graph alias, falling back to its family identity."""

    is_current: bool
    """Whether the selected snapshot equals the graph head."""

    status: Literal["staging", "validating", "ready", "failed"] | None
    """Selected snapshot lifecycle status, or None without a snapshot."""


class GraphPreviewNode(TypedDict):
    """Graph node enriched with its visible exact Skill version."""

    node_id: str
    """Logical node identity within the snapshot."""

    node_type: GraphNodeType
    """Whether this node describes a Skill or Scene."""

    owner_node_id: str | None
    """Owning Skill node for a Scene; None for a Skill."""

    skill_version_id: str
    """Exact source version bound to the node's owning Skill."""

    skill_id: str
    """Source Skill family identity."""

    skill_name: str
    """Display name of the source version."""

    name: str
    """Node display name."""

    version_label: str | None
    """Source version's family-local display label."""

    status: str
    """Source version lifecycle status."""

    description: str
    """Node capability description."""

    one_line_capability: str
    """Short retrieval-oriented capability statement."""

    inputs: list[str]
    """Named inputs described by the node."""

    outputs: list[str]
    """Named outputs described by the node."""

    domain_tags: list[str]
    """Domains used for semantic retrieval."""

    tooling: list[str]
    """Tools required by this node."""

    scene_metadata: dict[str, JsonValue]
    """Available scene_id, scene_type, entry_conditions, and exit_conditions values."""


class GraphPreviewEdge(TypedDict):
    """Directed relation connecting two preview nodes."""

    edge_id: UUID
    """Immutable edge identity."""

    source_node_id: str
    """Logical identity of the source node."""

    target_node_id: str
    """Logical identity of the target node."""

    edge_type: str
    """Relation category."""

    weight: float
    """Relation strength used in graph ranking."""

    condition: NotRequired[JsonValue]
    """Optional condition copied from the relation metadata."""

    evidence: NotRequired[JsonValue]
    """Optional evidence copied from the relation metadata."""


class GraphPreview(TypedDict):
    """Topology preview without embeddings or raw source content."""

    nodes: list[GraphPreviewNode]
    """Validated Skill and Scene nodes."""

    edges: list[GraphPreviewEdge]
    """Relations whose endpoints belong to this snapshot."""

    graph_version_id: Nonempty
    """Prefixed text identity of the selected immutable snapshot."""


class GraphScopeWrite(RecordModel):
    """Graph input identity whose project comes from the repository scope."""

    graph_id: Nonempty
    """Logical graph identity within the project."""


class GraphScopeRecord(GraphScopeWrite):
    """Complete persisted graph identity with project ownership."""

    project_id: Nonempty
    """Project owning this graph."""


class SkillGraphRecord(GraphScopeRecord):
    """Graph identity and its current publication pointer."""

    alias: str | None = None
    """Project-unique exact graph shortcut."""

    name: Nonempty
    """Human-readable graph name."""

    description: str | None = None
    """Optional purpose of this graph."""

    current_version_id: Nonempty | None = None
    """Selected ready graph snapshot; None means unpublished."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the graph identity was created."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Time graph metadata was last changed."""

    is_deleted: bool = False
    """Whether this graph is hidden from business operations."""

    deleted_at: AwareDatetime | None = None
    """Time of soft deletion; None while active."""

    @model_validator(mode="after")
    def validate_times(self):
        """Reject graph timestamps in reverse order."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class SkillGraphVersionRecord(GraphScopeRecord):
    """One independent graph content snapshot."""

    graph_version_id: Nonempty
    """Immutable graph snapshot identity."""

    status: Literal["staging", "validating", "ready", "failed"] = "staging"
    """Candidate lifecycle stage."""

    node_count: int | None = Field(default=None, ge=0)
    """Validated number of nodes."""

    edge_count: int | None = Field(default=None, ge=0)
    """Validated number of directed edges."""

    vector_count: int | None = Field(default=None, ge=0)
    """Validated number of non-null embeddings."""

    snapshot_hash: Digest | None = None
    """SHA-256 of canonical graph content."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Effective algorithm and embedding configuration."""

    request_hash: Digest | None = None
    """Canonical build request identity for idempotent retries."""

    actor_id: Nonempty | None = None
    """Actor who created this snapshot, when known."""

    message: str | None = None
    """Explanation supplied when this snapshot was created."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time this candidate was created."""

    ready_at: AwareDatetime | None = None
    """Time content validation finished."""

    @model_validator(mode="after")
    def validate_snapshot(self):
        """Require complete validation results for ready snapshots."""
        if self.status == "ready" and any(
            value is None
            for value in (self.node_count, self.edge_count, self.vector_count, self.snapshot_hash, self.ready_at)
        ):
            raise ValueError("Ready snapshots require counts, a digest, and ready_at")
        if self.vector_count is not None and (self.node_count is None or self.vector_count > self.node_count):
            raise ValueError("vector_count requires node_count and cannot exceed it")
        if self.ready_at is not None and self.ready_at < self.created_at:
            raise ValueError("ready_at must not precede created_at")
        return self


class SkillGraphNodeRecord(RecordModel):
    """Complete persisted graph node with ownership and retrieval features."""

    graph_id: Nonempty
    """Logical graph identity within the project."""

    graph_version_id: Nonempty
    """Snapshot containing this node."""

    node_id: Nonempty
    """Stable logical node identity across snapshots."""

    node_type: GraphNodeType = GraphNodeType.SKILL
    """Node category."""

    skill_version_id: Nonempty | None = None
    """Local immutable Skill version; required for Skill nodes."""

    owner_node_id: Nonempty | None = None
    """Owning Skill node in the same snapshot; required for scenes."""

    name: Nonempty
    """Display name; not a unique identity."""

    description: str = ""
    """Parsed capability description."""

    one_line_capability: str = ""
    """Short capability statement."""

    inputs: list[str] = Field(default_factory=list)
    """Input descriptions used for dependency linking."""

    outputs: list[str] = Field(default_factory=list)
    """Output descriptions used for dependency linking."""

    domain_tags: list[str] = Field(default_factory=list)
    """Domain labels used for retrieval."""

    tooling: list[str] = Field(default_factory=list)
    """Tool keywords; not installation instructions."""

    example_tasks: list[str] = Field(default_factory=list)
    """Explicit example task descriptions."""

    rendered_snippet: str = ""
    """Parsed snippet used for display and embedding."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Non-core node attributes."""

    embedding: list[float] | None = None
    """Finite nonzero float32 vector; dimensions vary by snapshot."""

    embedding_input_hash: Digest | None = None
    """SHA-256 of the exact embedding input."""

    project_id: Nonempty
    """Project owning this node."""

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, value):
        """Normalize embeddings to finite, nonzero float32 values."""
        return None if value is None else Vector().validate(value)

    @model_validator(mode="after")
    def validate_references(self):
        """Validate node ownership and paired embedding data."""
        if self.node_type == GraphNodeType.SKILL:
            if self.skill_version_id is None or self.owner_node_id is not None:
                raise ValueError("Skill nodes require a Skill version and no owner")
        elif self.skill_version_id is not None or self.owner_node_id is None or self.owner_node_id == self.node_id:
            raise ValueError("Scene nodes require a distinct owner and no Skill version")
        if (self.embedding is None) != (self.embedding_input_hash is None):
            raise ValueError("Embedding and input hash must both be present or absent")
        return self


class SkillGraphEdgeWrite(GraphScopeWrite):
    """One directed relationship within a graph snapshot."""

    graph_version_id: Nonempty
    """Snapshot containing both endpoints."""

    edge_id: UUID
    """Stable relationship identity."""

    source_node_id: Nonempty
    """Source node in this snapshot."""

    target_node_id: Nonempty
    """Target node in this snapshot."""

    edge_type: Literal["dependency", "containment", "flow", "semantic", "workflow", "alternative"] = "dependency"
    """Relationship meaning."""

    weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    """Finite nonnegative transition weight."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Evidence and non-core relationship attributes."""

    @model_validator(mode="after")
    def validate_endpoints(self):
        """Reject persisted self loops."""
        if self.source_node_id == self.target_node_id:
            raise ValueError("Edge endpoints must differ")
        return self


class SkillGraphEdgeRecord(SkillGraphEdgeWrite):
    """Complete persisted graph edge with project ownership."""

    project_id: Nonempty
    """Project owning this edge."""
