"""Validated graph records; lifecycle transitions belong to the future data service."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, field_validator, model_validator

from ....common.time_utils import utcnow
from ....infra.database import Vector
from ...typing import RecordModel

Nonempty = Annotated[str, Field(min_length=1)]
"""A nonempty external or logical identity."""
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
"""A lowercase SHA-256 hexadecimal digest."""


class GraphScopeRecord(RecordModel):
    """Project-isolated graph identity shared by all graph records."""

    project_id: Nonempty
    """Globally unique project owning this graph."""

    graph_id: Nonempty
    """Logical graph identity within the project."""


class SkillGraphRecord(GraphScopeRecord):
    """Graph identity and its current publication pointer."""

    name: Nonempty
    """Human-readable graph name."""

    description: str | None = None
    """Optional purpose of this graph."""

    current_revision: int | None = Field(default=None, gt=0)
    """Current published revision; None means unpublished."""

    created_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the graph identity was created."""

    updated_at: AwareDatetime = Field(default_factory=utcnow)
    """Time graph metadata was last changed."""

    @model_validator(mode="after")
    def validate_times(self):
        """Reject graph timestamps in reverse order."""
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self


class SkillGraphVersionRecord(GraphScopeRecord):
    """One independent graph content snapshot."""

    graph_version_id: UUID
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


class SkillGraphNodeRecord(GraphScopeRecord):
    """One node and its retrieval features within a snapshot."""

    graph_version_id: UUID
    """Snapshot containing this node."""

    node_id: Nonempty
    """Stable logical node identity across snapshots."""

    node_type: Literal["skill", "scene"] = "skill"
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

    @field_validator("embedding", mode="before")
    @classmethod
    def validate_embedding(cls, value):
        """Normalize embeddings to finite, nonzero float32 values."""
        return None if value is None else Vector().validate(value)

    @model_validator(mode="after")
    def validate_references(self):
        """Validate node ownership and paired embedding data."""
        if self.node_type == "skill":
            if self.skill_version_id is None or self.owner_node_id is not None:
                raise ValueError("Skill nodes require a Skill version and no owner")
        elif self.skill_version_id is not None or self.owner_node_id is None or self.owner_node_id == self.node_id:
            raise ValueError("Scene nodes require a distinct owner and no Skill version")
        if (self.embedding is None) != (self.embedding_input_hash is None):
            raise ValueError("Embedding and input hash must both be present or absent")
        return self


class SkillGraphEdgeRecord(GraphScopeRecord):
    """One directed relationship within a graph snapshot."""

    graph_version_id: UUID
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


class SkillGraphRevisionRecord(GraphScopeRecord):
    """One successful publication and its idempotent result."""

    revision: int = Field(gt=0)
    """Monotonic publication number within the graph."""

    graph_version_id: UUID
    """Immutable content selected by this publication."""

    idempotency_key: Nonempty
    """Caller-provided retry identity, unique within the graph."""

    request_hash: Digest
    """SHA-256 of the canonical request and its contract version."""

    action: Literal["build", "upsert", "remove", "restore"]
    """Business action that produced this publication."""

    restored_from_revision: int | None = Field(default=None, gt=0)
    """Earlier publication restored by a restore action."""

    actor_id: Nonempty
    """Authenticated actor; not an isolation key."""

    message: str | None = None
    """Optional publication explanation."""

    published_at: AwareDatetime = Field(default_factory=utcnow)
    """Time the publication was recorded."""

    @model_validator(mode="after")
    def validate_restore(self):
        """Require an earlier revision exactly for restore actions."""
        if self.action == "restore":
            if self.restored_from_revision is None or self.restored_from_revision >= self.revision:
                raise ValueError("Restore requires an earlier revision")
        elif self.restored_from_revision is not None:
            raise ValueError("Only restore actions may reference a restored revision")
        return self
