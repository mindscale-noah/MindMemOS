"""Public dictionary results returned by graph management pipelines."""

from datetime import datetime
from typing import Literal, TypedDict

from pydantic import JsonValue

from ..persistence.db_services.graphs.typing import GraphPreview


class GraphTopology(GraphPreview):
    """Published topology with the resolved owning graph identity."""

    graph_id: str
    """Exact graph family identity inside the authorized project."""


class GraphListItem(TypedDict):
    """Stored graph identity, publication pointer, and lifecycle timestamps."""

    graph_id: str
    """Logical graph identity within the project."""

    project_id: str
    """Project owning this graph."""

    alias: str | None
    """Project-unique exact graph shortcut."""

    name: str
    """Human-readable graph name."""

    description: str | None
    """Optional purpose of this graph."""

    current_version_id: str | None
    """Selected ready graph snapshot; None means unpublished."""

    created_at: datetime
    """Time the graph identity was created."""

    updated_at: datetime
    """Time graph metadata was last changed."""

    is_deleted: bool
    """Whether this graph is hidden from business operations."""

    deleted_at: datetime | None
    """Time of soft deletion; None while active."""


class GraphSnapshotListItem(TypedDict):
    """Stored snapshot identity, validation results, and build metadata."""

    graph_id: str
    """Logical graph identity within the project."""

    project_id: str
    """Project owning this graph."""

    graph_version_id: str
    """Immutable graph snapshot identity."""

    status: Literal["staging", "validating", "ready", "failed"]
    """Candidate lifecycle stage."""

    node_count: int | None
    """Validated number of nodes."""

    edge_count: int | None
    """Validated number of directed edges."""

    vector_count: int | None
    """Validated number of non-null embeddings."""

    snapshot_hash: str | None
    """SHA-256 of canonical graph content."""

    metadata: dict[str, JsonValue]
    """Effective algorithm and embedding configuration."""

    request_hash: str | None
    """Canonical build request identity for idempotent retries."""

    actor_id: str | None
    """Actor who created this snapshot, when known."""

    message: str | None
    """Explanation supplied when this snapshot was created."""

    created_at: datetime
    """Time this candidate was created."""

    ready_at: datetime | None
    """Time content validation finished."""
