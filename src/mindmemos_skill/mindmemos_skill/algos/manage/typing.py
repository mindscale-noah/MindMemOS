"""Graph data exchanged between management algorithms and publication."""

from pydantic import BaseModel, JsonValue

from ...persistence.db_services.graphs.typing import SkillGraphEdgeRecord, SkillGraphNodeRecord


class GraphSnapshot(BaseModel):
    """Complete immutable graph data captured for reuse or publication."""

    snapshot_id: str
    """Identity shared by the snapshot's nodes and edges."""

    nodes: list[SkillGraphNodeRecord]
    """Complete Skill and Scene nodes, including available embeddings."""

    edges: list[SkillGraphEdgeRecord]
    """Complete relationships between nodes in this snapshot."""

    metadata: dict[str, JsonValue]
    """Build settings, integrity digests, and model usage recorded for this snapshot."""


class GeneratedGraph(GraphSnapshot):
    """Complete build result awaiting atomic publication."""

    request_hash: str
    """Idempotency identity for the exact selected sources and build settings."""
