"""Shared vector provenance for graph construction and publication."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .typing import Nonempty


class EmbeddingProfile(BaseModel):
    """Identify compatible vector spaces without credentials or transport settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject unknown settings and freeze the profile."""

    space_id: Nonempty
    """Non-secret provider/deployment revision; change when model weights change."""

    model: Nonempty
    """Explicit embedding model name."""

    dimensions: int = Field(gt=0, strict=True)
    """Required dimension of every node and query vector."""

    metric: Literal["cosine"] = "cosine"
    """Search returns cosine distance; similarity is 1 minus distance."""

    normalization: Literal["none", "l2"] = "none"
    """Pre-storage vector normalization policy."""

    node_template: Nonempty = "scene_graph_nodes_v2"
    """Version of the deterministic node text renderer."""

    query_template: Nonempty = "scene_graph_query_v2"
    """Version of the query renderer compatible with this vector space."""
