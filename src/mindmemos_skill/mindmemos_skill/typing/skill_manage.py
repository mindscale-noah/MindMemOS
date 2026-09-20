"""Algorithm-side contracts for the ``skill_manage`` capability.

These DTOs cross the algorithm protocol boundary only. They must never carry
database objects, concrete storage drivers, or application persistence state:
the pure algorithm core consumes in-memory snapshots and returns plans/results,
while :class:`SkillManageService` decides when side effects happen.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EdgeType(StrEnum):
    """Closed, frozen set of graph edge types."""

    DEPENDENCY = "dependency"
    CONTAINMENT = "containment"
    FLOW = "flow"
    SEMANTIC = "semantic"
    WORKFLOW = "workflow"
    ALTERNATIVE = "alternative"


class RetrievalMode(StrEnum):
    """Frozen retrieval modes migrated from the GOS engine."""

    SKILL_ONLY = "skill_only"
    UNIFIED_PPR = "unified_ppr"
    SEPARATE = "separate"
    FUSE = "fuse"


class SkillManageScope(_FrozenModel):
    """Logical graph scope: every read/write carries ``project_id + graph_id``."""

    project_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)


class SkillVersionRef(_FrozenModel):
    """Exact, immutable Skill version identity accepted by a full build."""

    skill_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)


class EmbeddingProfile(_FrozenModel):
    """Identity of the embedding model, vector shape and input template for one generation.

    ``template_version`` names the algorithm's embedding-text template; the
    algorithm rejects a profile whose template it does not produce, so a
    template change always forces a new generation instead of mixing vectors.
    """

    model: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    normalization: str = Field(default="nfc", min_length=1)
    template_version: str = Field(default="1", min_length=1)


class SkillGraphNode(_FrozenModel):
    """One node in a graph snapshot."""

    graph_node_id: str = Field(min_length=1)
    node_type: Literal["skill", "scene"]
    skill_id: str | None = None
    version_id: str | None = None
    parent_skill_id: str | None = None
    name: str = ""
    description: str = ""
    one_line_capability: str = ""
    inputs: tuple[str, ...] = Field(default_factory=tuple)
    outputs: tuple[str, ...] = Field(default_factory=tuple)
    domain_tags: tuple[str, ...] = Field(default_factory=tuple)
    tooling: tuple[str, ...] = Field(default_factory=tuple)
    example_tasks: tuple[str, ...] = Field(default_factory=tuple)
    rendered_snippet: str = ""
    content_hash: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class SkillGraphEdge(_FrozenModel):
    """One directed edge in a graph snapshot."""

    edge_id: str = Field(min_length=1)
    edge_type: EdgeType
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    weight: float = Field(default=1.0, ge=0)
    metadata: dict[str, object] = Field(default_factory=dict)


class SkillGraphMembership(_FrozenModel):
    """Which immutable Skill version participates in one graph snapshot."""

    skill_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    graph_node_id: str = Field(min_length=1)


class SkillGraphEmbeddingText(_FrozenModel):
    """Embedding input for one node, as produced by the algorithm's template."""

    graph_node_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SkillGraphEmbedding(_FrozenModel):
    """One stored node vector, qualified by the profile that produced it."""

    graph_node_id: str = Field(min_length=1)
    node_type: Literal["skill", "scene"] = "skill"
    skill_id: str | None = None
    version_id: str | None = None
    content_hash: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_profile_version: str = Field(min_length=1)
    vector: tuple[float, ...] = Field(min_length=1)


class SkillGraphVectorHit(_FrozenModel):
    """One KNN hit; ``score`` is the provider-normalized vector similarity."""

    graph_node_id: str = Field(min_length=1)
    skill_id: str | None = None
    version_id: str | None = None
    score: float


class SkillGraphSnapshot(_FrozenModel):
    """Immutable, generation-qualified graph view handed to the algorithm."""

    generation_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    nodes: tuple[SkillGraphNode, ...] = Field(default_factory=tuple)
    edges: tuple[SkillGraphEdge, ...] = Field(default_factory=tuple)
    memberships: tuple[SkillGraphMembership, ...] = Field(default_factory=tuple)


class SkillDocumentInput(_FrozenModel):
    """One raw Skill document handed to the parser inside a build request."""

    skill_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_path: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


class SkillManageBuildInput(_FrozenModel):
    """Pure inputs for a full graph build plan."""

    scope: SkillManageScope
    version_refs: tuple[SkillVersionRef, ...] = Field(default_factory=tuple)
    documents: tuple[SkillDocumentInput, ...] = Field(default_factory=tuple)
    embedding_profile: EmbeddingProfile
    build_scene: bool = True
    llm_linking: bool = False
    dependency_match_threshold: float = Field(default=0.5, gt=0, le=1.0)


class SkillManageUpsertInput(_FrozenModel):
    """Pure inputs for a single-Skill upsert plan."""

    scope: SkillManageScope
    version_ref: SkillVersionRef
    snapshot: SkillGraphSnapshot
    embedding_profile: EmbeddingProfile
    llm_linking: bool = False
    dependency_match_threshold: float = Field(default=0.5, gt=0, le=1.0)
    document: SkillDocumentInput | None = None


class SkillManageRemoveSelector(_FrozenModel):
    """Discriminated remove selector: exactly one of version or family."""

    kind: Literal["version", "family"]
    skill_id: str = Field(min_length=1)
    version_id: str | None = None


class SkillManageRemoveInput(_FrozenModel):
    """Pure inputs for a single-Skill remove plan."""

    scope: SkillManageScope
    selector: SkillManageRemoveSelector
    snapshot: SkillGraphSnapshot


class SkillManageRetrieveInput(_FrozenModel):
    """Pure inputs for one retrieval over a fixed snapshot.

    ``vector_seed`` lists node names in KNN order (rank encodes the seed
    weight); ``vector_similarity`` carries each seed's normalized similarity so
    hits can report it without the algorithm touching the vector store.
    ``score_threshold`` is informational here: the service has already applied
    it to the seeds' vector similarity, so algorithms must not filter PPR or
    rerank scores with it.
    """

    scope: SkillManageScope
    snapshot: SkillGraphSnapshot
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, gt=0)
    mode: RetrievalMode = RetrievalMode.SKILL_ONLY
    score_threshold: float | None = None
    vector_seed: tuple[str, ...] = Field(default_factory=tuple)
    vector_similarity: dict[str, float] = Field(default_factory=dict)
    lexical_seed: tuple[str, ...] = Field(default_factory=tuple)


class SkillManageEmbedInput(_FrozenModel):
    """Texts to embed under one profile through the algorithm's injected model.

    ``run_id`` groups the model calls in the LLM call sink (operation ID for
    writes, a per-query ID for retrieval); without it nothing is recorded.
    """

    embedding_profile: EmbeddingProfile
    texts: tuple[str, ...] = Field(min_length=1)
    task: str = Field(default="skill_manage.build", min_length=1)
    run_id: str | None = Field(default=None, min_length=1)


class SkillManageEmbedOutput(_FrozenModel):
    """Validated vectors, one per input text and in input order."""

    embedding_profile: EmbeddingProfile
    vectors: tuple[tuple[float, ...], ...]


class SkillManageHit(_FrozenModel):
    """One retrieved Skill with its decomposed scores."""

    skill_id: str = Field(min_length=1)
    version_id: str = ""
    name: str = ""
    graph_node_id: str | None = None
    score: float
    vector_score: float | None = None
    ppr_score: float | None = None
    rerank_score: float | None = None


class SkillManageWarning(_FrozenModel):
    """Stable warning envelope with a safe, bounded details payload."""

    code: str = Field(min_length=1)
    message: str = ""
    details: dict[str, object] = Field(default_factory=dict)


class SkillManageBuildPlan(_FrozenModel):
    """Algorithm-computed full-build change plan (no side effects).

    ``embedding_texts`` covers every skill node in ``nodes``; the service
    decides which of them must actually be embedded versus reused from the base
    generation by ``content_hash``.
    """

    scope: SkillManageScope
    nodes: tuple[SkillGraphNode, ...] = Field(default_factory=tuple)
    edges: tuple[SkillGraphEdge, ...] = Field(default_factory=tuple)
    memberships: tuple[SkillGraphMembership, ...] = Field(default_factory=tuple)
    embedding_texts: tuple[SkillGraphEmbeddingText, ...] = Field(default_factory=tuple)
    warnings: tuple[SkillManageWarning, ...] = Field(default_factory=tuple)


class SkillManageMutationPlan(SkillManageBuildPlan):
    """Algorithm-computed upsert/remove change plan (no side effects)."""


class SkillManageRetrieveOutput(_FrozenModel):
    """Pure retrieval result over a fixed snapshot."""

    generation_id: str = Field(min_length=1)
    revision: int = Field(ge=0)
    hits: tuple[SkillManageHit, ...] = Field(default_factory=tuple)
    warnings: tuple[SkillManageWarning, ...] = Field(default_factory=tuple)


__all__ = [
    "EdgeType",
    "EmbeddingProfile",
    "RetrievalMode",
    "SkillGraphEdge",
    "SkillGraphEmbedding",
    "SkillGraphEmbeddingText",
    "SkillGraphMembership",
    "SkillGraphNode",
    "SkillGraphSnapshot",
    "SkillGraphVectorHit",
    "SkillManageBuildInput",
    "SkillManageBuildPlan",
    "SkillManageEmbedInput",
    "SkillManageEmbedOutput",
    "SkillManageHit",
    "SkillManageMutationPlan",
    "SkillManageRemoveInput",
    "SkillManageRemoveSelector",
    "SkillManageRetrieveInput",
    "SkillManageRetrieveOutput",
    "SkillManageScope",
    "SkillManageUpsertInput",
    "SkillManageWarning",
    "SkillDocumentInput",
    "SkillVersionRef",
]
