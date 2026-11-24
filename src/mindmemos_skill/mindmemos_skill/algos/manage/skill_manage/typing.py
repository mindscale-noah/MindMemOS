"""Graph construction and retrieval contracts."""

import json
from dataclasses import dataclass
from math import isclose
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator
from typing_extensions import NotRequired, TypedDict

from ....agents import OpenAIChatConfig
from ....agents.typing import AgentResult
from ....persistence.db_services.graphs.typing import (
    GraphNodeType,
    Nonempty,
    SkillGraphEdgeRecord,
    SkillGraphNodeRecord,
)
from ....persistence.db_services.graphs.writer import graph_digest
from ....scene.typing import SceneContractModel
from .config import EmbeddingProfile
from .constant import (
    METADATA_ALLOWED_TOOLS,
    METADATA_COMPATIBILITY,
    METADATA_EDGES,
    METADATA_EMBEDDING,
    METADATA_EXTERNAL_DEPENDENCIES,
    METADATA_INTENT_SIGNATURE,
    METADATA_KEY,
    METADATA_LOCAL_SCRIPTS,
    SCENE_GRAPH_REPRESENTATION_VERSION,
)

EnrichmentText = Annotated[str, Field(max_length=1000)]
EnrichmentItems = Annotated[tuple[EnrichmentText, ...], Field(max_length=20)]


class ExtractionSummary(SceneContractModel):
    """Payload-free accounting retained after transient extraction is released."""

    attempts: int = Field(default=0, ge=0)
    """Structured model calls, excluding transport retries inside each call."""

    input_tokens: int = Field(default=0, ge=0)
    """Sum of reported prompt tokens, including rejected outputs."""

    output_tokens: int = Field(default=0, ge=0)
    """Sum of reported completion tokens, including rejected outputs."""

    total_tokens: int = Field(default=0, ge=0)
    """Sum of reported total tokens; zero does not establish absence of consumption."""

    elapsed_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    """Wall time for extraction, parsing and transport retries."""

    error_type: Literal["invalid_output", "transport", "missing_model", "insufficient_source"] | None = None
    """Safe failure category; never raw exception text or model output."""


class SceneGraphScore(SceneContractModel):
    """Public Skill score components, separate from vector distance and rank."""

    member_id: Nonempty
    """Selected collection Skill ID; a Scene is never a standalone result."""

    ppr_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    """Skill node's normalized PPR mass before child aggregation."""

    scene_boost: float = Field(ge=0, allow_inf_nan=False)
    """alpha times max or mean child Scene PPR; zero for no children or skill_only."""

    graph_score: float = Field(ge=0, allow_inf_nan=False)
    """ppr_score plus scene_boost; this is not a probability or cosine similarity."""

    rerank_score: float = Field(ge=0, allow_inf_nan=False)
    """Field-based final score, only computed for graph-selected candidates."""

    @model_validator(mode="after")
    def validate_total(self):
        """Prevent mislabeled or inconsistent public score components."""
        if not isclose(self.graph_score, self.ppr_score + self.scene_boost, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("graph_score must equal ppr_score plus scene_boost")
        return self


class DerivedNodeMetadata(SceneContractModel):
    """Allowlisted derived attributes safe for graph persistence and re-embedding."""

    collection_skill_id: Nonempty
    """Collection owner key shared by the Skill and its Scenes."""

    family_id: Nonempty
    """Resolved canonical family in the graph's project."""

    scene_id: Nonempty | None = None
    """Local Scene identity; absent on Skill nodes."""

    scene_type: str = ""
    """Derived Scene category, absent on Skill nodes."""

    intent: str = ""
    """Source-grounded goal merged from normalized Skill or Scene semantics."""

    intent_signature: str = Field(default="", max_length=1000)
    """Independent compact retrieval signature, empty when unsupported by the source."""

    local_scripts: tuple[str, ...] = ()
    """Portable script paths from registered resources or explicit source documentation."""

    external_dependencies: tuple[str, ...] = ()
    """Documented external packages with available version constraints, never local paths."""

    compatibility: tuple[str, ...] = ()
    """Documented runtime requirements used as retrieval evidence, not execution policy."""

    allowed_tools: tuple[str, ...] = ()
    """Documented tool declarations used as retrieval features, not enforced permissions."""

    is_entry: bool = False
    """Whether this Scene is the owning Skill's entry; False on Skill nodes."""

    flow_text: str = Field(default="", max_length=800)
    """Bounded semantic flow description, never serialized logic_steps or raw SSL."""

    entry_conditions: tuple[str, ...] = ()
    """Source-grounded conditions for entering the Scene."""

    exit_conditions: tuple[str, ...] = ()
    """Source-grounded terminal or exit conditions."""


class DerivedEdgeMetadata(SceneContractModel):
    """Allowlisted relationship evidence with no arbitrary model response payload."""

    evidence: str = Field(default="", max_length=1000)
    """Bounded, source-grounded relationship explanation."""

    condition: str = ""
    """Branch condition for flow edges."""

    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    """Model confidence when present; effective edge weight already includes it."""


class SceneGraphContent(SceneContractModel):
    """Logical graph content independent of physical UUIDs and publication history."""

    profile: str = SCENE_GRAPH_REPRESENTATION_VERSION
    """Schema and algorithm interpretation, never inferred from old graph edges."""

    embedding: EmbeddingProfile
    """Vector compatibility and text-rendering identity."""

    nodes: tuple[SkillGraphNodeRecord, ...]
    """Full node content including exact versions, derived fields and vectors."""

    edges: tuple[SkillGraphEdgeRecord, ...]
    """Directed logical relationships; physical edge UUIDs do not define equivalence."""

    @model_validator(mode="after")
    def validate_content(self):
        """Validate scope, ownership and allowlisted persistence attributes."""
        if self.profile != SCENE_GRAPH_REPRESENTATION_VERSION:
            raise ValueError("Graph profile must match the current representation version")
        nodes = {node.node_id: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError("Graph node IDs must be unique")
        scopes = {(item.project_id, item.graph_id, item.graph_version_id) for item in (*self.nodes, *self.edges)}
        if len(scopes) > 1:
            raise ValueError("Graph content must belong to one project, graph and snapshot")
        families = set()
        members = set()
        versions = set()
        metadata = {node.node_id: DerivedNodeMetadata.model_validate(node.metadata) for node in self.nodes}
        for node in self.nodes:
            derived = metadata[node.node_id]
            expected_id = (
                [GraphNodeType.SKILL, derived.collection_skill_id]
                if node.node_type == GraphNodeType.SKILL
                else [GraphNodeType.SCENE, derived.collection_skill_id, derived.scene_id]
            )
            if node.node_id != json.dumps(expected_id, ensure_ascii=False, separators=(",", ":")):
                raise ValueError("Node identity must use the canonical member and local Scene encoding")
            if node.embedding is None or len(node.embedding) != self.embedding.dimensions:
                raise ValueError("Every node requires a vector matching the embedding profile")
            if node.node_type == GraphNodeType.SKILL:
                if (
                    derived.scene_id is not None
                    or derived.family_id in families
                    or derived.collection_skill_id in members
                    or node.skill_version_id in versions
                ):
                    raise ValueError(
                        "Skill nodes require distinct members, families and versions, and no Scene identity"
                    )
                families.add(derived.family_id)
                members.add(derived.collection_skill_id)
                versions.add(node.skill_version_id)
            else:
                owner = nodes.get(node.owner_node_id)
                if owner is None or owner.node_type != GraphNodeType.SKILL or derived.scene_id is None:
                    raise ValueError("Scene nodes require a Skill owner in the same snapshot")
                parent = metadata[owner.node_id]
                if (derived.collection_skill_id, derived.family_id) != (parent.collection_skill_id, parent.family_id):
                    raise ValueError("Scene ownership must preserve member and canonical family")
        logical_edges = set()
        for edge in self.edges:
            DerivedEdgeMetadata.model_validate(edge.metadata)
            if edge.source_node_id not in nodes or edge.target_node_id not in nodes:
                raise ValueError("Graph edge endpoints must exist")
            key = (edge.source_node_id, edge.target_node_id, edge.edge_type)
            if key in logical_edges:
                raise ValueError("Logical relationships must be unique")
            logical_edges.add(key)
            source, target = nodes[edge.source_node_id], nodes[edge.target_node_id]
            if edge.edge_type == "containment" and not (
                source.node_type == GraphNodeType.SCENE and source.owner_node_id == target.node_id
            ):
                raise ValueError("Containment must point from a Scene to its owning Skill")
            if edge.edge_type == "flow" and not (
                source.node_type == target.node_type == GraphNodeType.SCENE
                and source.owner_node_id == target.owner_node_id
            ):
                raise ValueError("Flow must connect Scenes of the same Skill")
            if edge.edge_type == "dependency" and (
                metadata[source.node_id].family_id == metadata[target.node_id].family_id
            ):
                raise ValueError("Dependency edges must cross canonical families")
        return self

    def content_digest(self) -> str:
        """Hash logical content without changing the legacy physical integrity digest."""
        nodes = [node.model_dump(mode="json", exclude={"graph_version_id"}) for node in self.nodes]
        edges = [edge.model_dump(mode="json", exclude={"graph_version_id", "edge_id"}) for edge in self.edges]
        for node in nodes:
            original = node[METADATA_KEY]
            normalized = DerivedNodeMetadata.model_validate(original).model_dump(mode="json")
            # Preserve the exact historical normalization of pre-v3 logical content.
            for key in (
                METADATA_INTENT_SIGNATURE,
                METADATA_LOCAL_SCRIPTS,
                METADATA_EXTERNAL_DEPENDENCIES,
                METADATA_COMPATIBILITY,
                METADATA_ALLOWED_TOOLS,
            ):
                if key not in original:
                    normalized.pop(key)
            node[METADATA_KEY] = normalized
        for edge in edges:
            edge[METADATA_KEY] = DerivedEdgeMetadata.model_validate(edge[METADATA_KEY]).model_dump(mode="json")
        return graph_digest(
            {
                "profile": self.profile,
                METADATA_EMBEDDING: self.embedding.model_dump(mode="json"),
                "nodes": sorted(nodes, key=lambda node: node["node_id"]),
                METADATA_EDGES: sorted(
                    edges, key=lambda edge: (edge["source_node_id"], edge["target_node_id"], edge["edge_type"])
                ),
            }
        )


class JointRelation(SceneContractModel):
    """Validated model relation between an explicitly supplied joint candidate pair."""

    source_id: Nonempty
    """Exact source node ID from the candidate batch."""

    target_id: Nonempty
    """Exact target node ID from the candidate batch."""

    @field_validator("source_id", "target_id", mode="before")
    @classmethod
    def normalize_json_encoded_node_id(cls, value):
        """Restore a canonical opaque ID when a model decodes it into a JSON array."""
        if isinstance(value, list) and value and all(isinstance(part, str) and part for part in value):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return value

    edge_type: Literal["dependency", "semantic", "workflow", "alternative"]
    """Supported inferred relation; containment and flow are deterministic only."""

    weight: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    """Strength required only by model_confidence; type_confidence ignores this optional value."""

    confidence: float = Field(default=1, ge=0, le=1, allow_inf_nan=False)
    """Confidence multiplier for either weighting policy; zero remains zero."""

    evidence: Nonempty = Field(max_length=1000)
    """Bounded explanation grounded in the candidate features."""


class JointRelations(SceneContractModel):
    """Strict envelope for optional joint relation inference."""

    relations: tuple[JointRelation, ...]
    """Only supported candidate relationships; empty when none are justified."""


class QuerySchema(BaseModel):
    """Public task intent consumed by vector search and field reranking."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized rewritten query fields."""

    goal: str = Field(min_length=1)
    """Task objective without inventing requirements."""

    task_name: str = ""
    """Short task label used by the source field-scoring rule."""

    domain: list[str] = Field(default_factory=list)
    """Relevant subject areas explicitly supported by the task."""

    operations: list[str] = Field(default_factory=list)
    """Requested actions."""

    artifacts: list[str] = Field(default_factory=list)
    """Input or output artifacts mentioned in the public task."""

    constraints: list[str] = Field(default_factory=list)
    """Explicit task restrictions that selection should preserve."""

    keywords: list[str] = Field(default_factory=list)
    """Search terms grounded in the public task."""


class RetrievalSeed(SceneContractModel):
    """One selected seed with independent route ranks and explicit prior mass."""

    node_id: Nonempty
    """Opaque node identity in the pinned snapshot."""

    dense_rank: int | None = Field(default=None, gt=0)
    """One-based accepted dense rank within this node type, if retrieved."""

    lexical_rank: int | None = Field(default=None, gt=0)
    """One-based lexical candidate rank within this node type, if retrieved."""

    fused_rank: int | None = Field(default=None, gt=0)
    """One-based selected rank within this node type; None omits the final rank bonus."""

    fusion_score: float = Field(ge=0, allow_inf_nan=False)
    """Field score plus reciprocal dense/lexical ranks and dual-route bonus."""

    weight: float = Field(gt=0, le=1, allow_inf_nan=False)
    """Normalized prior mass after balancing the nonempty node types."""


class RetrievalCandidate(SceneContractModel):
    """Safe route evidence for one candidate before seed truncation."""

    node_id: Nonempty
    """Candidate identity in the pinned graph, without node content or metadata."""

    node_type: GraphNodeType
    """Independent candidate route to which this node belongs."""

    dense_rank: int | None = Field(default=None, gt=0)
    """Accepted cosine-neighbor rank after the dense threshold, if present."""

    lexical_rank: int | None = Field(default=None, gt=0)
    """Rank among lexical candidates, if present."""

    fused_rank: int = Field(gt=0)
    """Rank after candidate fusion and before the seed budget is applied."""

    selected: bool
    """Whether the candidate enters graph personalization."""


class SceneGraphVectorScore(SceneContractModel):
    """One owner selected solely by its closest eligible vector."""

    member_id: Nonempty
    """Collection Skill member owning the matched node."""

    node_id: Nonempty
    """Best matching Skill or Scene node in this pinned snapshot."""

    node_type: GraphNodeType
    """Whether the winning vector belongs to the Skill or one of its Scenes."""

    skill_version_id: Nonempty | None = None
    """Exact local Skill version bound to the winning owner, when available."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Node metadata projection returned by the narrow vector query."""

    similarity: float = Field(allow_inf_nan=False)
    """Cosine similarity, computed as one minus stored cosine distance; higher is better."""


class SceneGraphVectorRetrieval(SceneContractModel):
    """Pure cosine ranking without rewriting, lexical fusion, graph walks or reranking."""

    strategy: Literal["vector"] = "vector"
    """The ranking uses only vector similarity over the pinned snapshot."""

    score_type: Literal["cosine_similarity"] = "cosine_similarity"
    """Each score's similarity is one minus cosine distance, with larger values ranked first."""

    snapshot_id: str
    """Immutable snapshot containing every returned vector and member."""

    scores: tuple[SceneGraphVectorScore, ...] = ()
    """Unique owners ordered by maximum eligible-node cosine similarity."""

    include_scenes: bool = False
    """Whether Scene vectors participate; their similarity maps directly to their owner."""

    scene_policy: Literal["max_owner"] = "max_owner"
    """Use each owner's best node similarity, with no sum, boost or rank transformation."""

    warnings: tuple[Literal["SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"], ...] = ()
    """Safe build warnings from the pinned snapshot."""


class SceneGraphRetrieval(SceneContractModel):
    """Detailed Skill-only output bound to the snapshot used throughout the query."""

    strategy: Literal["graph"] = "graph"
    """The ranking uses the configured graph retrieval mode."""

    score_type: Literal["graph_components"] = "graph_components"
    """Scores expose graph and reranking components, not cosine similarities."""

    snapshot_id: str
    """Immutable graph queried, independent of later changes to the graph head."""

    scores: tuple[SceneGraphScore, ...] = ()
    """Final ordered Skill members and their separately labeled score components."""

    warnings: tuple[Literal["SCENE_GRAPH_ENRICHMENT_RULE_FALLBACK"], ...] = ()
    """Safe build warnings from this pinned snapshot; no model exception text."""


class SkillEnrichment(SceneContractModel):
    """Allowlisted missing Skill fields; never identity, source content, or SSL."""

    description: EnrichmentText = ""
    """Source-grounded description, used only when the parsed field is absent."""

    one_line_capability: EnrichmentText = ""
    """Short capability summary, used only when absent."""

    intent: EnrichmentText = ""
    """Skill intent, used only when SSL has not already supplied it."""

    inputs: EnrichmentItems = ()
    """Documented inputs, used only when the input list is empty."""

    outputs: EnrichmentItems = ()
    """Documented outputs, used only when the output list is empty."""

    tooling: EnrichmentItems = ()
    """Explicit source tools, used only when the tooling list is empty."""

    domain_tags: EnrichmentItems = ()
    """Source-grounded domain labels, used only when absent."""

    example_tasks: EnrichmentItems = ()
    """Tasks supported by the document, used only when absent."""


class ChatModel(Protocol):
    """Structural port shared by the existing chat agent and test doubles."""

    config: OpenAIChatConfig
    """Recorded model settings; injected clients must match configured settings."""

    async def run(self, *, system: str, user: str, response_format: dict) -> AgentResult:
        """Return structured JSON text for a graph operation.

        Args:
            system: Operation instructions with the expected output contract.
            user: Explicit public source text or task payload.
            response_format: OpenAI-compatible response constraint.
        """
        ...


@dataclass(frozen=True)
class OrganizedGraph:
    """One validated immutable graph bound to an explicit candidate scope."""

    request_hash: str
    """Fingerprint of source membership and construction settings."""

    snapshot_id: str
    """Pinned persisted graph content identity."""

    nodes: list[SkillGraphNodeRecord]
    """Validated Skill and Scene records loaded from storage."""

    edges: list[SkillGraphEdgeRecord]
    """Validated relationships within this snapshot."""

    skill_ids: frozenset[str]
    """Complete allowed collection member IDs."""

    embedding_profile: EmbeddingProfile | None = None
    """Joint vector provenance for safe in-memory reuse; None on legacy graphs."""

    warnings: tuple[str, ...] = ()
    """Allowlisted build warnings bound to this immutable snapshot."""

    skill_versions: dict[str, str] | None = None
    """Exact owner version projection for pure vector queries; None on full graphs."""


class JointEdgeDraft(TypedDict):
    """Structural or semantic edge before storage identities are attached."""

    source: str
    """Source node identity within the graph being built."""

    target: str
    """Target node identity within the graph being built."""

    type: Literal["dependency", "containment", "flow", "semantic", "workflow", "alternative"]
    """Relationship category used together with endpoints for deduplication."""

    weight: float
    """Effective relationship strength, including model confidence when applicable."""

    description: NotRequired[str]
    """Evidence produced by dependency matching or relation inference."""

    evidence: NotRequired[str]
    """Stored evidence carried by reused structural edges."""

    condition: NotRequired[str]
    """Scene transition condition; absent or empty for unconditional edges."""

    confidence: NotRequired[float | None]
    """Model confidence when present; None for deterministic structural edges."""


class JointRepresentation(BaseModel):
    """Nodes and structural edges prepared before enrichment and linking."""

    nodes: list[SkillGraphNodeRecord]
    """Skill and Scene nodes in deterministic member order."""

    structural_edges: list[JointEdgeDraft]
    """Containment and flow edges owned by individual members."""

    reused_members: set[str]
    """Member identities whose captured nodes should skip enrichment."""


class JointLinkResult(BaseModel):
    """Final graph edges and the measured relation candidate count."""

    edges: list[SkillGraphEdgeRecord]
    """Deduplicated edge records bound to the new snapshot."""

    candidate_pairs: int
    """Number of unique unordered node pairs selected for model inference."""


class InferredEdge(TypedDict):
    """Accepted model relation before graph storage identities are attached."""

    source: str
    """Source node identity from the supplied candidate pair."""

    target: str
    """Target node identity from the supplied candidate pair."""

    type: Literal["dependency", "semantic", "workflow", "alternative"]
    """Accepted relationship category."""

    weight: float
    """Relationship strength multiplied by model confidence."""

    confidence: float
    """Model confidence for the accepted relationship."""

    description: str
    """Source-grounded evidence returned by the model."""


class BuildSettings(TypedDict):
    """Persisted construction inputs used for graph reuse and restoration."""

    representation: str
    """Stored graph representation version."""

    config: dict[str, JsonValue]
    """Serialized build options and embedding profile, excluding query options and secrets."""

    linking_version: str
    """Candidate selection and relationship construction version."""

    enrichment: dict[str, str] | None
    """Prompt and schema hashes when enrichment is enabled; otherwise None."""

    scenes: dict[str, str]
    """Canonical Skill IDs mapped to their captured Scene content digests."""
