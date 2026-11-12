"""Configuration for persisted graph-based Skill selection."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents import AgentConfig, OpenAIChatConfig
from ....infra.embedding import OpenAIEmbeddingConfig
from ....persistence.db_services.graphs.config import EmbeddingProfile as EmbeddingProfile


class RetrievalStrategyConfig(BaseModel):
    """Shared optional walk and query-view ablations."""

    scene_ppr: bool = True
    """In separate mode, diffuse Scene seeds over the Scene subgraph."""

    fusion_lambda: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    """Skill-subgraph share in fuse; the remaining share uses the full graph."""

    scene_count_damp: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    """Divide each Scene seed by its owner's child count raised to this power."""

    scene_uniqueness: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    """Divide Scene seeds by one plus this factor times cross-family Scene degree."""

    dependency_filter: Literal["none", "hub_df", "no_substring", "topk", "hub_df_topk", "idf_reweight"] = "none"
    """Query-only dependency filtering; skill_only and Scene-free graphs bypass it."""

    dependency_hub_fraction: float = Field(default=0.05, gt=0, le=1, allow_inf_nan=False)
    """Minimum node or edge frequency fraction defining an I/O hub driver."""

    dependency_top_k: int = Field(default=10, ge=0, strict=True)
    """Per-Scene incident dependency cap with endpoint-union semantics."""


class JointLinkingConfig(BaseModel):
    """Bound each joint candidate route and the final per-source union."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject unknown settings and freeze candidate budgets."""

    candidate_strategy: Literal["rrf", "field_evidence"] = "rrf"
    """Keep rank fusion by default, or require structured evidence and rank weighted fields."""

    dense_candidates: int = Field(default=20, ge=0, strict=True)
    """Maximum cross-family vector candidates per source; zero disables the route."""

    lexical_candidates: int = Field(default=20, ge=0, strict=True)
    """Maximum candidates with positive lexical overlap per source."""

    io_candidates: int = Field(default=20, ge=0, strict=True)
    """Maximum candidates with positive bidirectional I/O overlap per source."""

    max_candidates: int = Field(default=20, gt=0, strict=True)
    """Final per-source cap after combining the enabled candidate routes."""

    model_candidates: int = Field(default=8, gt=0, strict=True)
    """Maximum model candidates per source after excluding deterministic dependency pairs."""

    relation_weighting: Literal["type_confidence", "model_confidence"] = "type_confidence"
    """Multiply confidence by the fixed relation-type weight or an explicit model-supplied weight."""

    relation_validation: Literal["strict", "drop_invalid_endpoints"] = "strict"
    """Abort on invalid model endpoints, or discard only those proposals and keep valid relations."""

    dependency_match_threshold: float = Field(default=0.6, ge=0, le=1, allow_inf_nan=False)
    """Minimum positive I/O overlap for a deterministic joint dependency."""


class SkillManageConfig(RetrievalStrategyConfig):
    """Configure graph construction, query rewriting, and ranked retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject misspelled settings and freeze algorithm options."""

    embedding: OpenAIEmbeddingConfig
    """Explicit model and dimension used for graph and query vectors."""

    embedding_profile: EmbeddingProfile | None = None
    """Joint vector provenance; None derives a space ID from the configured endpoint."""

    joint_linking: JointLinkingConfig = Field(default_factory=JointLinkingConfig)
    """Joint candidate and dependency settings; legacy SceneRecord builds ignore these."""

    chat: Annotated[OpenAIChatConfig, AgentConfig(allowed=("openai_chat",), namespace="skill_manage")] | None = None
    """Chat settings required when relation inference or query rewriting is enabled."""

    build_scene: bool = True
    """Include validated Scene nodes; False builds only Skill nodes."""

    scene_flow_weight: float = Field(default=0.3, gt=0, le=1, allow_inf_nan=False)
    """Weight of directed transitions between Scenes in one Skill; must be in (0, 1]."""

    llm_linking: bool = True
    """Infer model relationships between Skill nodes; deterministic edges remain enabled."""

    scene_llm_linking: bool = False
    """Independently infer relationships involving a Scene, including mixed Skill/Scene pairs."""

    query_rewrite: bool = False
    """Extract a structured public task query before embedding and reranking."""

    enrich_skill: bool = False
    """Fill missing source-grounded Skill fields before node embedding."""

    enrichment_concurrency: int = Field(default=4, gt=0, strict=True)
    """Maximum number of Skill nodes enriched concurrently during one graph build."""

    enrichment_failure: Literal["error", "rules"] = "error"
    """Raise on enrichment failure, or explicitly keep rule fields with a warning."""

    mode: Literal["skill_only", "unified_ppr", "separate", "fuse"] = "separate"
    """Rank Skills alone or propagate through the joint Skill/Scene graph."""

    skill_candidates: int = Field(default=20, gt=0, strict=True)
    """Maximum Skill candidates per dense and lexical route."""

    scene_candidates: int = Field(default=20, gt=0, strict=True)
    """Maximum Scene candidates per dense and lexical route."""

    skill_seeds: int = Field(default=5, gt=0, strict=True)
    """Maximum Skill seeds after field and rank fusion."""

    scene_seeds: int = Field(default=5, ge=0, strict=True)
    """Maximum Scene seeds; zero disables Scene seeding, not joint propagation."""

    alpha: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    """Multiplier applied to the aggregated child Scene PPR mass."""

    aggregation: Literal["max", "mean"] = "max"
    """Child Scene reduction before applying alpha."""

    score_threshold: float = Field(default=-1, ge=-1, le=1)
    """Minimum cosine similarity for a vector seed; -1 includes all neighbors."""

    damping: float = Field(default=0.2, gt=0, le=1)
    """Restart probability in the source personalized PageRank recurrence."""

    max_iter: int = Field(default=50, gt=0, strict=True)
    """Maximum PageRank updates before returning the last estimate."""

    tolerance: float = Field(default=1e-6, gt=0, allow_inf_nan=False)
    """L1 convergence threshold for consecutive PageRank estimates."""

    max_input_chars: int = Field(default=1_000_000, gt=0, strict=True)
    """Maximum canonical characters accepted in one source Skill document."""

    llm_batch_size: int = Field(default=32, gt=0, strict=True)
    """Maximum candidate node pairs sent in one relation-inference request."""

    @model_validator(mode="after")
    def require_chat(self):
        """Require an explicitly configured chat model for enabled model stages."""
        if self.embedding_profile is not None:
            profile = self.embedding_profile
            if (profile.model, profile.dimensions) != (self.embedding.model, self.embedding.dimensions):
                raise ValueError("Embedding profile must match the configured model and dimensions")
            if profile.node_template not in ("scene_graph_nodes_v2", "scene_graph_nodes_v3"):
                raise ValueError("Unsupported node embedding template")
            if profile.query_template not in ("legacy_query_v1", "scene_graph_query_v2"):
                raise ValueError("Unsupported query embedding template")
        if self.embedding.space_id is not None and (
            self.embedding_profile is None or self.embedding_profile.space_id != self.embedding.space_id
        ):
            raise ValueError("Pinned Platform embeddings require their matching embedding profile")
        if self.skill_seeds > self.skill_candidates or self.scene_seeds > self.scene_candidates:
            raise ValueError("Seed budgets cannot exceed candidate budgets")
        if (
            self.llm_linking or self.scene_llm_linking or self.query_rewrite or self.enrich_skill
        ) and self.chat is None:
            raise ValueError("llm_linking, scene_llm_linking, query_rewrite and enrich_skill require chat settings")
        return self
