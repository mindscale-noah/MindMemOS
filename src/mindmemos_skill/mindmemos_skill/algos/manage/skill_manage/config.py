"""Configuration for persisted graph-based Skill selection."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents import OpenAIChatConfig
from ....infra.embedding import OpenAIEmbeddingConfig


class SkillManageConfig(BaseModel):
    """Configure graph construction, query rewriting, and ranked retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject misspelled settings and freeze algorithm options."""

    embedding: OpenAIEmbeddingConfig
    """Explicit model and dimension used for graph and query vectors."""

    chat: OpenAIChatConfig | None = None
    """Chat settings required when relation inference or query rewriting is enabled."""

    build_scene: bool = True
    """Build and embed scenes from caller-supplied records."""

    llm_linking: bool = True
    """Add model-proposed semantic, workflow, and alternative relations."""

    query_rewrite: bool = True
    """Extract a structured public task query before embedding and reranking."""

    mode: Literal["skill_only", "unified_ppr"] = "unified_ppr"
    """Rank Skills alone or propagate through the joint Skill/Scene graph."""

    dependency_match_threshold: float = Field(default=0.5, ge=0, le=1)
    """Minimum schema-overlap score for a deterministic dependency edge."""

    seed_top_k: int = Field(default=20, gt=0, strict=True)
    """Maximum vector-neighbor nodes used to seed personalized PageRank."""

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
    """Maximum candidate Skill pairs sent in one relation-inference request."""

    @model_validator(mode="after")
    def require_chat(self):
        """Require an explicitly configured chat model for enabled model stages."""
        if (self.llm_linking or self.query_rewrite) and self.chat is None:
            raise ValueError("llm_linking and query_rewrite require chat settings")
        return self
