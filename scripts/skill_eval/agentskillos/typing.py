"""Validated source and HTTP result schemas for retrieval evaluation."""

from pydantic import BaseModel, Field, model_validator


class QueryRecord(BaseModel):
    """One public benchmark instruction."""

    task_id: str = Field(min_length=1)
    """Unique query identity from the source JSONL."""

    instruction_text: str = Field(min_length=1)
    """Public retrieval input; excludes private relevance labels."""


class RelevanceRecord(BaseModel):
    """Legacy binary relevance annotations."""

    core_gt_ids: list[str] | None = None
    """Primary source Skill IDs; None falls back to gt_skill_ids."""

    gt_skill_ids: list[str] = Field(default_factory=list)
    """All annotated source Skill IDs."""

    task_type: str = "clean"
    """generic_only queries are excluded from core scoring."""

    @model_validator(mode="after")
    def validate_core_labels(self):
        """Reject ambiguous labels and empty non-generic reference sets."""
        labels = self.core_gt_ids if self.core_gt_ids is not None else self.gt_skill_ids
        if len(labels) != len(set(labels)) or any(not item.strip() for item in labels):
            raise ValueError("Core labels must contain unique nonblank IDs")
        if not labels and self.task_type != "generic_only":
            raise ValueError("Non-generic queries require core labels")
        return self


class PoolMember(BaseModel):
    """Source directory and benchmark identity of one candidate."""

    source_skill_id: str = Field(min_length=1)
    """Source identity used in private annotations."""

    member_id: str = Field(min_length=1)
    """Canonical benchmark identity supplied by the source join."""

    relative_path: str
    """Skill directory relative to the dataset root."""


class PoolManifest(BaseModel):
    """Prepared dataset candidate pool."""

    skills: list[PoolMember] = Field(min_length=1)
    """Complete candidate collection, including distractors."""


class RegistrationResult(BaseModel):
    """Exact identities returned by SDK registration."""

    skill_id: str
    """Service-assigned Skill family identity."""

    skill_version_id: str
    """Immutable registered version used for building and scoring."""


class GraphResult(BaseModel):
    """Immutable publication returned by SDK organization."""

    graph_id: str
    """Service-assigned graph identity."""

    graph_version_id: str
    """Pinned snapshot for every query in this run."""

    skill_version_ids: list[str]
    """Complete published candidate version collection."""


class RecommendationScore(BaseModel):
    """Ordered identity from a strategy-specific score record."""

    skill_version_id: str
    """Immutable candidate version; other score fields are ignored."""


class RecommendationResult(BaseModel):
    """Retrieval result needed to verify snapshot and ranking."""

    graph_version_id: str
    """Snapshot actually queried by the service."""

    scores: list[RecommendationScore]
    """Candidate versions in descending recommendation order."""
