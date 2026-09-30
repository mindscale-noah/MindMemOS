"""Validated Scene records and structured model outputs."""

from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ....agents import OpenAIChatConfig
from ....agents.typing import AgentResult
from ....persistence.db_services.skill_management.typing import SkillGraphEdgeRecord, SkillGraphNodeRecord


class Scene(BaseModel):
    """One caller-authored scenario within a Skill."""

    model_config = ConfigDict(extra="allow", frozen=True)
    """Retain additional caller-supplied SSL fields without interpreting them."""

    scene_id: str = Field(min_length=1)
    """Stable scene identity within its owning Skill."""

    scene_name: str = Field(min_length=1)
    """Short display name and capability statement."""

    scene_goal: str = Field(min_length=1)
    """Task goal used for scene embedding and retrieval."""

    scene_type: str = ""
    """Optional caller-defined scene category."""

    next_scene_rules: list[str] = Field(default_factory=list)
    """IDs of subsequent scenes in the same caller-supplied record."""

    @field_validator("next_scene_rules", mode="before")
    @classmethod
    def normalize_rules(cls, values):
        """Accept source SSL rule objects as well as explicit target IDs.

        Args:
            values: Rule strings or objects containing next_scene_id.
        """
        return [str(value.get("next_scene_id", "")) if isinstance(value, dict) else str(value) for value in values]


class SceneRecord(BaseModel):
    """A complete explicit Scene graph for one collection member."""

    model_config = ConfigDict(extra="allow", frozen=True)
    """Retain caller-supplied SSL metadata and logic steps."""

    scenes: list[Scene] = Field(default_factory=list)
    """Ordered scenes belonging to one Skill."""

    @model_validator(mode="after")
    def validate_links(self):
        """Reject duplicate scene identities and dangling flow references."""
        ids = {scene.scene_id for scene in self.scenes}
        if len(ids) != len(self.scenes):
            raise ValueError("Scene IDs must be unique within a Skill")
        for scene in self.scenes:
            if any(target not in ids or target == scene.scene_id for target in scene.next_scene_rules):
                raise ValueError("Scene flow targets must identify another scene in the same record")
        return self


class Relation(BaseModel):
    """One model-proposed directed relation using opaque node IDs."""

    model_config = ConfigDict(extra="forbid")
    """Reject unrecognized model-generated relationship attributes."""

    source_id: str
    """Exact source node ID from the supplied candidate pair."""

    target_id: str
    """Exact destination node ID from the supplied candidate pair."""

    edge_type: Literal["semantic", "workflow", "alternative"]
    """Additional relation type; deterministic dependency edges are preserved."""

    weight: float = Field(gt=0, le=1, allow_inf_nan=False)
    """Finite relationship strength used for graph propagation."""

    evidence: str = Field(min_length=1)
    """Explanation grounded in the supplied source descriptions."""


class Relations(BaseModel):
    """Structured relationship response for a bounded candidate batch."""

    model_config = ConfigDict(extra="forbid")
    """Reject unknown response envelope fields."""

    relations: list[Relation]
    """Supported relationships, or an empty list when no relation is justified."""


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


class ChatModel(Protocol):
    """Structural port shared by the existing chat agent and test doubles."""

    config: OpenAIChatConfig
    """Recorded model settings; injected clients must match configured settings."""

    def run(self, *, system: str, user: str, response_format: dict) -> AgentResult:
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

    snapshot_id: UUID
    """Pinned persisted graph content identity."""

    nodes: list[SkillGraphNodeRecord]
    """Validated Skill and Scene records loaded from storage."""

    edges: list[SkillGraphEdgeRecord]
    """Validated relationships within this snapshot."""

    skill_ids: frozenset[str]
    """Complete allowed collection member IDs."""
