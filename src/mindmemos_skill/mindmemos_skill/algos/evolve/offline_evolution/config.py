"""Evidence and change gate execution settings."""

import sys
from collections.abc import Collection
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ....agents.buitin_agents.claude.config import ClaudeConfig
from ....agents.buitin_agents.codex.config import CodexConfig
from ....agents.config import AgentConfig


class EvidenceGateConfig(BaseModel):
    """Configure collector execution without model connections or credentials."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)
    """Reject unsupported collector settings and expose field documentation."""

    script_timeout_s: float = Field(default=900, gt=0, allow_inf_nan=False)
    """Maximum seconds to execute one evidence script."""

    max_invalid_share: float = Field(default=0.1, ge=0, le=1)
    """Invalid finding fraction allowed per collection; at least one invalid entry is tolerated."""

    python: str = Field(default=sys.executable, min_length=1)
    """Collector interpreter for both formal and preview evidence gates."""


class OfflineEvolutionConfig(EvidenceGateConfig):
    """Configure shared coding sessions and deterministic gate execution."""

    agent: Annotated[
        ClaudeConfig | CodexConfig, AgentConfig(allowed=("claude", "codex"), namespace="offline_evolution")
    ] = Field(default_factory=lambda: ClaudeConfig(timeout_s=3600))
    """Shared native Agent settings or a preset; defaults to Claude."""

    retries: int = Field(default=1, ge=0)
    """Feedback rounds per stage before retaining the final gated result."""


class EvolutionOptions(BaseModel):
    """Separate readable inputs from editable members and creation permission."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_attribute_docstrings=True)
    """Freeze policy values so algorithms cannot widen the caller's permissions."""

    editable_skills: tuple[str, ...] | None = None
    """Editable input IDs; None permits all inputs, while [] makes all inputs read-only."""

    allow_create_skills: bool = True
    """Whether new independent Skills may be proposed."""

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Reject blank or duplicate member identities."""
        if self.editable_skills is not None:
            if any(not item.strip() or item != item.strip() for item in self.editable_skills):
                raise ValueError("Editable Skill IDs must be nonempty and have no surrounding whitespace")
            if len(set(self.editable_skills)) != len(self.editable_skills):
                raise ValueError("Editable Skill IDs must be unique")
        return self

    def allowed_ids(self, input_ids: Collection[str]) -> frozenset[str]:
        """Resolve and validate the editable subset of this call.

        Args:
            input_ids: Stable identities of the supplied input collection.

        Returns:
            Input identities whose contents may be modified.

        Raises:
            ValueError: The policy names a member outside the input collection.
        """
        allowed = frozenset(input_ids if self.editable_skills is None else self.editable_skills)
        unknown = allowed.difference(input_ids)
        if unknown:
            raise ValueError(f"Editable Skills are outside the input collection: {sorted(unknown)}")
        return allowed
