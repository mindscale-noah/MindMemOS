"""Shared record validation and portable UTC timestamp types."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecordModel(BaseModel):
    """Reject unknown fields and preserve complete immutable input records."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent accidental field reassignment after validation."""


class TransactionState(BaseModel):
    """Remember a failed repository operation until the shared transaction exits."""

    failed: bool = False
    """Whether an operation failed, including an exception caught by its caller."""


class TrajectoryCounts(BaseModel):
    """Counts of stored attempts for one exact Skill version."""

    total: int = Field(ge=0)
    """Number of local and reported trajectories associated with this version."""

    consumed: int = Field(ge=0)
    """Number of distinct trajectories successfully processed by evolution."""

    unconsumed: int = Field(ge=0)
    """Number of trajectories never successfully processed by evolution."""


class LegacySkillIdMapping(RecordModel):
    """Map an imported Skill or version within its original project."""

    project_id: str
    """Project shared by the original and imported records."""

    kind: Literal["family", "version"]
    """Identity namespace of the imported record."""

    legacy_id: str
    """Exact ID in the original Memory service."""

    target_id: str
    """Stable ID in the independent Skill service."""
