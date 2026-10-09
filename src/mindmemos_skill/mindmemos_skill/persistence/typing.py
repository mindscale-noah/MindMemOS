"""Shared record validation and portable UTC timestamp types."""

from pydantic import BaseModel, ConfigDict, Field


class RecordModel(BaseModel):
    """Reject unknown fields and preserve complete immutable input records."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent accidental field reassignment after validation."""


class TrajectoryCounts(BaseModel):
    """Counts of stored attempts for one exact Skill version."""

    total: int = Field(ge=0)
    """Number of local and reported trajectories associated with this version."""

    consumed: int = Field(ge=0)
    """Number of distinct trajectories successfully processed by evolution."""

    unconsumed: int = Field(ge=0)
    """Number of trajectories never successfully processed by evolution."""
