"""Shared record validation and portable UTC timestamp types."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


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


class LegacySkillSyncProject(RecordModel):
    """Checkpoint for one namespace authorized by the Platform registry."""

    project_id: str
    ready: bool = False
    generation: int = 0
    lease_until: AwareDatetime | None = None
    trajectory_cursor: str | int | None = None
    last_checked_at: AwareDatetime | None = None
    failures: int = 0


class LegacySkillSyncItem(RecordModel):
    """Stable mapping and most recent source observation for one legacy object."""

    project_id: str
    kind: Literal["family", "version", "trajectory"]
    legacy_id: str
    target_id: str | None = None
    generation: int = 0
    pending: bool = False
    source_deleted: bool = False
    published_version_id: str | None = None
