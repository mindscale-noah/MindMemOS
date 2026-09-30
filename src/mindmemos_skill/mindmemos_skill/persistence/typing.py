"""Shared record validation and portable UTC timestamp types."""

from pydantic import BaseModel, ConfigDict


class RecordModel(BaseModel):
    """Reject unknown fields and preserve complete immutable input records."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent accidental field reassignment after validation."""
