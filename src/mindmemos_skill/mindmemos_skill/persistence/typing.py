"""Shared record validation and portable UTC timestamp types."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict
from sqlalchemy import DateTime
from sqlalchemy.types import TypeDecorator


class RecordModel(BaseModel):
    """Reject unknown fields and preserve complete immutable input records."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Prevent accidental field reassignment after validation."""


class UTCDateTime(TypeDecorator):
    """Normalize aware timestamps and restore UTC on SQLite reads."""

    impl = DateTime(timezone=True)
    cache_ok = True

    @property
    def python_type(self):
        """Expose the native type to infra record validation."""
        return datetime

    def process_bind_param(self, value, dialect):
        """Normalize an aware timestamp before storage.

        Args:
            value: Timestamp or None.
            dialect: Active SQLAlchemy backend dialect.
        """
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamps must include a timezone")
        return value.astimezone(UTC)

    def process_result_value(self, value, dialect):
        """Restore UTC when the backend returns a naive datetime.

        Args:
            value: Stored timestamp or None.
            dialect: Active SQLAlchemy backend dialect.
        """
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
