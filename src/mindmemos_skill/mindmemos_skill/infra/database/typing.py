"""Registered schemas, query conditions, and portable search results."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import DateTime, Table
from sqlalchemy.types import TypeDecorator


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


class TableSpec(BaseModel):
    """Bind a business-owned table to its record schema."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, extra="forbid")
    """Keep registration bindings immutable and accept SQLAlchemy tables."""

    name: str = Field(min_length=1)
    """Public table name used by data operations."""

    table: Table
    """Business-owned SQLAlchemy table, including constraints and indexes."""

    insert_model: type[BaseModel]
    """Pydantic schema for complete insertable records."""

    @model_validator(mode="after")
    def check_schema(self):
        """Reject unusable schema bindings before accessing a database."""
        if self.name != self.name.strip() or not len(self.table.primary_key):
            raise ValueError("A trimmed name and a primary key are required")
        columns = set(self.table.c.keys())
        if not set(self.insert_model.model_fields) <= columns:
            raise ValueError("Validation model contains fields absent from the table")
        if any(f.alias or f.validation_alias or f.serialization_alias for f in self.insert_model.model_fields.values()):
            raise ValueError("Table validation models must use column names without aliases")
        for column in self.table.c:
            generated = (
                column.default is not None
                or column.server_default is not None
                or column is self.table.autoincrement_column
            )
            if not column.nullable and not generated and column.key not in self.insert_model.model_fields:
                raise ValueError(f"Required column {column.key!r} is absent from insert_model")
        return self


class Filter(BaseModel):
    """One condition; multiple conditions are combined with AND."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject misspelled query options."""

    field: str
    """Registered column name."""

    op: Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "is_null", "not_null", "starts_with"] = "eq"
    """Comparison operator."""

    value: Any = None
    """Comparison operand; omitted for null checks."""


class SearchHit(BaseModel):
    """A matching record and its cosine distance."""

    record: dict[str, Any]
    """Requested fields from the matching row."""

    distance: float
    """Cosine distance; smaller values indicate closer matches."""


class Scope(BaseModel):
    """Immutable equality boundaries for arbitrary registered columns."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject undeclared options and prevent scope changes during a request."""

    fields: tuple[tuple[str, Any], ...]
    """Column names and required values, stored as an immutable snapshot."""

    @field_validator("fields", mode="before")
    @classmethod
    def normalize_fields(cls, value):
        """Accept a mapping while keeping the validated scope immutable."""
        if isinstance(value, Mapping):
            return tuple(value.items())
        return value

    @model_validator(mode="after")
    def check_fields(self):
        """Reject empty, duplicate, or unusable isolation conditions."""
        names = [name for name, _ in self.fields]
        if (
            not names
            or len(names) != len(set(names))
            or any(
                not isinstance(name, str) or not name or name != name.strip() or value is None or value == ""
                for name, value in self.fields
            )
        ):
            raise ValueError("Scope fields must be unique, nonempty name-value pairs")
        for _, value in self.fields:
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError("Scope values must be immutable scalar values") from exc
        object.__setattr__(self, "fields", tuple(sorted(self.fields)))
        return self

    def require(self, field: str) -> Any:
        """Return a required dimension selected by the owning layer."""
        for name, value in self.fields:
            if name == field:
                return value
        raise ValueError(f"Scope is missing required field {field!r}")
