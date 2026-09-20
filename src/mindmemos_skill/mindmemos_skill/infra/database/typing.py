"""Registered schemas, query conditions, and portable search results."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import Table


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

    op: Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "is_null", "not_null"] = "eq"
    """Comparison operator."""

    value: Any = None
    """Comparison operand; omitted for null checks."""


class SearchHit(BaseModel):
    """A matching record and its cosine distance."""

    record: dict[str, Any]
    """Requested fields from the matching row."""

    distance: float
    """Cosine distance; smaller values indicate closer matches."""
