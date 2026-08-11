"""Business-neutral contracts for structured database backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping, TypeAlias

from .scope import DatabaseScope

ComparisonOperator = Literal[
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "not_in",
    "contains",
    "icontains",
    "is_empty",
    "is_null",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class Predicate:
    field: str
    op: ComparisonOperator
    value: Any = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FilterGroup:
    operator: Literal["and", "or", "not"] = "and"
    clauses: tuple["FilterExpression", ...] = field(default_factory=tuple)


FilterExpression: TypeAlias = Predicate | FilterGroup


@dataclass(frozen=True, slots=True, kw_only=True)
class Sort:
    field: str
    direction: Literal["asc", "desc"] = "asc"


@dataclass(frozen=True, slots=True, kw_only=True)
class Page:
    limit: int = 50
    cursor: str | None = None

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("page limit must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordQuery:
    scope: DatabaseScope = field(default_factory=DatabaseScope)
    filters: FilterExpression | None = None
    sort: tuple[Sort, ...] = field(default_factory=tuple)
    page: Page = field(default_factory=Page)


@dataclass(frozen=True, slots=True, kw_only=True)
class Record:
    table: str
    record_id: str
    payload: Mapping[str, Any]
    scope: DatabaseScope = field(default_factory=DatabaseScope)

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("record table must not be empty")
        if not self.record_id:
            raise ValueError("record_id must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class DatabaseCapabilities:
    metadata_filtering: bool = True
    batch_record_io: bool = True
    atomic_batch_write: bool = False
    transactions: bool = False
    compare_and_swap: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class DatabaseRequirements:
    metadata_filtering: bool = False
    batch_record_io: bool = False
    atomic_batch_write: bool = False
    transactions: bool = False
    compare_and_swap: bool = False

    def missing_from(self, available: DatabaseCapabilities) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name in (
                "metadata_filtering",
                "batch_record_io",
                "atomic_batch_write",
                "transactions",
                "compare_and_swap",
            )
            if getattr(self, field_name) and not getattr(available, field_name)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DatabaseConfig:
    provider: str = "sqlite"
    options: Mapping[str, Any] = field(default_factory=dict)
    required: DatabaseRequirements = field(default_factory=DatabaseRequirements)


class FieldType(StrEnum):
    UUID = "uuid"
    TEXT = "text"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT_ARRAY = "text_array"
    UUID_ARRAY = "uuid_array"
    JSON = "json"


class IndexKind(StrEnum):
    BTREE = "btree"
    FULL_TEXT = "full_text"


@dataclass(frozen=True, slots=True, kw_only=True)
class FieldSpec:
    name: str
    field_type: FieldType
    nullable: bool = True
    default: Any = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexSpec:
    name: str
    fields: tuple[str, ...]
    unique: bool = False
    kind: IndexKind = IndexKind.BTREE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("index name must not be empty")
        if not self.fields:
            raise ValueError("an index requires at least one field")


@dataclass(frozen=True, slots=True, kw_only=True)
class TableSpec:
    """One structured logical table; persistence owns concrete declarations."""

    name: str
    primary_key: str
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)
    indexes: tuple[IndexSpec, ...] = field(default_factory=tuple)
    scope_scoped: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.primary_key:
            raise ValueError("table name and primary key must not be empty")
        field_names = {spec.name for spec in self.fields}
        if len(field_names) != len(self.fields):
            raise ValueError(f"table {self.name!r} contains duplicate fields")
        if self.primary_key not in field_names:
            raise ValueError(f"table {self.name!r} must declare its primary key field {self.primary_key!r}")
        index_names = [index.name for index in self.indexes]
        duplicate_indexes = sorted({name for name in index_names if index_names.count(name) > 1})
        if duplicate_indexes:
            raise ValueError(f"table {self.name!r} contains duplicate indexes: {duplicate_indexes}")
        for index in self.indexes:
            unknown = set(index.fields) - field_names
            if unknown:
                raise ValueError(f"index {index.name!r} references unknown fields: {sorted(unknown)}")


@dataclass(frozen=True, slots=True, kw_only=True)
class SchemaMigration:
    """One immutable, ordered, forward-only schema migration.

    ``tables`` identifies the latest catalog entries affected by the step.
    Backend statements are intentionally stored on the migration itself so its
    checksum never depends on a mutable latest :class:`TableSpec`.

    An initial migration may omit statements: a fresh database is created from
    the latest catalog and all registered migrations are stamped atomically.
    Existing databases execute only statements for versions not yet recorded.
    """

    namespace: str
    version: int
    name: str
    tables: tuple[str, ...]
    sqlite_statements: tuple[str, ...] = field(default_factory=tuple)
    postgres_statements: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise ValueError("schema migration namespace must not be empty")
        if self.version <= 0:
            raise ValueError("schema migration version must be positive")
        if not self.name.strip():
            raise ValueError("schema migration name must not be empty")
        if not self.tables:
            raise ValueError("schema migration must reference at least one table")
        if len(self.tables) != len(set(self.tables)):
            raise ValueError("schema migration tables may not contain duplicates")
        if any(not statement.strip() for statement in self.sqlite_statements):
            raise ValueError("sqlite migration statements must not be empty")
        if any(not statement.strip() for statement in self.postgres_statements):
            raise ValueError("postgres migration statements must not be empty")

    def statements_for(self, provider: str) -> tuple[str, ...]:
        if provider == "sqlite":
            return self.sqlite_statements
        if provider in {"postgres", "postgresql"}:
            return self.postgres_statements
        raise ValueError(f"unsupported migration provider: {provider!r}")


__all__ = [
    "ComparisonOperator",
    "DatabaseCapabilities",
    "DatabaseConfig",
    "DatabaseRequirements",
    "FieldSpec",
    "FieldType",
    "FilterExpression",
    "FilterGroup",
    "IndexKind",
    "IndexSpec",
    "Page",
    "Predicate",
    "Record",
    "RecordQuery",
    "SchemaMigration",
    "Sort",
    "TableSpec",
]
