"""Backend-neutral storage structures.

These types preserve the semantics currently consumed by search and dreaming
without carrying Qdrant models, Cypher fragments, SQL expressions, or driver
objects across the database boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping, TypeAlias

from .scope import DatabaseScope

JsonObject = dict[str, Any]
GraphDirection = Literal["out", "in", "both"]

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


def scope_predicates(scope: DatabaseScope) -> tuple[Predicate, ...]:
    """Translate all non-null scope values into exact-match predicates."""

    return tuple(Predicate(field=field_name, op="eq", value=value) for field_name, value in scope.items())


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
    scope: DatabaseScope
    filters: FilterExpression | None = None
    sort: tuple[Sort, ...] = field(default_factory=tuple)
    page: Page = field(default_factory=Page)


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorQuery:
    table: str
    scope: DatabaseScope
    vector_name: str
    dense_vector: tuple[float, ...] | None = None
    sparse_indices: tuple[int, ...] | None = None
    sparse_values: tuple[float, ...] | None = None
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    filters: FilterExpression | None = None
    top_k: int = 10
    dense_limit: int | None = None
    sparse_limit: int | None = None
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.dense_limit is not None and self.dense_limit <= 0:
            raise ValueError("dense_limit must be positive when provided")
        if self.sparse_limit is not None and self.sparse_limit <= 0:
            raise ValueError("sparse_limit must be positive when provided")
        if self.mode in {"dense", "hybrid"} and self.dense_vector is None:
            raise ValueError(f"{self.mode} search requires a dense vector")
        if self.mode in {"sparse", "hybrid"}:
            if self.sparse_indices is None or self.sparse_values is None:
                raise ValueError(f"{self.mode} search requires sparse indices and values")
            if len(self.sparse_indices) != len(self.sparse_values):
                raise ValueError("sparse query indices and values must have the same length")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphStep:
    """One repeatable segment in a portable graph path pattern.

    ``target_filters`` fields address the source business record referenced by
    a candidate node. ``edge_filters`` fields use ``key.<name>`` or
    ``properties.<name>`` to distinguish immutable edge identity from mutable
    graph properties. An empty relation or kind tuple means "any". The same
    constraints apply to every repeated hop.
    """

    relations: tuple[str, ...] = field(default_factory=tuple)
    direction: GraphDirection = "both"
    target_kinds: tuple[str, ...] = field(default_factory=tuple)
    edge_filters: FilterExpression | None = None
    target_filters: FilterExpression | None = None
    min_hops: int = 1
    max_hops: int = 1

    def __post_init__(self) -> None:
        if self.direction not in {"out", "in", "both"}:
            raise ValueError(f"unsupported graph step direction: {self.direction!r}")
        if self.min_hops <= 0:
            raise ValueError("graph step min_hops must be positive")
        if self.max_hops < self.min_hops:
            raise ValueError("graph step max_hops must be greater than or equal to min_hops")
        if any(not relation for relation in self.relations):
            raise ValueError("graph step relations must not contain empty names")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphSort:
    """Portable ordering over a returned path."""

    scope: Literal["end_node", "last_edge", "path"]
    field: str
    direction: Literal["asc", "desc"] = "asc"
    nulls: Literal["first", "last"] = "last"

    def __post_init__(self) -> None:
        if self.scope not in {"end_node", "last_edge", "path"}:
            raise ValueError(f"unsupported graph sort scope: {self.scope!r}")
        if not self.field:
            raise ValueError("graph sort field must not be empty")
        if self.scope == "path" and self.field != "length":
            raise ValueError("the only portable path sort field is 'length'")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphTraversalQuery:
    """A bounded, backend-neutral multi-segment graph traversal.

    ``VectorDBService`` evaluates ``steps`` in order using backend record
    primitives. ``truncated`` must be true when any result, per-seed,
    expansion, or timeout bound prevented complete evaluation, so callers
    never confuse a partial traversal with a full one.
    """

    scope: DatabaseScope
    seeds: tuple[GraphNodeRef, ...]
    steps: tuple[GraphStep, ...]
    path_uniqueness: Literal["node", "edge", "walk"] = "node"
    result_uniqueness: Literal["path", "end_node"] = "path"
    order_by: tuple[GraphSort, ...] = field(default_factory=tuple)
    limit: int = 200
    limit_per_seed: int | None = None
    max_expansions: int = 10_000
    timeout_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("graph traversal requires at least one seed")
        if not self.steps:
            raise ValueError("graph traversal requires at least one step")
        if any(not self.scope.matches(seed.scope) for seed in self.seeds):
            raise ValueError("all graph traversal seeds must match the query scope")
        if self.path_uniqueness not in {"node", "edge", "walk"}:
            raise ValueError(f"unsupported path uniqueness: {self.path_uniqueness!r}")
        if self.result_uniqueness not in {"path", "end_node"}:
            raise ValueError(f"unsupported result uniqueness: {self.result_uniqueness!r}")
        if self.limit <= 0:
            raise ValueError("graph traversal limit must be positive")
        if self.limit_per_seed is not None and self.limit_per_seed <= 0:
            raise ValueError("graph traversal limit_per_seed must be positive")
        if self.max_expansions <= 0:
            raise ValueError("graph traversal max_expansions must be positive")
        if self.timeout_ms is not None and self.timeout_ms <= 0:
            raise ValueError("graph traversal timeout_ms must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendCapabilities:
    """Features offered by one vector/document backend instance.

    Graph traversal is intentionally absent.  It is implemented by
    ``VectorDBService`` from record and metadata operations, so a backend only
    needs to advertise the primitives that service composes.
    """

    dense_vector: bool = True
    sparse_vector: bool = False
    hybrid_search: bool = False
    metadata_filtering: bool = True
    batch_record_io: bool = True
    atomic_batch_write: bool = False
    max_vector_dimensions: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendRequirements:
    """Features required by one application/backend configuration.

    Requirements use false defaults so selecting a backend without an
    explicit capability profile does not accidentally require every optional
    vector feature.
    """

    dense_vector: bool = False
    sparse_vector: bool = False
    hybrid_search: bool = False
    metadata_filtering: bool = False
    batch_record_io: bool = False
    atomic_batch_write: bool = False
    max_vector_dimensions: int | None = None

    def missing_from(self, available: BackendCapabilities) -> tuple[str, ...]:
        missing: list[str] = []
        for field_name in (
            "dense_vector",
            "sparse_vector",
            "hybrid_search",
            "metadata_filtering",
            "batch_record_io",
            "atomic_batch_write",
        ):
            if getattr(self, field_name) and not getattr(available, field_name):
                missing.append(field_name)
        if self.max_vector_dimensions is not None and (
            available.max_vector_dimensions is None or available.max_vector_dimensions < self.max_vector_dimensions
        ):
            missing.append(f"vector_dimensions>={self.max_vector_dimensions}")
        return tuple(missing)


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendConfig:
    provider: str
    options: Mapping[str, Any] = field(default_factory=dict)
    required: BackendRequirements = field(default_factory=BackendRequirements)


@dataclass(frozen=True, slots=True, kw_only=True)
class SparseVector:
    """Sparse vector independent of a vector database SDK."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("sparse vector indices and values must have the same length")
        if any(index < 0 for index in self.indices):
            raise ValueError("sparse vector indices must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorValue:
    """Named dense/sparse vectors attached to one logical record."""

    dense: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    sparse: Mapping[str, SparseVector] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Record:
    """One row/document/point in a logical table."""

    table: str
    record_id: str
    scope: DatabaseScope
    payload: Mapping[str, Any]
    vectors: VectorValue | None = None

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError("record table must not be empty")
        if not self.record_id:
            raise ValueError("record_id must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorHit:
    """Backend-neutral vector or hybrid-search result."""

    record: Record
    score: float
    source: str
    debug: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphNodeRef:
    """Canonical identity of a node in every supported backend."""

    scope: DatabaseScope
    kind: str
    node_id: str

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("graph node references require kind")
        if not self.node_id:
            raise ValueError("graph node references require node_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphNode:
    """Graph-owned node identity pointing at an application record."""

    ref: GraphNodeRef


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphEdge:
    """A typed, directed edge with a stable identity key.

    ``edge_key`` contains application-defined immutable identity dimensions.
    It is deliberately separate from mutable ``properties`` so every backend
    can implement idempotent upsert without knowing business relation types.
    """

    source: GraphNodeRef
    target: GraphNodeRef
    relation: str
    edge_key: Mapping[str, Any] = field(default_factory=dict)
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source.scope != self.target.scope:
            raise ValueError("cross-scope graph edges are not allowed")
        if not self.relation:
            raise ValueError("graph relation must not be empty")

    @property
    def scope(self) -> DatabaseScope:
        return self.source.scope


@dataclass(frozen=True, slots=True, kw_only=True)
class TraversedGraphEdge:
    """One stored edge together with the direction used while traversing it."""

    edge: GraphEdge
    direction: Literal["out", "in"]

    def __post_init__(self) -> None:
        if self.direction not in {"out", "in"}:
            raise ValueError(f"unsupported traversed edge direction: {self.direction!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphPath:
    """A backend-neutral path; ``nodes[0]`` is always the originating seed."""

    seed: GraphNodeRef
    nodes: tuple[GraphNode, ...]
    edges: tuple[TraversedGraphEdge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a graph path requires at least its seed node")
        if self.nodes[0].ref != self.seed:
            raise ValueError("the first graph path node must match its seed")
        if len(self.nodes) != len(self.edges) + 1:
            raise ValueError("a graph path must contain exactly one more node than edge")
        for index, traversed in enumerate(self.edges):
            current = self.nodes[index].ref
            following = self.nodes[index + 1].ref
            if traversed.direction == "out":
                expected = (current, following)
            else:
                expected = (following, current)
            if (traversed.edge.source, traversed.edge.target) != expected:
                raise ValueError("graph path edge orientation does not connect adjacent nodes")

    @property
    def end(self) -> GraphNode:
        return self.nodes[-1]


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphTraversalResult:
    """Bounded traversal output plus an explicit partial-result indicator."""

    paths: tuple[GraphPath, ...]
    truncated: bool = False
    expanded_nodes: int = 0

    def __post_init__(self) -> None:
        if self.expanded_nodes < 0:
            raise ValueError("expanded_nodes must not be negative")


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
    HASH = "hash"

@dataclass(frozen=True, slots=True, kw_only=True)
class FieldSpec:
    name: str
    field_type: FieldType
    nullable: bool = True
    default: Any = None


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexSpec:
    name: str
    fields: tuple[str, ...]
    unique: bool = False
    kind: IndexKind = IndexKind.BTREE

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("an index requires at least one field")


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorFieldSpec:
    name: str
    dimensions: int
    distance: Literal["cosine", "euclidean", "dot"] = "cosine"
    sparse: bool = False

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("vector dimensions must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class TableSpec:
    """One logical table; adapters decide its physical representation.

    For ``scope_scoped`` tables, ``Record.scope`` is an envelope outside the
    declared payload fields. Adapters must apply primary-key and unique-index
    semantics inside that scope and may store its arbitrary dimensions as
    JSON, metadata paths, typed columns, or another native representation.
    """

    name: str
    primary_key: str
    fields: tuple[FieldSpec, ...] = field(default_factory=tuple)
    indexes: tuple[IndexSpec, ...] = field(default_factory=tuple)
    vectors: tuple[VectorFieldSpec, ...] = field(default_factory=tuple)
    scope_scoped: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.primary_key:
            raise ValueError("table name and primary key must not be empty")
        field_names = {spec.name for spec in self.fields}
        if len(field_names) != len(self.fields):
            raise ValueError(f"table {self.name!r} contains duplicate fields")
        vector_names = {spec.name for spec in self.vectors}
        if len(vector_names) != len(self.vectors):
            raise ValueError(f"table {self.name!r} contains duplicate vector fields")
        for index in self.indexes:
            unknown = set(index.fields) - field_names - {self.primary_key}
            if unknown:
                raise ValueError(f"index {index.name!r} references unknown fields: {sorted(unknown)}")
