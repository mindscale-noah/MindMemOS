"""Vanilla-memory persistence over the backend-neutral vector DB service.

This module is the business/storage translation boundary.  Migrated vanilla
components keep using their original ``MemoryDb*`` DTOs while this class maps
them to persistence-v2 tables, scoped records, named vectors, and portable
graph primitives.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from ..components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ..config import TextProcessingConfig, get_config
from ..errors import MemoryUpdateError
from ..infra.vector_store import (
    DatabaseScope,
    FilterExpression,
    FilterGroup,
    GraphEdge,
    GraphNode,
    GraphNodeRef,
    GraphStep,
    GraphTraversalQuery,
    Page,
    Predicate,
    Record,
    RecordQuery,
    ScopeValue,
    Sort,
    SparseVector,
    VectorDBService,
    VectorQuery,
    VectorValue,
)
from ..llm import EmbedClient, get_embed_client
from ..logging import get_logger, traced
from ..typing import (
    FieldCondition,
    MemoryDbDeleteCommand,
    MemoryDbMemoryUpdateCommand,
    MemoryDbMutationPlan,
    MemoryDbMutationResult,
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryDbSearchResult,
    MemoryDbWriteResult,
    MemoryRequestContext,
    MemoryView,
    REL_TASK_EXPERIENCE,
    SearchFilter,
)
from .v2 import ENTITY_TABLE, MEMORY_TABLE, SOURCE_TABLE

logger = get_logger(__name__)

_SCOPE_FIELDS = (
    "account_id",
    "project_id",
    "api_key_uuid",
    "user_id",
    "app_id",
    "session_id",
    "agent_id",
)
_QUERY_CONTEXT_SCOPE_FIELDS = frozenset({"project_id"})
_TRUSTED_WRITE_SCOPE_FIELDS = frozenset({"account_id", "project_id", "api_key_uuid"})
_SCOPE_METADATA_KEY = "__scope"


class MemoryPersistence:
    """Implement the memory reads, searches, and mutations required by Lite pipelines.

    This is Lite's single business/storage translation API. Every operation is
    expressed through ``VectorDBService`` and the tables declared under
    ``persistence.v2``.
    """

    _PROTECTED_BOUND_SCOPE_FIELDS = _TRUSTED_WRITE_SCOPE_FIELDS

    def __init__(
        self,
        service: VectorDBService,
        *,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        embed_client: EmbedClient | None = None,
        _bound_scope: DatabaseScope | None = None,
    ) -> None:
        self._service = service
        self._text_config = text_config
        self._text_preprocessor = text_preprocessor
        self._sparse_encoder = sparse_encoder
        self._embed_client = embed_client
        resolved_bound_scope = _bound_scope if _bound_scope is not None else DatabaseScope()
        protected = self._PROTECTED_BOUND_SCOPE_FIELDS.intersection(dict(resolved_bound_scope.items()))
        if protected:
            names = ", ".join(sorted(protected))
            raise ValueError(f"{names} is trusted context scope and cannot be supplied through _bound_scope")
        self._bound_scope = resolved_bound_scope

    @property
    def service(self) -> VectorDBService:
        return self._service

    @property
    def bound_scope(self) -> DatabaseScope:
        """Return the immutable scope dimensions attached by ``bind``."""

        return self._bound_scope

    def bind(
        self,
        scope: Mapping[str, ScopeValue | None] | None = None,
        /,
        **dimensions: ScopeValue | None,
    ) -> "MemoryPersistence":
        """Return an isolated persistence view with additional forced scope dimensions.

        Trusted tenant dimensions remain owned by ``MemoryRequestContext`` and
        cannot be rebound by an algorithm. Bound dimensions are additive and
        conflicts fail before any database operation.
        """

        supplied = dict(scope or {})
        for name, value in dimensions.items():
            if name in supplied and supplied[name] != value:
                raise ValueError(f"scope dimension {name!r} was supplied more than once with different values")
            supplied[name] = value
        protected = self._PROTECTED_BOUND_SCOPE_FIELDS.intersection(supplied)
        if protected:
            names = ", ".join(sorted(protected))
            raise ValueError(f"{names} is trusted context scope and cannot be bound by an algorithm")
        for name, value in supplied.items():
            if value is None or value == "":
                raise ValueError(f"bound scope dimension {name!r} must have a non-empty scalar value")

        return type(self)(
            self._service,
            text_config=self._text_config,
            text_preprocessor=self._text_preprocessor,
            sparse_encoder=self._sparse_encoder,
            embed_client=self._embed_client,
            _bound_scope=_merge_scopes(self._bound_scope, DatabaseScope(supplied)),
        )

    def _scope(self, ctx: MemoryRequestContext, value: Any | None = None) -> DatabaseScope:
        """Resolve the project and optional bound scope used by read paths."""

        value_scope = (
            DatabaseScope({field: getattr(value, field, None) for field in _SCOPE_FIELDS})
            if value is not None
            else DatabaseScope()
        )
        mandatory_scope = DatabaseScope({field: getattr(ctx, field, None) for field in _QUERY_CONTEXT_SCOPE_FIELDS})
        missing = mandatory_scope.missing_required_fields(_QUERY_CONTEXT_SCOPE_FIELDS)
        if missing:
            raise ValueError(f"missing mandatory persistence scope dimensions: {', '.join(missing)}")
        return _merge_scopes(value_scope, mandatory_scope, self._bound_scope)

    def _write_scope(self, ctx: MemoryRequestContext, value: Any) -> DatabaseScope:
        """Resolve and validate scope before a value crosses the write boundary."""

        for field in sorted(_TRUSTED_WRITE_SCOPE_FIELDS):
            context_value = getattr(ctx, field, None)
            if not isinstance(context_value, str) or not context_value.strip():
                raise ValueError(f"trusted write scope dimension {field!r} must be a non-empty string")

        value_scope = DatabaseScope({field: getattr(value, field, None) for field in _SCOPE_FIELDS})
        supported_fields = tuple(field for field in _SCOPE_FIELDS if hasattr(value, field))
        context_dimensions: dict[str, ScopeValue] = {}
        for field in supported_fields:
            context_value = getattr(ctx, field, None)
            value_value = getattr(value, field, None)
            if field in _TRUSTED_WRITE_SCOPE_FIELDS:
                if not isinstance(value_value, str) or not value_value.strip():
                    raise ValueError(f"trusted write scope dimension {field!r} must be a non-empty string")
                if value_value != context_value:
                    raise ValueError(
                        f"write scope dimension {field!r} conflicts with trusted request context: "
                        f"value {value_value!r}, context {context_value!r}"
                    )
            if context_value is not None:
                context_dimensions[field] = context_value

        return _merge_scopes(value_scope, DatabaseScope(context_dimensions), self._bound_scope)

    def _validate_relationship_write_scope(self, ctx: MemoryRequestContext, relationship: Any) -> None:
        self._write_scope(ctx, relationship)
        self._write_scope(ctx, relationship.source)
        self._write_scope(ctx, relationship.target)

    def _ensure_text_components(self) -> tuple[TextPreprocessor, SparseVectorEncoder]:
        if self._text_preprocessor is None or self._sparse_encoder is None:
            config = self._text_config or get_config().algo_config.text_processing
            self._text_preprocessor = self._text_preprocessor or get_text_preprocessor(config)
            self._sparse_encoder = self._sparse_encoder or SparseVectorEncoder(config)
        return self._text_preprocessor, self._sparse_encoder

    def _ensure_embed_client(self) -> EmbedClient:
        if self._embed_client is None:
            self._embed_client = get_embed_client()
        return self._embed_client

    @traced("persistence.memory.get")
    async def get_memory(self, ctx: MemoryRequestContext, memory_id: str) -> MemoryView | None:
        record = await self._find_record(MEMORY_TABLE, ctx, memory_id)
        return _memory_view(record) if record is not None else None

    @traced("persistence.memory.get_many")
    async def get_memories(self, ctx: MemoryRequestContext, memory_ids: list[str]) -> list[MemoryView]:
        if not memory_ids:
            return []
        records = await self._records_for_ids(MEMORY_TABLE, ctx, memory_ids)
        by_id = {record.record_id: record for record in records}
        return [_memory_view(by_id[memory_id]) for memory_id in memory_ids if memory_id in by_id]

    @traced("persistence.memory.list")
    async def list_memories(
        self,
        ctx: MemoryRequestContext,
        *,
        filters: SearchFilter | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> tuple[list[MemoryView], str | None]:
        records, next_cursor = await self._service.query_records(
            MEMORY_TABLE,
            RecordQuery(
                scope=self._scope(ctx),
                filters=_with_memory_mode(ctx, _filter_expression(filters)),
                sort=(Sort(field="created_at", direction="desc"),),
                page=Page(limit=max(1, limit), cursor=cursor),
            ),
        )
        return [_memory_view(record) for record in records], next_cursor

    @traced("persistence.memory.search_sparse")
    async def search_sparse(
        self,
        ctx: MemoryRequestContext,
        query: MemoryDbSearchQuery,
        *,
        indices: list[int],
        values: list[float],
    ) -> MemoryDbSearchResult:
        hits = await self._service.search_vectors(
            VectorQuery(
                table=MEMORY_TABLE,
                scope=self._scope(ctx),
                vector_name="bm25",
                sparse_indices=tuple(indices),
                sparse_values=tuple(values),
                mode="sparse",
                filters=_with_memory_mode(ctx, _filter_expression(query.filters)),
                top_k=max(1, query.top_k),
            )
        )
        return _search_result(query.query, hits)

    @traced("persistence.memory.search_dense")
    async def search_dense(
        self,
        ctx: MemoryRequestContext,
        query: MemoryDbSearchQuery,
        *,
        dense_vector: list[float],
    ) -> MemoryDbSearchResult:
        hits = await self._service.search_vectors(
            VectorQuery(
                table=MEMORY_TABLE,
                scope=self._scope(ctx),
                vector_name="semantic",
                dense_vector=tuple(dense_vector),
                mode="dense",
                filters=_with_memory_mode(ctx, _filter_expression(query.filters)),
                top_k=max(1, query.top_k),
            )
        )
        return _search_result(query.query, hits)

    @traced("persistence.memory.search_hybrid")
    async def search_hybrid(
        self,
        ctx: MemoryRequestContext,
        query: MemoryDbSearchQuery,
        *,
        dense_vector: list[float],
        sparse_vector,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
    ) -> MemoryDbSearchResult:
        hits = await self._service.search_vectors(
            VectorQuery(
                table=MEMORY_TABLE,
                scope=self._scope(ctx),
                vector_name="semantic",
                dense_vector=tuple(dense_vector),
                sparse_indices=tuple(sparse_vector.indices),
                sparse_values=tuple(sparse_vector.values),
                mode="hybrid",
                filters=_with_memory_mode(ctx, _filter_expression(query.filters)),
                top_k=max(1, query.top_k),
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
            )
        )
        return _search_result(query.query, hits)

    @traced("persistence.memory.apply_mutation_plan")
    async def apply_mutation_plan(
        self,
        ctx: MemoryRequestContext,
        plan: MemoryDbMutationPlan,
        *,
        consistency: str = "fast",
    ) -> MemoryDbWriteResult:
        errors: list[str] = []
        graph_pending = False
        mutations: list[MemoryDbMutationResult] = []

        async def execute(label: str, operation) -> None:
            nonlocal graph_pending
            try:
                await operation()
            except Exception as exc:
                if consistency == "strong":
                    raise
                logger.warning("memory_persistence_write_failed", stage=label, error=str(exc), exc_info=True)
                errors.append(f"{label}: {exc}")
                graph_pending = graph_pending or label == "graph"

        # The orchestration context is the source of truth for mode ownership.
        # Child algorithms may keep their own extraction labels, but they do
        # not need to know how mixed-mode persistence is represented.
        memory_commands = [
            command.model_copy(
                update={
                    "memory": command.memory.model_copy(
                        update={"memory_mode": ctx.memory_algorithm or command.memory.memory_mode}
                    )
                }
            )
            for command in plan.memory_writes
        ]
        entity_commands = [*plan.entity_writes, *plan.entity_updates_as_writes()]
        source_commands = [command for command in plan.source_writes if command.source.persist_payload]
        memory_records = [
            _memory_record(command.memory, command.vector, self._write_scope(ctx, command.memory))
            for command in memory_commands
        ]
        entity_records = [
            _entity_record(command.entity, command.core_vector, self._write_scope(ctx, command.entity))
            for command in entity_commands
        ]
        source_records = [
            _source_record(command.source, self._write_scope(ctx, command.source)) for command in source_commands
        ]
        if self._service.graph_enabled:
            for command in plan.relationship_writes:
                self._validate_relationship_write_scope(ctx, command.relationship)

        await execute(
            "memory",
            lambda: self._service.upsert_records(MEMORY_TABLE, memory_records),
        )
        await execute(
            "entity",
            lambda: self._service.upsert_records(ENTITY_TABLE, entity_records),
        )
        await execute(
            "source",
            lambda: self._service.upsert_records(SOURCE_TABLE, source_records),
        )

        if self._service.graph_enabled and plan.relationship_writes:
            await execute("graph", lambda: self._write_graph(ctx, plan))

        for command in plan.memory_updates:
            try:
                mutations.append(await self._update_memory(ctx, command))
            except Exception as exc:
                if command.consistency == "strong":
                    raise
                errors.append(f"memory update {command.memory_id}: {exc}")

        for command in plan.memory_deletes:
            try:
                mutations.append(await self._delete_memory(ctx, command))
            except Exception as exc:
                if command.consistency == "strong":
                    raise
                errors.append(f"memory delete {command.memory_id}: {exc}")

        unsupported = (
            len([command for command in plan.entity_updates if command.entity is None])
            + len(plan.entity_deletes)
            + len(plan.source_updates)
            + len(plan.source_deletes)
            + len(plan.relationship_deletes)
        )
        if unsupported:
            errors.append(f"unsupported mutation commands: {unsupported}")

        return MemoryDbWriteResult(
            memory_ids=[command.memory.memory_id for command in memory_commands],
            entity_ids=[command.entity.entity_id for command in entity_commands],
            source_ids=[command.source.source_id for command in source_commands],
            mutations=mutations,
            graph_pending=graph_pending,
            errors=errors,
        )

    async def update_memory(
        self,
        ctx: MemoryRequestContext,
        req: MemoryDbMemoryUpdateCommand,
    ) -> MemoryDbMutationResult:
        result = await self.apply_mutation_plan(
            ctx,
            MemoryDbMutationPlan(memory_updates=[req]),
            consistency=req.consistency,
        )
        if result.mutations:
            return result.mutations[0]
        raise RuntimeError("; ".join(result.errors) or f"memory update {req.memory_id!r} produced no result")

    async def delete_memory(
        self,
        ctx: MemoryRequestContext,
        req: MemoryDbDeleteCommand,
    ) -> MemoryDbMutationResult:
        result = await self.apply_mutation_plan(
            ctx,
            MemoryDbMutationPlan(memory_deletes=[req]),
            consistency=req.consistency,
        )
        if result.mutations:
            return result.mutations[0]
        raise RuntimeError("; ".join(result.errors) or f"memory delete {req.memory_id!r} produced no result")

    async def get_memory_lineage(
        self,
        ctx: MemoryRequestContext,
        memory_ids: list[str],
    ) -> dict[str, list[str]]:
        seed_ids = _dedupe(memory_ids)
        if self._service.graph_enabled and seed_ids:
            graph_scope = self._scope(ctx)
            result = await self._service.traverse(
                GraphTraversalQuery(
                    scope=graph_scope,
                    seeds=tuple(
                        GraphNodeRef(scope=graph_scope, kind="Memory", node_id=memory_id) for memory_id in seed_ids
                    ),
                    steps=(
                        GraphStep(
                            relations=("DERIVED_FROM",),
                            direction="out",
                            target_kinds=("Memory",),
                            min_hops=1,
                            max_hops=10_000,
                        ),
                    ),
                    result_uniqueness="path",
                    limit=10_000,
                    max_expansions=100_000,
                )
            )
            lineage: dict[str, list[str]] = {memory_id: [] for memory_id in seed_ids}
            for path in result.paths:
                ancestors = lineage[path.seed.node_id]
                ancestor_id = path.end.ref.node_id
                if ancestor_id not in ancestors:
                    ancestors.append(ancestor_id)
            return lineage

        records = await self._records_for_ids(MEMORY_TABLE, ctx, memory_ids)
        result: dict[str, list[str]] = {}
        for record in records:
            parents = record.payload.get("parent_ids") or record.payload.get("root_id") or []
            result[record.record_id] = _dedupe(str(value) for value in parents)
        return result

    async def get_related_memory_ids(
        self,
        ctx: MemoryRequestContext,
        memory_ids: list[str],
        *,
        steps: Sequence[GraphStep] | None = None,
        result_uniqueness: Literal["path", "end_node"] = "end_node",
        active_only: bool = True,
        limit_per_memory: int | None = 20,
        max_candidates: int = 200,
    ) -> list[dict[str, Any]]:
        """List visible memories reached through caller-specified graph relations.

        The default is one bidirectional ``RELATES_TO`` hop. Multi-step callers,
        such as Dreaming's shared-entity grouping, receive the traversed node
        path plus optional entity metadata without adding another persistence
        operation.
        """

        if (
            not self._service.graph_enabled
            or not memory_ids
            or max_candidates <= 0
            or (limit_per_memory is not None and limit_per_memory <= 0)
        ):
            return []
        graph_scope = self._scope(ctx)
        seeds = tuple(GraphNodeRef(scope=graph_scope, kind="Memory", node_id=value) for value in _dedupe(memory_ids))
        resolved_steps = tuple(steps) if steps is not None else (
            GraphStep(
                relations=("RELATES_TO",),
                direction="both",
                target_kinds=("Memory",),
            ),
        )
        result = await self._service.traverse(
            GraphTraversalQuery(
                scope=graph_scope,
                seeds=seeds,
                steps=resolved_steps,
                result_uniqueness=result_uniqueness,
                limit=max_candidates,
                limit_per_seed=limit_per_memory,
            )
        )
        memory_paths = [path for path in result.paths if path.end.ref.kind == "Memory"]
        visible = {
            memory.memory_id
            for memory in await self.get_memories(ctx, [path.end.ref.node_id for path in memory_paths])
            if not active_only or memory.status == "active"
        }
        entity_ids = _dedupe(
            node.ref.node_id
            for path in memory_paths
            for node in path.nodes[1:-1]
            if node.ref.kind == "Entity"
        )
        entity_records = await self._records_for_ids(ENTITY_TABLE, ctx, entity_ids)
        entity_by_id = {record.record_id: record for record in entity_records}
        rows: list[dict[str, Any]] = []
        for path in memory_paths:
            memory_id = path.end.ref.node_id
            if memory_id not in visible:
                continue
            row: dict[str, Any] = {
                "memory_id": memory_id,
                "seed_memory_id": path.seed.node_id,
                "path": tuple(
                    {"kind": node.ref.kind, "node_id": node.ref.node_id}
                    for node in path.nodes
                ),
            }
            entity_node = next((node for node in path.nodes[1:-1] if node.ref.kind == "Entity"), None)
            if entity_node is not None:
                entity_id = entity_node.ref.node_id
                entity_record = entity_by_id.get(entity_id)
                row.update(
                    {
                        "entity_id": entity_id,
                        "entity_name": entity_record.payload.get("entity_name") if entity_record is not None else None,
                        "entity_type": entity_record.payload.get("entity_type") if entity_record is not None else None,
                    }
                )
            rows.append(row)
        return rows

    async def find_task_entity(
        self,
        ctx: MemoryRequestContext,
        task_text: str,
        *,
        limit: int = 5,
    ):
        """Find task entities matching a task text.

        The task text is matched exactly against ``entity_name`` first; when no
        exact row exists a case-insensitive substring match is used so partial /
        paraphrase task texts still surface existing tasks.
        """
        if not task_text.strip():
            return []
        scope = self._scope(ctx)
        page = Page(limit=max(1, limit))
        task_filter = lambda operator: FilterGroup(
            operator="and",
            clauses=(
                Predicate(field="entity_type", op="eq", value="task"),
                Predicate(field="entity_name", op=operator, value=task_text),
            ),
        )
        records, _ = await self._service.query_records(
            ENTITY_TABLE,
            RecordQuery(scope=scope, filters=task_filter("eq"), page=page),
        )
        if records:
            return records
        records, _ = await self._service.query_records(
            ENTITY_TABLE,
            RecordQuery(scope=scope, filters=task_filter("icontains"), page=page),
        )
        # When several tasks contain the query text, prefer the shortest name:
        # it is the most specific (least likely to be an unrelated superstring).
        return sorted(
            records,
            key=lambda record: len(str(record.payload.get("entity_name") or "")),
        )

    async def search_task_entities(
        self,
        ctx: MemoryRequestContext,
        task_text: str,
        *,
        dense_vector: list[float] | None = None,
        sparse_indices: list[int] | None = None,
        sparse_values: list[float] | None = None,
        top_k: int = 10,
        score_threshold: float | None = None,
    ):
        """Retrieve task entities closest to the query.

        Mirrors vanilla's hybrid search: when dense + BM25 sparse inputs are both
        available it fuses the two channels over the entity table (RRF, like
        vanilla search); otherwise it degrades to dense-only, then to name
        lookup. Candidates below the dense ``score_threshold`` floor are dropped
        so unrelated queries return nothing.
        """
        scope = self._scope(ctx)
        filters = _filter_expression(
            SearchFilter(
                must=[
                    FieldCondition(field="entity_type", op="match", value="task"),
                    FieldCondition(field="status", op="match", value="active"),
                ]
            )
        )

        use_hybrid = dense_vector is not None and sparse_indices is not None and sparse_values is not None
        if dense_vector is None and sparse_indices is not None:
            # No dense channel available; a lone sparse channel is meaningless here.
            return await self.find_task_entity(ctx, task_text, limit=top_k)
        if dense_vector is None:
            return await self.find_task_entity(ctx, task_text, limit=top_k)

        mode = "hybrid" if use_hybrid else "dense"
        prefetch = max(top_k, 10)
        hits = await self._service.search_vectors(
            VectorQuery(
                table=ENTITY_TABLE,
                scope=scope,
                vector_name="semantic",
                dense_vector=tuple(dense_vector),
                sparse_indices=tuple(sparse_indices) if use_hybrid else None,
                sparse_values=tuple(sparse_values) if use_hybrid else None,
                mode=mode,
                filters=filters,
                top_k=prefetch,
                dense_limit=prefetch,
                sparse_limit=prefetch if use_hybrid else None,
            )
        )

        records: list = []
        for hit in hits:
            if score_threshold is not None:
                native = None
                if use_hybrid:
                    native = (hit.debug.get("native_scores") or {}).get("dense")
                else:
                    native = hit.score
                if native is None or native < score_threshold:
                    continue
            records.append(hit.record)
        return records[: max(1, top_k)]

    async def get_task_experiences(
        self,
        ctx: MemoryRequestContext,
        task_entity_id: str,
        *,
        limit: int = 20,
    ) -> list[MemoryView]:
        """Return the experiences reachable one ``TASK_EXPERIENCE`` hop away.

        This is the "task → its lessons" read: seed the graph at the task
        entity, walk the TASK_EXPERIENCE edges outward, and hydrate the
        experience memories. Requires the graph tables to be enabled.
        """
        if not self._service.graph_enabled:
            return []
        graph_scope = self._scope(ctx)
        result = await self._service.traverse(
            GraphTraversalQuery(
                scope=graph_scope,
                seeds=(GraphNodeRef(scope=graph_scope, kind="Entity", node_id=task_entity_id),),
                steps=(
                    GraphStep(
                        relations=(REL_TASK_EXPERIENCE,),
                        direction="out",
                        target_kinds=("Memory",),
                        min_hops=1,
                        max_hops=1,
                    ),
                ),
                result_uniqueness="end_node",
                limit=max(1, limit),
            )
        )
        memory_ids = [path.end.ref.node_id for path in result.paths if path.end.ref.kind == "Memory"]
        return await self.get_memories(ctx, memory_ids)

    async def _write_graph(self, ctx: MemoryRequestContext, plan: MemoryDbMutationPlan) -> None:
        relationships = [command.relationship for command in plan.relationship_writes]
        refs: dict[tuple[str, str], GraphNodeRef] = {}
        edges: list[GraphEdge] = []
        for relationship in relationships:
            scope = self._write_scope(ctx, relationship)
            source = GraphNodeRef(
                scope=scope,
                kind=relationship.source.kind,
                node_id=relationship.source.node_id,
            )
            target = GraphNodeRef(
                scope=scope,
                kind=relationship.target.kind,
                node_id=relationship.target.node_id,
            )
            refs[(source.kind, source.node_id)] = source
            refs[(target.kind, target.node_id)] = target
            identity = {
                key: value
                for key, value in {
                    "property_name": relationship.property_name,
                    "edge_type": relationship.edge_type,
                    "relation_type": relationship.relation_type,
                    "entity_id": relationship.entity_id,
                    "extraction_position": relationship.extraction_position,
                }.items()
                if value is not None
            }
            properties = {
                key: value
                for key, value in {
                    "property_name": relationship.property_name,
                    "edge_type": relationship.edge_type,
                    "relation_type": relationship.relation_type,
                    "entity_id": relationship.entity_id,
                    "extraction_position": relationship.extraction_position,
                    "mention_count": relationship.mention_count,
                    "metadata": relationship.metadata,
                }.items()
                if value is not None
            }
            edges.append(
                GraphEdge(
                    source=source,
                    target=target,
                    relation=relationship.rel_type,
                    edge_key=identity,
                    properties=properties,
                )
            )
        await self._service.upsert_nodes([GraphNode(ref=ref) for ref in refs.values()])
        await self._service.upsert_edges(edges)

    async def _update_memory(
        self,
        ctx: MemoryRequestContext,
        command: MemoryDbMemoryUpdateCommand,
    ) -> MemoryDbMutationResult:
        record = await self._find_record(MEMORY_TABLE, ctx, command.memory_id, with_vectors=True)
        if record is None:
            return MemoryDbMutationResult(memory_id=command.memory_id, changed=False)
        payload = dict(record.payload)
        metadata = _clean_metadata(payload.get("metadata"))
        if command.dedup_metadata_key and command.dedup_metadata_key in command.metadata_patch:
            if metadata.get(command.dedup_metadata_key) == command.metadata_patch[command.dedup_metadata_key]:
                return MemoryDbMutationResult(memory_id=command.memory_id, changed=False)

        now = datetime.now(UTC)
        payload.update(command.payload_patch)
        payload["update_at"] = now
        resolved_dense = _dense_from_command(command)
        resolved_sparse = _sparse_from_command(command)
        if command.content is not None:
            preprocessor, encoder = self._ensure_text_components()
            preprocessed = preprocessor.preprocess_text(
                command.content,
                segment_id="update",
                include_entities=False,
            )
            payload["content"] = preprocessed.normalized_text
            metadata.update(
                {
                    "content_hash": preprocessed.content_hash,
                    "bm25_text": preprocessed.bm25_text,
                    "tokens": list(preprocessed.tokens),
                    "lang": preprocessed.lang,
                }
            )
            if resolved_sparse is None:
                encoded = encoder.encode_document(preprocessed.tokens)
                resolved_sparse = SparseVector(
                    indices=tuple(encoded.indices),
                    values=tuple(encoded.values),
                )
            if resolved_dense is None:
                embed_response = await self._ensure_embed_client().embed(
                    task="memory.update",
                    text=preprocessed.normalized_text,
                )
                if not embed_response.embeddings or not embed_response.embeddings[0]:
                    raise MemoryUpdateError("memory update embedding returned empty vector")
                resolved_dense = list(embed_response.embeddings[0])
        if command.reinforcement_count is not None:
            payload["reinforcement_count"] = command.reinforcement_count
        if command.reinforcement_count_delta:
            payload["reinforcement_count"] = int(payload.get("reinforcement_count") or 0) + (
                command.reinforcement_count_delta
            )
        if command.status is not None:
            payload["status"] = command.status
            payload["status_changed_at"] = now
        metadata.update(command.metadata_patch)
        metadata[_SCOPE_METADATA_KEY] = dict(record.scope.items())
        payload["metadata"] = metadata

        vectors = record.vectors or VectorValue()
        dense = dict(vectors.dense)
        sparse = dict(vectors.sparse)
        if resolved_dense is not None:
            dense["semantic"] = tuple(resolved_dense)
        if resolved_sparse is not None:
            sparse["bm25"] = resolved_sparse
        await self._service.upsert_records(
            MEMORY_TABLE,
            [
                Record(
                    table=record.table,
                    record_id=record.record_id,
                    scope=record.scope,
                    payload=payload,
                    vectors=VectorValue(dense=dense, sparse=sparse),
                )
            ],
        )
        return MemoryDbMutationResult(memory_id=command.memory_id, changed=True)

    async def _delete_memory(
        self,
        ctx: MemoryRequestContext,
        command: MemoryDbDeleteCommand,
    ) -> MemoryDbMutationResult:
        record = await self._find_record(MEMORY_TABLE, ctx, command.memory_id)
        if record is None:
            return MemoryDbMutationResult(
                memory_id=command.memory_id,
                changed=False,
                hard=command.hard,
            )
        if command.hard:
            await self._service.delete_records(MEMORY_TABLE, record.scope, [record.record_id])
            if self._service.graph_enabled:
                await self._service.delete_node(
                    GraphNodeRef(
                        scope=self._scope(ctx),
                        kind="Memory",
                        node_id=record.record_id,
                    )
                )
        else:
            metadata = _clean_metadata(record.payload.get("metadata"))
            metadata.update({"archived_reason": command.reason, _SCOPE_METADATA_KEY: dict(record.scope.items())})
            await self._service.patch_record(
                MEMORY_TABLE,
                record.scope,
                record.record_id,
                {
                    "status": "archived",
                    "status_changed_at": datetime.now(UTC),
                    "metadata": metadata,
                },
            )
        return MemoryDbMutationResult(
            memory_id=command.memory_id,
            changed=True,
            hard=command.hard,
        )

    async def _find_record(
        self,
        table: str,
        ctx: MemoryRequestContext,
        record_id: str,
        *,
        with_vectors: bool = False,
    ) -> Record | None:
        query = RecordQuery(
            scope=self._scope(ctx),
            filters=_table_scoped_filter(
                table,
                ctx,
                Predicate(field=_primary_key(table), op="eq", value=record_id),
            ),
            page=Page(limit=2),
        )
        if with_vectors:
            records, _ = await self._service.scroll(table, query, with_vectors=True)
        else:
            records, _ = await self._service.query_records(table, query)
        if len(records) > 1:
            raise RuntimeError(f"{table} record {record_id!r} is ambiguous in the supplied scope")
        return records[0] if records else None

    async def _records_for_ids(
        self,
        table: str,
        ctx: MemoryRequestContext,
        record_ids: Sequence[str],
    ) -> list[Record]:
        ids = _dedupe(record_ids)
        if not ids:
            return []
        records, _ = await self._service.query_records(
            table,
            RecordQuery(
                scope=self._scope(ctx),
                filters=_table_scoped_filter(
                    table,
                    ctx,
                    Predicate(field=_primary_key(table), op="in", value=tuple(ids)),
                ),
                page=Page(limit=max(1, len(ids))),
            ),
        )
        return records


def _primary_key(table: str) -> str:
    return {
        MEMORY_TABLE: "memory_id",
        ENTITY_TABLE: "entity_id",
        SOURCE_TABLE: "source_id",
    }[table]


def _merge_scopes(*scopes: DatabaseScope) -> DatabaseScope:
    """Merge scope dimensions without allowing a later source to override one."""

    merged: dict[str, ScopeValue] = {}
    for scope in scopes:
        for name, value in scope.items():
            if name in merged and merged[name] != value:
                raise ValueError(
                    f"scope dimension {name!r} conflicts: existing value {merged[name]!r}, new value {value!r}"
                )
            merged[name] = value
    return DatabaseScope(merged)

def _with_memory_mode(
    ctx: MemoryRequestContext,
    filters: FilterExpression | None,
) -> FilterExpression | None:
    """Force the selected mode into every memory-table read."""

    if not ctx.memory_algorithm:
        return filters
    mode_filter = Predicate(field="memory_mode", op="eq", value=ctx.memory_algorithm)
    if filters is None:
        return mode_filter
    return FilterGroup(operator="and", clauses=(mode_filter, filters))


def _table_scoped_filter(
    table: str,
    ctx: MemoryRequestContext,
    filters: FilterExpression,
) -> FilterExpression:
    if table != MEMORY_TABLE:
        return filters
    return _with_memory_mode(ctx, filters) or filters


def _dense_from_command(command: MemoryDbMemoryUpdateCommand) -> list[float] | None:
    dense = command.embedding if command.embedding is not None else command.dense_vector
    if dense is None:
        return None
    if not dense:
        raise MemoryUpdateError("memory update embedding cannot be empty")
    return list(dense)


def _sparse_from_command(command: MemoryDbMemoryUpdateCommand) -> SparseVector | None:
    if command.bm25_indices is not None:
        return SparseVector(
            indices=tuple(command.bm25_indices),
            values=tuple([1.0] * len(command.bm25_indices)),
        )
    if command.sparse_vectors is None:
        return None

    indices = command.sparse_vectors.get("bm25_indices")
    values = command.sparse_vectors.get("bm25_values")
    if (indices is None) != (values is None):
        raise ValueError("sparse_vectors bm25_indices and bm25_values must be updated together")
    if indices is None or values is None:
        return None
    return SparseVector(indices=tuple(indices), values=tuple(values))


def _with_scope_metadata(metadata: Any, scope: DatabaseScope) -> dict[str, Any]:
    result = dict(metadata or {})
    result[_SCOPE_METADATA_KEY] = dict(scope.items())
    return result


def _clean_metadata(metadata: Any) -> dict[str, Any]:
    result = dict(metadata or {})
    result.pop(_SCOPE_METADATA_KEY, None)
    return result


def _memory_record(memory, vector, scope: DatabaseScope) -> Record:
    payload = memory.model_dump(mode="python", exclude=set(_SCOPE_FIELDS))
    payload["schema_version"] = 2
    payload["metadata"] = _with_scope_metadata(payload.get("metadata"), scope)
    dense = {}
    sparse = {}
    if vector is not None:
        if vector.semantic_vector is not None:
            dense["semantic"] = tuple(vector.semantic_vector)
        if vector.bm25_indices:
            sparse["bm25"] = SparseVector(
                indices=tuple(vector.bm25_indices),
                values=tuple(vector.bm25_values or [1.0] * len(vector.bm25_indices)),
            )
    return Record(
        table=MEMORY_TABLE,
        record_id=memory.memory_id,
        scope=scope,
        payload=payload,
        vectors=VectorValue(dense=dense, sparse=sparse),
    )


def _entity_record(entity, vector, scope: DatabaseScope) -> Record:
    payload = entity.model_dump(mode="python", exclude=set(_SCOPE_FIELDS))
    payload["schema_version"] = 2
    payload["metadata"] = _with_scope_metadata(payload.get("metadata"), scope)
    dense = {}
    sparse = {}
    if vector is not None:
        if vector.semantic_vector is not None:
            dense["semantic"] = tuple(vector.semantic_vector)
        if vector.bm25_indices:
            sparse["bm25"] = SparseVector(
                indices=tuple(vector.bm25_indices),
                values=tuple(vector.bm25_values or [1.0] * len(vector.bm25_indices)),
            )
    return Record(
        table=ENTITY_TABLE,
        record_id=entity.entity_id,
        scope=scope,
        payload=payload,
        vectors=VectorValue(dense=dense, sparse=sparse),
    )


def _source_record(source, scope: DatabaseScope) -> Record:
    payload = source.model_dump(mode="python", exclude={*set(_SCOPE_FIELDS), "persist_payload"})
    payload["schema_version"] = 2
    payload["metadata"] = _with_scope_metadata(payload.get("metadata"), scope)
    return Record(
        table=SOURCE_TABLE,
        record_id=source.source_id,
        scope=scope,
        payload=payload,
    )


def _memory_view(record: Record) -> MemoryView:
    payload = dict(record.payload)
    payload["memory_id"] = record.record_id
    payload["metadata"] = _clean_metadata(payload.get("metadata"))
    payload.pop("schema_version", None)
    for field in ("request_id", "entity_id"):
        if payload.get(field) is not None:
            payload[field] = str(payload[field])
    for field in ("parent_ids", "root_id"):
        payload[field] = [str(value) for value in payload.get(field) or []]
    scope = dict(record.scope.items())
    for field in _SCOPE_FIELDS:
        payload[field] = scope.get(field)
    return MemoryView.model_validate(payload)


def _search_result(query: str, hits) -> MemoryDbSearchResult:
    values = [
        MemoryDbSearchHit(
            memory_id=hit.record.record_id,
            score=hit.score,
            memory=_memory_view(hit.record),
            source=hit.source,
            rank=index,
            debug={"rank": index},
        )
        for index, hit in enumerate(hits, start=1)
    ]
    return MemoryDbSearchResult(query=query, hits=values, total=len(values))


def _filter_expression(search_filter: SearchFilter | None) -> FilterExpression | None:
    if search_filter is None:
        return None
    clauses: list[FilterExpression] = []
    clauses.extend(_condition_expression(value) for value in search_filter.must)
    if search_filter.should:
        clauses.append(
            FilterGroup(
                operator="or",
                clauses=tuple(_condition_expression(value) for value in search_filter.should),
            )
        )
    if search_filter.must_not:
        clauses.append(
            FilterGroup(
                operator="not",
                clauses=tuple(_condition_expression(value) for value in search_filter.must_not),
            )
        )
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else FilterGroup(operator="and", clauses=tuple(clauses))


def _condition_expression(value: FieldCondition | SearchFilter) -> FilterExpression:
    if isinstance(value, SearchFilter):
        return _filter_expression(value) or FilterGroup(operator="and")
    field = f"metadata.{_SCOPE_METADATA_KEY}.{value.field}" if value.field in _SCOPE_FIELDS else value.field
    if value.op == "match":
        return Predicate(field=field, op="eq", value=value.value)
    if value.op == "any":
        return Predicate(field=field, op="in", value=tuple(value.values or ()))
    if value.op == "except":
        return Predicate(field=field, op="not_in", value=tuple(value.values or ()))
    if value.op == "text":
        return Predicate(field=field, op="icontains", value=value.value)
    if value.op in {"range", "datetime"}:
        bounds = [
            Predicate(field=field, op=operator, value=getattr(value, operator))
            for operator in ("gt", "gte", "lt", "lte")
            if getattr(value, operator) is not None
        ]
        return bounds[0] if len(bounds) == 1 else FilterGroup(operator="and", clauses=tuple(bounds))
    if value.op == "is_empty":
        return Predicate(field=field, op="is_empty")
    if value.op == "is_null":
        return Predicate(field=field, op="is_null", value=True)
    raise ValueError(f"unsupported memory filter operator: {value.op}")


def _dedupe(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


__all__ = ["MemoryPersistence"]
