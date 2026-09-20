"""Domain-level storage port for the ``skill_manage`` capability.

The adapter composes a structured ``ScopedDatabase`` (graph state) with a
``ScopedVectorStore`` (pgvector embeddings) and returns domain DTOs only. It
never leaks psycopg rows, SQL, pgvector or fast-graphrag types across the
boundary, and it enforces the frozen scope rules: head/operation/lease are
scoped by ``project_id + graph_id``, snapshot records additionally by
``generation_id``.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from ...typing.skill_manage import (
    SkillGraphEdge,
    SkillGraphEmbedding,
    SkillGraphNode,
    SkillGraphSnapshot,
    SkillGraphVectorHit,
    SkillManageScope,
)
from ..database import DatabaseScope

# P0-frozen namespaces. Changing either breaks generation-qualified identity
# and requires an ADR update.
SKILL_MANAGE_RECORD_NAMESPACE = uuid.UUID("a5c8f7e2-4d3b-5a1c-9e8f-7d6c5b4a3210")
SKILL_MANAGE_GENERATION_NAMESPACE = uuid.UUID("b6d9e8f3-5e4c-6b2d-af9e-8e7d6c5b4321")


def physical_record_id(generation_id: str, record_kind: str, logical_id: str) -> str:
    """Frozen physical record identity for a snapshot record.

    ``uuid5(RECORD_NAMESPACE, generation_id + ":" + kind + ":" + logical_id)``
    keeps the same logical node/edge/mapping distinct across generations.
    """

    payload = f"{generation_id}:{record_kind}:{logical_id}"
    return str(uuid.uuid5(SKILL_MANAGE_RECORD_NAMESPACE, payload))


def generation_id_from_operation(operation_id: str, attempt_no: int) -> str:
    """Frozen deterministic generation identity for one operation attempt."""

    payload = f"{operation_id}:{attempt_no}"
    return str(uuid.uuid5(SKILL_MANAGE_GENERATION_NAMESPACE, payload))


def generation_counts_hash(
    *,
    node_ids: Iterable[str],
    edge_ids: Iterable[str],
    memberships: Iterable[tuple[str, str, str]],
    vector_node_ids: Iterable[str],
) -> str:
    """Order-independent digest of a generation's record identities.

    Stored on the generation row at ``ready`` time so a later reconcile can
    detect a partially written or tampered snapshot without recounting.
    """

    payload = json.dumps(
        {
            "nodes": sorted(node_ids),
            "edges": sorted(edge_ids),
            "memberships": sorted(list(item) for item in memberships),
            "vectors": sorted(vector_node_ids),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def head_scope(scope: SkillManageScope) -> DatabaseScope:
    """Physical scope for head, operation and writer-lease records."""

    return DatabaseScope(project_id=scope.project_id, graph_id=scope.graph_id)


def snapshot_scope(scope: SkillManageScope, generation_id: str) -> DatabaseScope:
    """Physical scope for generation-qualified snapshot records."""

    return DatabaseScope(project_id=scope.project_id, graph_id=scope.graph_id, generation_id=generation_id)


@runtime_checkable
class SkillGraphStorageAdapter(Protocol):
    """Domain storage port used by :class:`SkillManageService`.

    Timestamps cross this port as timezone-aware ISO-8601 strings in both
    directions: ``now`` on the way in, and every ``*_at`` field of the row
    dicts returned by ``get_*``/``list_*`` on the way out, whatever the
    backend stores natively. Adapters must never read the wall clock when one
    is supplied, so the service's injected clock governs lease expiry,
    retention age and every recorded instant.
    """

    async def ensure_schema(self) -> None: ...

    async def is_ready(self) -> bool: ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def get_graph_head(self, scope: SkillManageScope) -> dict[str, Any] | None: ...

    async def create_head_if_absent(
        self,
        scope: SkillManageScope,
        head: dict[str, Any],
    ) -> bool: ...

    async def get_operation(self, scope: SkillManageScope, operation_id: str) -> dict[str, Any] | None: ...

    async def reserve_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        idempotency_key: str,
        request_hash: str,
        operation_type: str,
        now: str | None = None,
    ) -> bool: ...

    async def get_writer_lease(self, scope: SkillManageScope) -> dict[str, Any] | None: ...

    async def acquire_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> int:
        """Return the fencing token; raise ``SkillManageWriterConflictError`` while another writer holds it."""
        ...

    async def renew_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        token: int,
        ttl_seconds: float,
        now: str | None = None,
    ) -> bool: ...

    async def release_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        token: int,
        now: str | None = None,
    ) -> bool: ...

    async def begin_initial_attempt_atomically(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        generation_id: str,
        fencing_token: int,
        base_revision: int | None = None,
        now: str | None = None,
    ) -> bool: ...

    async def reserve_retry_attempt(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        now: str | None = None,
    ) -> bool:
        """CAS ``failed(retryable)`` -> ``pending_repair(retry_attempt_reservation)``."""
        ...

    async def begin_retry_attempt_atomically(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        attempt_no: int,
        generation_id: str,
        fencing_token: int,
        base_revision: int | None = None,
        now: str | None = None,
    ) -> bool: ...

    async def begin_validation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        fencing_token: int | None = None,
        result_generation_id: str | None = None,
        now: str | None = None,
    ) -> bool:
        """CAS ``running`` -> ``validating``; with ``fencing_token`` only the attempt holding it succeeds.

        ``result_generation_id`` repoints the row's ``current_generation_id``
        at the generation the attempt will report (a no-op mutation reports
        the base generation it left in place), so crash recovery converges
        the row from that generation's status rather than from a candidate
        that was never staged.
        """
        ...

    async def claim_crash_recovery(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        expected_attempt_no: int,
        expected_generation_id: str | None,
        now: str | None = None,
    ) -> bool:
        """CAS exactly the observed attempt from ``running``/``validating`` to ``pending_repair(crash_recovery)``."""
        ...

    async def repair_crash_recovery(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        attempt_no: int,
        generation_id: str | None,
        now: str | None = None,
    ) -> str: ...

    async def stage_generation(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        operation_id: str,
        attempt_no: int,
        base_revision: int | None,
        target_revision: int,
        now: str | None = None,
    ) -> None:
        """Insert the generation row as ``staging``; raise ``SkillManageConsistencyError`` if it exists."""
        ...

    async def get_generation(self, scope: SkillManageScope, generation_id: str) -> dict[str, Any] | None: ...

    async def write_snapshot(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        nodes: list[SkillGraphNode],
        edges: list[SkillGraphEdge],
        memberships: list[dict[str, Any]],
    ) -> None: ...

    async def write_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        embeddings: list[SkillGraphEmbedding],
    ) -> None: ...

    async def load_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
    ) -> tuple[SkillGraphEmbedding, ...]: ...

    async def count_embeddings(self, scope: SkillManageScope, generation_id: str) -> int: ...

    async def search_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        query_vector: tuple[float, ...],
        top_k: int,
        score_threshold: float | None = None,
    ) -> tuple[SkillGraphVectorHit, ...]:
        """Cosine KNN over the generation's vectors, best first, at most ``top_k``.

        A pair whose similarity is undefined (a zero-norm vector on either
        side) is not a hit on any backend, whatever ``score_threshold`` is.
        """
        ...

    async def mark_generation_ready(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        node_count: int,
        edge_count: int,
        vector_count: int,
        counts_hash: str,
        now: str | None = None,
    ) -> bool:
        """CAS ``staging`` -> ``ready`` and freeze the generation summary."""
        ...

    async def mark_generation_failed(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        now: str | None = None,
    ) -> bool:
        """CAS ``ready``/``staging`` -> ``failed``; ``False`` once the candidate is published, missing or already failed.

        This is the fence that settles an in-doubt publish: a row update
        queues behind a publish transaction whose commit is still in flight
        and is re-evaluated against the committed row, so winning the CAS
        proves the candidate was never published and can no longer be.
        ``discard_generation`` and ``repair_crash_recovery`` apply it before
        deleting or settling anything.
        """
        ...

    async def publish_generation(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        expected_revision: int | None,
        new_revision: int,
        embedding_profile: dict[str, Any] | None = None,
        owner_operation_id: str | None = None,
        fencing_token: int | None = None,
        algorithm_name: str | None = None,
        algorithm_version: str | None = None,
        now: str | None = None,
    ) -> bool:
        """One transaction: fencing check, head create/CAS, new ``active``, old ``retired``.

        Returns ``False`` when the head CAS loses. Raises
        ``SkillManageWriterConflictError`` when ``owner_operation_id`` /
        ``fencing_token`` no longer match an unexpired lease, and
        ``SkillManageConsistencyError`` when the generation is not ``ready``
        — checked again by the ``ready -> active`` CAS inside the
        transaction, so a candidate fenced by a reconciler meanwhile rolls
        the whole switch back.
        """
        ...

    async def record_operation_result(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        base_revision: int | None,
        result_revision: int,
        generation_id: str,
        result_payload: dict[str, Any] | None = None,
        fencing_token: int | None = None,
        now: str | None = None,
    ) -> bool:
        """CAS ``validating`` -> ``committed`` together with the published outcome."""
        ...

    async def fail_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        error_code: str,
        error_message: str,
        error_retryable: bool | None,
        fencing_token: int | None = None,
        now: str | None = None,
    ) -> bool: ...

    async def conflict_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        error_code: str,
        error_message: str,
        fencing_token: int | None = None,
        now: str | None = None,
    ) -> bool: ...

    async def cancel_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        error_code: str,
        fencing_token: int | None = None,
        now: str | None = None,
    ) -> bool: ...

    async def discard_generation(self, scope: SkillManageScope, generation_id: str) -> None:
        """Delete an unpublished candidate's rows after fencing it; a published generation is never touched."""
        ...

    async def gc_retired_generations(
        self,
        scope: SkillManageScope,
        *,
        keep: int = 2,
        min_age_seconds: float = 0.0,
        now: str | None = None,
    ) -> list[str]: ...

    async def list_operations(
        self,
        scope: SkillManageScope,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Every operation row in the scope, optionally restricted to ``statuses``; follows the cursor to the end."""
        ...

    async def list_generations(
        self,
        scope: SkillManageScope,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Every generation row in the scope, optionally restricted to ``statuses``; follows the cursor to the end."""
        ...

    async def list_snapshot_generation_ids(self, scope: SkillManageScope) -> set[str]:
        """Every generation id that still owns snapshot, mapping or vector records in the scope.

        Ids without a generation row are orphans (records written by a
        superseded writer after its candidate was discarded, or left by an
        interrupted discard) that only an explicit sweep can reclaim.
        """
        ...

    async def load_snapshot(self, scope: SkillManageScope, generation_id: str) -> SkillGraphSnapshot: ...


__all__ = [
    "SKILL_MANAGE_GENERATION_NAMESPACE",
    "SKILL_MANAGE_RECORD_NAMESPACE",
    "SkillGraphStorageAdapter",
    "generation_counts_hash",
    "generation_id_from_operation",
    "head_scope",
    "physical_record_id",
    "snapshot_scope",
]
