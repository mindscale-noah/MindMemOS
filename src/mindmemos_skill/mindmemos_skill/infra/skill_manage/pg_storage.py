"""PostgreSQL/pgvector storage adapter for ``skill_manage``.

Composes one structured ``ScopedDatabase`` (graph/head/operation/generation
state) with one pgvector ``ScopedVectorStore`` (embeddings). Every method takes
the domain ``SkillManageScope`` and converts it to the correct physical scope:
head/operation/lease use ``project_id + graph_id``; snapshot records use
``project_id + graph_id + generation_id``.

Operation and generation transitions go through the frozen state machine;
``publish_generation`` is the single visibility switch and runs the fencing
check, the head create/CAS and the new-active/old-retired flips in one
structured transaction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from ...errors import (
    SkillManageConsistencyError,
    SkillManageGcPartialFailureError,
    SkillManageWriterConflictError,
)
from ...infra.database import Page, Record, RecordQuery, ScopedDatabase
from ...infra.database.scope import DatabaseScope
from ...infra.vector_store import DatabaseScope as VectorScope
from ...infra.vector_store import Page as VectorPage
from ...infra.vector_store import Record as VectorRecord
from ...infra.vector_store import RecordQuery as VectorRecordQuery
from ...infra.vector_store import ScopedVectorStore, VectorQuery, VectorValue
from ...typing.skill_manage import (
    SkillGraphEdge,
    SkillGraphEmbedding,
    SkillGraphMembership,
    SkillGraphNode,
    SkillGraphSnapshot,
    SkillGraphVectorHit,
    SkillManageScope,
)
from .storage import head_scope, physical_record_id, snapshot_scope
from .vector_tables import EMBEDDING_VECTOR_NAME, SKILL_MANAGE_EMBEDDINGS

_GENERATIONS = "skill_manage_generations"
_GRAPHS = "skill_manage_graphs"
_LEASES = "skill_manage_writer_leases"
_OPERATIONS = "skill_manage_operations"
_OPERATION_RESULT_SCHEMA_VERSION = 1
_PAGE_LIMIT = 10000
_STRANDED_STATUSES = ["pending_repair", "running", "validating"]


def _now_iso(now: str | None) -> str:
    return now or datetime.now(UTC).isoformat()


def _row(record: Record) -> dict[str, Any]:
    """A row dict as the port promises it: DATETIME columns as ISO-8601 strings."""

    return {key: value.isoformat() if isinstance(value, datetime) else value for key, value in record.payload.items()}


def _as_datetime(value: Any) -> datetime | None:
    """Postgres restores DATETIME fields as datetime; Python-side callers pass ISO strings."""

    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _as_iso(value: Any) -> str | None:
    parsed = _as_datetime(value)
    return parsed.isoformat() if parsed is not None else None


def _add_seconds(iso_value: str, seconds: float) -> str:
    parsed = _as_datetime(iso_value)
    assert parsed is not None
    return (parsed + timedelta(seconds=seconds)).isoformat()


def _is_expired(expires_at: Any, now: str) -> bool:
    expires = _as_datetime(expires_at)
    if expires is None:
        return True
    current = _as_datetime(now)
    assert current is not None
    return expires <= current


def _fence(fencing_token: int | None) -> dict[str, Any] | None:
    return {"writer_fencing_token": fencing_token} if fencing_token is not None else None


def _vector_scope(scope: SkillManageScope, generation_id: str) -> VectorScope:
    return VectorScope(project_id=scope.project_id, graph_id=scope.graph_id, generation_id=generation_id)


class PgSkillGraphStorageAdapter:
    """Domain storage adapter combining structured graph state and embeddings."""

    def __init__(
        self,
        *,
        database: ScopedDatabase,
        vector_store: ScopedVectorStore,
        owns_database: bool = True,
        page_size: int = _PAGE_LIMIT,
    ) -> None:
        self._database = database
        self._vector_store = vector_store
        self._owns_database = owns_database
        self._page_size = page_size
        self._closed = False

    async def _query_structured_paged(self, table: str, scope: DatabaseScope) -> list[Record]:
        """Loop ``query_records`` until the cursor exhausts, returning each record at most once.

        A single ``Page`` call truncates at ``page_size`` rows, which would make
        generation read-back and GC silently drop everything past the cap. The
        backend cursor is an OFFSET, so a row inserted concurrently before the
        boundary shifts the next page and re-yields the previous page's last
        row; deduplicating by ``record_id`` keeps a list walk from counting a
        generation twice (which would let GC delete inside the keep window). A
        row skipped by a concurrent delete is simply picked up on the next pass.
        """

        records: list[Record] = []
        seen: set[str] = set()
        cursor: str | None = None
        while True:
            batch, cursor = await self._database.query_records(
                table, RecordQuery(scope=scope, page=Page(limit=self._page_size, cursor=cursor))
            )
            for record in batch:
                if record.record_id not in seen:
                    seen.add(record.record_id)
                    records.append(record)
            if cursor is None:
                return records

    async def _query_vector_paged(
        self,
        table: str,
        scope: VectorScope,
        *,
        with_vectors: bool = False,
    ) -> list[VectorRecord]:
        """Loop ``scroll`` until the cursor exhausts (see :meth:`_query_structured_paged`)."""

        records: list[VectorRecord] = []
        seen: set[str] = set()
        cursor: str | None = None
        while True:
            batch, cursor = await self._vector_store.scroll(
                table,
                VectorRecordQuery(scope=scope, page=VectorPage(limit=self._page_size, cursor=cursor)),
                with_vectors=with_vectors,
            )
            for record in batch:
                if record.record_id not in seen:
                    seen.add(record.record_id)
                    records.append(record)
            if cursor is None:
                return records

    async def ensure_schema(self) -> None:
        return None

    async def is_ready(self) -> bool:
        return not self._closed

    async def start(self) -> None:
        await self.ensure_schema()

    # ------------------------------------------------------------------ head

    async def get_graph_head(self, scope: SkillManageScope) -> dict[str, Any] | None:
        records = await self._database.get_records(_GRAPHS, head_scope(scope), (scope.graph_id,))
        return _row(records[0]) if records else None

    async def create_head_if_absent(self, scope: SkillManageScope, head: dict[str, Any]) -> bool:
        now = datetime.now(UTC).isoformat()
        payload = {"graph_id": scope.graph_id, "created_at": now, "updated_at": now, **head}
        return await self._database.insert_record_if_absent(
            _GRAPHS,
            Record(table=_GRAPHS, record_id=scope.graph_id, scope=head_scope(scope), payload=payload),
        )

    # ------------------------------------------------------------ operations

    async def get_operation(self, scope: SkillManageScope, operation_id: str) -> dict[str, Any] | None:
        records = await self._database.get_records(_OPERATIONS, head_scope(scope), (operation_id,))
        return _row(records[0]) if records else None

    async def reserve_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        idempotency_key: str,
        request_hash: str,
        operation_type: str,
        now: str | None = None,
    ) -> bool:
        cur_now = _now_iso(now)
        return await self._database.insert_record_if_absent(
            _OPERATIONS,
            Record(
                table=_OPERATIONS,
                record_id=operation_id,
                scope=head_scope(scope),
                payload={
                    "operation_id": operation_id,
                    "idempotency_key": idempotency_key,
                    "request_hash": request_hash,
                    "operation_type": operation_type,
                    "status": "pending",
                    "attempt_no": 1,
                    "created_at": cur_now,
                    "updated_at": cur_now,
                },
            ),
        )

    async def transition_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        expected_status: list[str],
        changes: dict[str, Any],
        expected_fields: dict[str, Any] | None = None,
        now: str | None = None,
    ) -> bool:
        """Transition an operation, enforcing the frozen state machine.

        ``expected_fields`` join the status in the CAS predicate so an attempt
        can fence its own transitions (by fencing token) and a recoverer can
        claim exactly the attempt it observed. Returns ``True`` when the
        transition was applied, ``False`` on a status/field conflict, a
        missing operation, or a state-machine-invalid transition.
        """
        from ...application.skill_manage.state_machine import OperationStatus, require_transition

        current = await self.get_operation(scope, operation_id)
        if current is None or current.get("status") not in expected_status:
            return False
        target = changes.get("status")
        if target is not None:
            try:
                require_transition(OperationStatus(current["status"]), OperationStatus(target))
            except (ValueError, TypeError):
                return False
        return await self._database.compare_and_swap_record(
            _OPERATIONS,
            head_scope(scope),
            operation_id,
            expected={"status": current["status"], **(expected_fields or {})},
            changes={**changes, "updated_at": _now_iso(now)},
        )

    async def begin_initial_attempt_atomically(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        generation_id: str,
        fencing_token: int,
        base_revision: int | None = None,
        now: str | None = None,
    ) -> bool:
        """CAS a pending operation into running with its initial generation."""

        cur_now = _now_iso(now)
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["pending"],
            changes={
                "status": "running",
                "current_generation_id": generation_id,
                "writer_fencing_token": fencing_token,
                "base_revision": base_revision,
                "first_started_at": cur_now,
                "attempt_started_at": cur_now,
            },
            now=cur_now,
        )

    async def reserve_retry_attempt(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        now: str | None = None,
    ) -> bool:
        current = await self.get_operation(scope, operation_id)
        if current is None or current.get("status") != "failed" or current.get("error_retryable") is not True:
            return False
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["failed"],
            changes={"status": "pending_repair", "pending_repair_reason": "retry_attempt_reservation"},
            expected_fields={"error_retryable": True},
            now=now,
        )

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
    ) -> bool:
        """CAS a pending_repair operation into a new running attempt.

        The CAS also requires the row to still be at ``attempt_no - 1``: a
        caller that computed the next attempt from a read taken before it
        stalled must not re-open an attempt number (and its deterministic
        generation id) that another same-key caller used meanwhile.
        """

        cur_now = _now_iso(now)
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["pending_repair"],
            changes={
                "status": "running",
                "attempt_no": attempt_no,
                "current_generation_id": generation_id,
                "writer_fencing_token": fencing_token,
                "base_revision": base_revision,
                "result_revision": None,
                "pending_repair_reason": None,
                "error_code": None,
                "error_message": None,
                "error_retryable": None,
                "attempt_started_at": cur_now,
                "validating_at": None,
                "completed_at": None,
            },
            expected_fields={"attempt_no": attempt_no - 1},
            now=cur_now,
        )

    async def begin_validation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        fencing_token: int | None = None,
        result_generation_id: str | None = None,
        now: str | None = None,
    ) -> bool:
        cur_now = _now_iso(now)
        changes: dict[str, Any] = {"status": "validating", "validating_at": cur_now}
        if result_generation_id is not None:
            changes["current_generation_id"] = result_generation_id
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["running"],
            changes=changes,
            expected_fields=_fence(fencing_token),
            now=cur_now,
        )

    async def claim_crash_recovery(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        expected_attempt_no: int,
        expected_generation_id: str | None,
        now: str | None = None,
    ) -> bool:
        """CAS the exact stranded attempt (``running``/``validating``) to pending_repair."""

        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["running", "validating"],
            changes={"status": "pending_repair", "pending_repair_reason": "crash_recovery"},
            expected_fields={"attempt_no": expected_attempt_no, "current_generation_id": expected_generation_id},
            now=now,
        )

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
        cur_now = _now_iso(now)
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["validating"],
            changes={
                "status": "committed",
                "base_revision": base_revision,
                "result_revision": result_revision,
                "current_generation_id": generation_id,
                "result_schema_version": _OPERATION_RESULT_SCHEMA_VERSION,
                "result_payload": dict(result_payload or {}),
                "completed_at": cur_now,
            },
            expected_fields=_fence(fencing_token),
            now=cur_now,
        )

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
    ) -> bool:
        cur_now = _now_iso(now)
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["pending", "running", "validating"],
            changes={
                "status": "failed",
                "error_code": error_code,
                "error_message": error_message,
                "error_retryable": error_retryable,
                "completed_at": cur_now,
            },
            expected_fields=_fence(fencing_token),
            now=cur_now,
        )

    async def conflict_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        error_code: str,
        error_message: str,
        fencing_token: int | None = None,
        now: str | None = None,
    ) -> bool:
        cur_now = _now_iso(now)
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["running", "validating"],
            changes={
                "status": "conflict",
                "error_code": error_code,
                "error_message": error_message,
                "error_retryable": False,
                "completed_at": cur_now,
            },
            expected_fields=_fence(fencing_token),
            now=cur_now,
        )

    async def cancel_operation(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        error_code: str,
        fencing_token: int | None = None,
        now: str | None = None,
    ) -> bool:
        cur_now = _now_iso(now)
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["pending", "running", "validating"],
            changes={
                "status": "cancelled",
                "error_code": error_code,
                "error_message": "cancelled before publish",
                "error_retryable": False,
                "completed_at": cur_now,
            },
            expected_fields=_fence(fencing_token),
            now=cur_now,
        )

    async def repair_crash_recovery(
        self,
        scope: SkillManageScope,
        operation_id: str,
        *,
        attempt_no: int,
        generation_id: str | None,
        now: str | None = None,
    ) -> str:
        """Reconcile a stranded operation against the head and candidate generation.

        Returns the converged status: ``committed`` when the candidate was
        published (it is the active generation, or was and has since been
        retired), ``failed`` (retryable) when the head is unchanged from the
        attempt's base or the graph is absent, ``conflict`` when the head
        moved elsewhere, or ``pending_repair`` when the state is ambiguous. An
        unpublished candidate is discarded on the failed/conflict branches.

        Only the attempt named by ``attempt_no``/``generation_id`` is settled:
        every transition CASes on those fields too, and when the row already
        belongs to another attempt (or moved underneath the CAS) the row's
        current status is returned untouched, so a reconciler working from a
        stale read can never terminate a live newer attempt. An unpublished
        candidate is fenced (``mark_generation_failed``) before the head is
        read: the fence waits behind a publish commit still in flight, so a
        landing publish is seen as ``active`` rather than proven "rolled
        back" from a stale read and then discarded.
        """

        cur_now = _now_iso(now)
        operation = await self.get_operation(scope, operation_id)
        if operation is None:
            return "pending_repair"
        if operation.get("status") not in _STRANDED_STATUSES:
            return str(operation.get("status"))
        attempt = {"attempt_no": attempt_no, "current_generation_id": generation_id}
        if any(operation.get(name) != value for name, value in attempt.items()):
            return str(operation.get("status"))
        # A retry reservation keeps the failed attempt's fields, so the reason
        # is the only thing that tells "still the row I claimed" from "settled
        # by someone else and reserved for the next attempt". A reservation is
        # the same key's to resume, never a stranded attempt to converge: it is
        # left alone however late this read was, and the reason also joins the
        # settle predicates so a read taken before the reservation loses too.
        reason = operation.get("pending_repair_reason")
        if reason == "retry_attempt_reservation":
            return str(operation.get("status"))
        attempt["pending_repair_reason"] = reason
        generation = await self.get_generation(scope, generation_id) if generation_id else None
        if generation is not None and generation.get("status") in ("staging", "ready"):
            if not await self.mark_generation_failed(scope, generation_id, now=cur_now):
                generation = await self.get_generation(scope, generation_id)
        candidate_status = generation.get("status") if generation is not None else None
        head = await self.get_graph_head(scope)
        repair_changes = {
            "completed_at": cur_now,
            "repaired_at": cur_now,
            "repair_count": int(operation.get("repair_count") or 0) + 1,
        }

        if head is not None and generation is not None and candidate_status in ("active", "retired"):
            settled = await self.transition_operation(
                scope,
                operation_id,
                expected_status=_STRANDED_STATUSES,
                changes={
                    "status": "committed",
                    "result_revision": generation.get("target_revision"),
                    "current_generation_id": generation_id,
                    "result_schema_version": _OPERATION_RESULT_SCHEMA_VERSION,
                    "result_payload": {
                        "node_count": int(generation.get("node_count") or 0),
                        "edge_count": int(generation.get("edge_count") or 0),
                        "vector_count": int(generation.get("vector_count") or 0),
                    },
                    "last_repair_action": "crash_recovery:committed",
                    **repair_changes,
                },
                expected_fields=attempt,
                now=cur_now,
            )
            return "committed" if settled else await self._current_status(scope, operation_id)
        if head is not None and head.get("active_generation_id") == generation_id:
            return "pending_repair"
        if head is None or (
            operation.get("base_revision") is not None and head.get("active_revision") == operation.get("base_revision")
        ):
            settled = await self.transition_operation(
                scope,
                operation_id,
                expected_status=_STRANDED_STATUSES,
                changes={
                    "status": "failed",
                    "error_code": "SKILL_MANAGE_STORAGE_UNAVAILABLE",
                    "error_message": "attempt did not reach publish",
                    "error_retryable": True,
                    "last_repair_action": "crash_recovery:failed",
                    **repair_changes,
                },
                expected_fields=attempt,
                now=cur_now,
            )
            if not settled:
                return await self._current_status(scope, operation_id)
            if generation_id is not None:
                await self.discard_generation(scope, generation_id)
            return "failed"
        settled = await self.transition_operation(
            scope,
            operation_id,
            expected_status=_STRANDED_STATUSES,
            changes={
                "status": "conflict",
                "error_code": "SKILL_MANAGE_REVISION_CONFLICT",
                "error_message": "head moved past this attempt's base revision",
                "error_retryable": False,
                "last_repair_action": "crash_recovery:conflict",
                **repair_changes,
            },
            expected_fields=attempt,
            now=cur_now,
        )
        if not settled:
            return await self._current_status(scope, operation_id)
        if generation_id is not None:
            await self.discard_generation(scope, generation_id)
        return "conflict"

    async def _current_status(self, scope: SkillManageScope, operation_id: str) -> str:
        operation = await self.get_operation(scope, operation_id)
        return str(operation.get("status")) if operation is not None else "pending_repair"

    # ----------------------------------------------------------------- lease

    async def get_writer_lease(self, scope: SkillManageScope) -> dict[str, Any] | None:
        records = await self._database.get_records(_LEASES, head_scope(scope), (scope.graph_id,))
        if not records:
            return None
        payload = records[0].payload
        return {
            "owner_operation_id": payload.get("owner_operation_id"),
            "fencing_token": int(payload.get("fencing_token") or 0),
            "acquired_at": _as_iso(payload.get("acquired_at")),
            "renewed_at": _as_iso(payload.get("renewed_at")),
            "expires_at": _as_iso(payload.get("expires_at")),
        }

    async def acquire_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> int:
        """Acquire or atomically take over the project+graph writer lease.

        Returns the strictly-monotonic fencing token. A takeover of an expired
        lease increments the prior token; a fresh lease starts at 1.
        """

        lease_scope = head_scope(scope)
        records = await self._database.get_records(_LEASES, lease_scope, (scope.graph_id,))
        prior = records[0].payload if records else None
        token = int(prior.get("fencing_token") or 0) + 1 if prior is not None else 1
        cur_now = _now_iso(now)
        expires = _add_seconds(cur_now, ttl_seconds)
        payload = {
            "graph_id": scope.graph_id,
            "owner_operation_id": owner_operation_id,
            "fencing_token": token,
            "acquired_at": cur_now,
            "renewed_at": cur_now,
            "expires_at": expires,
        }
        if prior is None:
            inserted = await self._database.insert_record_if_absent(
                _LEASES,
                Record(table=_LEASES, record_id=scope.graph_id, scope=lease_scope, payload=payload),
            )
            if not inserted:
                # Lost a create race; retry the takeover path.
                return await self.acquire_writer_lease(
                    scope, owner_operation_id=owner_operation_id, ttl_seconds=ttl_seconds, now=now
                )
            return token
        if prior.get("owner_operation_id") is not None and not _is_expired(prior.get("expires_at"), cur_now):
            raise SkillManageWriterConflictError(
                graph_id=scope.graph_id, owner_operation_id=prior.get("owner_operation_id")
            )
        swapped = await self._database.compare_and_swap_record(
            _LEASES,
            lease_scope,
            scope.graph_id,
            expected={
                "fencing_token": int(prior.get("fencing_token") or 0),
                "owner_operation_id": prior.get("owner_operation_id"),
            },
            changes={
                "owner_operation_id": owner_operation_id,
                "fencing_token": token,
                "acquired_at": cur_now,
                "renewed_at": cur_now,
                "expires_at": expires,
            },
        )
        if not swapped:
            raise SkillManageWriterConflictError(graph_id=scope.graph_id)
        return token

    async def renew_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        token: int,
        ttl_seconds: float,
        now: str | None = None,
    ) -> bool:
        cur_now = _now_iso(now)
        return await self._database.compare_and_swap_record(
            _LEASES,
            head_scope(scope),
            scope.graph_id,
            expected={"owner_operation_id": owner_operation_id, "fencing_token": token},
            changes={"renewed_at": cur_now, "expires_at": _add_seconds(cur_now, ttl_seconds)},
        )

    async def release_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        token: int,
        now: str | None = None,
    ) -> bool:
        cur_now = _now_iso(now)
        return await self._database.compare_and_swap_record(
            _LEASES,
            head_scope(scope),
            scope.graph_id,
            expected={"owner_operation_id": owner_operation_id, "fencing_token": token},
            changes={"owner_operation_id": None, "expires_at": cur_now},
        )

    # ------------------------------------------------------------ generations

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
        inserted = await self._database.insert_record_if_absent(
            _GENERATIONS,
            Record(
                table=_GENERATIONS,
                record_id=generation_id,
                scope=snapshot_scope(scope, generation_id),
                payload={
                    "record_id": generation_id,
                    "generation_id": generation_id,
                    "operation_id": operation_id,
                    "attempt_no": attempt_no,
                    "base_revision": base_revision,
                    "target_revision": target_revision,
                    "status": "staging",
                    "created_at": _now_iso(now),
                },
            ),
        )
        if not inserted:
            raise SkillManageConsistencyError(f"generation {generation_id} is already staged")

    async def get_generation(self, scope: SkillManageScope, generation_id: str) -> dict[str, Any] | None:
        records = await self._database.get_records(_GENERATIONS, snapshot_scope(scope, generation_id), (generation_id,))
        return _row(records[0]) if records else None

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
        return await self._database.compare_and_swap_record(
            _GENERATIONS,
            snapshot_scope(scope, generation_id),
            generation_id,
            expected={"status": "staging"},
            changes={
                "status": "ready",
                "node_count": node_count,
                "edge_count": edge_count,
                "vector_count": vector_count,
                "chunk_count": 0,
                "counts_hash": counts_hash,
                "ready_at": _now_iso(now),
            },
        )

    async def mark_generation_failed(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        now: str | None = None,
    ) -> bool:
        changes = {"status": "failed", "failed_at": _now_iso(now)}
        # ``staging`` is tried first so a candidate flipping staging -> ready
        # between the two CASes is still caught by the second one.
        for expected in ("staging", "ready"):
            if await self._database.compare_and_swap_record(
                _GENERATIONS,
                snapshot_scope(scope, generation_id),
                generation_id,
                expected={"status": expected},
                changes=changes,
            ):
                return True
        return False

    async def write_snapshot(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        nodes: list[SkillGraphNode],
        edges: list[SkillGraphEdge],
        memberships: list[dict[str, Any]],
    ) -> None:
        node_records = [
            Record(
                table="skill_manage_nodes",
                record_id=physical_record_id(generation_id, "node", node.graph_node_id),
                scope=snapshot_scope(scope, generation_id),
                payload={
                    "record_id": physical_record_id(generation_id, "node", node.graph_node_id),
                    "graph_node_id": node.graph_node_id,
                    "generation_id": generation_id,
                    "node_type": node.node_type,
                    "skill_id": node.skill_id,
                    "version_id": node.version_id,
                    "parent_node_id": node.parent_skill_id,
                    "payload": node.model_dump(mode="python"),
                    "content_hash": node.content_hash,
                },
            )
            for node in nodes
        ]
        edge_records = [
            Record(
                table="skill_manage_edges",
                record_id=physical_record_id(generation_id, "edge", edge.edge_id),
                scope=snapshot_scope(scope, generation_id),
                payload={
                    "record_id": physical_record_id(generation_id, "edge", edge.edge_id),
                    "edge_id": edge.edge_id,
                    "generation_id": generation_id,
                    "source_node_id": edge.source_id,
                    "target_node_id": edge.target_id,
                    "edge_type": edge.edge_type.value if hasattr(edge.edge_type, "value") else str(edge.edge_type),
                    "weight": edge.weight,
                    "confidence": None,
                    "payload": edge.model_dump(mode="python"),
                },
            )
            for edge in edges
        ]
        membership_records = [
            Record(
                table="skill_manage_memberships",
                record_id=physical_record_id(generation_id, "membership", membership["skill_id"]),
                scope=snapshot_scope(scope, generation_id),
                payload={
                    "record_id": physical_record_id(generation_id, "membership", membership["skill_id"]),
                    "membership_id": membership["skill_id"],
                    "skill_id": membership["skill_id"],
                    "version_id": membership["version_id"],
                    "graph_node_id": membership.get("graph_node_id", ""),
                    "generation_id": generation_id,
                    "status": "active",
                },
            )
            for membership in memberships
        ]
        await self._database.upsert_records("skill_manage_nodes", node_records)
        await self._database.upsert_records("skill_manage_edges", edge_records)
        await self._database.upsert_records("skill_manage_memberships", membership_records)

    async def write_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        embeddings: list[SkillGraphEmbedding],
    ) -> None:
        vector_scope = _vector_scope(scope, generation_id)
        vector_records: list[VectorRecord] = []
        mapping_records: list[Record] = []
        for embedding in embeddings:
            vector_record_id = physical_record_id(generation_id, "embedding", embedding.graph_node_id)
            vector_records.append(
                VectorRecord(
                    table=SKILL_MANAGE_EMBEDDINGS,
                    record_id=vector_record_id,
                    scope=vector_scope,
                    payload={
                        "vector_record_id": vector_record_id,
                        "graph_node_id": embedding.graph_node_id,
                        "skill_id": embedding.skill_id,
                        "version_id": embedding.version_id,
                        "node_type": embedding.node_type,
                        "generation_id": generation_id,
                        "content_hash": embedding.content_hash,
                        "embedding_model": embedding.embedding_model,
                        "embedding_profile_version": embedding.embedding_profile_version,
                        "metadata": {},
                    },
                    vectors=VectorValue(dense={EMBEDDING_VECTOR_NAME: embedding.vector}),
                )
            )
            mapping_id = physical_record_id(generation_id, "mapping", embedding.graph_node_id)
            mapping_records.append(
                Record(
                    table="skill_manage_id_mappings",
                    record_id=mapping_id,
                    scope=snapshot_scope(scope, generation_id),
                    payload={
                        "record_id": mapping_id,
                        "graph_node_id": embedding.graph_node_id,
                        "generation_id": generation_id,
                        "vector_record_id": vector_record_id,
                        "mapping_type": "node",
                    },
                )
            )
        await self._vector_store.upsert_records(SKILL_MANAGE_EMBEDDINGS, vector_records)
        await self._database.upsert_records("skill_manage_id_mappings", mapping_records)

    async def load_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
    ) -> tuple[SkillGraphEmbedding, ...]:
        records = await self._query_vector_paged(
            SKILL_MANAGE_EMBEDDINGS,
            _vector_scope(scope, generation_id),
            with_vectors=True,
        )
        embeddings: list[SkillGraphEmbedding] = []
        for record in records:
            vector = (record.vectors.dense if record.vectors is not None else {}).get(EMBEDDING_VECTOR_NAME)
            if not vector:
                continue
            payload = record.payload
            embeddings.append(
                SkillGraphEmbedding(
                    graph_node_id=payload["graph_node_id"],
                    node_type=payload.get("node_type") or "skill",
                    skill_id=payload.get("skill_id"),
                    version_id=payload.get("version_id"),
                    content_hash=payload["content_hash"],
                    embedding_model=payload["embedding_model"],
                    embedding_profile_version=payload["embedding_profile_version"],
                    vector=tuple(vector),
                )
            )
        return tuple(embeddings)

    async def count_embeddings(self, scope: SkillManageScope, generation_id: str) -> int:
        records = await self._query_vector_paged(
            SKILL_MANAGE_EMBEDDINGS,
            _vector_scope(scope, generation_id),
        )
        return len(records)

    async def search_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        query_vector: tuple[float, ...],
        top_k: int,
        score_threshold: float | None = None,
    ) -> tuple[SkillGraphVectorHit, ...]:
        hits = await self._vector_store.search_vectors(
            VectorQuery(
                table=SKILL_MANAGE_EMBEDDINGS,
                scope=_vector_scope(scope, generation_id),
                vector_name=EMBEDDING_VECTOR_NAME,
                dense_vector=tuple(query_vector),
                top_k=top_k,
                score_threshold=score_threshold,
            )
        )
        # pgvector reports an undefined cosine (a zero-norm vector on either
        # side) as NaN, which PostgreSQL orders after every number and lets
        # through any threshold; such a pair is not a hit.
        return tuple(
            SkillGraphVectorHit(
                graph_node_id=hit.record.payload["graph_node_id"],
                skill_id=hit.record.payload.get("skill_id"),
                version_id=hit.record.payload.get("version_id"),
                score=float(hit.score),
            )
            for hit in hits
            if math.isfinite(hit.score)
        )

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
        fail_after_publish: bool = False,
    ) -> bool:
        cur_now = _now_iso(now)
        async with self._database.transaction() as transaction:
            if owner_operation_id is not None or fencing_token is not None:
                lease_records = await transaction.get_records(_LEASES, head_scope(scope), (scope.graph_id,))
                lease = lease_records[0].payload if lease_records else None
                if (
                    lease is None
                    or lease.get("owner_operation_id") != owner_operation_id
                    or int(lease.get("fencing_token") or 0) != fencing_token
                    or _is_expired(lease.get("expires_at"), cur_now)
                ):
                    raise SkillManageWriterConflictError(
                        graph_id=scope.graph_id,
                        owner_operation_id=lease.get("owner_operation_id") if lease else None,
                    )
            generation_records = await transaction.get_records(
                _GENERATIONS, snapshot_scope(scope, generation_id), (generation_id,)
            )
            generation = generation_records[0].payload if generation_records else None
            if generation is None or generation.get("status") != "ready":
                status = generation.get("status") if generation else None
                raise SkillManageConsistencyError(f"generation {generation_id} is {status or 'missing'}, not ready")
            head_records = await transaction.get_records(_GRAPHS, head_scope(scope), (scope.graph_id,))
            previous = head_records[0].payload if head_records else None
            previous_generation_id = previous.get("active_generation_id") if previous else None

            if expected_revision is None:
                published = await transaction.insert_record_if_absent(
                    _GRAPHS,
                    Record(
                        table=_GRAPHS,
                        record_id=scope.graph_id,
                        scope=head_scope(scope),
                        payload={
                            "graph_id": scope.graph_id,
                            "active_revision": new_revision,
                            "active_generation_id": generation_id,
                            "algorithm_name": algorithm_name,
                            "algorithm_version": algorithm_version,
                            "embedding_profile": dict(embedding_profile or {}),
                            "created_at": cur_now,
                            "updated_at": cur_now,
                            "status": "active",
                        },
                    ),
                )
            else:
                changes: dict[str, Any] = {
                    "active_revision": new_revision,
                    "active_generation_id": generation_id,
                    "updated_at": cur_now,
                }
                if embedding_profile is not None:
                    changes["embedding_profile"] = dict(embedding_profile)
                if algorithm_name is not None:
                    changes["algorithm_name"] = algorithm_name
                if algorithm_version is not None:
                    changes["algorithm_version"] = algorithm_version
                published = await transaction.compare_and_swap_record(
                    _GRAPHS,
                    head_scope(scope),
                    scope.graph_id,
                    expected={"active_revision": expected_revision},
                    changes=changes,
                )
            if not published:
                return False
            # The flip is a CAS on ``ready``, not a blind patch: a reconciler that
            # took the lease over while this writer was stalled inside the
            # transaction has fenced the candidate to ``failed`` (and possibly
            # deleted it) without ever touching the head, so the head CAS alone
            # cannot detect it. Losing here rolls the whole switch back.
            activated = await transaction.compare_and_swap_record(
                _GENERATIONS,
                snapshot_scope(scope, generation_id),
                generation_id,
                expected={"status": "ready"},
                changes={"status": "active", "activated_at": cur_now},
            )
            if not activated:
                raise SkillManageConsistencyError(f"generation {generation_id} is no longer ready at publish")
            if previous_generation_id is not None and previous_generation_id != generation_id:
                old_records = await transaction.get_records(
                    _GENERATIONS, snapshot_scope(scope, previous_generation_id), (previous_generation_id,)
                )
                if old_records and old_records[0].payload.get("status") == "active":
                    await transaction.patch_record(
                        _GENERATIONS,
                        snapshot_scope(scope, previous_generation_id),
                        previous_generation_id,
                        {"status": "retired", "retired_at": cur_now},
                    )
            if fail_after_publish:
                # Fault-injection hook (AC-P3-008): raise after the head-swap and
                # generation flips but before the enclosing transaction commits.
                raise RuntimeError("injected publish failure")
            return True

    async def discard_generation(self, scope: SkillManageScope, generation_id: str) -> None:
        """Drop an unpublished generation's rows; a published one is never touched.

        The candidate is fenced to ``failed`` first. The fence is a row update
        that waits behind a publish commit still in flight, so a candidate
        that turns out to be ``active`` when the fence loses is left alone
        instead of having its rows deleted underneath the head.
        """

        head = await self.get_graph_head(scope)
        if head is not None and head.get("active_generation_id") == generation_id:
            return
        generation = await self.get_generation(scope, generation_id)
        if generation is not None:
            status = generation.get("status")
            if status in ("active", "retired"):
                return
            if status in ("staging", "ready") and not await self.mark_generation_failed(scope, generation_id):
                refreshed = await self.get_generation(scope, generation_id)
                if refreshed is not None and refreshed.get("status") in ("active", "retired"):
                    return
        await self._delete_generation_records(scope, generation_id)

    async def load_snapshot(self, scope: SkillManageScope, generation_id: str) -> SkillGraphSnapshot:
        generation = await self.get_generation(scope, generation_id)
        target_revision = generation.get("target_revision") if generation is not None else None
        sscope = snapshot_scope(scope, generation_id)
        node_records = await self._query_structured_paged("skill_manage_nodes", sscope)
        edge_records = await self._query_structured_paged("skill_manage_edges", sscope)
        membership_records = await self._query_structured_paged("skill_manage_memberships", sscope)
        nodes = tuple(SkillGraphNode.model_validate(record.payload.get("payload", {})) for record in node_records)
        edges = tuple(SkillGraphEdge.model_validate(record.payload.get("payload", {})) for record in edge_records)
        memberships = tuple(
            SkillGraphMembership(
                skill_id=record.payload["skill_id"],
                version_id=record.payload["version_id"],
                graph_node_id=record.payload.get("graph_node_id", ""),
            )
            for record in membership_records
        )
        return SkillGraphSnapshot(
            generation_id=generation_id,
            revision=int(target_revision) if target_revision is not None else 0,
            nodes=nodes,
            edges=edges,
            memberships=memberships,
        )

    async def gc_retired_generations(
        self,
        scope: SkillManageScope,
        *,
        keep: int = 2,
        min_age_seconds: float = 0.0,
        now: str | None = None,
    ) -> list[str]:
        """Delete retired generations beyond ``keep`` that retired at least ``min_age_seconds`` ago.

        Re-reads the head first so a just-reactivated generation is never
        deleted, and never deletes a generation still referenced by a
        non-terminal operation: a ``validating`` row whose publish landed and
        was later retired must keep its generation row so repair can still
        converge it to ``committed``. Returns the removed IDs, oldest first.
        Generations are deleted independently: a failure defers only that
        generation, and the pass ends by raising
        ``SkillManageGcPartialFailureError`` carrying both lists so the
        caller can report exactly what was removed.
        """

        cur_now = _as_datetime(_now_iso(now))
        assert cur_now is not None
        head = await self.get_graph_head(scope)
        active_generation_id = head.get("active_generation_id") if head else None
        referenced = {
            row.get("current_generation_id")
            for row in await self.list_operations(scope, statuses=_STRANDED_STATUSES)
            if row.get("current_generation_id") is not None
        }
        records = await self._query_structured_paged(_GENERATIONS, head_scope(scope))
        retired: list[tuple[datetime, str]] = []
        tombstoned: list[str] = []
        for record in records:
            payload = record.payload
            generation_id = payload.get("generation_id")
            if generation_id == active_generation_id or generation_id in referenced:
                continue
            if payload.get("status") == "deleted":
                # A deletion interrupted after its tombstone: finish it
                # regardless of retention, nothing can read it any more.
                tombstoned.append(payload["generation_id"])
                continue
            if payload.get("status") != "retired":
                continue
            retired_at = _as_datetime(payload.get("retired_at"))
            if retired_at is None:
                continue
            retired.append((retired_at, payload["generation_id"]))
        retired.sort()
        beyond_keep = retired[: max(0, len(retired) - keep)]
        cutoff = cur_now - timedelta(seconds=min_age_seconds)
        removed: list[str] = []
        deferred: list[tuple[str, str]] = []
        targets = sorted(tombstoned) + [
            generation_id for retired_at, generation_id in beyond_keep if retired_at <= cutoff
        ]
        for generation_id in targets:
            try:
                await self._delete_generation_records(scope, generation_id, now=now)
            except Exception as exc:
                # The row goes last, so a lost acknowledgement of a delete
                # that did land shows as a row that is gone: report it as
                # removed rather than deferring it forever. If the store is
                # down the re-read fails too and the id stays deferred.
                if not await self._generation_gone(scope, generation_id):
                    deferred.append((generation_id, type(exc).__name__))
                    continue
            removed.append(generation_id)
        if deferred:
            raise SkillManageGcPartialFailureError(removed=removed, deferred=deferred)
        return removed

    async def _generation_gone(self, scope: SkillManageScope, generation_id: str) -> bool:
        try:
            return await self.get_generation(scope, generation_id) is None
        except Exception:
            return False

    async def list_operations(
        self,
        scope: SkillManageScope,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        wanted = set(statuses) if statuses is not None else None
        records = await self._query_structured_paged(_OPERATIONS, head_scope(scope))
        return [_row(record) for record in records if wanted is None or record.payload.get("status") in wanted]

    async def list_generations(
        self,
        scope: SkillManageScope,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        wanted = set(statuses) if statuses is not None else None
        records = await self._query_structured_paged(_GENERATIONS, head_scope(scope))
        return [_row(record) for record in records if wanted is None or record.payload.get("status") in wanted]

    async def list_snapshot_generation_ids(self, scope: SkillManageScope) -> set[str]:
        referenced: set[str] = set()
        for table in (
            "skill_manage_nodes",
            "skill_manage_edges",
            "skill_manage_memberships",
            "skill_manage_id_mappings",
            "skill_manage_chunks",
        ):
            for record in await self._query_structured_paged(table, head_scope(scope)):
                generation_id = record.scope.get("generation_id")
                if generation_id is not None:
                    referenced.add(str(generation_id))
        vector_scope = VectorScope(project_id=scope.project_id, graph_id=scope.graph_id)
        for record in await self._query_vector_paged(SKILL_MANAGE_EMBEDDINGS, vector_scope):
            generation_id = record.payload.get("generation_id")
            if generation_id is not None:
                referenced.add(str(generation_id))
        return referenced

    async def _delete_generation_records(
        self, scope: SkillManageScope, generation_id: str, *, now: str | None = None
    ) -> None:
        """Tombstone the generation row, then delete its records, the row itself last.

        The ``deleted`` tombstone lands before any record is removed, so a
        reader that finds the row still ``active``/``retired`` after loading
        the snapshot knows every load ran before the deletion started; a pass
        interrupted after the tombstone is resumed by the next GC/repair pass.
        A row in any other status is not a deletion target and is left alone.
        """

        sscope = snapshot_scope(scope, generation_id)
        generation = await self.get_generation(scope, generation_id)
        if generation is not None and generation.get("status") != "deleted":
            tombstoned = False
            for expected in ("retired", "failed"):
                if await self._database.compare_and_swap_record(
                    _GENERATIONS,
                    sscope,
                    generation_id,
                    expected={"status": expected},
                    changes={"status": "deleted", "deleted_at": _now_iso(now)},
                ):
                    tombstoned = True
                    break
            if not tombstoned:
                return
        vector_scope = _vector_scope(scope, generation_id)
        vector_records = await self._query_vector_paged(SKILL_MANAGE_EMBEDDINGS, vector_scope)
        if vector_records:
            await self._vector_store.delete_records(
                SKILL_MANAGE_EMBEDDINGS, vector_scope, [record.record_id for record in vector_records]
            )
        for table in (
            "skill_manage_id_mappings",
            "skill_manage_chunks",
            "skill_manage_edges",
            "skill_manage_nodes",
            "skill_manage_memberships",
        ):
            records = await self._query_structured_paged(table, sscope)
            ids = [record.record_id for record in records]
            if ids:
                await self._database.delete_records(table, sscope, ids)
        await self._database.delete_records(_GENERATIONS, sscope, [generation_id])

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._vector_store.close()
        if self._owns_database:
            await self._database.close()


__all__ = ["PgSkillGraphStorageAdapter"]
