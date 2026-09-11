"""AC-P3-002 (fake wrapper): run the shared contract against an in-memory adapter."""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from mindmemos_skill.application.skill_manage.state_machine import OperationStatus, require_transition
from mindmemos_skill.errors import (
    SkillManageConsistencyError,
    SkillManageGcPartialFailureError,
    SkillManageWriterConflictError,
)
from mindmemos_skill.infra.skill_manage import SkillGraphStorageAdapter
from mindmemos_skill.typing import (
    SkillGraphEdge,
    SkillGraphEmbedding,
    SkillGraphNode,
    SkillGraphSnapshot,
    SkillGraphVectorHit,
    SkillManageScope,
)
from skill_manage_storage_contract import run_storage_contract

_STRANDED = ("pending_repair", "running", "validating")


def _parse(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _now_iso(now: str | None) -> str:
    return now or datetime.now(UTC).isoformat()


def _add_seconds(iso_value: str, seconds: float) -> str:
    parsed = _parse(iso_value)
    assert parsed is not None
    return (parsed + timedelta(seconds=seconds)).isoformat()


def _fence(fencing_token: int | None) -> dict[str, Any] | None:
    return {"writer_fencing_token": fencing_token} if fencing_token is not None else None


def _is_expired(expires_at: str | None, now: str) -> bool:
    expires = _parse(expires_at)
    current = _parse(now)
    assert current is not None
    return expires is None or expires <= current


class InMemorySkillGraphStorageAdapter:
    """Deterministic fake exercising the same protocol surface as the PG adapter."""

    def __init__(self) -> None:
        self._heads: dict[tuple[str, str], dict[str, Any]] = {}
        self._operations: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._generations: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._snapshots: dict[tuple[str, str, str], SkillGraphSnapshot] = {}
        self._embeddings: dict[tuple[str, str, str], list[SkillGraphEmbedding]] = {}
        self._leases: dict[tuple[str, str], dict[str, Any]] = {}

    async def ensure_schema(self) -> None:
        return None

    async def is_ready(self) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def get_graph_head(self, scope: SkillManageScope) -> dict[str, Any] | None:
        # A read is a snapshot, as it is on PostgreSQL: a row dict handed out
        # must not change under its holder when the row is written later.
        head = self._heads.get((scope.project_id, scope.graph_id))
        return dict(head) if head is not None else None

    async def create_head_if_absent(self, scope: SkillManageScope, head: dict[str, Any]) -> bool:
        key = (scope.project_id, scope.graph_id)
        if key in self._heads:
            return False
        self._heads[key] = dict(head)
        return True

    async def get_operation(self, scope: SkillManageScope, operation_id: str) -> dict[str, Any] | None:
        op = self._operations.get((scope.project_id, scope.graph_id, operation_id))
        return dict(op) if op is not None else None

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
        key = (scope.project_id, scope.graph_id, operation_id)
        if key in self._operations:
            return False
        self._operations[key] = {
            "operation_id": operation_id,
            "idempotency_key": idempotency_key,
            "request_hash": request_hash,
            "operation_type": operation_type,
            "status": "pending",
            "attempt_no": 1,
            "created_at": _now_iso(now),
            "updated_at": _now_iso(now),
        }
        return True

    async def get_writer_lease(self, scope: SkillManageScope) -> dict[str, Any] | None:
        current = self._leases.get((scope.project_id, scope.graph_id))
        return dict(current) if current is not None else None

    async def acquire_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> int:
        key = (scope.project_id, scope.graph_id)
        cur_now = _now_iso(now)
        current = self._leases.get(key)
        if current is None:
            token = 1
        elif current["owner_operation_id"] is not None and not _is_expired(current["expires_at"], cur_now):
            raise SkillManageWriterConflictError(
                graph_id=scope.graph_id, owner_operation_id=current["owner_operation_id"]
            )
        else:
            token = int(current["fencing_token"]) + 1
        self._leases[key] = {
            "owner_operation_id": owner_operation_id,
            "fencing_token": token,
            "acquired_at": cur_now,
            "renewed_at": cur_now,
            "expires_at": _add_seconds(cur_now, ttl_seconds),
        }
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
        current = self._leases.get((scope.project_id, scope.graph_id))
        if current is None or current["owner_operation_id"] != owner_operation_id or current["fencing_token"] != token:
            return False
        cur_now = _now_iso(now)
        current["renewed_at"] = cur_now
        current["expires_at"] = _add_seconds(cur_now, ttl_seconds)
        return True

    async def release_writer_lease(
        self,
        scope: SkillManageScope,
        *,
        owner_operation_id: str,
        token: int,
        now: str | None = None,
    ) -> bool:
        current = self._leases.get((scope.project_id, scope.graph_id))
        if current is None or current["owner_operation_id"] != owner_operation_id or current["fencing_token"] != token:
            return False
        current["owner_operation_id"] = None
        current["expires_at"] = _now_iso(now)
        return True

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
        op = self._operations.get((scope.project_id, scope.graph_id, operation_id))
        if op is None or op["status"] not in expected_status:
            return False
        if any(op.get(name) != value for name, value in (expected_fields or {}).items()):
            return False
        target = changes.get("status")
        if target is not None:
            try:
                require_transition(OperationStatus(op["status"]), OperationStatus(target))
            except (ValueError, TypeError):
                return False
        op.update(changes)
        op["updated_at"] = _now_iso(now)
        return True

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
        op = await self.get_operation(scope, operation_id)
        if op is None or op["status"] != "failed" or op.get("error_retryable") is not True:
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
        return await self.transition_operation(
            scope,
            operation_id,
            expected_status=["running", "validating"],
            changes={"status": "pending_repair", "pending_repair_reason": "crash_recovery"},
            expected_fields={"attempt_no": expected_attempt_no, "current_generation_id": expected_generation_id},
            now=now,
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
        cur_now = _now_iso(now)
        op = await self.get_operation(scope, operation_id)
        if op is None:
            return "pending_repair"
        if op["status"] not in _STRANDED:
            return str(op["status"])
        attempt = {"attempt_no": attempt_no, "current_generation_id": generation_id}
        if any(op.get(name) != value for name, value in attempt.items()):
            return str(op["status"])
        # A retry reservation keeps the attempt fields and is never settled
        # here; the reason also joins the settle predicate (see the PG adapter).
        reason = op.get("pending_repair_reason")
        if reason == "retry_attempt_reservation":
            return str(op["status"])
        attempt["pending_repair_reason"] = reason
        generation = await self.get_generation(scope, generation_id) if generation_id else None
        if generation is not None and generation["status"] in ("staging", "ready"):
            if not await self.mark_generation_failed(scope, generation_id, now=cur_now):
                generation = await self.get_generation(scope, generation_id)
        candidate_status = generation["status"] if generation is not None else None
        head = await self.get_graph_head(scope)
        repair_changes = {
            "completed_at": cur_now,
            "repaired_at": cur_now,
            "repair_count": int(op.get("repair_count") or 0) + 1,
        }
        if head is not None and generation is not None and candidate_status in ("active", "retired"):
            settled = await self.transition_operation(
                scope,
                operation_id,
                expected_status=list(_STRANDED),
                changes={
                    "status": "committed",
                    "result_revision": generation.get("target_revision"),
                    "current_generation_id": generation_id,
                    "result_schema_version": 1,
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
            op.get("base_revision") is not None and head.get("active_revision") == op.get("base_revision")
        ):
            settled = await self.transition_operation(
                scope,
                operation_id,
                expected_status=list(_STRANDED),
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
            expected_status=list(_STRANDED),
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
        op = await self.get_operation(scope, operation_id)
        return str(op["status"]) if op is not None else "pending_repair"

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
        key = (scope.project_id, scope.graph_id, generation_id)
        if key in self._generations:
            raise SkillManageConsistencyError(f"generation {generation_id} is already staged")
        self._generations[key] = {
            "generation_id": generation_id,
            "operation_id": operation_id,
            "attempt_no": attempt_no,
            "base_revision": base_revision,
            "target_revision": target_revision,
            "status": "staging",
            "node_count": 0,
            "edge_count": 0,
            "vector_count": 0,
            "counts_hash": None,
            "created_at": _now_iso(now),
            "ready_at": None,
            "activated_at": None,
            "retired_at": None,
        }

    async def get_generation(self, scope: SkillManageScope, generation_id: str) -> dict[str, Any] | None:
        generation = self._generations.get((scope.project_id, scope.graph_id, generation_id))
        return dict(generation) if generation is not None else None

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
        generation = self._generations.get((scope.project_id, scope.graph_id, generation_id))
        if generation is None or generation["status"] != "staging":
            return False
        generation.update(
            {
                "status": "ready",
                "node_count": node_count,
                "edge_count": edge_count,
                "vector_count": vector_count,
                "counts_hash": counts_hash,
                "ready_at": _now_iso(now),
            }
        )
        return True

    async def mark_generation_failed(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        now: str | None = None,
    ) -> bool:
        generation = self._generations.get((scope.project_id, scope.graph_id, generation_id))
        if generation is None or generation["status"] not in ("staging", "ready"):
            return False
        generation.update({"status": "failed", "failed_at": _now_iso(now)})
        return True

    async def write_snapshot(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        nodes: list[SkillGraphNode],
        edges: list[SkillGraphEdge],
        memberships: list[dict[str, Any]],
    ) -> None:
        generation = self._generations.get((scope.project_id, scope.graph_id, generation_id)) or {}
        self._snapshots[(scope.project_id, scope.graph_id, generation_id)] = SkillGraphSnapshot(
            generation_id=generation_id,
            revision=int(generation.get("target_revision") or 0),
            nodes=tuple(nodes),
            edges=tuple(edges),
            memberships=tuple(memberships),
        )

    async def write_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        embeddings: list[SkillGraphEmbedding],
    ) -> None:
        self._embeddings[(scope.project_id, scope.graph_id, generation_id)] = list(embeddings)

    async def load_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
    ) -> tuple[SkillGraphEmbedding, ...]:
        return tuple(self._embeddings.get((scope.project_id, scope.graph_id, generation_id), ()))

    async def count_embeddings(self, scope: SkillManageScope, generation_id: str) -> int:
        return len(self._embeddings.get((scope.project_id, scope.graph_id, generation_id), ()))

    async def search_embeddings(
        self,
        scope: SkillManageScope,
        generation_id: str,
        *,
        query_vector: tuple[float, ...],
        top_k: int,
        score_threshold: float | None = None,
    ) -> tuple[SkillGraphVectorHit, ...]:
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        scored: list[SkillGraphVectorHit] = []
        for embedding in self._embeddings.get((scope.project_id, scope.graph_id, generation_id), ()):
            norm = math.sqrt(sum(value * value for value in embedding.vector))
            if query_norm == 0 or norm == 0:
                continue
            if len(embedding.vector) != len(query_vector):
                score = 0.0
            else:
                score = sum(a * b for a, b in zip(query_vector, embedding.vector)) / (query_norm * norm)
            if score_threshold is not None and score < score_threshold:
                continue
            scored.append(
                SkillGraphVectorHit(
                    graph_node_id=embedding.graph_node_id,
                    skill_id=embedding.skill_id,
                    version_id=embedding.version_id,
                    score=score,
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.graph_node_id))
        return tuple(scored[:top_k])

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
        cur_now = _now_iso(now)
        key = (scope.project_id, scope.graph_id)
        if owner_operation_id is not None or fencing_token is not None:
            lease = self._leases.get(key)
            if (
                lease is None
                or lease["owner_operation_id"] != owner_operation_id
                or lease["fencing_token"] != fencing_token
                or _is_expired(lease["expires_at"], cur_now)
            ):
                raise SkillManageWriterConflictError(
                    graph_id=scope.graph_id, owner_operation_id=lease["owner_operation_id"] if lease else None
                )
        generation = self._generations.get((*key, generation_id))
        if generation is None or generation["status"] != "ready":
            status = generation["status"] if generation else None
            raise SkillManageConsistencyError(f"generation {generation_id} is {status or 'missing'}, not ready")
        current = self._heads.get(key)
        head = {
            "active_revision": new_revision,
            "active_generation_id": generation_id,
            "algorithm_name": algorithm_name,
            "algorithm_version": algorithm_version,
            "embedding_profile": dict(embedding_profile or {}),
        }
        if current is None and expected_revision is None:
            self._heads[key] = head
        elif current is not None and current.get("active_revision") == expected_revision:
            if embedding_profile is None:
                head["embedding_profile"] = dict(current.get("embedding_profile") or {})
            if algorithm_name is None:
                head["algorithm_name"] = current.get("algorithm_name")
            if algorithm_version is None:
                head["algorithm_version"] = current.get("algorithm_version")
            self._heads[key] = head
        else:
            return False
        generation["status"] = "active"
        generation["activated_at"] = cur_now
        previous_generation_id = current.get("active_generation_id") if current else None
        if previous_generation_id is not None and previous_generation_id != generation_id:
            previous = self._generations.get((*key, previous_generation_id))
            if previous is not None and previous["status"] == "active":
                previous["status"] = "retired"
                previous["retired_at"] = cur_now
        return True

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
                "result_schema_version": 1,
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

    async def discard_generation(self, scope: SkillManageScope, generation_id: str) -> None:
        head = self._heads.get((scope.project_id, scope.graph_id))
        if head is not None and head.get("active_generation_id") == generation_id:
            return
        generation = self._generations.get((scope.project_id, scope.graph_id, generation_id))
        if generation is not None:
            if generation["status"] in ("active", "retired"):
                return
            if generation["status"] in ("staging", "ready") and not await self.mark_generation_failed(
                scope, generation_id
            ):
                refreshed = self._generations.get((scope.project_id, scope.graph_id, generation_id))
                if refreshed is not None and refreshed["status"] in ("active", "retired"):
                    return
        await self._delete_generation_records(scope, generation_id)

    async def gc_retired_generations(
        self,
        scope: SkillManageScope,
        *,
        keep: int = 2,
        min_age_seconds: float = 0.0,
        now: str | None = None,
    ) -> list[str]:
        cur_now = _parse(_now_iso(now))
        assert cur_now is not None
        head = self._heads.get((scope.project_id, scope.graph_id))
        active = head.get("active_generation_id") if head else None
        referenced = {
            op.get("current_generation_id")
            for op in await self.list_operations(scope, statuses=_STRANDED)
            if op.get("current_generation_id") is not None
        }
        retired: list[tuple[datetime, str]] = []
        tombstoned: list[str] = []
        for (project_id, graph_id, generation_id), generation in self._generations.items():
            if (project_id, graph_id) != (scope.project_id, scope.graph_id):
                continue
            if generation_id == active or generation_id in referenced:
                continue
            if generation["status"] == "deleted":
                tombstoned.append(generation_id)
                continue
            retired_at = _parse(generation.get("retired_at"))
            if generation["status"] != "retired" or retired_at is None:
                continue
            retired.append((retired_at, generation_id))
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

    async def _delete_generation_records(
        self, scope: SkillManageScope, generation_id: str, *, now: str | None = None
    ) -> None:
        key = (scope.project_id, scope.graph_id, generation_id)
        generation = self._generations.get(key)
        if generation is not None and generation["status"] != "deleted":
            if generation["status"] not in ("retired", "failed"):
                return
            generation.update({"status": "deleted", "deleted_at": _now_iso(now)})
        self._generations.pop(key, None)
        self._snapshots.pop(key, None)
        self._embeddings.pop(key, None)

    async def load_snapshot(self, scope: SkillManageScope, generation_id: str) -> SkillGraphSnapshot:
        snapshot = self._snapshots.get((scope.project_id, scope.graph_id, generation_id))
        if snapshot is None:
            return SkillGraphSnapshot(generation_id=generation_id, revision=0)
        return snapshot

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
        return [
            dict(op)
            for (project_id, graph_id, _), op in self._operations.items()
            if (project_id, graph_id) == (scope.project_id, scope.graph_id)
            and (wanted is None or op.get("status") in wanted)
        ]

    async def list_generations(
        self,
        scope: SkillManageScope,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        wanted = set(statuses) if statuses is not None else None
        return [
            dict(generation)
            for (project_id, graph_id, _), generation in self._generations.items()
            if (project_id, graph_id) == (scope.project_id, scope.graph_id)
            and (wanted is None or generation.get("status") in wanted)
        ]

    async def list_snapshot_generation_ids(self, scope: SkillManageScope) -> set[str]:
        return {
            generation_id
            for store in (self._snapshots, self._embeddings)
            for (project_id, graph_id, generation_id) in store
            if (project_id, graph_id) == (scope.project_id, scope.graph_id)
        }

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_fake_adapter_passes_shared_storage_contract() -> None:
    adapter = InMemorySkillGraphStorageAdapter()
    for case in run_storage_contract(lambda: _noop(adapter)):
        await case(adapter)


async def _noop(adapter: SkillGraphStorageAdapter) -> SkillGraphStorageAdapter:
    return adapter
