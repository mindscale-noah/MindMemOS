"""Shared storage contract exercised by fake and real adapters (AC-P3-002).

Every adapter implementation must pass this suite unchanged. The suite only
touches the ``SkillGraphStorageAdapter`` protocol surface, so it can run
against an in-memory fake or a real PostgreSQL/pgvector backend.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable

import pytest
from mindmemos_skill.errors import (
    SkillManageConsistencyError,
    SkillManageGcPartialFailureError,
    SkillManageWriterConflictError,
)
from mindmemos_skill.infra.skill_manage import SkillGraphStorageAdapter
from mindmemos_skill.typing import SkillGraphEdge, SkillGraphEmbedding, SkillGraphNode, SkillManageScope

T0 = "2026-09-03T00:00:00+00:00"
# The real backend's vector table is fixed at this width; the fake accepts any.
VECTOR_DIMENSION = 3072


async def _stage_ready(
    adapter: SkillGraphStorageAdapter,
    scope: SkillManageScope,
    generation_id: str,
    *,
    operation_id: str,
    attempt_no: int = 1,
    base_revision: int | None,
    target_revision: int,
    nodes: list[SkillGraphNode] | None = None,
    edges: list[SkillGraphEdge] | None = None,
    memberships: list[dict[str, Any]] | None = None,
    now: str = T0,
) -> None:
    """Stage, write and mark ready: the steps a publish requires."""

    await adapter.stage_generation(
        scope,
        generation_id,
        operation_id=operation_id,
        attempt_no=attempt_no,
        base_revision=base_revision,
        target_revision=target_revision,
        now=now,
    )
    await adapter.write_snapshot(
        scope, generation_id, nodes=nodes or [], edges=edges or [], memberships=memberships or []
    )
    assert await adapter.mark_generation_ready(
        scope,
        generation_id,
        node_count=len(nodes or []),
        edge_count=len(edges or []),
        vector_count=0,
        counts_hash="h",
        now=now,
    )


def run_storage_contract(
    make_adapter: Callable[[], Awaitable[SkillGraphStorageAdapter]],
) -> list[Callable[[SkillGraphStorageAdapter], Awaitable[None]]]:
    """Return the ordered list of contract cases for one adapter instance."""

    async def head_create_and_idempotency(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p1", graph_id="g1")
        assert await adapter.get_graph_head(scope) is None
        assert await adapter.create_head_if_absent(scope, {"active_revision": 0}) is True
        assert await adapter.create_head_if_absent(scope, {"active_revision": 0}) is False
        head = await adapter.get_graph_head(scope)
        assert head is not None
        assert head["active_revision"] == 0

    async def operation_reserve_idempotency(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p1", graph_id="g1")
        assert (
            await adapter.reserve_operation(
                scope,
                "op-1",
                idempotency_key="k1",
                request_hash="h1",
                operation_type="build",
            )
            is True
        )
        assert (
            await adapter.reserve_operation(
                scope,
                "op-1",
                idempotency_key="k1",
                request_hash="h1",
                operation_type="build",
            )
            is False
        )
        operation = await adapter.get_operation(scope, "op-1")
        assert operation is not None
        assert operation["operation_id"] == "op-1"
        assert operation["idempotency_key"] == "k1"

    async def scope_isolation(adapter: SkillGraphStorageAdapter) -> None:
        a = SkillManageScope(project_id="p1", graph_id="g1")
        b = SkillManageScope(project_id="p1", graph_id="g2")
        await adapter.create_head_if_absent(a, {"active_revision": 0})
        assert await adapter.get_graph_head(b) is None

    async def cross_project_same_graph_name_is_isolated(adapter: SkillGraphStorageAdapter) -> None:
        a = SkillManageScope(project_id="proj-a", graph_id="same")
        b = SkillManageScope(project_id="proj-b", graph_id="same")
        await adapter.create_head_if_absent(a, {"active_revision": 0})
        head_a = await adapter.get_graph_head(a)
        head_b = await adapter.get_graph_head(b)
        assert head_a is not None and head_a["active_revision"] == 0
        assert head_b is None
        # Writes into A must not leak into B.
        op_a = await adapter.reserve_operation(
            a, "op-a", idempotency_key="ka", request_hash="ha", operation_type="build"
        )
        assert op_a is True
        assert await adapter.get_operation(b, "op-a") is None

    async def same_logical_node_exists_in_two_generations(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p7", graph_id="g7")
        node = SkillGraphNode(graph_node_id="n-1", node_type="skill", skill_id="s1", version_id="v1", name="One")
        await _stage_ready(
            adapter, scope, "gen-a", operation_id="op-a", base_revision=None, target_revision=0, nodes=[node]
        )
        await _stage_ready(
            adapter, scope, "gen-b", operation_id="op-b", base_revision=None, target_revision=0, nodes=[node]
        )
        snap_a = await adapter.load_snapshot(scope, "gen-a")
        snap_b = await adapter.load_snapshot(scope, "gen-b")
        assert [n.graph_node_id for n in snap_a.nodes] == ["n-1"]
        assert [n.graph_node_id for n in snap_b.nodes] == ["n-1"]
        # Publishing generation A must leave generation B intact.
        assert await adapter.publish_generation(scope, "gen-a", expected_revision=None, new_revision=0, now=T0)
        snap_b_after = await adapter.load_snapshot(scope, "gen-b")
        assert [n.graph_node_id for n in snap_b_after.nodes] == ["n-1"]

    async def snapshot_load_roundtrip(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p1", graph_id="g1")
        snapshot = await adapter.load_snapshot(scope, "gen-1")
        assert snapshot.generation_id == "gen-1"

    async def snapshot_write_publish_load(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p2", graph_id="g2")
        await adapter.reserve_operation(
            scope, "op-rt", idempotency_key="k-rt", request_hash="h-rt", operation_type="build"
        )
        await _stage_ready(
            adapter,
            scope,
            "gen-rt",
            operation_id="op-rt",
            base_revision=None,
            target_revision=0,
            nodes=[
                SkillGraphNode(graph_node_id="n-1", node_type="skill", skill_id="s1", version_id="v1", name="One"),
                SkillGraphNode(graph_node_id="n-2", node_type="skill", skill_id="s2", version_id="v1", name="Two"),
            ],
            edges=[
                SkillGraphEdge(edge_id="e-1", edge_type="dependency", source_id="n-1", target_id="n-2", weight=1.0),
            ],
            memberships=[{"skill_id": "s1", "version_id": "v1", "graph_node_id": "n-1"}],
        )
        assert await adapter.publish_generation(
            scope,
            "gen-rt",
            expected_revision=None,
            new_revision=0,
            algorithm_name="skill_manage",
            algorithm_version="1",
        )
        snapshot = await adapter.load_snapshot(scope, "gen-rt")
        # Record order is the backend's own (PG returns physical id order); the port promises the set.
        assert sorted(node.graph_node_id for node in snapshot.nodes) == ["n-1", "n-2"]
        assert [edge.edge_id for edge in snapshot.edges] == ["e-1"]
        assert [(m.skill_id, m.version_id, m.graph_node_id) for m in snapshot.memberships] == [("s1", "v1", "n-1")]
        head = await adapter.get_graph_head(scope)
        assert head is not None
        assert (head["algorithm_name"], head["algorithm_version"]) == ("skill_manage", "1")

    async def generation_lifecycle_and_retirement(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p10", graph_id="g10")
        await adapter.stage_generation(
            scope, "gen-1", operation_id="op-1", attempt_no=1, base_revision=None, target_revision=0, now=T0
        )
        staged = await adapter.get_generation(scope, "gen-1")
        assert staged is not None and staged["status"] == "staging"
        # A second stage of the same generation is a consistency violation, not an upsert.
        with pytest.raises(SkillManageConsistencyError):
            await adapter.stage_generation(
                scope, "gen-1", operation_id="op-1", attempt_no=1, base_revision=None, target_revision=0, now=T0
            )
        # Publishing a generation that never reached ready is rejected without touching the head.
        with pytest.raises(SkillManageConsistencyError):
            await adapter.publish_generation(scope, "gen-1", expected_revision=None, new_revision=0, now=T0)
        assert await adapter.get_graph_head(scope) is None
        await adapter.write_snapshot(scope, "gen-1", nodes=[], edges=[], memberships=[])
        assert await adapter.mark_generation_ready(
            scope,
            "gen-1",
            node_count=0,
            edge_count=0,
            vector_count=0,
            counts_hash="h1",
            now="2026-09-03T00:00:01+00:00",
        )
        assert not await adapter.mark_generation_ready(
            scope,
            "gen-1",
            node_count=0,
            edge_count=0,
            vector_count=0,
            counts_hash="h1",
            now="2026-09-03T00:00:01+00:00",
        )
        ready = await adapter.get_generation(scope, "gen-1")
        assert ready is not None and ready["status"] == "ready" and ready["counts_hash"] == "h1"
        assert ready["ready_at"] is not None
        assert await adapter.publish_generation(
            scope, "gen-1", expected_revision=None, new_revision=0, now="2026-09-03T00:00:02+00:00"
        )
        active = await adapter.get_generation(scope, "gen-1")
        assert active is not None and active["status"] == "active" and active["activated_at"] is not None

        await _stage_ready(adapter, scope, "gen-2", operation_id="op-2", base_revision=0, target_revision=1)
        assert await adapter.publish_generation(
            scope, "gen-2", expected_revision=0, new_revision=1, now="2026-09-03T00:00:03+00:00"
        )
        old = await adapter.get_generation(scope, "gen-1")
        new = await adapter.get_generation(scope, "gen-2")
        assert old is not None and old["status"] == "retired" and old["retired_at"] is not None
        assert new is not None and new["status"] == "active"
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_generation_id"] == "gen-2" and head["active_revision"] == 1
        # A retired generation stays readable until GC.
        assert (await adapter.load_snapshot(scope, "gen-1")).generation_id == "gen-1"

    async def writer_lease_and_fencing_takeover(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p3", graph_id="g3")
        assert await adapter.get_writer_lease(scope) is None
        token1 = await adapter.acquire_writer_lease(scope, owner_operation_id="op-a", ttl_seconds=60.0, now=T0)
        assert token1 >= 1
        lease = await adapter.get_writer_lease(scope)
        assert lease is not None
        assert lease["owner_operation_id"] == "op-a" and lease["fencing_token"] == token1
        # The port promises an aware instant, not a UTC rendering: a backend
        # whose session zone is not UTC returns the same instant with another offset.
        assert datetime.fromisoformat(lease["expires_at"]) == datetime.fromisoformat("2026-09-03T00:01:00+00:00")
        # A live lease cannot be acquired by another writer.
        with pytest.raises(SkillManageWriterConflictError) as excinfo:
            await adapter.acquire_writer_lease(
                scope, owner_operation_id="op-b", ttl_seconds=60.0, now="2026-09-03T00:00:10+00:00"
            )
        assert excinfo.value.error_code == "SKILL_MANAGE_WRITER_CONFLICT"
        assert excinfo.value.owner_operation_id == "op-a"
        assert await adapter.renew_writer_lease(
            scope, owner_operation_id="op-a", token=token1, ttl_seconds=60.0, now="2026-09-03T00:00:30+00:00"
        )
        # Non-owner renewal must fail.
        assert not await adapter.renew_writer_lease(
            scope, owner_operation_id="op-b", token=token1, ttl_seconds=60.0, now="2026-09-03T00:00:40+00:00"
        )
        # Expired takeover yields a strictly greater token. Renewal at t0+30s
        # with TTL 60 expires at t0+90s, so take over at t0+95s.
        token2 = await adapter.acquire_writer_lease(
            scope, owner_operation_id="op-b", ttl_seconds=60.0, now="2026-09-03T00:01:35+00:00"
        )
        assert token2 > token1
        # Stale owner (op-a, token1) cannot renew or release now.
        assert not await adapter.renew_writer_lease(
            scope, owner_operation_id="op-a", token=token1, ttl_seconds=60.0, now="2026-09-03T00:01:40+00:00"
        )
        assert not await adapter.release_writer_lease(
            scope, owner_operation_id="op-a", token=token1, now="2026-09-03T00:01:41+00:00"
        )
        assert await adapter.release_writer_lease(
            scope, owner_operation_id="op-b", token=token2, now="2026-09-03T00:02:00+00:00"
        )
        released = await adapter.get_writer_lease(scope)
        assert released is not None and released["owner_operation_id"] is None
        # A released lease is immediately acquirable, still with a greater token.
        token3 = await adapter.acquire_writer_lease(
            scope, owner_operation_id="op-c", ttl_seconds=60.0, now="2026-09-03T00:02:01+00:00"
        )
        assert token3 > token2
        assert await adapter.release_writer_lease(
            scope, owner_operation_id="op-c", token=token3, now="2026-09-03T00:02:02+00:00"
        )

    async def publish_rejects_stale_or_missing_fencing(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p11", graph_id="g11")
        token_a = await adapter.acquire_writer_lease(scope, owner_operation_id="op-a", ttl_seconds=60.0, now=T0)
        await _stage_ready(adapter, scope, "gen-a", operation_id="op-a", base_revision=None, target_revision=0)
        # Wrong token, wrong owner and an expired lease are all fenced out.
        for owner, token, now in (
            ("op-a", token_a + 1, "2026-09-03T00:00:01+00:00"),
            ("op-x", token_a, "2026-09-03T00:00:01+00:00"),
            ("op-a", token_a, "2026-09-03T00:01:00+00:00"),
        ):
            with pytest.raises(SkillManageWriterConflictError):
                await adapter.publish_generation(
                    scope,
                    "gen-a",
                    expected_revision=None,
                    new_revision=0,
                    owner_operation_id=owner,
                    fencing_token=token,
                    now=now,
                )
            assert await adapter.get_graph_head(scope) is None
            staged = await adapter.get_generation(scope, "gen-a")
            assert staged is not None and staged["status"] == "ready"
        # The live owner publishes; a stale writer whose lease was taken over cannot.
        assert await adapter.publish_generation(
            scope,
            "gen-a",
            expected_revision=None,
            new_revision=0,
            owner_operation_id="op-a",
            fencing_token=token_a,
            now="2026-09-03T00:00:02+00:00",
        )
        token_b = await adapter.acquire_writer_lease(
            scope, owner_operation_id="op-b", ttl_seconds=60.0, now="2026-09-03T00:02:00+00:00"
        )
        await _stage_ready(
            adapter, scope, "gen-stale", operation_id="op-a", attempt_no=2, base_revision=0, target_revision=1
        )
        with pytest.raises(SkillManageWriterConflictError):
            await adapter.publish_generation(
                scope,
                "gen-stale",
                expected_revision=0,
                new_revision=1,
                owner_operation_id="op-a",
                fencing_token=token_a,
                now="2026-09-03T00:02:01+00:00",
            )
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_generation_id"] == "gen-a" and head["active_revision"] == 0
        await _stage_ready(adapter, scope, "gen-b", operation_id="op-b", base_revision=0, target_revision=1)
        assert await adapter.publish_generation(
            scope,
            "gen-b",
            expected_revision=0,
            new_revision=1,
            owner_operation_id="op-b",
            fencing_token=token_b,
            now="2026-09-03T00:02:02+00:00",
        )
        assert await adapter.release_writer_lease(
            scope, owner_operation_id="op-b", token=token_b, now="2026-09-03T00:02:03+00:00"
        )

    async def concurrent_create_head_exactly_once(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p5", graph_id="g5")
        results = await asyncio.gather(
            *[
                adapter.create_head_if_absent(scope, {"active_revision": 0, "active_generation_id": f"g-{i}"})
                for i in range(4)
            ]
        )
        assert results.count(True) == 1
        head = await adapter.get_graph_head(scope)
        assert head is not None
        assert head["active_revision"] == 0

    async def concurrent_head_cas_only_one_succeeds(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p6", graph_id="g6")
        await _stage_ready(adapter, scope, "g0", operation_id="op-0", base_revision=None, target_revision=0)
        assert await adapter.publish_generation(scope, "g0", expected_revision=None, new_revision=0, now=T0)
        for i in range(4):
            await _stage_ready(adapter, scope, f"g-{i}", operation_id=f"op-{i}", base_revision=0, target_revision=1)
        results = await asyncio.gather(
            *[adapter.publish_generation(scope, f"g-{i}", expected_revision=0, new_revision=1) for i in range(4)]
        )
        assert results.count(True) == 1
        head = await adapter.get_graph_head(scope)
        assert head is not None
        assert head["active_revision"] == 1
        statuses = {}
        for generation_id in ("g0", "g-0", "g-1", "g-2", "g-3"):
            generation = await adapter.get_generation(scope, generation_id)
            assert generation is not None
            statuses[generation_id] = generation["status"]
        assert statuses["g0"] == "retired"
        assert statuses[head["active_generation_id"]] == "active"
        assert sorted(statuses.values()) == ["active", "ready", "ready", "ready", "retired"]

    async def operation_state_machine_transitions(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p8", graph_id="g8")
        assert await adapter.reserve_operation(
            scope, "op-sm", idempotency_key="k-sm", request_hash="h-sm", operation_type="build"
        )
        # pending -> running (initial attempt)
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-sm", generation_id="gen-1", fencing_token=1, base_revision=None, now=T0
        )
        op = await adapter.get_operation(scope, "op-sm")
        assert op is not None and op["status"] == "running" and op["current_generation_id"] == "gen-1"
        assert op["writer_fencing_token"] == 1 and op["first_started_at"] is not None
        # A second begin from pending must fail (already running).
        assert not await adapter.begin_initial_attempt_atomically(
            scope, "op-sm", generation_id="gen-2", fencing_token=2, now="2026-09-03T00:00:01+00:00"
        )
        # running -> validating -> committed with the published outcome
        assert not await adapter.record_operation_result(
            scope, "op-sm", base_revision=None, result_revision=0, generation_id="gen-1"
        )
        assert await adapter.begin_validation(scope, "op-sm", now="2026-09-03T00:00:02+00:00")
        op = await adapter.get_operation(scope, "op-sm")
        assert op is not None and op["status"] == "validating" and op["validating_at"] is not None
        assert await adapter.record_operation_result(
            scope,
            "op-sm",
            base_revision=None,
            result_revision=0,
            generation_id="gen-1",
            result_payload={"node_count": 2},
            now="2026-09-03T00:00:03+00:00",
        )
        op = await adapter.get_operation(scope, "op-sm")
        assert op is not None and op["status"] == "committed"
        assert op["result_revision"] == 0 and op["current_generation_id"] == "gen-1"
        assert op["result_payload"] == {"node_count": 2} and op["completed_at"] is not None
        # Terminal committed -> no further transitions allowed by the machine.
        assert not await adapter.transition_operation(
            scope,
            "op-sm",
            expected_status=["committed"],
            changes={"status": "failed"},
            now="2026-09-03T00:00:04+00:00",
        )
        assert not await adapter.fail_operation(
            scope, "op-sm", error_code="X", error_message="x", error_retryable=False
        )

    async def operation_failure_and_conflict_outcomes(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p12", graph_id="g12")
        # A reserved-but-unstarted operation can still fail (pending -> failed).
        assert await adapter.reserve_operation(
            scope, "op-f0", idempotency_key="k-f0", request_hash="h-f0", operation_type="build"
        )
        assert await adapter.fail_operation(
            scope, "op-f0", error_code="SKILL_MANAGE_INVALID_REQUEST", error_message="x", error_retryable=False
        )
        # running -> failed carries the typed code and retryable flag.
        assert await adapter.reserve_operation(
            scope, "op-f1", idempotency_key="k-f1", request_hash="h-f1", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-f1", generation_id="gen-f1", fencing_token=1)
        assert await adapter.fail_operation(
            scope,
            "op-f1",
            error_code="SKILL_MANAGE_EMBEDDING_PROVIDER_ERROR",
            error_message="SkillManageError",
            error_retryable=True,
            now="2026-09-03T00:00:01+00:00",
        )
        op = await adapter.get_operation(scope, "op-f1")
        assert op is not None and op["status"] == "failed" and op["error_retryable"] is True
        assert op["completed_at"] is not None
        # pending -> conflict is not a legal edge; validating -> conflict is.
        assert await adapter.reserve_operation(
            scope, "op-c1", idempotency_key="k-c1", request_hash="h-c1", operation_type="build"
        )
        assert not await adapter.conflict_operation(
            scope, "op-c1", error_code="SKILL_MANAGE_REVISION_CONFLICT", error_message="x"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-c1", generation_id="gen-c1", fencing_token=1)
        assert await adapter.begin_validation(scope, "op-c1")
        assert await adapter.conflict_operation(
            scope, "op-c1", error_code="SKILL_MANAGE_REVISION_CONFLICT", error_message="head moved"
        )
        op = await adapter.get_operation(scope, "op-c1")
        assert op is not None and op["status"] == "conflict"
        assert op["error_code"] == "SKILL_MANAGE_REVISION_CONFLICT" and op["error_retryable"] is False
        # running -> cancelled is terminal: no retry reservation, no further failure.
        assert await adapter.reserve_operation(
            scope, "op-x1", idempotency_key="k-x1", request_hash="h-x1", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-x1", generation_id="gen-x1", fencing_token=1)
        assert await adapter.cancel_operation(scope, "op-x1", error_code="SKILL_MANAGE_OPERATION_CANCELLED")
        op = await adapter.get_operation(scope, "op-x1")
        assert op is not None and op["status"] == "cancelled" and op["completed_at"] is not None
        assert not await adapter.reserve_retry_attempt(scope, "op-x1")
        assert not await adapter.fail_operation(scope, "op-x1", error_code="X", error_message="x", error_retryable=True)
        # Attempt-side transitions are fenced by the token the attempt began with.
        assert await adapter.reserve_operation(
            scope, "op-fn", idempotency_key="k-fn", request_hash="h-fn", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-fn", generation_id="gen-fn", fencing_token=7)
        assert not await adapter.begin_validation(scope, "op-fn", fencing_token=6)
        assert not await adapter.fail_operation(
            scope, "op-fn", error_code="X", error_message="x", error_retryable=True, fencing_token=6
        )
        assert not await adapter.cancel_operation(scope, "op-fn", error_code="X", fencing_token=6)
        assert await adapter.begin_validation(scope, "op-fn", fencing_token=7)
        assert not await adapter.conflict_operation(scope, "op-fn", error_code="X", error_message="x", fencing_token=6)
        assert not await adapter.record_operation_result(
            scope, "op-fn", base_revision=None, result_revision=0, generation_id="gen-fn", fencing_token=6
        )
        assert await adapter.record_operation_result(
            scope, "op-fn", base_revision=None, result_revision=0, generation_id="gen-fn", fencing_token=7
        )
        op = await adapter.get_operation(scope, "op-fn")
        assert op is not None and op["status"] == "committed"

    async def operation_retry_reservation_and_crash_recovery(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p9", graph_id="g9")
        assert await adapter.reserve_operation(
            scope, "op-rt2", idempotency_key="k-rt2", request_hash="h-rt2", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-rt2", generation_id="gen-a", fencing_token=1, now=T0
        )
        # A stranded running operation with an expired lease -> crash recovery
        # claim, which must name exactly the attempt it observed.
        assert not await adapter.claim_crash_recovery(
            scope, "op-rt2", expected_attempt_no=2, expected_generation_id="gen-a", now="2026-09-03T00:05:00+00:00"
        )
        assert not await adapter.claim_crash_recovery(
            scope, "op-rt2", expected_attempt_no=1, expected_generation_id="gen-z", now="2026-09-03T00:05:00+00:00"
        )
        assert await adapter.claim_crash_recovery(
            scope,
            "op-rt2",
            expected_attempt_no=1,
            expected_generation_id="gen-a",
            now="2026-09-03T00:05:00+00:00",
        )
        op = await adapter.get_operation(scope, "op-rt2")
        assert op is not None and op["status"] == "pending_repair"
        assert op["pending_repair_reason"] == "crash_recovery"
        # Retry reservation: pending_repair -> new running attempt with a new generation.
        assert await adapter.begin_retry_attempt_atomically(
            scope,
            "op-rt2",
            attempt_no=2,
            generation_id="gen-b",
            fencing_token=2,
            now="2026-09-03T00:05:01+00:00",
        )
        op = await adapter.get_operation(scope, "op-rt2")
        assert op is not None and op["status"] == "running" and op["attempt_no"] == 2
        assert op["current_generation_id"] == "gen-b" and op["pending_repair_reason"] is None

        # failed(retryable) -> pending_repair(retry_attempt_reservation) -> running attempt 3.
        assert await adapter.fail_operation(
            scope, "op-rt2", error_code="SKILL_MANAGE_STORAGE_UNAVAILABLE", error_message="x", error_retryable=True
        )
        assert await adapter.reserve_retry_attempt(scope, "op-rt2", now="2026-09-03T00:06:00+00:00")
        op = await adapter.get_operation(scope, "op-rt2")
        assert op is not None and op["status"] == "pending_repair"
        assert op["pending_repair_reason"] == "retry_attempt_reservation"
        # Only one reservation wins; the state is no longer failed.
        assert not await adapter.reserve_retry_attempt(scope, "op-rt2", now="2026-09-03T00:06:01+00:00")
        assert await adapter.begin_retry_attempt_atomically(
            scope, "op-rt2", attempt_no=3, generation_id="gen-c", fencing_token=3, now="2026-09-03T00:06:02+00:00"
        )
        op = await adapter.get_operation(scope, "op-rt2")
        assert op is not None and op["status"] == "running" and op["attempt_no"] == 3
        assert op.get("error_code") is None and op.get("completed_at") is None
        # A non-retryable failure cannot reserve a retry.
        assert await adapter.fail_operation(
            scope, "op-rt2", error_code="SKILL_MANAGE_VALIDATION_FAILED", error_message="x", error_retryable=False
        )
        assert not await adapter.reserve_retry_attempt(scope, "op-rt2")

    async def retry_begin_requires_the_row_to_still_be_at_the_previous_attempt(
        adapter: SkillGraphStorageAdapter,
    ) -> None:
        """A caller that computed ``attempt_no`` before stalling cannot re-open a number already used."""

        scope = SkillManageScope(project_id="p21", graph_id="g21")
        assert await adapter.reserve_operation(
            scope, "op-a", idempotency_key="k-a", request_hash="h-a", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-a", generation_id="gen-1", fencing_token=1)
        assert await adapter.fail_operation(
            scope, "op-a", error_code="SKILL_MANAGE_STORAGE_UNAVAILABLE", error_message="x", error_retryable=True
        )
        # Another same-key caller already ran attempt 2 and failed it retryably.
        assert await adapter.reserve_retry_attempt(scope, "op-a")
        assert await adapter.begin_retry_attempt_atomically(
            scope, "op-a", attempt_no=2, generation_id="gen-2", fencing_token=2
        )
        assert await adapter.fail_operation(
            scope,
            "op-a",
            error_code="SKILL_MANAGE_STORAGE_UNAVAILABLE",
            error_message="x",
            error_retryable=True,
            fencing_token=2,
        )
        # The stalled caller, still believing the row is at attempt 1, reserves
        # (the reservation itself is harmless) but must not re-open attempt 2.
        assert await adapter.reserve_retry_attempt(scope, "op-a")
        assert not await adapter.begin_retry_attempt_atomically(
            scope, "op-a", attempt_no=2, generation_id="gen-2", fencing_token=3
        )
        op = await adapter.get_operation(scope, "op-a")
        assert op is not None and op["status"] == "pending_repair" and op["attempt_no"] == 2
        assert op["pending_repair_reason"] == "retry_attempt_reservation"
        # The reservation is resumed with the number derived from the current row.
        assert await adapter.begin_retry_attempt_atomically(
            scope, "op-a", attempt_no=3, generation_id="gen-3", fencing_token=3
        )
        op = await adapter.get_operation(scope, "op-a")
        assert op is not None and op["status"] == "running" and op["attempt_no"] == 3
        assert op["current_generation_id"] == "gen-3" and op["writer_fencing_token"] == 3

    async def crash_recovery_repair_branches(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p13", graph_id="g13")
        # Graph absent: the stranded attempt fails retryable and its candidate is discarded.
        assert await adapter.reserve_operation(
            scope, "op-x", idempotency_key="k-x", request_hash="h-x", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-x", generation_id="gen-x", fencing_token=1)
        await adapter.stage_generation(
            scope, "gen-x", operation_id="op-x", attempt_no=1, base_revision=None, target_revision=0
        )
        assert await adapter.claim_crash_recovery(scope, "op-x", expected_attempt_no=1, expected_generation_id="gen-x")
        assert await adapter.repair_crash_recovery(scope, "op-x", attempt_no=1, generation_id="gen-x") == "failed"
        op = await adapter.get_operation(scope, "op-x")
        assert op is not None and op["status"] == "failed" and op["error_retryable"] is True
        assert await adapter.get_generation(scope, "gen-x") is None
        assert await adapter.repair_crash_recovery(scope, "op-x", attempt_no=1, generation_id="gen-x") == "failed"

        # Head points at the candidate: converge to committed.
        assert await adapter.reserve_operation(
            scope, "op-y", idempotency_key="k-y", request_hash="h-y", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-y", generation_id="gen-y", fencing_token=2)
        await _stage_ready(adapter, scope, "gen-y", operation_id="op-y", base_revision=None, target_revision=0)
        assert await adapter.publish_generation(scope, "gen-y", expected_revision=None, new_revision=0, now=T0)
        assert await adapter.begin_validation(scope, "op-y")
        assert await adapter.claim_crash_recovery(scope, "op-y", expected_attempt_no=1, expected_generation_id="gen-y")
        assert await adapter.repair_crash_recovery(scope, "op-y", attempt_no=1, generation_id="gen-y") == "committed"
        op = await adapter.get_operation(scope, "op-y")
        assert op is not None and op["status"] == "committed" and op["result_revision"] == 0

        # Head unchanged from the attempt's base: failed retryable.
        assert await adapter.reserve_operation(
            scope, "op-z", idempotency_key="k-z", request_hash="h-z", operation_type="upsert"
        )
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-z", generation_id="gen-z", fencing_token=3, base_revision=0
        )
        await adapter.stage_generation(
            scope, "gen-z", operation_id="op-z", attempt_no=1, base_revision=0, target_revision=1
        )
        assert await adapter.claim_crash_recovery(scope, "op-z", expected_attempt_no=1, expected_generation_id="gen-z")
        assert await adapter.repair_crash_recovery(scope, "op-z", attempt_no=1, generation_id="gen-z") == "failed"
        assert await adapter.get_generation(scope, "gen-z") is None

        # Head moved past the attempt's base: conflict.
        assert await adapter.reserve_operation(
            scope, "op-w", idempotency_key="k-w", request_hash="h-w", operation_type="upsert"
        )
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-w", generation_id="gen-w", fencing_token=4, base_revision=0
        )
        await adapter.stage_generation(
            scope, "gen-w", operation_id="op-w", attempt_no=1, base_revision=0, target_revision=1
        )
        await _stage_ready(adapter, scope, "gen-other", operation_id="op-other", base_revision=0, target_revision=1)
        assert await adapter.publish_generation(scope, "gen-other", expected_revision=0, new_revision=1, now=T0)
        assert await adapter.claim_crash_recovery(scope, "op-w", expected_attempt_no=1, expected_generation_id="gen-w")
        assert await adapter.repair_crash_recovery(scope, "op-w", attempt_no=1, generation_id="gen-w") == "conflict"
        op = await adapter.get_operation(scope, "op-w")
        assert op is not None and op["status"] == "conflict" and op["error_code"] == "SKILL_MANAGE_REVISION_CONFLICT"
        assert await adapter.get_generation(scope, "gen-w") is None
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_generation_id"] == "gen-other"

        # A candidate that was published and later retired by another writer
        # still converges to committed (with its counts) and is never discarded.
        assert await adapter.reserve_operation(
            scope, "op-v", idempotency_key="k-v", request_hash="h-v", operation_type="upsert"
        )
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-v", generation_id="gen-v", fencing_token=5, base_revision=1
        )
        await _stage_ready(adapter, scope, "gen-v", operation_id="op-v", base_revision=1, target_revision=2)
        assert await adapter.begin_validation(scope, "op-v")
        assert await adapter.publish_generation(scope, "gen-v", expected_revision=1, new_revision=2, now=T0)
        await _stage_ready(adapter, scope, "gen-after", operation_id="op-after", base_revision=2, target_revision=3)
        assert await adapter.publish_generation(scope, "gen-after", expected_revision=2, new_revision=3, now=T0)
        retired = await adapter.get_generation(scope, "gen-v")
        assert retired is not None and retired["status"] == "retired"
        assert await adapter.claim_crash_recovery(scope, "op-v", expected_attempt_no=1, expected_generation_id="gen-v")
        assert await adapter.repair_crash_recovery(scope, "op-v", attempt_no=1, generation_id="gen-v") == "committed"
        op = await adapter.get_operation(scope, "op-v")
        assert op is not None and op["status"] == "committed" and op["result_revision"] == 2
        assert op["result_payload"] == {"node_count": 0, "edge_count": 0, "vector_count": 0}
        assert (await adapter.get_generation(scope, "gen-v"))["status"] == "retired"

    async def gc_respects_keep_and_min_age(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p14", graph_id="g14")
        stamps = [f"2026-09-03T00:0{i}:00+00:00" for i in range(5)]
        for index, stamp in enumerate(stamps):
            await _stage_ready(
                adapter,
                scope,
                f"gen-{index}",
                operation_id=f"op-{index}",
                base_revision=index - 1 if index else None,
                target_revision=index,
                now=stamp,
            )
            assert await adapter.publish_generation(
                scope, f"gen-{index}", expected_revision=index - 1 if index else None, new_revision=index, now=stamp
            )
        statuses = {}
        for index in range(5):
            generation = await adapter.get_generation(scope, f"gen-{index}")
            assert generation is not None
            statuses[f"gen-{index}"] = generation["status"]
        assert statuses == {
            "gen-0": "retired",
            "gen-1": "retired",
            "gen-2": "retired",
            "gen-3": "retired",
            "gen-4": "active",
        }
        # Nothing is old enough yet: min age wins over keep.
        assert (
            await adapter.gc_retired_generations(scope, keep=2, min_age_seconds=3600, now="2026-09-03T00:10:00+00:00")
            == []
        )
        # A keep larger than the retired count protects everything.
        assert await adapter.gc_retired_generations(scope, keep=5, now="2026-09-03T00:10:00+00:00") == []
        assert await adapter.gc_retired_generations(scope, keep=6, now="2026-09-03T00:10:00+00:00") == []
        for index in range(4):
            assert await adapter.get_generation(scope, f"gen-{index}") is not None
        # Beyond keep=2 (gen-0, gen-1) and older than 5 minutes at 00:07 -> both go, oldest first.
        assert await adapter.gc_retired_generations(
            scope, keep=2, min_age_seconds=300, now="2026-09-03T00:07:00+00:00"
        ) == ["gen-0", "gen-1"]
        assert await adapter.get_generation(scope, "gen-0") is None
        assert await adapter.get_generation(scope, "gen-1") is None
        # keep=0 with no age floor collects every retired generation but never the active one.
        assert await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T00:10:00+00:00") == [
            "gen-2",
            "gen-3",
        ]
        assert await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T00:10:00+00:00") == []
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_generation_id"] == "gen-4"
        active = await adapter.get_generation(scope, "gen-4")
        assert active is not None and active["status"] == "active"

    async def single_active_generation_invariant(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p4", graph_id="g4")
        await adapter.reserve_operation(scope, "op-a", idempotency_key="ka", request_hash="ha", operation_type="build")
        await _stage_ready(adapter, scope, "g-a", operation_id="op-a", base_revision=None, target_revision=0)
        assert await adapter.publish_generation(scope, "g-a", expected_revision=None, new_revision=0)
        # A second publish with a wrong expected revision must not create a second active generation.
        await adapter.reserve_operation(scope, "op-b", idempotency_key="kb", request_hash="hb", operation_type="build")
        await _stage_ready(adapter, scope, "g-b", operation_id="op-b", base_revision=0, target_revision=1)
        assert not await adapter.publish_generation(scope, "g-b", expected_revision=5, new_revision=1)
        loser = await adapter.get_generation(scope, "g-b")
        assert loser is not None and loser["status"] == "ready"
        winner = await adapter.get_generation(scope, "g-a")
        assert winner is not None and winner["status"] == "active"

    async def skewed_clock_takeover_is_safe_and_fences_the_old_writer(adapter: SkillGraphStorageAdapter) -> None:
        """Clock skew can only make a takeover *early*, never unsafe.

        A writer whose clock runs ahead of everyone else's may judge a live
        lease expired and take it over. That is admissible because: the token
        is strictly monotonic and CAS'd, so the old writer can never renew,
        release or publish once it happens; and a writer whose clock runs
        behind can never take over a lease that is genuinely still held. The
        invariant that holds regardless of skew is: acquire succeeds only when
        the row has no owner or its ``expires_at`` is at or before the
        acquirer's own ``now``.
        """

        scope = SkillManageScope(project_id="p16", graph_id="g16")
        await _stage_ready(adapter, scope, "base", operation_id="op-base", base_revision=None, target_revision=0)
        assert await adapter.publish_generation(scope, "base", expected_revision=None, new_revision=0, now=T0)
        # Writer A holds the lease until T0+60s by its own clock.
        token_a = await adapter.acquire_writer_lease(scope, owner_operation_id="op-a", ttl_seconds=60.0, now=T0)
        await _stage_ready(adapter, scope, "gen-a", operation_id="op-a", base_revision=0, target_revision=1)
        await _stage_ready(adapter, scope, "gen-b", operation_id="op-b", base_revision=0, target_revision=1)

        # A slow-clocked writer B (its now is *before* the lease expiry) must
        # not take over: the lease is genuinely held.
        with pytest.raises(SkillManageWriterConflictError):
            await adapter.acquire_writer_lease(
                scope, owner_operation_id="op-b", ttl_seconds=60.0, now="2026-09-03T00:00:30+00:00"
            )
        # A fast-clocked writer B (its now is past the expiry A wrote) takes
        # over early with a strictly greater token.
        token_b = await adapter.acquire_writer_lease(
            scope, owner_operation_id="op-b", ttl_seconds=60.0, now="2026-09-03T00:05:00+00:00"
        )
        assert token_b > token_a

        # A is still running by its own (slow) clock and believes its lease is
        # live, but its token is superseded: every fenced step is rejected.
        assert not await adapter.renew_writer_lease(
            scope, owner_operation_id="op-a", token=token_a, ttl_seconds=60.0, now="2026-09-03T00:00:31+00:00"
        )
        with pytest.raises(SkillManageWriterConflictError):
            await adapter.publish_generation(
                scope,
                "gen-a",
                expected_revision=0,
                new_revision=1,
                owner_operation_id="op-a",
                fencing_token=token_a,
                now="2026-09-03T00:00:32+00:00",
            )
        # The head and A's candidate are untouched by the rejection.
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_revision"] == 0 and head["active_generation_id"] == "base"
        assert (await adapter.get_generation(scope, "gen-a"))["status"] == "ready"
        # B, the current holder, publishes.
        assert await adapter.publish_generation(
            scope,
            "gen-b",
            expected_revision=0,
            new_revision=1,
            owner_operation_id="op-b",
            fencing_token=token_b,
            now="2026-09-03T00:05:01+00:00",
        )
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_revision"] == 1 and head["active_generation_id"] == "gen-b"
        assert await adapter.release_writer_lease(
            scope, owner_operation_id="op-b", token=token_b, now="2026-09-03T00:05:02+00:00"
        )

    async def gc_keeps_generations_referenced_by_non_terminal_operations(adapter: SkillGraphStorageAdapter) -> None:
        """A published-then-retired candidate of a stranded ``validating`` row must survive GC.

        Otherwise ``repair_crash_recovery`` can no longer see that the publish
        landed and would converge the row to ``conflict`` instead of
        ``committed``.
        """

        scope = SkillManageScope(project_id="p17", graph_id="g17")
        stamps = [f"2026-09-03T00:0{i}:00+00:00" for i in range(4)]
        for index, stamp in enumerate(stamps):
            assert await adapter.reserve_operation(
                scope, f"op-{index}", idempotency_key=f"k-{index}", request_hash=f"h-{index}", operation_type="build"
            )
            await _stage_ready(
                adapter,
                scope,
                f"gen-{index}",
                operation_id=f"op-{index}",
                base_revision=index - 1 if index else None,
                target_revision=index,
                now=stamp,
            )
            assert await adapter.publish_generation(
                scope, f"gen-{index}", expected_revision=index - 1 if index else None, new_revision=index, now=stamp
            )
        # op-1 published gen-1 but died before finalize: it is still validating.
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-1", generation_id="gen-1", fencing_token=1, base_revision=0, now=stamps[1]
        )
        assert await adapter.begin_validation(scope, "op-1", fencing_token=1, now=stamps[1])
        assert (await adapter.get_generation(scope, "gen-1"))["status"] == "retired"

        # keep=0, no age floor: gen-0 and gen-2 go, gen-1 is protected by op-1.
        assert await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T01:00:00+00:00") == [
            "gen-0",
            "gen-2",
        ]
        assert (await adapter.get_generation(scope, "gen-1"))["status"] == "retired"
        # Repair can still prove the publish landed.
        assert await adapter.claim_crash_recovery(
            scope, "op-1", expected_attempt_no=1, expected_generation_id="gen-1", now="2026-09-03T01:00:01+00:00"
        )
        assert (
            await adapter.repair_crash_recovery(
                scope, "op-1", attempt_no=1, generation_id="gen-1", now="2026-09-03T01:00:02+00:00"
            )
            == "committed"
        )
        assert (await adapter.get_operation(scope, "op-1"))["result_revision"] == 1
        # Terminal now: the generation is collectable.
        assert await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T01:00:03+00:00") == ["gen-1"]

    async def list_operations_and_generations_by_status_within_scope(adapter: SkillGraphStorageAdapter) -> None:
        scope = SkillManageScope(project_id="p15", graph_id="g15")
        other = SkillManageScope(project_id="p15", graph_id="g15-other")
        for key in ("a", "b", "c"):
            assert await adapter.reserve_operation(
                scope, f"op-{key}", idempotency_key=f"k-{key}", request_hash=f"h-{key}", operation_type="build"
            )
        assert await adapter.reserve_operation(
            other, "op-other", idempotency_key="k-other", request_hash="h-other", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-b", generation_id="gen-b", fencing_token=1, base_revision=None, now=T0
        )
        assert await adapter.begin_initial_attempt_atomically(
            scope, "op-c", generation_id="gen-c", fencing_token=2, base_revision=None, now=T0
        )
        assert await adapter.begin_validation(scope, "op-c", fencing_token=2, now=T0)

        by_id = {row["operation_id"]: row["status"] for row in await adapter.list_operations(scope)}
        assert by_id == {"op-a": "pending", "op-b": "running", "op-c": "validating"}
        stranded = {
            row["operation_id"] for row in await adapter.list_operations(scope, statuses=("running", "validating"))
        }
        assert stranded == {"op-b", "op-c"}
        assert await adapter.list_operations(scope, statuses=("committed",)) == []
        # Scope isolation: the other graph's row is never listed under this scope.
        assert "op-other" not in by_id

        await _stage_ready(adapter, scope, "gen-b", operation_id="op-b", base_revision=None, target_revision=0)
        await adapter.stage_generation(
            scope, "gen-c", operation_id="op-c", attempt_no=1, base_revision=None, target_revision=0, now=T0
        )
        assert await adapter.publish_generation(scope, "gen-b", expected_revision=None, new_revision=0, now=T0)
        generations = {row["generation_id"]: row["status"] for row in await adapter.list_generations(scope)}
        assert generations == {"gen-b": "active", "gen-c": "staging"}
        assert {
            row["generation_id"] for row in await adapter.list_generations(scope, statuses=("staging", "ready"))
        } == {"gen-c"}
        assert await adapter.list_generations(other) == []

    async def in_doubt_publish_fence_settles_the_candidate(adapter: SkillGraphStorageAdapter) -> None:
        """``mark_generation_failed`` wins only against a never-published candidate.

        The service uses it to settle a publish whose acknowledgement was
        lost, and ``discard_generation``/``repair_crash_recovery`` use it
        before deleting anything: winning proves the publish did not land and
        can no longer land (the candidate is no longer ``ready``); losing
        means the candidate's own status now says whether it was published.
        """

        scope = SkillManageScope(project_id="p18", graph_id="g18")
        await _stage_ready(adapter, scope, "gen-a", operation_id="op-a", base_revision=None, target_revision=0)
        assert await adapter.mark_generation_failed(scope, "gen-a", now="2026-09-03T00:00:01+00:00")
        fenced = await adapter.get_generation(scope, "gen-a")
        assert fenced is not None and fenced["status"] == "failed" and fenced["failed_at"] is not None
        assert not await adapter.mark_generation_failed(scope, "gen-a", now="2026-09-03T00:00:02+00:00")
        # A fenced candidate can never be published afterwards; the head is untouched.
        with pytest.raises(SkillManageConsistencyError):
            await adapter.publish_generation(scope, "gen-a", expected_revision=None, new_revision=0, now=T0)
        assert await adapter.get_graph_head(scope) is None
        await adapter.discard_generation(scope, "gen-a")
        assert await adapter.get_generation(scope, "gen-a") is None
        assert not await adapter.mark_generation_failed(scope, "gen-a")
        # A published candidate (active, then retired) loses the fence, keeps its
        # status and is never discarded.
        await _stage_ready(adapter, scope, "gen-b", operation_id="op-b", base_revision=None, target_revision=0)
        assert await adapter.publish_generation(scope, "gen-b", expected_revision=None, new_revision=0, now=T0)
        assert not await adapter.mark_generation_failed(scope, "gen-b")
        assert (await adapter.get_generation(scope, "gen-b"))["status"] == "active"
        await _stage_ready(adapter, scope, "gen-c", operation_id="op-c", base_revision=0, target_revision=1)
        assert await adapter.publish_generation(scope, "gen-c", expected_revision=0, new_revision=1, now=T0)
        assert not await adapter.mark_generation_failed(scope, "gen-b")
        await adapter.discard_generation(scope, "gen-b")
        assert (await adapter.get_generation(scope, "gen-b"))["status"] == "retired"
        assert (await adapter.load_snapshot(scope, "gen-b")).generation_id == "gen-b"
        # A staging candidate is fenced the same way: it was never publishable.
        await adapter.stage_generation(
            scope, "gen-d", operation_id="op-d", attempt_no=1, base_revision=1, target_revision=2, now=T0
        )
        assert await adapter.mark_generation_failed(scope, "gen-d")
        assert (await adapter.get_generation(scope, "gen-d"))["status"] == "failed"
        await adapter.discard_generation(scope, "gen-d")
        assert await adapter.get_generation(scope, "gen-d") is None

    async def repair_crash_recovery_only_settles_the_attempt_it_was_given(adapter: SkillGraphStorageAdapter) -> None:
        """A reconciler working from a stale read must never terminate a newer live attempt."""

        scope = SkillManageScope(project_id="p19", graph_id="g19")
        assert await adapter.reserve_operation(
            scope, "op-s", idempotency_key="k-s", request_hash="h-s", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-s", generation_id="gen-1", fencing_token=1)
        assert await adapter.claim_crash_recovery(scope, "op-s", expected_attempt_no=1, expected_generation_id="gen-1")
        # A same-key retry already opened attempt 2 under a new token.
        assert await adapter.begin_retry_attempt_atomically(
            scope, "op-s", attempt_no=2, generation_id="gen-2", fencing_token=2, now="2026-09-03T00:05:00+00:00"
        )
        await adapter.stage_generation(
            scope, "gen-2", operation_id="op-s", attempt_no=2, base_revision=None, target_revision=0
        )
        # Reconciling attempt 1 is a no-op that reports the row's actual status.
        assert await adapter.repair_crash_recovery(scope, "op-s", attempt_no=1, generation_id="gen-1") == "running"
        op = await adapter.get_operation(scope, "op-s")
        assert op is not None and op["status"] == "running" and op["attempt_no"] == 2
        assert op["current_generation_id"] == "gen-2" and op.get("repair_count") in (None, 0)
        assert (await adapter.get_generation(scope, "gen-2"))["status"] == "staging"
        # Against a validating live attempt the answer is the same.
        assert await adapter.begin_validation(scope, "op-s", fencing_token=2)
        assert await adapter.repair_crash_recovery(scope, "op-s", attempt_no=1, generation_id="gen-1") == "validating"
        assert (await adapter.get_operation(scope, "op-s"))["status"] == "validating"
        # The attempt that was actually observed still reconciles normally.
        assert await adapter.claim_crash_recovery(scope, "op-s", expected_attempt_no=2, expected_generation_id="gen-2")
        assert await adapter.repair_crash_recovery(scope, "op-s", attempt_no=2, generation_id="gen-2") == "failed"
        op = await adapter.get_operation(scope, "op-s")
        assert op is not None and op["status"] == "failed" and op["error_retryable"] is True
        assert await adapter.get_generation(scope, "gen-2") is None

    async def repair_crash_recovery_does_not_settle_a_row_that_moved_into_a_retry_reservation(
        adapter: SkillGraphStorageAdapter,
    ) -> None:
        """A reservation keeps the attempt fields, so the reason must be part of the settle predicate.

        A reconciler that stalled after reading the row it claimed must lose
        its CAS once a same-key retry has settled that attempt and reserved
        the next one; landing it would turn the resumable reservation into a
        terminal row the same key could never get past.
        """

        scope = SkillManageScope(project_id="p22", graph_id="g22")
        assert await adapter.reserve_operation(
            scope, "op-r", idempotency_key="k-r", request_hash="h-r", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-r", generation_id="gen-1", fencing_token=1)
        assert await adapter.claim_crash_recovery(scope, "op-r", expected_attempt_no=1, expected_generation_id="gen-1")
        original_get_operation = adapter.get_operation
        moved = False

        async def move_after_the_reconcile_reads(scope_, operation_id):
            nonlocal moved
            row = await original_get_operation(scope_, operation_id)
            if not moved and row is not None and row.get("pending_repair_reason") == "crash_recovery":
                # The same-key retry settles attempt 1 and reserves attempt 2
                # between the reconcile's read and its settle CAS.
                moved = True
                assert await adapter.transition_operation(
                    scope_,
                    operation_id,
                    expected_status=["pending_repair"],
                    changes={
                        "status": "failed",
                        "error_code": "SKILL_MANAGE_STORAGE_UNAVAILABLE",
                        "error_retryable": True,
                    },
                )
                assert await adapter.reserve_retry_attempt(scope_, operation_id)
            return row

        adapter.get_operation = move_after_the_reconcile_reads  # type: ignore[method-assign]
        try:
            converged = await adapter.repair_crash_recovery(scope, "op-r", attempt_no=1, generation_id="gen-1")
        finally:
            adapter.get_operation = original_get_operation  # type: ignore[method-assign]
        assert moved and converged == "pending_repair"
        op = await adapter.get_operation(scope, "op-r")
        assert op is not None and op["status"] == "pending_repair" and op["attempt_no"] == 1
        assert op["pending_repair_reason"] == "retry_attempt_reservation"
        # A reconcile whose own read already sees the reservation leaves it alone as well.
        assert (
            await adapter.repair_crash_recovery(scope, "op-r", attempt_no=1, generation_id="gen-1") == "pending_repair"
        )
        op = await adapter.get_operation(scope, "op-r")
        assert op is not None and op["status"] == "pending_repair"
        assert op["pending_repair_reason"] == "retry_attempt_reservation" and op.get("repair_count") in (None, 0)
        # The reservation is still resumable by the same key.
        assert await adapter.begin_retry_attempt_atomically(
            scope, "op-r", attempt_no=2, generation_id="gen-2", fencing_token=2
        )
        op = await adapter.get_operation(scope, "op-r")
        assert op is not None and op["status"] == "running" and op["attempt_no"] == 2

    async def attempt_transitions_are_rejected_on_a_claimed_row(adapter: SkillGraphStorageAdapter) -> None:
        """``expected_status`` is enforced even where the state machine would allow the target.

        A recoverer's claim keeps the writer's fencing token, so the token alone
        cannot stop the stale writer's own bookkeeping from landing on the
        claimed row; only the status check does.
        """

        scope = SkillManageScope(project_id="p23", graph_id="g23")
        assert await adapter.reserve_operation(
            scope, "op-c", idempotency_key="k-c", request_hash="h-c", operation_type="build"
        )
        assert await adapter.begin_initial_attempt_atomically(scope, "op-c", generation_id="gen-1", fencing_token=1)
        assert await adapter.claim_crash_recovery(scope, "op-c", expected_attempt_no=1, expected_generation_id="gen-1")
        assert not await adapter.begin_validation(scope, "op-c", fencing_token=1)
        assert not await adapter.record_operation_result(
            scope, "op-c", base_revision=None, result_revision=0, generation_id="gen-1", fencing_token=1
        )
        assert not await adapter.fail_operation(
            scope, "op-c", error_code="X", error_message="x", error_retryable=None, fencing_token=1
        )
        assert not await adapter.conflict_operation(scope, "op-c", error_code="X", error_message="x", fencing_token=1)
        assert not await adapter.cancel_operation(scope, "op-c", error_code="X", fencing_token=1)
        op = await adapter.get_operation(scope, "op-c")
        assert op is not None and op["status"] == "pending_repair" and op["pending_repair_reason"] == "crash_recovery"
        assert op.get("error_code") is None and op.get("completed_at") is None

    async def stale_token_of_the_same_owner_cannot_renew_or_release(adapter: SkillGraphStorageAdapter) -> None:
        """A same-key takeover reuses the owner id; only the token tells the old holder from the new one."""

        scope = SkillManageScope(project_id="p24", graph_id="g24")
        token1 = await adapter.acquire_writer_lease(scope, owner_operation_id="op-a", ttl_seconds=60.0, now=T0)
        token2 = await adapter.acquire_writer_lease(
            scope, owner_operation_id="op-a", ttl_seconds=60.0, now="2026-09-03T00:01:01+00:00"
        )
        assert token2 > token1
        before = await adapter.get_writer_lease(scope)
        assert not await adapter.renew_writer_lease(
            scope, owner_operation_id="op-a", token=token1, ttl_seconds=600.0, now="2026-09-03T00:01:02+00:00"
        )
        assert not await adapter.release_writer_lease(
            scope, owner_operation_id="op-a", token=token1, now="2026-09-03T00:01:03+00:00"
        )
        assert await adapter.get_writer_lease(scope) == before
        assert await adapter.release_writer_lease(
            scope, owner_operation_id="op-a", token=token2, now="2026-09-03T00:01:04+00:00"
        )
        assert (await adapter.get_writer_lease(scope))["owner_operation_id"] is None

    async def record_deletion_requires_a_tombstonable_row(adapter: SkillGraphStorageAdapter) -> None:
        """A generation that cannot be tombstoned (staging/ready/active) keeps every record."""

        scope = SkillManageScope(project_id="p25", graph_id="g25")
        node = SkillGraphNode(graph_node_id="n-1", node_type="skill", skill_id="s1", version_id="v1", name="One")
        await adapter.stage_generation(
            scope, "gen-s", operation_id="op-s", attempt_no=1, base_revision=None, target_revision=0
        )
        await adapter.write_snapshot(scope, "gen-s", nodes=[node], edges=[], memberships=[])
        await adapter._delete_generation_records(scope, "gen-s")  # type: ignore[attr-defined]
        assert (await adapter.get_generation(scope, "gen-s"))["status"] == "staging"
        assert [n.graph_node_id for n in (await adapter.load_snapshot(scope, "gen-s")).nodes] == ["n-1"]

    async def concurrent_lease_acquire_hands_out_exactly_one_token(adapter: SkillGraphStorageAdapter) -> None:
        """A create race and an expired-lease takeover race each admit exactly one writer.

        Two writers holding one token is fatal: the loser of the next row CAS
        releases the winner's lease (same owner id, same token), the winner's
        renewal fails and it records a terminal conflict on a healthy attempt.
        """

        scope = SkillManageScope(project_id="p26", graph_id="g26")

        async def acquire(index: int, now: str) -> int | None:
            try:
                return await adapter.acquire_writer_lease(
                    scope, owner_operation_id=f"op-{index}", ttl_seconds=60.0, now=now
                )
            except SkillManageWriterConflictError:
                return None

        # Create race: no lease row exists yet.
        results = await asyncio.gather(*(acquire(index, T0) for index in range(8)))
        tokens = [token for token in results if token is not None]
        assert len(tokens) == 1, results
        lease = await adapter.get_writer_lease(scope)
        assert lease is not None and lease["fencing_token"] == tokens[0]
        # Takeover race: the lease is expired for every contender.
        later = "2026-09-03T00:05:00+00:00"
        results = await asyncio.gather(*(acquire(10 + index, later) for index in range(8)))
        tokens = [token for token in results if token is not None]
        assert len(tokens) == 1 and tokens[0] > lease["fencing_token"], results
        lease = await adapter.get_writer_lease(scope)
        assert lease is not None and lease["fencing_token"] == tokens[0]
        assert lease["owner_operation_id"] == f"op-{10 + results.index(tokens[0])}"

    async def discard_after_a_lost_fence_never_deletes_a_published_generation(
        adapter: SkillGraphStorageAdapter,
    ) -> None:
        """The fence loses only to a publish already landing; the re-read must then leave the generation alone.

        By the re-read the candidate may even have been superseded (``retired``),
        and ``_delete_generation_records`` would tombstone a retired row: the
        status re-read is what keeps discard from bypassing GC's retention.
        """

        scope = SkillManageScope(project_id="p27", graph_id="g27")
        node = SkillGraphNode(graph_node_id="n-a", node_type="skill", skill_id="sa", version_id="v1", name="A")
        await _stage_ready(
            adapter, scope, "gen-1", operation_id="op-1", base_revision=None, target_revision=0, nodes=[node]
        )
        original_fence = adapter.mark_generation_failed
        fenced = 0

        async def publish_twice_then_fence(scope_, generation_id, **kwargs):
            nonlocal fenced
            if generation_id == "gen-1" and fenced == 0:
                fenced += 1
                assert await adapter.publish_generation(scope_, "gen-1", expected_revision=None, new_revision=0, now=T0)
                await _stage_ready(adapter, scope_, "gen-2", operation_id="op-2", base_revision=0, target_revision=1)
                assert await adapter.publish_generation(scope_, "gen-2", expected_revision=0, new_revision=1, now=T0)
            return await original_fence(scope_, generation_id, **kwargs)

        adapter.mark_generation_failed = publish_twice_then_fence  # type: ignore[method-assign]
        try:
            await adapter.discard_generation(scope, "gen-1")
        finally:
            adapter.mark_generation_failed = original_fence  # type: ignore[method-assign]
        assert fenced == 1
        row = await adapter.get_generation(scope, "gen-1")
        assert row is not None and row["status"] == "retired"
        assert [n.graph_node_id for n in (await adapter.load_snapshot(scope, "gen-1")).nodes] == ["n-a"]
        # The same window with a single publish (candidate merely ``active``) is protected too.
        await _stage_ready(adapter, scope, "gen-3", operation_id="op-3", base_revision=1, target_revision=2)

        async def publish_once_then_fence(scope_, generation_id, **kwargs):
            if generation_id == "gen-3":
                assert await adapter.publish_generation(scope_, "gen-3", expected_revision=1, new_revision=2, now=T0)
            return await original_fence(scope_, generation_id, **kwargs)

        adapter.mark_generation_failed = publish_once_then_fence  # type: ignore[method-assign]
        try:
            await adapter.discard_generation(scope, "gen-3")
        finally:
            adapter.mark_generation_failed = original_fence  # type: ignore[method-assign]
        assert (await adapter.get_generation(scope, "gen-3"))["status"] == "active"
        assert (await adapter.get_graph_head(scope))["active_generation_id"] == "gen-3"

    async def gc_deletes_generations_independently_and_reports_a_partial_pass(
        adapter: SkillGraphStorageAdapter,
    ) -> None:
        """A generation that cannot be deleted defers only itself; the pass reports both lists.

        Both adapters delete one generation's records through the same
        ``_delete_generation_records`` hook, which the case fails for one
        generation to model a mid-pass storage error.
        """

        scope = SkillManageScope(project_id="p20", graph_id="g20")
        stamps = [f"2026-09-03T00:0{i}:00+00:00" for i in range(4)]
        for index, stamp in enumerate(stamps):
            await _stage_ready(
                adapter,
                scope,
                f"gen-{index}",
                operation_id=f"op-{index}",
                base_revision=index - 1 if index else None,
                target_revision=index,
                now=stamp,
            )
            assert await adapter.publish_generation(
                scope, f"gen-{index}", expected_revision=index - 1 if index else None, new_revision=index, now=stamp
            )
        original_delete = adapter._delete_generation_records  # type: ignore[attr-defined]

        async def delete_unless_gen_1(scope_, generation_id, **kwargs):
            if generation_id == "gen-1":
                raise RuntimeError("storage blip")
            await original_delete(scope_, generation_id, **kwargs)

        adapter._delete_generation_records = delete_unless_gen_1  # type: ignore[attr-defined]
        try:
            with pytest.raises(SkillManageGcPartialFailureError) as excinfo:
                await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T01:00:00+00:00")
        finally:
            adapter._delete_generation_records = original_delete  # type: ignore[attr-defined]
        assert excinfo.value.removed == ["gen-0", "gen-2"]
        assert excinfo.value.deferred == [("gen-1", "RuntimeError")]
        assert await adapter.get_generation(scope, "gen-0") is None
        assert await adapter.get_generation(scope, "gen-2") is None
        assert (await adapter.get_generation(scope, "gen-1"))["status"] == "retired"
        assert (await adapter.get_generation(scope, "gen-3"))["status"] == "active"
        assert (await adapter.get_graph_head(scope))["active_generation_id"] == "gen-3"
        # The next clean pass collects exactly what was deferred.
        assert await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T01:00:01+00:00") == ["gen-1"]
        assert await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T01:00:02+00:00") == []

        # A delete whose acknowledgement is lost after it landed is a removal,
        # not a deferral: the row is gone, so no later pass could report it.
        await _stage_ready(adapter, scope, "gen-4", operation_id="op-gc-4", base_revision=3, target_revision=4)
        assert await adapter.publish_generation(
            scope, "gen-4", expected_revision=3, new_revision=4, now="2026-09-03T01:00:03+00:00"
        )

        async def delete_then_lose_the_ack(scope_, generation_id, **kwargs):
            await original_delete(scope_, generation_id, **kwargs)
            raise ConnectionResetError("acknowledgement lost")

        adapter._delete_generation_records = delete_then_lose_the_ack  # type: ignore[attr-defined]
        try:
            removed = await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T02:00:04+00:00")
        finally:
            adapter._delete_generation_records = original_delete  # type: ignore[attr-defined]
        assert removed == ["gen-3"]
        assert await adapter.get_generation(scope, "gen-3") is None

        # An outage during the pass: the delete and the re-read both fail, so
        # nothing can be claimed as removed and the partial report survives.
        await _stage_ready(adapter, scope, "gen-5", operation_id="op-gc-5", base_revision=4, target_revision=5)
        assert await adapter.publish_generation(
            scope, "gen-5", expected_revision=4, new_revision=5, now="2026-09-03T01:00:05+00:00"
        )
        original_get = adapter.get_generation

        async def store_is_down(*args, **kwargs):
            raise ConnectionResetError("store down")

        adapter._delete_generation_records = store_is_down  # type: ignore[attr-defined]
        adapter.get_generation = store_is_down  # type: ignore[method-assign]
        try:
            with pytest.raises(SkillManageGcPartialFailureError) as outage:
                await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T02:00:06+00:00")
        finally:
            adapter._delete_generation_records = original_delete  # type: ignore[attr-defined]
            adapter.get_generation = original_get  # type: ignore[method-assign]
        assert outage.value.removed == [] and outage.value.deferred == [("gen-4", "ConnectionResetError")]
        assert (await adapter.get_generation(scope, "gen-4"))["status"] == "retired"
        assert await adapter.gc_retired_generations(scope, keep=0, now="2026-09-03T02:00:07+00:00") == ["gen-4"]

    async def orphan_records_are_listed_and_reclaimable(adapter: SkillGraphStorageAdapter) -> None:
        """Records under an id with no generation row are visible to the sweep and deletable."""

        scope = SkillManageScope(project_id="p22", graph_id="g22")
        node = SkillGraphNode(graph_node_id="n-1", node_type="skill", skill_id="s1", version_id="v1", name="One")
        await _stage_ready(
            adapter, scope, "gen-live", operation_id="op-live", base_revision=None, target_revision=0, nodes=[node]
        )
        assert await adapter.publish_generation(scope, "gen-live", expected_revision=None, new_revision=0, now=T0)
        # A late write from a superseded writer: records without a generation row.
        await adapter.write_snapshot(scope, "gen-orphan", nodes=[node], edges=[], memberships=[])
        assert await adapter.get_generation(scope, "gen-orphan") is None
        assert await adapter.list_snapshot_generation_ids(scope) == {"gen-live", "gen-orphan"}
        await adapter.discard_generation(scope, "gen-orphan")
        assert await adapter.list_snapshot_generation_ids(scope) == {"gen-live"}
        assert (await adapter.load_snapshot(scope, "gen-orphan")).nodes == ()
        # The published generation is untouched by the same call.
        await adapter.discard_generation(scope, "gen-live")
        assert [n.graph_node_id for n in (await adapter.load_snapshot(scope, "gen-live")).nodes] == ["n-1"]
        assert await adapter.list_snapshot_generation_ids(SkillManageScope(project_id="p22", graph_id="other")) == set()

    async def undefined_similarity_is_not_a_hit(adapter: SkillGraphStorageAdapter) -> None:
        """A zero-norm vector on either side has no cosine; it never surfaces as a hit."""

        scope = SkillManageScope(project_id="p23", graph_id="g23")
        await adapter.stage_generation(
            scope, "gen-z", operation_id="op-z", attempt_no=1, base_revision=None, target_revision=0, now=T0
        )

        def embedding(node_id: str, vector: tuple[float, ...]) -> SkillGraphEmbedding:
            return SkillGraphEmbedding(
                graph_node_id=node_id,
                skill_id=node_id,
                version_id="v1",
                content_hash=f"h-{node_id}",
                embedding_model="m",
                embedding_profile_version="1",
                vector=vector,
            )

        unit = (1.0,) + (0.0,) * (VECTOR_DIMENSION - 1)
        zero = (0.0,) * VECTOR_DIMENSION
        await adapter.write_embeddings(
            scope, "gen-z", embeddings=[embedding("n-unit", unit), embedding("n-zero", zero)]
        )
        assert await adapter.count_embeddings(scope, "gen-z") == 2

        for threshold in (None, 0.0, 0.5):
            hits = await adapter.search_embeddings(
                scope, "gen-z", query_vector=unit, top_k=5, score_threshold=threshold
            )
            assert [(hit.graph_node_id, round(hit.score, 6)) for hit in hits] == [("n-unit", 1.0)], threshold
        assert await adapter.search_embeddings(scope, "gen-z", query_vector=zero, top_k=5) == ()
        await adapter.discard_generation(scope, "gen-z")

    async def timestamps_cross_the_port_as_aware_iso_strings(adapter: SkillGraphStorageAdapter) -> None:
        """Every ``*_at`` a read returns is a tz-aware ISO-8601 string, whatever the backend stores."""

        scope = SkillManageScope(project_id="p24", graph_id="g24")
        assert await adapter.reserve_operation(
            scope, "op-ts", idempotency_key="k", request_hash="h", operation_type="build", now=T0
        )
        await adapter.begin_initial_attempt_atomically(scope, "op-ts", generation_id="gen-ts", fencing_token=1, now=T0)
        await _stage_ready(adapter, scope, "gen-ts", operation_id="op-ts", base_revision=None, target_revision=0)
        assert await adapter.publish_generation(scope, "gen-ts", expected_revision=None, new_revision=0, now=T0)

        rows = [
            await adapter.get_operation(scope, "op-ts"),
            await adapter.get_generation(scope, "gen-ts"),
            await adapter.get_graph_head(scope),
            *await adapter.list_operations(scope),
            *await adapter.list_generations(scope),
        ]
        seen = 0
        for row in rows:
            assert row is not None
            for key, value in row.items():
                if not key.endswith("_at") or value is None:
                    continue
                assert isinstance(value, str), (key, value)
                assert datetime.fromisoformat(value).tzinfo is not None, (key, value)
                seen += 1
        assert seen >= 8

    return [
        head_create_and_idempotency,
        operation_reserve_idempotency,
        scope_isolation,
        cross_project_same_graph_name_is_isolated,
        same_logical_node_exists_in_two_generations,
        snapshot_load_roundtrip,
        snapshot_write_publish_load,
        generation_lifecycle_and_retirement,
        writer_lease_and_fencing_takeover,
        publish_rejects_stale_or_missing_fencing,
        concurrent_create_head_exactly_once,
        concurrent_head_cas_only_one_succeeds,
        operation_state_machine_transitions,
        operation_failure_and_conflict_outcomes,
        operation_retry_reservation_and_crash_recovery,
        retry_begin_requires_the_row_to_still_be_at_the_previous_attempt,
        crash_recovery_repair_branches,
        gc_respects_keep_and_min_age,
        single_active_generation_invariant,
        skewed_clock_takeover_is_safe_and_fences_the_old_writer,
        gc_keeps_generations_referenced_by_non_terminal_operations,
        list_operations_and_generations_by_status_within_scope,
        in_doubt_publish_fence_settles_the_candidate,
        repair_crash_recovery_only_settles_the_attempt_it_was_given,
        repair_crash_recovery_does_not_settle_a_row_that_moved_into_a_retry_reservation,
        attempt_transitions_are_rejected_on_a_claimed_row,
        stale_token_of_the_same_owner_cannot_renew_or_release,
        record_deletion_requires_a_tombstonable_row,
        concurrent_lease_acquire_hands_out_exactly_one_token,
        discard_after_a_lost_fence_never_deletes_a_published_generation,
        gc_deletes_generations_independently_and_reports_a_partial_pass,
        orphan_records_are_listed_and_reclaimable,
        undefined_similarity_is_not_a_hit,
        timestamps_cross_the_port_as_aware_iso_strings,
    ]
