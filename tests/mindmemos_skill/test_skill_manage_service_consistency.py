"""Write-path consistency chain: lease/fencing, generation and operation state machines, deferred GC.

Every instant comes from an injected clock and every wait from an injected
``sleep`` that advances that clock, so lease expiry, poll timeouts and
retention ages are exercised without real time passing.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from mindmemos_skill.application.skill_manage.models import SkillManageOperationResult, SkillManageServiceSettings
from mindmemos_skill.application.skill_manage.service import SkillManageService
from mindmemos_skill.application.skill_manage.state_machine import OperationStatus
from mindmemos_skill.errors import (
    SkillManageOperationCancelledError,
    SkillManageOperationPendingError,
    SkillManageOperationPendingRepairError,
    SkillManageValidationError,
)
from mindmemos_skill.infra.skill_manage import generation_id_from_operation
from mindmemos_skill.typing import (
    SkillGraphEdge,
    SkillManageBuildInput,
    SkillManageBuildPlan,
    SkillManageEmbedInput,
)
from test_skill_manage_service_embedding import (
    PROFILE,
    REFS_ALPHA_V2,
    SCOPE,
    FakeVersionLoader,
    RecordingAlgorithm,
    _build,
    _head,
    _operation_id,
    _remove,
    _upsert,
)
from test_skill_manage_storage_contract import InMemorySkillGraphStorageAdapter

T0 = datetime(2026, 9, 4, 0, 0, tzinfo=UTC)
SETTINGS = SkillManageServiceSettings(
    operation_wait_timeout_seconds=1.0,
    operation_poll_interval_seconds=0.1,
    writer_lease_ttl_seconds=60.0,
    writer_lease_renew_interval_seconds=20.0,
    retired_generations_to_keep=2,
    retired_generation_min_age_seconds=0.0,
)


class FakeClock:
    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def iso(self) -> str:
        return self.now.isoformat()

    async def sleep(self, seconds: float) -> None:
        """Advance the clock instead of waiting, but still yield to the event loop."""

        self.advance(seconds)
        await asyncio.sleep(0)


class NamedAlgorithm(RecordingAlgorithm):
    algorithm_name = "recording"
    algorithm_version = "7"


def _service(
    storage: InMemorySkillGraphStorageAdapter,
    algorithm: RecordingAlgorithm,
    clock: FakeClock,
    *,
    settings: SkillManageServiceSettings = SETTINGS,
) -> SkillManageService:
    return SkillManageService(
        algorithm=algorithm,
        storage=storage,
        embedding_profile=PROFILE,
        settings=settings,
        clock=clock,
        sleep=clock.sleep,
        version_loader=FakeVersionLoader(),
    )


async def _generation_statuses(storage: InMemorySkillGraphStorageAdapter) -> dict[str, str]:
    statuses = {}
    for (project_id, graph_id, generation_id), generation in storage._generations.items():
        if (project_id, graph_id) == (SCOPE.project_id, SCOPE.graph_id):
            statuses[generation_id] = generation["status"]
    return statuses


@pytest.mark.asyncio
async def test_write_path_walks_generation_and_operation_state_machines() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = NamedAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)

    built = await service.build_skill_index(_build("k1"))
    clock.advance(10)
    upserted = await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    clock.advance(10)
    removed = await service.remove_skill_from_index(_remove("r1", "gamma"))
    assert [r.status for r in (built, upserted, removed)] == [OperationStatus.COMMITTED] * 3
    assert [r.result_revision for r in (built, upserted, removed)] == [0, 1, 2]

    # Generation lifecycle: staging -> ready -> active, and the previous active
    # is retired in the same publish; counts are frozen at ready time.
    assert await _generation_statuses(storage) == {
        built.generation_id: "retired",
        upserted.generation_id: "retired",
        removed.generation_id: "active",
    }
    for result, expected_counts in ((built, (3, 0, 3)), (upserted, (3, 0, 3)), (removed, (2, 0, 2))):
        generation = await storage.get_generation(SCOPE, result.generation_id)
        assert generation is not None
        assert (generation["node_count"], generation["edge_count"], generation["vector_count"]) == expected_counts
        assert generation["counts_hash"] and generation["ready_at"] and generation["activated_at"]
    first_retired = (await storage.get_generation(SCOPE, built.generation_id))["retired_at"]
    second_retired = (await storage.get_generation(SCOPE, upserted.generation_id))["retired_at"]
    assert first_retired == (await storage.get_generation(SCOPE, upserted.generation_id))["activated_at"]
    assert second_retired == (await storage.get_generation(SCOPE, removed.generation_id))["activated_at"]
    assert first_retired < second_retired
    assert (await storage.get_generation(SCOPE, removed.generation_id))["retired_at"] is None

    # Operation lifecycle: pending -> running -> validating -> committed with
    # a strictly increasing fencing token and the attempt timestamps.
    for index, result in enumerate((built, upserted, removed), start=1):
        row = await storage.get_operation(SCOPE, result.operation_id)
        assert row is not None
        assert row["status"] == "committed"
        assert row["attempt_no"] == 1
        assert row["writer_fencing_token"] == index
        assert (
            row["current_generation_id"] == result.generation_id == generation_id_from_operation(result.operation_id, 1)
        )
        assert row["result_revision"] == result.result_revision
        assert row["first_started_at"] and row["validating_at"] and row["completed_at"]
        assert row["result_payload"]["vector_count"] == result.vector_count

    # The lease is released after each write and the head names the algorithm.
    lease = await storage.get_writer_lease(SCOPE)
    assert lease is not None and lease["owner_operation_id"] is None and lease["fencing_token"] == 3
    head = await _head(storage)
    assert (head["algorithm_name"], head["algorithm_version"]) == ("recording", "7")

    # Memberships: one per skill node, repointed by the upsert, filtered by the remove.
    memberships = {
        result.generation_id: {
            (m.skill_id, m.version_id) for m in (await storage.load_snapshot(SCOPE, result.generation_id)).memberships
        }
        for result in (built, upserted, removed)
    }
    assert memberships[built.generation_id] == {("alpha", "v1"), ("beta", "v1"), ("gamma", "v1")}
    assert memberships[upserted.generation_id] == {("alpha", "v2"), ("beta", "v1"), ("gamma", "v1")}
    assert memberships[removed.generation_id] == {("alpha", "v2"), ("beta", "v1")}


@pytest.mark.asyncio
async def test_lease_held_by_another_writer_is_a_writer_conflict_until_it_expires() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    await storage.acquire_writer_lease(SCOPE, owner_operation_id="other-writer", ttl_seconds=60.0, now=clock.iso())

    blocked = await service.build_skill_index(_build("k1"))
    assert blocked.status is OperationStatus.CONFLICT
    assert blocked.error_code == "SKILL_MANAGE_WRITER_CONFLICT"
    assert blocked.error_retryable is True
    assert algorithm.plan_build_calls == []
    # The reservation stays unstarted; nothing was staged.
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["status"] == "pending"
    assert await _generation_statuses(storage) == {}
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] == "other-writer"

    # Once the other writer's lease expires, the same key starts attempt 1 of the same operation.
    clock.advance(61)
    result = await service.build_skill_index(_build("k1"))
    assert result.status is OperationStatus.COMMITTED
    assert result.operation_id == _operation_id("k1")
    assert result.generation_id == generation_id_from_operation(result.operation_id, 1)
    row = await storage.get_operation(SCOPE, result.operation_id)
    assert row is not None and row["status"] == "committed" and row["writer_fencing_token"] == 2


@pytest.mark.asyncio
async def test_lease_taken_over_mid_write_is_fenced_out_at_publish() -> None:
    clock = FakeClock()

    class TakeoverStorage(InMemorySkillGraphStorageAdapter):
        """Another writer takes the lease over while this write is embedding."""

        async def write_embeddings(self, scope, generation_id, *, embeddings):
            await super().write_embeddings(scope, generation_id, embeddings=embeddings)
            clock.advance(61)
            await self.acquire_writer_lease(scope, owner_operation_id="intruder", ttl_seconds=60.0, now=clock.iso())

    storage = TakeoverStorage()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, clock)

    fenced = await service.build_skill_index(_build("k1"))
    assert fenced.status is OperationStatus.CONFLICT
    assert fenced.error_code == "SKILL_MANAGE_WRITER_CONFLICT"
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None
    assert row["status"] == "conflict" and row["error_code"] == "SKILL_MANAGE_WRITER_CONFLICT"
    assert await storage.get_graph_head(SCOPE) is None
    assert await _generation_statuses(storage) == {}
    # The intruder's lease is untouched by the fenced-out writer.
    lease = await storage.get_writer_lease(SCOPE)
    assert lease is not None and lease["owner_operation_id"] == "intruder" and lease["fencing_token"] == 2


@pytest.mark.asyncio
async def test_lease_is_renewed_while_the_write_runs() -> None:
    clock = FakeClock()
    renewals: list[str] = []

    class RecordingLeaseStorage(InMemorySkillGraphStorageAdapter):
        async def renew_writer_lease(self, scope, *, owner_operation_id, token, ttl_seconds, now=None):
            renewals.append(now)
            return await super().renew_writer_lease(
                scope, owner_operation_id=owner_operation_id, token=token, ttl_seconds=ttl_seconds, now=now
            )

    class SlowEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            # Yield long enough for the renewal task to fire several times.
            for _ in range(5):
                await asyncio.sleep(0)
            return await super().embed(request)

    storage = RecordingLeaseStorage()
    service = _service(storage, SlowEmbedAlgorithm(), clock)

    result = await service.build_skill_index(_build("k1"))

    assert result.status is OperationStatus.COMMITTED
    assert renewals, "the lease must be renewed while the write is in flight"
    lease = await storage.get_writer_lease(SCOPE)
    assert lease is not None and lease["owner_operation_id"] is None
    # Every renewal happened one renew interval after the previous instant.
    stamps = [datetime.fromisoformat(value) for value in renewals]
    assert all((b - a).total_seconds() == 20.0 for a, b in zip(stamps, stamps[1:]))


@pytest.mark.asyncio
async def test_validation_failure_is_recorded_as_non_retryable_and_discards_the_generation() -> None:
    class DanglingEdgeAlgorithm(RecordingAlgorithm):
        async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
            plan = await super().plan_build(request)
            edge = SkillGraphEdge(
                edge_id="e-dangling", edge_type="dependency", source_id="skill:alpha", target_id="skill:nope"
            )
            return plan.model_copy(update={"edges": (edge,)})

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, DanglingEdgeAlgorithm(), clock)

    with pytest.raises(SkillManageValidationError) as excinfo:
        await service.build_skill_index(_build("k1"))
    assert excinfo.value.error_code == "SKILL_MANAGE_VALIDATION_FAILED"

    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None
    assert row["status"] == "failed"
    assert row["error_code"] == "SKILL_MANAGE_VALIDATION_FAILED"
    assert row["error_retryable"] is False
    assert await _generation_statuses(storage) == {}
    assert await storage.get_graph_head(SCOPE) is None
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None

    replay = await service.build_skill_index(_build("k1"))
    assert replay.status is OperationStatus.FAILED
    assert replay.error_code == "SKILL_MANAGE_VALIDATION_FAILED"


@pytest.mark.asyncio
async def test_duplicate_edge_ids_fail_validation_instead_of_committing_an_overstated_edge_count() -> None:
    """Edge rows are keyed by edge_id, so a duplicate would be collapsed by the write while edge_count kept the plan's."""

    class DuplicateEdgeAlgorithm(RecordingAlgorithm):
        async def plan_build(self, request: SkillManageBuildInput) -> SkillManageBuildPlan:
            plan = await super().plan_build(request)
            edges = (
                SkillGraphEdge(
                    edge_id="e-dup", edge_type="dependency", source_id="skill:alpha", target_id="skill:beta", weight=0.9
                ),
                SkillGraphEdge(
                    edge_id="e-dup",
                    edge_type="dependency",
                    source_id="skill:alpha",
                    target_id="skill:gamma",
                    weight=0.1,
                ),
            )
            return plan.model_copy(update={"edges": edges})

    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, DuplicateEdgeAlgorithm(), FakeClock())

    with pytest.raises(SkillManageValidationError, match="duplicate edge_id"):
        await service.build_skill_index(_build("k1"))
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["status"] == "failed" and row["error_retryable"] is False
    assert await _generation_statuses(storage) == {}
    assert await storage.get_graph_head(SCOPE) is None


@pytest.mark.asyncio
async def test_a_naive_clock_is_refused_before_anything_is_written() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, RecordingAlgorithm(), FakeClock(T0.replace(tzinfo=None)))

    with pytest.raises(ValueError, match="timezone-aware"):
        await service.build_skill_index(_build("k1"))
    assert await storage.get_operation(SCOPE, _operation_id("k1")) is None
    assert await storage.get_graph_head(SCOPE) is None


@pytest.mark.asyncio
async def test_running_operation_with_a_live_lease_is_waited_on_until_the_timeout() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    request = _build("k1")
    operation_id = _operation_id("k1")
    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    assert await storage.reserve_operation(
        SCOPE,
        operation_id,
        idempotency_key="k1",
        request_hash=compute_skill_manage_request_hash(request),
        operation_type="build",
    )
    token = await storage.acquire_writer_lease(
        SCOPE, owner_operation_id=operation_id, ttl_seconds=600.0, now=clock.iso()
    )
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, operation_id, generation_id="gen-elsewhere", fencing_token=token, now=clock.iso()
    )

    with pytest.raises(SkillManageOperationPendingError) as excinfo:
        await service.build_skill_index(request)
    assert excinfo.value.operation_id == operation_id
    assert excinfo.value.status == "running"
    assert excinfo.value.error_code == "SKILL_MANAGE_OPERATION_PENDING"
    # The wait consumed the configured timeout in poll intervals and touched nothing.
    assert clock.now >= T0 + timedelta(seconds=1.0)
    assert algorithm.plan_build_calls == []
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "running"


@pytest.mark.asyncio
async def test_stranded_running_operation_with_expired_lease_is_recovered_and_retried() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    request = _build("k1")
    operation_id = _operation_id("k1")
    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    # A writer that died after staging: running row, staged generation, expired lease.
    assert await storage.reserve_operation(
        SCOPE,
        operation_id,
        idempotency_key="k1",
        request_hash=compute_skill_manage_request_hash(request),
        operation_type="build",
    )
    token = await storage.acquire_writer_lease(
        SCOPE, owner_operation_id=operation_id, ttl_seconds=60.0, now=clock.iso()
    )
    stranded_generation = generation_id_from_operation(operation_id, 1)
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, operation_id, generation_id=stranded_generation, fencing_token=token, now=clock.iso()
    )
    await storage.stage_generation(
        SCOPE, stranded_generation, operation_id=operation_id, attempt_no=1, base_revision=None, target_revision=0
    )
    clock.advance(61)

    result = await service.build_skill_index(request)

    assert result.status is OperationStatus.COMMITTED
    assert result.operation_id == operation_id
    assert result.generation_id == generation_id_from_operation(operation_id, 2)
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None
    assert row["status"] == "committed" and row["attempt_no"] == 2 and row["writer_fencing_token"] == 2
    assert await _generation_statuses(storage) == {result.generation_id: "active"}
    assert len(algorithm.plan_build_calls) == 1


@pytest.mark.asyncio
async def test_stranded_operation_whose_publish_landed_is_reconciled_to_committed() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    request = _build("k1")
    operation_id = _operation_id("k1")
    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    # A writer that died between publish and finalize: head points at its
    # generation, the row is still validating and its lease has lapsed.
    assert await storage.reserve_operation(
        SCOPE,
        operation_id,
        idempotency_key="k1",
        request_hash=compute_skill_manage_request_hash(request),
        operation_type="build",
    )
    token = await storage.acquire_writer_lease(
        SCOPE, owner_operation_id=operation_id, ttl_seconds=60.0, now=clock.iso()
    )
    generation_id = generation_id_from_operation(operation_id, 1)
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, operation_id, generation_id=generation_id, fencing_token=token, now=clock.iso()
    )
    await storage.stage_generation(
        SCOPE, generation_id, operation_id=operation_id, attempt_no=1, base_revision=None, target_revision=0
    )
    await storage.write_snapshot(SCOPE, generation_id, nodes=[], edges=[], memberships=[])
    assert await storage.mark_generation_ready(
        SCOPE, generation_id, node_count=0, edge_count=0, vector_count=0, counts_hash="h", now=clock.iso()
    )
    assert await storage.begin_validation(SCOPE, operation_id, now=clock.iso())
    assert await storage.publish_generation(
        SCOPE,
        generation_id,
        expected_revision=None,
        new_revision=0,
        owner_operation_id=operation_id,
        fencing_token=token,
        now=clock.iso(),
    )
    clock.advance(61)

    result = await service.build_skill_index(request)

    assert result.status is OperationStatus.COMMITTED
    assert result.generation_id == generation_id
    assert result.result_revision == 0
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 1
    assert algorithm.plan_build_calls == []
    assert (await _head(storage))["active_generation_id"] == generation_id


@pytest.mark.asyncio
async def test_pending_repair_reservation_is_left_to_its_live_owner_and_resumed_once_it_dies() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    request = _build("k1")
    operation_id = _operation_id("k1")
    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    # Another caller is between reserve_retry_attempt and begin_retry_attempt
    # and still holds the lease: this caller must not touch the row.
    assert await storage.reserve_operation(
        SCOPE,
        operation_id,
        idempotency_key="k1",
        request_hash=compute_skill_manage_request_hash(request),
        operation_type="build",
    )
    assert await storage.begin_initial_attempt_atomically(SCOPE, operation_id, generation_id="g", fencing_token=1)
    assert await storage.fail_operation(
        SCOPE, operation_id, error_code="SKILL_MANAGE_STORAGE_UNAVAILABLE", error_message="x", error_retryable=True
    )
    await storage.acquire_writer_lease(SCOPE, owner_operation_id=operation_id, ttl_seconds=60.0, now=clock.iso())
    assert await storage.reserve_retry_attempt(SCOPE, operation_id)

    with pytest.raises(SkillManageOperationPendingRepairError) as excinfo:
        await service.build_skill_index(request)
    assert excinfo.value.operation_id == operation_id
    assert excinfo.value.reason == "retry_attempt_reservation"
    assert algorithm.plan_build_calls == []

    # Once that caller's lease lapses the reservation is resumed as attempt 2.
    clock.advance(61)
    result = await service.build_skill_index(request)
    assert result.status is OperationStatus.COMMITTED
    assert result.generation_id == generation_id_from_operation(operation_id, 2)
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 2
    assert row["writer_fencing_token"] == 2


@pytest.mark.asyncio
async def test_publish_whose_commit_landed_but_raised_is_finalized_from_the_head() -> None:
    class AckLostStorage(InMemorySkillGraphStorageAdapter):
        async def publish_generation(self, scope, generation_id, **kwargs):
            published = await super().publish_generation(scope, generation_id, **kwargs)
            assert published
            raise ConnectionResetError("commit acknowledgement lost")

    storage = AckLostStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)

    result = await service.build_skill_index(_build("k1"))

    assert result.status is OperationStatus.COMMITTED
    row = await storage.get_operation(SCOPE, result.operation_id)
    assert row is not None and row["status"] == "committed" and row["result_revision"] == 0
    assert (await _head(storage))["active_generation_id"] == result.generation_id
    assert await _generation_statuses(storage) == {result.generation_id: "active"}
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_publish_confirmed_rolled_back_is_retryable_and_retried_as_a_new_attempt() -> None:
    class RollbackStorage(InMemorySkillGraphStorageAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 1

        async def publish_generation(self, scope, generation_id, **kwargs):
            if self.failures:
                self.failures -= 1
                raise ConnectionResetError("transaction rolled back")
            return await super().publish_generation(scope, generation_id, **kwargs)

    storage = RollbackStorage()
    clock = FakeClock()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, clock)

    with pytest.raises(ConnectionResetError):
        await service.build_skill_index(_build("k1"))
    operation_id = _operation_id("k1")
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None
    assert row["status"] == "failed" and row["error_code"] == "SKILL_MANAGE_STORAGE_UNAVAILABLE"
    assert row["error_retryable"] is True
    assert await storage.get_graph_head(SCOPE) is None
    assert await _generation_statuses(storage) == {}
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None

    result = await service.build_skill_index(_build("k1"))
    assert result.status is OperationStatus.COMMITTED
    assert result.generation_id == generation_id_from_operation(operation_id, 2)
    assert len(algorithm.plan_build_calls) == 2


@pytest.mark.asyncio
async def test_finalize_storage_blip_is_retried_and_a_persistent_one_leaves_the_row_for_repair() -> None:
    class FlakyFinalizeStorage(InMemorySkillGraphStorageAdapter):
        def __init__(self, failures: int) -> None:
            super().__init__()
            self.failures = failures

        async def record_operation_result(self, scope, operation_id, **kwargs):
            if self.failures:
                self.failures -= 1
                raise ConnectionResetError("finalize unavailable")
            return await super().record_operation_result(scope, operation_id, **kwargs)

    clock = FakeClock()
    storage = FlakyFinalizeStorage(failures=1)
    result = await _service(storage, RecordingAlgorithm(), clock).build_skill_index(_build("k1"))
    assert result.status is OperationStatus.COMMITTED
    assert (await storage.get_operation(SCOPE, result.operation_id))["status"] == "committed"

    storage = FlakyFinalizeStorage(failures=10)
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, clock)
    with pytest.raises(SkillManageOperationPendingRepairError) as excinfo:
        await service.build_skill_index(_build("k2"))
    operation_id = _operation_id("k2")
    assert excinfo.value.operation_id == operation_id
    assert excinfo.value.reason == "finalize_unavailable"
    # The publish landed, the row waits in validating, the lease is released.
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "validating"
    generation_id = generation_id_from_operation(operation_id, 1)
    assert (await _head(storage))["active_generation_id"] == generation_id
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    assert not any("_renew_forever" in repr(task) for task in asyncio.all_tasks() if not task.done())

    # Storage is back: the same key reconciles the stranded row to committed.
    storage.failures = 0
    recovered = await service.build_skill_index(_build("k2"))
    assert recovered.status is OperationStatus.COMMITTED
    assert recovered.generation_id == generation_id
    assert recovered.node_count == 3 and recovered.vector_count == 3
    assert (await storage.get_operation(SCOPE, operation_id))["status"] == "committed"
    assert len(algorithm.plan_build_calls) == 1


@pytest.mark.asyncio
async def test_superseded_attempt_cannot_touch_the_row_of_the_attempt_that_replaced_it() -> None:
    clock = FakeClock()
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, clock)
    request = _build("k1")
    operation_id = _operation_id("k1")
    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    # Attempt 1 stalled: its row is running, its lease lapsed, nothing staged.
    assert await storage.reserve_operation(
        SCOPE,
        operation_id,
        idempotency_key="k1",
        request_hash=compute_skill_manage_request_hash(request),
        operation_type="build",
    )
    stale_token = await storage.acquire_writer_lease(
        SCOPE, owner_operation_id=operation_id, ttl_seconds=60.0, now=clock.iso()
    )
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, operation_id, generation_id=generation_id_from_operation(operation_id, 1), fencing_token=stale_token
    )
    clock.advance(61)

    # Attempt 2 recovers and commits under a new token.
    result = await service.build_skill_index(request)
    assert result.status is OperationStatus.COMMITTED
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["attempt_no"] == 2 and row["writer_fencing_token"] == 2

    # The stale attempt wakes up: every transition it tries is fenced out.
    assert not await storage.begin_validation(SCOPE, operation_id, fencing_token=stale_token)
    assert not await storage.fail_operation(
        SCOPE, operation_id, error_code="X", error_message="x", error_retryable=True, fencing_token=stale_token
    )
    assert not await storage.conflict_operation(
        SCOPE, operation_id, error_code="X", error_message="x", fencing_token=stale_token
    )
    assert not await storage.cancel_operation(SCOPE, operation_id, error_code="X", fencing_token=stale_token)
    assert (await storage.get_operation(SCOPE, operation_id))["status"] == "committed"
    assert not await storage.renew_writer_lease(
        SCOPE, owner_operation_id=operation_id, token=stale_token, ttl_seconds=60.0, now=clock.iso()
    )


@pytest.mark.asyncio
async def test_deferred_gc_keeps_recent_and_young_retired_generations() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    settings = SETTINGS.model_copy(
        update={"retired_generations_to_keep": 1, "retired_generation_min_age_seconds": 300.0}
    )
    service = _service(storage, algorithm, clock, settings=settings)

    results = []
    for index in range(4):
        results.append(await service.build_skill_index(_build(f"k{index}")))
        clock.advance(200)
    assert all(result.warnings == () for result in results)

    # After the fourth publish (t ~ 600s): g0 (retired ~200s) and g1 (retired
    # ~400s) are beyond keep=1, but only g0 has reached the 300s minimum age.
    statuses = await _generation_statuses(storage)
    assert results[0].generation_id not in statuses
    assert statuses == {
        results[1].generation_id: "retired",
        results[2].generation_id: "retired",
        results[3].generation_id: "active",
    }
    assert (await _head(storage))["active_generation_id"] == results[3].generation_id

    # A keep larger than the retired count never deletes anything.
    generous = SETTINGS.model_copy(update={"retired_generations_to_keep": 10})
    fifth = await _service(storage, algorithm, clock, settings=generous).build_skill_index(_build("k5"))
    assert fifth.status is OperationStatus.COMMITTED
    assert set(await _generation_statuses(storage)) == {r.generation_id for r in results[1:]} | {fifth.generation_id}


@pytest.mark.asyncio
async def test_gc_failure_only_warns_and_keeps_the_committed_result() -> None:
    class BrokenGcStorage(InMemorySkillGraphStorageAdapter):
        async def gc_retired_generations(self, scope, *, keep=2, min_age_seconds=0.0, now=None):
            raise RuntimeError("gc unavailable")

    storage = BrokenGcStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)

    result = await service.build_skill_index(_build("k1"))

    assert result.status is OperationStatus.COMMITTED
    assert [warning["code"] for warning in result.warnings] == ["SKILL_MANAGE_GC_DEFERRED"]
    assert result.warnings[0]["details"] == {"reason": "RuntimeError"}
    row = await storage.get_operation(SCOPE, result.operation_id)
    assert row is not None and row["status"] == "committed"
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    # The recorded outcome is the committed result, GC warning excluded.
    replay = await service.build_skill_index(_build("k1"))
    assert replay.status is OperationStatus.COMMITTED and replay.warnings == ()


@pytest.mark.asyncio
async def test_cancellation_before_publish_marks_the_operation_cancelled() -> None:
    class CancelledEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            raise asyncio.CancelledError()

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, CancelledEmbedAlgorithm(), clock)

    with pytest.raises(asyncio.CancelledError):
        await service.build_skill_index(_build("k1"))

    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None
    assert row["status"] == "cancelled" and row["error_code"] == "SKILL_MANAGE_OPERATION_CANCELLED"
    assert await _generation_statuses(storage) == {}
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None

    with pytest.raises(SkillManageOperationCancelledError) as excinfo:
        await service.build_skill_index(_build("k1"))
    assert excinfo.value.operation_id == _operation_id("k1")
    assert excinfo.value.error_code == "SKILL_MANAGE_OPERATION_CANCELLED"


@pytest.mark.asyncio
async def test_noop_remove_walks_the_operation_state_machine_without_a_generation() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    built = await service.build_skill_index(_build("k1"))

    absent = await service.remove_skill_from_index(_remove("r-absent", "zeta"))

    assert absent.status is OperationStatus.COMMITTED
    assert absent.generation_id == built.generation_id
    # The result describes the generation it names, vectors included, so a
    # replay settled by crash recovery from that generation's row says the same.
    assert (absent.node_count, absent.edge_count, absent.vector_count) == (3, 0, 3)
    row = await storage.get_operation(SCOPE, absent.operation_id)
    assert row is not None
    assert row["status"] == "committed" and row["validating_at"] and row["completed_at"]
    assert row["result_payload"]["vector_count"] == 3
    assert row["writer_fencing_token"] == 2
    assert await _generation_statuses(storage) == {built.generation_id: "active"}
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_explicit_expected_revision_is_an_optimistic_lock_checked_against_the_head() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    built = await service.build_skill_index(_build("k1"))
    assert built.result_revision == 0

    # A stale expected_revision is a conflict, not a silent rebase: the write
    # must not create a generation and must leave the head untouched.
    stale = await service.upsert_skill_index(_upsert("u-stale", REFS_ALPHA_V2[0], expected_revision=5))
    assert stale.status is OperationStatus.CONFLICT
    assert stale.error_code == "SKILL_MANAGE_REVISION_CONFLICT"
    assert stale.operation_id is not None
    row = await storage.get_operation(SCOPE, stale.operation_id)
    assert row is not None and row["status"] == "conflict" and row["error_retryable"] is False
    assert row["base_revision"] == 0
    # No generation rows were staged for the rejected write.
    assert await _generation_statuses(storage) == {built.generation_id: "active"}
    assert (await _head(storage))["active_revision"] == 0

    # The matching expected_revision proceeds and advances the revision.
    matched = await service.upsert_skill_index(_upsert("u-match", REFS_ALPHA_V2[0], expected_revision=0))
    assert matched.status is OperationStatus.COMMITTED
    assert matched.result_revision == 1
    assert matched.base_revision == 0


@pytest.mark.asyncio
async def test_dry_run_upsert_and_remove_do_not_persist_or_advance_the_revision() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    built = await service.build_skill_index(_build("k1"))
    assert built.result_revision == 0

    dry_upsert = await service.upsert_skill_index(_upsert(None, REFS_ALPHA_V2[0], commit_policy="dry_run"))
    dry_remove = await service.remove_skill_from_index(_remove(None, "gamma", commit_policy="dry_run"))

    assert dry_upsert.status is OperationStatus.COMMITTED
    assert dry_upsert.operation_id is None
    assert dry_upsert.result_revision is None
    assert dry_remove.status is OperationStatus.COMMITTED
    assert dry_remove.operation_id is None
    assert dry_remove.result_revision is None

    # No operation rows, no new generations, head untouched.
    assert {key[2] for key in storage._operations} == {built.operation_id}
    assert await _generation_statuses(storage) == {built.generation_id: "active"}
    assert (await _head(storage))["active_revision"] == 0
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_same_process_same_key_concurrent_writes_merge_via_the_inflight_registry() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            entered.set()
            await release.wait()
            return await super().embed(request)

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, SlowEmbedAlgorithm(), clock)

    first = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await entered.wait()
    second = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    release.set()

    first_result = await first
    second_result = await second
    assert first_result == second_result
    assert first_result.status is OperationStatus.COMMITTED
    assert first_result.operation_id == second_result.operation_id
    # Exactly one operation row and one live generation: the follower joined.
    assert {key[2] for key in storage._operations} == {first_result.operation_id}
    assert await _generation_statuses(storage) == {first_result.generation_id: "active"}

    # A different request hash on the same key joins nothing and conflicts.
    conflicting = await service.build_skill_index(_build("k1", REFS_ALPHA_V2))
    assert conflicting.status is OperationStatus.CONFLICT
    assert conflicting.error_code == "SKILL_MANAGE_IDEMPOTENCY_CONFLICT"


def test_plan_unchanged_is_order_independent_for_nodes_edges_and_memberships() -> None:
    from mindmemos_skill.application.skill_manage.service import _plan_unchanged
    from mindmemos_skill.typing import SkillGraphSnapshot, SkillManageMutationPlan
    from test_skill_manage_service_embedding import _memberships, _node

    nodes = (_node("alpha", "v1", "alpha"), _node("beta", "v1", "beta"), _node("gamma", "v1", "gamma"))
    edges = (
        SkillGraphEdge(
            edge_id="e-1", edge_type="dependency", source_id="skill:alpha", target_id="skill:beta", weight=1.0
        ),
        SkillGraphEdge(
            edge_id="e-2", edge_type="dependency", source_id="skill:beta", target_id="skill:gamma", weight=1.0
        ),
    )
    plan = SkillManageMutationPlan(
        scope=SCOPE, nodes=nodes, edges=edges, memberships=_memberships(nodes), embedding_texts=()
    )
    # A loaded snapshot returns rows in physical record order, which can
    # differ from the plan's derivation order; that must still be a no-op.
    reversed_snapshot = SkillGraphSnapshot(
        generation_id="g", revision=0, nodes=nodes[::-1], edges=edges[::-1], memberships=_memberships(nodes)[::-1]
    )
    assert _plan_unchanged(plan, reversed_snapshot)

    # Any real difference (a dropped node) is still detected.
    dropped = SkillGraphSnapshot(
        generation_id="g", revision=0, nodes=nodes[:2], edges=edges[:1], memberships=_memberships(nodes[:2])
    )
    assert not _plan_unchanged(plan, dropped)


@pytest.mark.asyncio
async def test_explicit_repair_converges_stranded_rows_and_discards_stale_candidates_idempotently() -> None:
    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    service = _service(storage, algorithm, clock)
    built = await service.build_skill_index(_build("k0"))
    head_before = await _head(storage)

    # A writer that died after staging: running row, staged candidate, lease that will lapse.
    dead_id = _operation_id("k-dead")
    assert await storage.reserve_operation(
        SCOPE,
        dead_id,
        idempotency_key="k-dead",
        request_hash=compute_skill_manage_request_hash(_build("k-dead")),
        operation_type="build",
    )
    dead_token = await storage.acquire_writer_lease(
        SCOPE, owner_operation_id=dead_id, ttl_seconds=60.0, now=clock.iso()
    )
    dead_generation = generation_id_from_operation(dead_id, 1)
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, dead_id, generation_id=dead_generation, fencing_token=dead_token, base_revision=0, now=clock.iso()
    )
    await storage.stage_generation(
        SCOPE, dead_generation, operation_id=dead_id, attempt_no=1, base_revision=0, target_revision=1, now=clock.iso()
    )
    # An orphan candidate that no operation row references any more.
    await storage.stage_generation(
        SCOPE,
        "orphan-gen",
        operation_id="op-vanished",
        attempt_no=1,
        base_revision=0,
        target_revision=1,
        now=clock.iso(),
    )

    # While the dead writer's lease is still alive nothing is touched.
    clock.advance(10)
    early = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert [(a.kind, a.operation_id, a.reason) for a in early.actions] == [
        ("operation_skipped", dead_id, "writer lease is alive")
    ]
    assert (await storage.get_operation(SCOPE, dead_id))["status"] == "running"
    assert "orphan-gen" in await _generation_statuses(storage)

    # After the lease lapses the row converges to failed(retryable) and both
    # the dead attempt's candidate and the orphan are discarded.
    clock.advance(60)
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert {(a.kind, a.operation_id, a.generation_id) for a in report.actions} == {
        ("operation_reconciled", dead_id, dead_generation),
        ("candidate_discarded", "op-vanished", "orphan-gen"),
    }
    reconciled = next(a for a in report.actions if a.kind == "operation_reconciled")
    assert (reconciled.from_status, reconciled.to_status) == ("running", "failed")
    row = await storage.get_operation(SCOPE, dead_id)
    assert row["status"] == "failed" and row["error_retryable"] is True
    assert row["last_repair_action"] == "crash_recovery:failed"
    assert await _generation_statuses(storage) == {built.generation_id: "active"}
    # The head and its active generation are never touched by repair.
    assert await _head(storage) == head_before
    assert report.warnings == ()
    # The lease taken over for the reconcile is released again.
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None

    # Idempotent: a second pass finds nothing to do.
    again = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert again.actions == () and again.warnings == ()

    # The same key can now be retried as attempt 2 by the normal write path.
    retried = await service.build_skill_index(_build("k-dead"))
    assert retried.status is OperationStatus.COMMITTED
    assert (await storage.get_operation(SCOPE, dead_id))["attempt_no"] == 2


@pytest.mark.asyncio
async def test_explicit_repair_keeps_the_taken_over_lease_until_the_reconcile_is_done() -> None:
    """A slow but not-yet-dead writer must stay fenced for the whole reconcile.

    The repair takes the lease over before it claims the row and only releases
    it after ``repair_crash_recovery`` has settled (and discarded) the
    candidate; a publish the old writer attempts in that window is rejected
    by fencing rather than racing the discard.
    """

    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash
    from mindmemos_skill.errors import SkillManageWriterConflictError

    class ObservingStorage(InMemorySkillGraphStorageAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.lease_during_repair: dict[str, object] | None = None
            self.stale_publish: str | None = None

        async def repair_crash_recovery(self, scope, operation_id, *, attempt_no, generation_id, now=None):
            self.lease_during_repair = await self.get_writer_lease(scope)
            # The old writer wakes up mid-reconcile and tries to publish its
            # candidate with the token it still believes it holds.
            try:
                await self.publish_generation(
                    scope,
                    generation_id,
                    expected_revision=0,
                    new_revision=1,
                    owner_operation_id=operation_id,
                    fencing_token=stale_token,
                    now=now,
                )
                self.stale_publish = "published"
            except SkillManageWriterConflictError:
                self.stale_publish = "fenced"
            return await super().repair_crash_recovery(
                scope, operation_id, attempt_no=attempt_no, generation_id=generation_id, now=now
            )

    storage = ObservingStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    built = await service.build_skill_index(_build("k0"))

    dead_id = _operation_id("k-slow")
    assert await storage.reserve_operation(
        SCOPE,
        dead_id,
        idempotency_key="k-slow",
        request_hash=compute_skill_manage_request_hash(_build("k-slow")),
        operation_type="build",
    )
    stale_token = await storage.acquire_writer_lease(
        SCOPE, owner_operation_id=dead_id, ttl_seconds=60.0, now=clock.iso()
    )
    candidate = generation_id_from_operation(dead_id, 1)
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, dead_id, generation_id=candidate, fencing_token=stale_token, base_revision=0, now=clock.iso()
    )
    await storage.stage_generation(
        SCOPE, candidate, operation_id=dead_id, attempt_no=1, base_revision=0, target_revision=1, now=clock.iso()
    )
    await storage.write_snapshot(SCOPE, candidate, nodes=[], edges=[], memberships=[])
    assert await storage.mark_generation_ready(
        SCOPE, candidate, node_count=0, edge_count=0, vector_count=0, counts_hash="h", now=clock.iso()
    )
    clock.advance(61)

    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)

    assert [(a.kind, a.to_status) for a in report.actions] == [("operation_reconciled", "failed")]
    # During the reconcile the lease belonged to the repair (a newer token), so
    # the stale writer's publish was fenced and its candidate was discarded.
    assert storage.lease_during_repair is not None
    assert storage.lease_during_repair["owner_operation_id"] == dead_id
    assert storage.lease_during_repair["fencing_token"] > stale_token
    assert storage.stale_publish == "fenced"
    assert await _generation_statuses(storage) == {built.generation_id: "active"}
    assert (await _head(storage))["active_revision"] == 0
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_explicit_maintenance_collects_retired_generations_without_a_write() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    algorithm = RecordingAlgorithm()
    clock = FakeClock()
    settings = SETTINGS.model_copy(
        update={"retired_generations_to_keep": 1, "retired_generation_min_age_seconds": 3600.0}
    )
    service = _service(storage, algorithm, clock, settings=settings)

    first = await service.build_skill_index(_build("k1"))
    clock.advance(10)
    second = await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    clock.advance(10)
    third = await service.remove_skill_from_index(_remove("r1", "gamma"))
    # Both retired generations are too young for the write-path GC to collect.
    assert await _generation_statuses(storage) == {
        first.generation_id: "retired",
        second.generation_id: "retired",
        third.generation_id: "active",
    }

    # Still too young: maintenance is a no-op and says so.
    early = await service.maintain_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert early.removed_generation_ids == () and early.warnings == ()

    # Past the minimum age, only the generation beyond keep=1 goes; the active one never does.
    clock.advance(3600)
    report = await service.maintain_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert report.removed_generation_ids == (first.generation_id,)
    assert await _generation_statuses(storage) == {second.generation_id: "retired", third.generation_id: "active"}
    assert (await _head(storage))["active_generation_id"] == third.generation_id
    # Idempotent.
    assert (
        await service.maintain_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    ).removed_generation_ids == ()


# ---------------------------------------------------------------------------
# Regressions from the 2026-09-04 adversarial review of the third round.


@pytest.mark.asyncio
async def test_same_key_different_hash_while_leader_in_flight_returns_the_idempotency_conflict() -> None:
    """The in-process path must report the same conflict shape as the durable row does."""

    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            entered.set()
            await release.wait()
            return await super().embed(request)

    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, SlowEmbedAlgorithm(), FakeClock())
    leader = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await entered.wait()

    conflicting = await service.build_skill_index(_build("k1", REFS_ALPHA_V2))

    assert conflicting.status is OperationStatus.CONFLICT
    assert conflicting.error_code == "SKILL_MANAGE_IDEMPOTENCY_CONFLICT"
    assert conflicting.operation_id == _operation_id("k1")
    release.set()
    assert (await leader).status is OperationStatus.COMMITTED
    # And the durable-row path afterwards reports the identical shape.
    assert await service.build_skill_index(_build("k1", REFS_ALPHA_V2)) == conflicting


@pytest.mark.asyncio
async def test_cancelled_leader_does_not_cancel_its_followers() -> None:
    """Only the task that was actually cancelled propagates ``CancelledError`` (spec §4).

    A follower that joined the cancelled leader re-runs against the durable
    row and receives the typed cancelled error carrying the operation id; its
    own task is not marked cancelled.
    """

    entered = asyncio.Event()

    class BlockingEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            entered.set()
            await asyncio.Event().wait()
            return await super().embed(request)

    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, BlockingEmbedAlgorithm(), FakeClock())
    leader = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await entered.wait()
    follower = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await asyncio.sleep(0)
    leader.cancel()
    await asyncio.gather(leader, follower, return_exceptions=True)

    assert leader.cancelled()
    assert not follower.cancelled()
    assert isinstance(follower.exception(), SkillManageOperationCancelledError)
    assert follower.exception().operation_id == _operation_id("k1")
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["status"] == "cancelled"
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_follower_of_a_leader_cancelled_before_reserving_performs_the_write_itself() -> None:
    entered = asyncio.Event()

    class BlockingReserveStorage(InMemorySkillGraphStorageAdapter):
        async def reserve_operation(self, scope, operation_id, **kwargs):
            if not entered.is_set():
                entered.set()
                await asyncio.Event().wait()
            return await super().reserve_operation(scope, operation_id, **kwargs)

    storage = BlockingReserveStorage()
    service = _service(storage, RecordingAlgorithm(), FakeClock())
    leader = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await entered.wait()
    follower = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await asyncio.sleep(0)
    leader.cancel()
    results = await asyncio.gather(leader, follower, return_exceptions=True)

    assert leader.cancelled()
    assert isinstance(results[1], SkillManageOperationResult)
    assert results[1].status is OperationStatus.COMMITTED
    assert (await _head(storage))["active_revision"] == 0


@pytest.mark.asyncio
async def test_failed_leader_write_leaves_no_unretrieved_future_exception(caplog) -> None:
    """A leader that fails with no followers must not trigger asyncio's never-retrieved log."""

    import gc
    import logging

    class FailingEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            raise RuntimeError("embedding provider down")

    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, FailingEmbedAlgorithm(), FakeClock())
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        with pytest.raises(RuntimeError):
            await service.build_skill_index(_build("k1"))
        gc.collect()
        await asyncio.sleep(0)
    assert not [record for record in caplog.records if "never retrieved" in record.getMessage()]


@pytest.mark.asyncio
async def test_repair_does_not_settle_a_newer_attempt_that_superseded_the_listed_row() -> None:
    """Repair must act on the row as it is under the lease, not as it was listed.

    Between the listing and the reconcile a same-key retry (another process)
    turns the listed ``pending_repair`` attempt 1 into a live, published
    attempt 2. Reconciling against the stale attempt would record the live
    attempt as ``conflict`` and discard its candidate; repair must skip it.
    """

    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    built = await service.build_skill_index(_build("k0"))

    # A stranded pending_repair(crash_recovery) attempt 1 with a dead lease.
    dead_id = _operation_id("k-dead")
    request = _build("k-dead")
    assert await storage.reserve_operation(
        SCOPE,
        dead_id,
        idempotency_key="k-dead",
        request_hash=compute_skill_manage_request_hash(request),
        operation_type="build",
    )
    token = await storage.acquire_writer_lease(SCOPE, owner_operation_id=dead_id, ttl_seconds=60.0, now=clock.iso())
    gen1 = generation_id_from_operation(dead_id, 1)
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, dead_id, generation_id=gen1, fencing_token=token, base_revision=0, now=clock.iso()
    )
    assert await storage.claim_crash_recovery(
        SCOPE, dead_id, expected_attempt_no=1, expected_generation_id=gen1, now=clock.iso()
    )
    clock.advance(61)

    original_list = storage.list_operations
    raced = False

    async def list_then_let_a_retry_win(scope, *, statuses=None):
        nonlocal raced
        rows = await original_list(scope, statuses=statuses)
        if not raced and any(row["operation_id"] == dead_id for row in rows):
            raced = True
            # Another process retries the same key: attempt 1 is reconciled to
            # failed, attempt 2 runs and publishes revision 1.
            other = _service(storage, RecordingAlgorithm(), clock)
            retried = await other.build_skill_index(request)
            assert retried.status is OperationStatus.COMMITTED and retried.result_revision == 1
        return rows

    storage.list_operations = list_then_let_a_retry_win  # type: ignore[method-assign]
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    storage.list_operations = original_list  # type: ignore[method-assign]

    assert [(a.kind, a.operation_id, a.reason) for a in report.actions] == [
        ("operation_skipped", dead_id, "operation moved since it was listed")
    ]
    row = await storage.get_operation(SCOPE, dead_id)
    assert row["status"] == "committed" and row["attempt_no"] == 2 and row["result_revision"] == 1
    gen2 = generation_id_from_operation(dead_id, 2)
    assert await _generation_statuses(storage) == {built.generation_id: "retired", gen2: "active"}
    assert (await _head(storage))["active_generation_id"] == gen2
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_repair_reports_an_ambiguous_row_as_unresolved_with_a_consistency_warning() -> None:
    """Spec §9: a row the rules cannot settle stays pending_repair and is flagged, not "reconciled"."""

    from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    await service.build_skill_index(_build("k0"))

    # The head points at the candidate but its generation row is gone: the
    # frozen repair rules cannot decide between committed and lost.
    dead_id = _operation_id("k-dead")
    assert await storage.reserve_operation(
        SCOPE,
        dead_id,
        idempotency_key="k-dead",
        request_hash=compute_skill_manage_request_hash(_build("k-dead")),
        operation_type="build",
    )
    token = await storage.acquire_writer_lease(SCOPE, owner_operation_id=dead_id, ttl_seconds=60.0, now=clock.iso())
    gen1 = generation_id_from_operation(dead_id, 1)
    assert await storage.begin_initial_attempt_atomically(
        SCOPE, dead_id, generation_id=gen1, fencing_token=token, base_revision=0, now=clock.iso()
    )
    storage._heads[(SCOPE.project_id, SCOPE.graph_id)]["active_generation_id"] = gen1
    clock.advance(61)

    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)

    assert [(a.kind, a.operation_id, a.to_status, a.reason) for a in report.actions] == [
        ("operation_unresolved", dead_id, "pending_repair", "SKILL_MANAGE_CONSISTENCY_VIOLATION")
    ]
    assert [w["code"] for w in report.warnings] == ["SKILL_MANAGE_CONSISTENCY_VIOLATION"]
    assert (await storage.get_operation(SCOPE, dead_id))["status"] == "pending_repair"
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_gc_keeps_a_retired_generation_still_referenced_by_a_validating_operation() -> None:
    """A publish that landed but was never finalized must survive GC so repair converges to committed."""

    x_id = _operation_id("kx")

    class StrandFinalizeStorage(InMemorySkillGraphStorageAdapter):
        armed = True

        async def record_operation_result(self, scope, operation_id, **kwargs):
            if self.armed and operation_id == x_id:
                raise ConnectionResetError("finalize unavailable")
            return await super().record_operation_result(scope, operation_id, **kwargs)

    storage = StrandFinalizeStorage()
    clock = FakeClock()
    settings = SETTINGS.model_copy(
        update={"retired_generations_to_keep": 1, "retired_generation_min_age_seconds": 10.0}
    )
    service = _service(storage, RecordingAlgorithm(), clock, settings=settings)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0
    clock.advance(30)
    with pytest.raises(SkillManageOperationPendingRepairError):
        await service.build_skill_index(_build("kx"))
    storage.armed = False
    gen_x = generation_id_from_operation(x_id, 1)
    assert (await storage.get_operation(SCOPE, x_id))["status"] == "validating"
    assert (await _head(storage))["active_generation_id"] == gen_x

    for key in ("k1", "k2", "k3"):
        clock.advance(30)
        assert (await service.build_skill_index(_build(key))).status is OperationStatus.COMMITTED
    # gen_x is retired and beyond keep=1, yet it is still referenced by the
    # validating row, so neither the write-path GC nor explicit maintenance may take it.
    assert (await _generation_statuses(storage))[gen_x] == "retired"
    assert (
        gen_x
        not in (
            await service.maintain_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
        ).removed_generation_ids
    )
    assert (await _generation_statuses(storage))[gen_x] == "retired"

    clock.advance(61)
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert [(a.kind, a.operation_id, a.to_status) for a in report.actions] == [
        ("operation_reconciled", x_id, "committed")
    ]
    assert (await storage.get_operation(SCOPE, x_id))["result_revision"] == 1
    assert (await service.build_skill_index(_build("kx"))).result_revision == 1
    # Once the operation is terminal the generation is collectable again.
    assert (
        gen_x
        in (
            await service.maintain_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
        ).removed_generation_ids
    )


@pytest.mark.asyncio
async def test_dry_run_reports_the_same_revision_conflict_a_persist_would() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, RecordingAlgorithm(), FakeClock())
    built = await service.build_skill_index(_build("k1"))

    stale = await service.upsert_skill_index(
        _upsert(None, REFS_ALPHA_V2[0], commit_policy="dry_run", expected_revision=5)
    )
    assert stale.status is OperationStatus.CONFLICT
    assert stale.error_code == "SKILL_MANAGE_REVISION_CONFLICT" and stale.operation_id is None
    matched = await service.upsert_skill_index(
        _upsert(None, REFS_ALPHA_V2[0], commit_policy="dry_run", expected_revision=0)
    )
    assert matched.status is OperationStatus.COMMITTED and matched.operation_id is None
    # A first build previews against an absent head the same way persist would.
    empty = InMemorySkillGraphStorageAdapter()
    fresh = _service(empty, RecordingAlgorithm(), FakeClock())
    assert (await fresh.build_skill_index(_build(None, commit_policy="dry_run", expected_revision=0))).status is (
        OperationStatus.CONFLICT
    )
    assert (await fresh.build_skill_index(_build(None, commit_policy="dry_run"))).status is OperationStatus.COMMITTED
    # Nothing was persisted by any of the previews.
    assert {key[2] for key in storage._operations} == {built.operation_id}
    assert await empty.get_graph_head(SCOPE) is None
