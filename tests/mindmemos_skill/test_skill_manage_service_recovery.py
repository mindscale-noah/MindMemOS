"""Fourth-round recovery regressions: in-doubt publish, fenced reconcile, cancellation, superseded attempts.

Each test pins one migration-side defect found by the adversarial review of
the write-consistency chain (bug records 22-28): an interrupted publish whose
commit is still landing must not be mistaken for a rollback; a reconciler
may only settle the attempt it observed, under the lease; a cancellation
landing anywhere in the failure bookkeeping must still stop the lease
renewal task; an attempt superseded by a same-key retry reports the row's
outcome rather than its own; repair leaves a retry reservation alone; a
no-op mutation's finalize blip is left for crash recovery; and maintenance
reports exactly what it removed even when the pass did not finish.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

import pytest
from mindmemos_skill.application.skill_manage.loader import SkillVersionNotFound
from mindmemos_skill.application.skill_manage.models import compute_skill_manage_request_hash
from mindmemos_skill.application.skill_manage.service import SkillManageService, _WriterLease
from mindmemos_skill.application.skill_manage.state_machine import OperationStatus
from mindmemos_skill.errors import (
    SkillManageOperationCancelledError,
    SkillManageOperationPendingError,
    SkillManageOperationPendingRepairError,
    SkillManageValidationError,
    SkillManageWriterConflictError,
)
from mindmemos_skill.infra.skill_manage import generation_id_from_operation
from mindmemos_skill.typing import SkillManageEmbedInput, SkillVersionRef
from test_skill_manage_service_consistency import SETTINGS, FakeClock, _generation_statuses, _service
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
    _retrieve,
    _upsert,
)
from test_skill_manage_storage_contract import InMemorySkillGraphStorageAdapter


def _renewal_tasks() -> list[asyncio.Task]:
    return [task for task in asyncio.all_tasks() if not task.done() and "_renew_forever" in repr(task)]


async def _stranded_row(
    storage: InMemorySkillGraphStorageAdapter,
    clock: FakeClock,
    key: str,
    *,
    base_revision: int | None,
) -> tuple[str, str, int]:
    """Reserve, lease and begin attempt 1 of ``key`` as a writer that then dies."""

    operation_id = _operation_id(key)
    assert await storage.reserve_operation(
        SCOPE,
        operation_id,
        idempotency_key=key,
        request_hash=compute_skill_manage_request_hash(_build(key)),
        operation_type="build",
    )
    token = await storage.acquire_writer_lease(
        SCOPE, owner_operation_id=operation_id, ttl_seconds=60.0, now=clock.iso()
    )
    generation_id = generation_id_from_operation(operation_id, 1)
    assert await storage.begin_initial_attempt_atomically(
        SCOPE,
        operation_id,
        generation_id=generation_id,
        fencing_token=token,
        base_revision=base_revision,
        now=clock.iso(),
    )
    return operation_id, generation_id, token


# ------------------------------------------------------------- bug 22


@pytest.mark.asyncio
async def test_in_doubt_publish_is_settled_by_fencing_the_candidate() -> None:
    """A commit that lands only after the client saw an error is finalized, never rolled back by hand.

    The fake models the real ordering: the publish call raises before its
    commit is visible, and the ``ready -> failed`` fence — a row update that
    on PostgreSQL queues behind the in-flight commit — observes the commit
    landing and loses. The outcome must then be read from the candidate, not
    inferred from a head that still showed the base revision.
    """

    class CommitStillLandingStorage(InMemorySkillGraphStorageAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.pending_publish: tuple | None = None
            self.fence_calls = 0

        async def publish_generation(self, scope, generation_id, **kwargs):
            if self.pending_publish is None:
                self.pending_publish = (scope, generation_id, kwargs)
                raise ConnectionResetError("commit acknowledgement lost")
            return await super().publish_generation(scope, generation_id, **kwargs)

        async def mark_generation_failed(self, scope, generation_id, *, now=None):
            self.fence_calls += 1
            if self.pending_publish is not None:
                scope, pending_id, kwargs = self.pending_publish
                self.pending_publish = None
                assert await super().publish_generation(scope, pending_id, **kwargs)
            return await super().mark_generation_failed(scope, generation_id, now=now)

    storage = CommitStillLandingStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)

    result = await service.build_skill_index(_build("k1"))

    assert storage.fence_calls == 1
    assert result.status is OperationStatus.COMMITTED and result.result_revision == 0
    row = await storage.get_operation(SCOPE, result.operation_id)
    assert row is not None and row["status"] == "committed" and row["result_revision"] == 0
    assert (await _head(storage))["active_generation_id"] == result.generation_id
    assert await _generation_statuses(storage) == {result.generation_id: "active"}
    assert len(await storage.load_embeddings(SCOPE, result.generation_id)) == 3
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_publish_proven_rolled_back_by_the_fence_is_retryable() -> None:
    """When the fence wins, the candidate was never published: failed(retryable), candidate gone."""

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
    service = _service(storage, RecordingAlgorithm(), clock)
    with pytest.raises(ConnectionResetError):
        await service.build_skill_index(_build("k1"))
    operation_id = _operation_id("k1")
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "failed" and row["error_retryable"] is True
    assert row["error_code"] == "SKILL_MANAGE_STORAGE_UNAVAILABLE"
    assert await _generation_statuses(storage) == {}
    assert await storage.get_graph_head(SCOPE) is None

    retried = await service.build_skill_index(_build("k1"))
    assert retried.status is OperationStatus.COMMITTED
    assert retried.generation_id == generation_id_from_operation(operation_id, 2)


@pytest.mark.asyncio
async def test_fenced_candidate_left_behind_by_a_failed_discard_is_swept_by_repair() -> None:
    """A candidate the fence settled but whose discard failed is collected regardless of its age."""

    class DiscardFailsOnceStorage(InMemorySkillGraphStorageAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.publish_failures = 0
            self.discard_failures = 0

        async def publish_generation(self, scope, generation_id, **kwargs):
            if self.publish_failures:
                self.publish_failures -= 1
                raise ConnectionResetError("transaction rolled back")
            return await super().publish_generation(scope, generation_id, **kwargs)

        async def discard_generation(self, scope, generation_id):
            if self.discard_failures:
                self.discard_failures -= 1
                raise ConnectionResetError("vector store unavailable")
            await super().discard_generation(scope, generation_id)

    storage = DiscardFailsOnceStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0
    storage.publish_failures = storage.discard_failures = 1
    with pytest.raises(ConnectionResetError):
        await service.build_skill_index(_build("k1"))
    operation_id = _operation_id("k1")
    gen1 = generation_id_from_operation(operation_id, 1)
    assert (await storage.get_operation(SCOPE, operation_id))["status"] == "failed"
    assert (await _generation_statuses(storage))[gen1] == "failed"
    assert len(await storage.load_embeddings(SCOPE, gen1)) == 3

    # Younger than the stale-candidate age, yet no longer publishable: swept now.
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)

    assert [(a.kind, a.operation_id, a.generation_id, a.from_status) for a in report.actions] == [
        ("candidate_discarded", operation_id, gen1, "failed")
    ]
    assert gen1 not in await _generation_statuses(storage)
    assert await storage.load_embeddings(SCOPE, gen1) == ()
    assert (await _head(storage))["active_revision"] == 0


# ------------------------------------------------------------- bug 23


@pytest.mark.asyncio
async def test_same_key_reconcile_of_a_claimed_row_holds_the_lease_and_settles_only_that_attempt() -> None:
    """A pending_repair(crash_recovery) row is reconciled under the lease, so no retry can open attempt 2 underneath."""

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0

    # A recoverer claimed attempt 1 and died: the row is pending_repair(crash_recovery),
    # its lease has lapsed, and the candidate was never published.
    operation_id, gen1, _ = await _stranded_row(storage, clock, "k1", base_revision=0)
    await storage.stage_generation(
        SCOPE, gen1, operation_id=operation_id, attempt_no=1, base_revision=0, target_revision=1, now=clock.iso()
    )
    assert await storage.claim_crash_recovery(
        SCOPE, operation_id, expected_attempt_no=1, expected_generation_id=gen1, now=clock.iso()
    )
    clock.advance(61)

    original_repair = storage.repair_crash_recovery
    interleaved: list[str] = []

    async def repair_with_a_concurrent_same_key_caller(scope, op_id, *, attempt_no, generation_id, now=None):
        # Another same-key caller arrives mid-reconcile. It must be fenced out
        # by the lease this reconcile holds rather than open attempt 2 under it.
        with pytest.raises(SkillManageWriterConflictError):
            await storage.acquire_writer_lease(SCOPE, owner_operation_id=op_id, ttl_seconds=60.0, now=clock.iso())
        interleaved.append((await storage.get_operation(SCOPE, op_id))["status"])
        return await original_repair(scope, op_id, attempt_no=attempt_no, generation_id=generation_id, now=now)

    storage.repair_crash_recovery = repair_with_a_concurrent_same_key_caller  # type: ignore[method-assign]
    result = await service.build_skill_index(_build("k1"))
    storage.repair_crash_recovery = original_repair  # type: ignore[method-assign]

    assert interleaved == ["pending_repair"]
    assert result.status is OperationStatus.COMMITTED and result.result_revision == 1
    assert result.generation_id == generation_id_from_operation(operation_id, 2)
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 2
    assert row["last_repair_action"] == "crash_recovery:failed"
    statuses = await _generation_statuses(storage)
    assert gen1 not in statuses and statuses[result.generation_id] == "active"
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_same_key_reconcile_waits_while_another_recoverer_holds_the_lease() -> None:
    """A live recoverer mid-reconcile is waited on like a live attempt, never raced."""

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, clock)
    operation_id, gen1, _ = await _stranded_row(storage, clock, "k1", base_revision=None)
    clock.advance(61)
    # A recoverer took the lease over and claimed the row, and is still alive.
    await storage.acquire_writer_lease(SCOPE, owner_operation_id=operation_id, ttl_seconds=60.0, now=clock.iso())
    assert await storage.claim_crash_recovery(
        SCOPE, operation_id, expected_attempt_no=1, expected_generation_id=gen1, now=clock.iso()
    )

    with pytest.raises(SkillManageOperationPendingError) as excinfo:
        await service.build_skill_index(_build("k1"))
    assert excinfo.value.status == "pending_repair"
    row = await storage.get_operation(SCOPE, operation_id)
    assert row["status"] == "pending_repair" and row["attempt_no"] == 1
    assert algorithm.plan_build_calls == []
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] == operation_id


# ------------------------------------------------------------- bug 24


@pytest.mark.asyncio
async def test_cancellation_during_failure_bookkeeping_still_stops_the_lease_renewal() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowFailStorage(InMemorySkillGraphStorageAdapter):
        async def fail_operation(self, scope, operation_id, **kwargs):
            entered.set()
            await release.wait()
            return await super().fail_operation(scope, operation_id, **kwargs)

    class BrokenEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            raise RuntimeError("provider down")

    storage = SlowFailStorage()
    clock = FakeClock()
    service = _service(storage, BrokenEmbedAlgorithm(), clock)
    task = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await entered.wait()
    assert len(_renewal_tasks()) == 1

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _renewal_tasks() == []
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


@pytest.mark.asyncio
async def test_repeated_cancellation_during_publish_still_reports_the_real_outcome() -> None:
    """Every cancellation after the visibility switch started is absorbed; the switch itself is never cancelled."""

    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowPublishStorage(InMemorySkillGraphStorageAdapter):
        async def publish_generation(self, scope, generation_id, **kwargs):
            entered.set()
            await release.wait()
            return await super().publish_generation(scope, generation_id, **kwargs)

    storage = SlowPublishStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    task = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await entered.wait()
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
    release.set()

    result = await task

    assert result.status is OperationStatus.COMMITTED and result.result_revision == 0
    assert (await storage.get_operation(SCOPE, result.operation_id))["status"] == "committed"
    assert (await _head(storage))["active_generation_id"] == result.generation_id
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    assert _renewal_tasks() == []


# ------------------------------------------------------------- bug 25


@pytest.mark.asyncio
async def test_writer_superseded_by_a_same_key_takeover_replays_the_row_instead_of_its_own_conflict() -> None:
    """Two processes: W1 stalls, its lease lapses, W2 takes over the same key and commits; W1 answers with the row."""

    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            entered.set()
            await release.wait()
            return await super().embed(request)

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    w1_service = _service(storage, SlowEmbedAlgorithm(), clock)
    w2_service = _service(storage, RecordingAlgorithm(), clock)
    operation_id = _operation_id("k1")

    w1 = asyncio.ensure_future(w1_service.build_skill_index(_build("k1")))
    await entered.wait()
    clock.advance(61)
    w2_result = await w2_service.build_skill_index(_build("k1"))
    assert w2_result.status is OperationStatus.COMMITTED
    assert w2_result.generation_id == generation_id_from_operation(operation_id, 2)

    release.set()
    w1_result = await w1

    assert w1_result == w2_result
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 2
    assert await _generation_statuses(storage) == {w2_result.generation_id: "active"}
    assert (await storage.get_writer_lease(SCOPE))["fencing_token"] == 2


@pytest.mark.asyncio
async def test_writer_superseded_after_staging_replays_the_row_instead_of_a_validation_failure() -> None:
    """W1's candidate is discarded by W2's crash recovery; W1's read-back validation must not become the answer."""

    entered = asyncio.Event()
    release = asyncio.Event()

    class SlowVectorStorage(InMemorySkillGraphStorageAdapter):
        async def write_embeddings(self, scope, generation_id, *, embeddings):
            await super().write_embeddings(scope, generation_id, embeddings=embeddings)
            if not entered.is_set():
                entered.set()
                await release.wait()

    storage = SlowVectorStorage()
    clock = FakeClock()
    w1_service = _service(storage, RecordingAlgorithm(), clock)
    w2_service = _service(storage, RecordingAlgorithm(), clock)
    operation_id = _operation_id("k1")

    w1 = asyncio.ensure_future(w1_service.build_skill_index(_build("k1")))
    await entered.wait()
    clock.advance(61)
    w2_result = await w2_service.build_skill_index(_build("k1"))
    assert w2_result.status is OperationStatus.COMMITTED
    assert generation_id_from_operation(operation_id, 1) not in await _generation_statuses(storage)

    release.set()
    w1_result = await w1

    assert w1_result == w2_result
    assert (await storage.get_operation(SCOPE, operation_id))["status"] == "committed"


@pytest.mark.asyncio
async def test_writer_superseded_by_a_still_running_attempt_reports_a_retryable_writer_conflict() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    w2_entered = asyncio.Event()
    w2_release = asyncio.Event()

    class GatedEmbedAlgorithm(RecordingAlgorithm):
        def __init__(self, gate_in: asyncio.Event, gate_out: asyncio.Event) -> None:
            super().__init__()
            self.gate_in, self.gate_out = gate_in, gate_out

        async def embed(self, request: SkillManageEmbedInput):
            self.gate_in.set()
            await self.gate_out.wait()
            return await super().embed(request)

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    # W1's process is stalled outright: its renewal loop never wakes up, so
    # its lease genuinely lapses while W2 runs.
    w1_service = _service(storage, GatedEmbedAlgorithm(entered, release), clock)
    w1_service._sleep = lambda seconds: asyncio.Event().wait()
    w2_service = _service(storage, GatedEmbedAlgorithm(w2_entered, w2_release), clock)
    operation_id = _operation_id("k1")

    w1 = asyncio.ensure_future(w1_service.build_skill_index(_build("k1")))
    await entered.wait()
    clock.advance(61)
    w2 = asyncio.ensure_future(w2_service.build_skill_index(_build("k1")))
    await w2_entered.wait()
    assert (await storage.get_operation(SCOPE, operation_id))["attempt_no"] == 2

    release.set()
    w1_result = await w1
    assert w1_result.status is OperationStatus.CONFLICT
    assert w1_result.error_code == "SKILL_MANAGE_WRITER_CONFLICT" and w1_result.error_retryable is True
    assert (await storage.get_operation(SCOPE, operation_id))["status"] == "running"
    assert generation_id_from_operation(operation_id, 1) not in await _generation_statuses(storage)

    w2_release.set()
    assert (await w2).status is OperationStatus.COMMITTED
    assert _renewal_tasks() == []


# ------------------------------------------------------------- bug 26


@pytest.mark.asyncio
async def test_repair_leaves_a_retry_reservation_taken_after_the_listing_alone() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0
    dead_id, gen1, _ = await _stranded_row(storage, clock, "k-dead", base_revision=0)
    clock.advance(61)

    original_list = storage.list_operations
    raced = False

    async def list_then_race(scope, *, statuses=None):
        nonlocal raced
        rows = await original_list(scope, statuses=statuses)
        if not raced and any(row["operation_id"] == dead_id for row in rows):
            raced = True
            # Between the listing and the re-read: another process recovers the
            # row, reserves its retry and dies with the reservation open, while
            # a different key moves the head.
            await storage.acquire_writer_lease(SCOPE, owner_operation_id=dead_id, ttl_seconds=60.0, now=clock.iso())
            assert await storage.claim_crash_recovery(
                SCOPE, dead_id, expected_attempt_no=1, expected_generation_id=gen1, now=clock.iso()
            )
            assert await storage.repair_crash_recovery(SCOPE, dead_id, attempt_no=1, generation_id=gen1) == "failed"
            assert await storage.reserve_retry_attempt(SCOPE, dead_id, now=clock.iso())
            clock.advance(61)
            other = await _service(storage, RecordingAlgorithm(), clock).build_skill_index(_build("k-other"))
            assert other.status is OperationStatus.COMMITTED and other.result_revision == 1
        return rows

    storage.list_operations = list_then_race  # type: ignore[method-assign]
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    storage.list_operations = original_list  # type: ignore[method-assign]

    assert [(a.kind, a.operation_id, a.from_status, a.reason) for a in report.actions] == [
        ("operation_skipped", dead_id, "pending_repair", "operation moved since it was listed")
    ]
    row = await storage.get_operation(SCOPE, dead_id)
    assert row["status"] == "pending_repair" and row["pending_repair_reason"] == "retry_attempt_reservation"
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    # The reservation is resumed by the retry rule, rebased on the moved head.
    resumed = await service.build_skill_index(_build("k-dead"))
    assert resumed.status is OperationStatus.COMMITTED and resumed.result_revision == 2
    assert (await storage.get_operation(SCOPE, dead_id))["attempt_no"] == 2


# ------------------------------------------------------------- bug 27


@pytest.mark.asyncio
async def test_noop_finalize_blip_leaves_the_row_for_crash_recovery_instead_of_failing_it() -> None:
    class FlakyFinalizeStorage(InMemorySkillGraphStorageAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.failures = 0

        async def record_operation_result(self, scope, operation_id, **kwargs):
            if self.failures:
                self.failures -= 1
                raise ConnectionResetError("finalize unavailable")
            return await super().record_operation_result(scope, operation_id, **kwargs)

    storage = FlakyFinalizeStorage()
    clock = FakeClock()
    algorithm = RecordingAlgorithm()
    service = _service(storage, algorithm, clock)
    built = await service.build_skill_index(_build("k0"))
    storage.failures = 2

    with pytest.raises(SkillManageOperationPendingRepairError) as excinfo:
        await service.remove_skill_from_index(_remove("r1", "absent"))
    operation_id = _operation_id("r1")
    assert excinfo.value.reason == "finalize_unavailable"
    row = await storage.get_operation(SCOPE, operation_id)
    assert row["status"] == "validating" and row.get("error_code") is None
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    assert _renewal_tasks() == []
    assert await _generation_statuses(storage) == {built.generation_id: "active"}

    # Crash recovery converges the stranded no-op once its lease lapsed.
    clock.advance(61)
    recovered = await service.remove_skill_from_index(_remove("r1", "absent"))
    assert recovered.status is OperationStatus.COMMITTED and recovered.result_revision == 0
    assert (await storage.get_operation(SCOPE, operation_id))["status"] == "committed"
    assert (await _head(storage))["active_revision"] == 0


# ------------------------------------------------------------- bug 28


@pytest.mark.asyncio
async def test_maintenance_reports_generations_removed_before_a_failing_one() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    generous = SETTINGS.model_copy(update={"retired_generations_to_keep": 10})
    service = _service(storage, RecordingAlgorithm(), clock, settings=generous)
    results = []
    for index in range(4):
        results.append(await service.build_skill_index(_build(f"k{index}")))
        clock.advance(100)
    g0, g1, g2, g3 = (result.generation_id for result in results)
    assert [(await _generation_statuses(storage))[g] for g in (g0, g1, g2, g3)] == ["retired"] * 3 + ["active"]

    original_delete = storage._delete_generation_records
    poisoned = {g1}

    async def delete_unless_poisoned(scope, generation_id, **kwargs):
        if generation_id in poisoned:
            raise RuntimeError("delete failed")
        await original_delete(scope, generation_id, **kwargs)

    storage._delete_generation_records = delete_unless_poisoned  # type: ignore[method-assign]
    aggressive = SETTINGS.model_copy(update={"retired_generations_to_keep": 0})
    service = _service(storage, RecordingAlgorithm(), clock, settings=aggressive)
    report = await service.maintain_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)

    assert report.removed_generation_ids == (g0, g2)
    assert [(w["code"], w["details"]) for w in report.warnings] == [
        ("SKILL_MANAGE_GC_DEFERRED", {"generation_id": g1, "reason": "RuntimeError"})
    ]
    assert await _generation_statuses(storage) == {g1: "retired", g3: "active"}

    poisoned.clear()
    second = await service.maintain_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert second.removed_generation_ids == (g1,) and second.warnings == ()
    assert await _generation_statuses(storage) == {g3: "active"}


# ------------------------------------------------------------- bug 32


@pytest.mark.asyncio
async def test_repair_sweep_keeps_the_candidate_of_an_attempt_begun_after_the_live_listing() -> None:
    """With a zero age threshold, a candidate staged between the two listings still belongs to a live attempt."""

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0

    original_list = storage.list_generations
    begun: dict[str, str] = {}

    async def begin_an_attempt_then_list(scope, *, statuses=None):
        if not begun:
            # A writer reserves, leases, begins and stages between the live
            # operation listing and the candidate listing.
            operation_id, generation_id, _ = await _stranded_row(storage, clock, "k-live", base_revision=0)
            await storage.stage_generation(
                SCOPE,
                generation_id,
                operation_id=operation_id,
                attempt_no=1,
                base_revision=0,
                target_revision=1,
                now=clock.iso(),
            )
            begun.update(operation_id=operation_id, generation_id=generation_id)
        return await original_list(scope, statuses=statuses)

    storage.list_generations = begin_an_attempt_then_list  # type: ignore[method-assign]
    report = await service.repair_skill_index(
        project_id=SCOPE.project_id, graph_id=SCOPE.graph_id, stale_candidate_age_seconds=0
    )
    storage.list_generations = original_list  # type: ignore[method-assign]

    assert report.actions == () and report.warnings == ()
    assert (await _generation_statuses(storage))[begun["generation_id"]] == "staging"
    assert (await storage.get_operation(SCOPE, begun["operation_id"]))["status"] == "running"


# ------------------------------------------------------------- bug 34


@pytest.mark.asyncio
async def test_reconciler_whose_taken_over_lease_lapsed_does_not_claim_the_attempt_that_replaced_it() -> None:
    """A lease taken over for a reconcile is not renewed in the background; once it lapses it proves nothing."""

    w2_entered, w2_release = asyncio.Event(), asyncio.Event()

    class GatedStagingStorage(InMemorySkillGraphStorageAdapter):
        gate_for: str | None = None

        async def write_snapshot(self, scope, generation_id, *, nodes, edges, memberships):
            if generation_id == self.gate_for:
                w2_entered.set()
                await w2_release.wait()
            await super().write_snapshot(scope, generation_id, nodes=nodes, edges=edges, memberships=memberships)

    storage = GatedStagingStorage()
    clock = FakeClock()
    algorithm_a = RecordingAlgorithm()
    caller_a = _service(storage, algorithm_a, clock)
    assert (await caller_a.build_skill_index(_build("k0"))).result_revision == 0
    operation_id, gen1, _ = await _stranded_row(storage, clock, "k1", base_revision=0)
    assert await storage.claim_crash_recovery(
        SCOPE, operation_id, expected_attempt_no=1, expected_generation_id=gen1, now=clock.iso()
    )
    clock.advance(61)
    storage.gate_for = generation_id_from_operation(operation_id, 2)

    caller_w2 = _service(storage, RecordingAlgorithm(), clock)
    w2: dict[str, asyncio.Future] = {}
    original_repair = storage.repair_crash_recovery

    async def repair_then_stall_past_the_ttl(scope, op_id, *, attempt_no, generation_id, now=None):
        converged = await original_repair(scope, op_id, attempt_no=attempt_no, generation_id=generation_id, now=now)
        # Caller A stalls here longer than the lease TTL; a same-key caller W2
        # takes the lapsed lease over, opens attempt 2 and is still in flight
        # (staging its candidate) when A resumes.
        clock.advance(61)
        w2["task"] = asyncio.ensure_future(caller_w2.build_skill_index(_build("k1")))
        await w2_entered.wait()
        return converged

    storage.repair_crash_recovery = repair_then_stall_past_the_ttl  # type: ignore[method-assign]
    outcome = await caller_a.build_skill_index(_build("k1"))
    storage.repair_crash_recovery = original_repair  # type: ignore[method-assign]

    assert outcome.status is OperationStatus.CONFLICT
    assert outcome.error_code == "SKILL_MANAGE_WRITER_CONFLICT" and outcome.error_retryable is True
    row = await storage.get_operation(SCOPE, operation_id)
    assert row["status"] == "running" and row["attempt_no"] == 2
    assert len(algorithm_a.plan_build_calls) == 1

    w2_release.set()
    committed = await w2["task"]
    assert committed.status is OperationStatus.COMMITTED and committed.result_revision == 1
    assert committed.generation_id == generation_id_from_operation(operation_id, 2)
    assert (await storage.get_operation(SCOPE, operation_id))["status"] == "committed"
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


# ------------------------------------------------------------- bug 35


@pytest.mark.asyncio
async def test_stalled_retry_cannot_reopen_an_attempt_number_another_caller_already_used() -> None:
    """A stalls between taking the lease and reserving; B runs attempt 2 meanwhile; A must open attempt 3."""

    class FlakyPublishStorage(InMemorySkillGraphStorageAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.publish_failures = 0

        async def publish_generation(self, scope, generation_id, **kwargs):
            if self.publish_failures:
                self.publish_failures -= 1
                raise ConnectionResetError("transaction rolled back")
            return await super().publish_generation(scope, generation_id, **kwargs)

    storage = FlakyPublishStorage()
    clock = FakeClock()
    # A's wait budget outlives its own stall so the outcome is decided by the
    # attempt bookkeeping, not by the poll deadline.
    patient = SETTINGS.model_copy(update={"operation_wait_timeout_seconds": 300.0})
    caller_a = _service(storage, RecordingAlgorithm(), clock, settings=patient)
    caller_b = _service(storage, RecordingAlgorithm(), clock)
    operation_id = _operation_id("k1")
    # attempt 1 failed retryably (publish rolled back once).
    storage.publish_failures = 1
    with pytest.raises(ConnectionResetError):
        await caller_a.build_skill_index(_build("k1"))
    assert (await storage.get_operation(SCOPE, operation_id))["attempt_no"] == 1

    original_reserve = storage.reserve_retry_attempt
    stalled_once = False

    async def stall_then_reserve(scope, op_id, *, now=None):
        nonlocal stalled_once
        if not stalled_once:
            stalled_once = True
            # A holds the lease but stalls past its TTL before reserving; B takes
            # the lease over, runs attempt 2 and fails it retryably.
            clock.advance(61)
            storage.publish_failures = 1
            with pytest.raises(ConnectionResetError):
                await caller_b.build_skill_index(_build("k1"))
            row = await storage.get_operation(SCOPE, op_id)
            assert row["status"] == "failed" and row["attempt_no"] == 2
        return await original_reserve(scope, op_id, now=now)

    storage.reserve_retry_attempt = stall_then_reserve  # type: ignore[method-assign]
    result = await caller_a.build_skill_index(_build("k1"))
    storage.reserve_retry_attempt = original_reserve  # type: ignore[method-assign]

    assert result.status is OperationStatus.COMMITTED and result.result_revision == 0
    assert result.generation_id == generation_id_from_operation(operation_id, 3)
    row = await storage.get_operation(SCOPE, operation_id)
    assert row["status"] == "committed" and row["attempt_no"] == 3
    assert await _generation_statuses(storage) == {result.generation_id: "active"}
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None


# ------------------------------------------------------------- bug 36


_EAGER_GC = SETTINGS.model_copy(update={"retired_generations_to_keep": 0, "retired_generation_min_age_seconds": 0.0})


class _HookedStorage(InMemorySkillGraphStorageAdapter):
    """Runs a one-shot hook before a chosen read so a concurrent writer can slip in."""

    def __init__(self) -> None:
        super().__init__()
        self.before_load_snapshot: Callable[[], Awaitable[None]] | None = None
        self.before_search: Callable[[], Awaitable[None]] | None = None

    async def load_snapshot(self, scope, generation_id):
        hook, self.before_load_snapshot = self.before_load_snapshot, None
        if hook is not None:
            await hook()
        return await super().load_snapshot(scope, generation_id)

    async def search_embeddings(self, scope, generation_id, **kwargs):
        hook, self.before_search = self.before_search, None
        if hook is not None:
            await hook()
        return await super().search_embeddings(scope, generation_id, **kwargs)


@pytest.mark.asyncio
async def test_reader_whose_generation_is_collected_mid_read_starts_over_from_the_fresh_head() -> None:
    """A generation retired and GC'd under a reader is never reported as an empty graph at revision 0."""

    storage = _HookedStorage()
    clock = FakeClock()
    writer = _service(storage, RecordingAlgorithm(), clock, settings=_EAGER_GC)
    reader = _service(storage, RecordingAlgorithm(), clock, settings=_EAGER_GC)
    built = await writer.build_skill_index(_build("k1"))
    assert built.result_revision == 0

    async def publish_and_collect_underneath() -> None:
        clock.advance(1)
        removed = await writer.remove_skill_from_index(_remove("r1", "gamma"))
        assert removed.result_revision == 1
        assert built.generation_id not in await _generation_statuses(storage)

    # Collected before the snapshot load: the read restarts from the new head.
    storage.before_load_snapshot = publish_and_collect_underneath
    result = await reader.retrieve_skills(_retrieve())
    assert result.revision == 1 and len(result.hits) == 2 and result.warnings == ()
    assert result.generation_id == generation_id_from_operation(_operation_id("r1"), 1)

    # Collected between the snapshot load and the KNN: same restart.
    async def publish_and_collect_again() -> None:
        clock.advance(1)
        assert (await writer.remove_skill_from_index(_remove("r2", "beta"))).result_revision == 2

    storage.before_search = publish_and_collect_again
    result = await reader.retrieve_skills(_retrieve())
    assert result.revision == 2 and len(result.hits) == 1
    # A dry-run mutation previews against a generation that is still readable.
    storage.before_load_snapshot = None
    preview = await reader.remove_skill_from_index(_remove(None, "alpha", commit_policy="dry_run"))
    assert preview.status is OperationStatus.COMMITTED and preview.node_count == 0


@pytest.mark.asyncio
async def test_interrupted_generation_deletion_is_tombstoned_first_and_finished_by_the_next_pass() -> None:
    class DeleteInterruptedStorage(InMemorySkillGraphStorageAdapter):
        interrupt = False

        async def _delete_generation_records(self, scope, generation_id, *, now=None):
            key = (scope.project_id, scope.graph_id, generation_id)
            generation = self._generations.get(key)
            if self.interrupt and generation is not None and generation["status"] == "retired":
                # The tombstone landed, then the records could not be deleted.
                generation.update({"status": "deleted", "deleted_at": now})
                raise ConnectionResetError("vector store unavailable")
            await super()._delete_generation_records(scope, generation_id, now=now)

    storage = DeleteInterruptedStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock, settings=_EAGER_GC)
    built = await service.build_skill_index(_build("k1"))
    clock.advance(1)
    storage.interrupt = True
    removed = await service.remove_skill_from_index(_remove("r1", "gamma"))
    assert [w["code"] for w in removed.warnings] == ["SKILL_MANAGE_GC_DEFERRED"]
    assert removed.warnings[0]["details"]["generation_id"] == built.generation_id
    assert (await _generation_statuses(storage))[built.generation_id] == "deleted"

    # A tombstoned generation is finished regardless of keep/min-age; it is
    # never counted toward the retained window.
    storage.interrupt = False
    lenient = SETTINGS.model_copy(
        update={"retired_generations_to_keep": 5, "retired_generation_min_age_seconds": 86400.0}
    )
    report = await _service(storage, RecordingAlgorithm(), clock, settings=lenient).maintain_skill_index(
        project_id=SCOPE.project_id, graph_id=SCOPE.graph_id
    )
    assert report.removed_generation_ids == (built.generation_id,) and report.warnings == ()
    assert await _generation_statuses(storage) == {removed.generation_id: "active"}


# ------------------------------------------------------------- bug 37


@pytest.mark.asyncio
async def test_loader_failure_during_build_or_upsert_creates_no_operation_row() -> None:
    """Document resolution is preflight (spec §2): a loader blip never burns the idempotency key."""

    class FlakyLoader:
        def __init__(self) -> None:
            self.failures = 0

        async def load(self, ref, *, project_id):
            if self.failures:
                self.failures -= 1
                raise RuntimeError("repository unavailable")
            return await FakeVersionLoader().load(ref, project_id=project_id)

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    loader = FlakyLoader()
    algorithm = RecordingAlgorithm()
    service = SkillManageService(
        algorithm=algorithm,
        storage=storage,
        embedding_profile=PROFILE,
        settings=SETTINGS,
        clock=clock,
        sleep=clock.sleep,
        version_loader=loader,
    )
    loader.failures = 1
    with pytest.raises(RuntimeError):
        await service.build_skill_index(_build("k1"))
    assert await storage.get_operation(SCOPE, _operation_id("k1")) is None
    assert await storage.get_writer_lease(SCOPE) is None
    assert algorithm.plan_build_calls == []

    # Once the repository is back the same key is simply a first attempt.
    built = await service.build_skill_index(_build("k1"))
    assert built.status is OperationStatus.COMMITTED and built.result_revision == 0
    assert (await storage.get_operation(SCOPE, built.operation_id))["attempt_no"] == 1

    loader.failures = 1
    with pytest.raises(RuntimeError):
        await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    assert await storage.get_operation(SCOPE, _operation_id("u1")) is None
    assert (await _head(storage))["active_revision"] == 0
    upserted = await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    assert upserted.status is OperationStatus.COMMITTED and upserted.result_revision == 1


@pytest.mark.asyncio
async def test_missing_version_on_build_raises_before_any_operation_row_exists() -> None:
    storage = InMemorySkillGraphStorageAdapter()
    service = _service(storage, RecordingAlgorithm(), FakeClock())
    missing = (SkillVersionRef(skill_id="alpha", version_id="v-missing"),)
    with pytest.raises(SkillVersionNotFound):
        await service.build_skill_index(_build("k1", refs=missing))
    assert await storage.get_operation(SCOPE, _operation_id("k1")) is None
    assert await storage.get_writer_lease(SCOPE) is None


@pytest.mark.asyncio
async def test_settled_operations_replay_without_consulting_the_loader() -> None:
    """Preflight must not turn a replay into a loader failure once the versions left the repository."""

    class RemovableLoader:
        def __init__(self) -> None:
            self.available = True
            self.calls = 0

        async def load(self, ref, *, project_id):
            self.calls += 1
            if not self.available:
                raise SkillVersionNotFound(ref.skill_id, ref.version_id)
            return await FakeVersionLoader().load(ref, project_id=project_id)

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    loader = RemovableLoader()
    service = SkillManageService(
        algorithm=RecordingAlgorithm(),
        storage=storage,
        embedding_profile=PROFILE,
        settings=SETTINGS,
        clock=clock,
        sleep=clock.sleep,
        version_loader=loader,
    )
    built = await service.build_skill_index(_build("k1"))
    upserted = await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0]))
    stale = await service.upsert_skill_index(_upsert("u2", REFS_ALPHA_V2[0], expected_revision=0))
    assert stale.status is OperationStatus.CONFLICT
    calls_before = loader.calls

    loader.available = False
    assert await service.build_skill_index(_build("k1")) == built
    assert await service.upsert_skill_index(_upsert("u1", REFS_ALPHA_V2[0])) == upserted
    assert await service.upsert_skill_index(_upsert("u2", REFS_ALPHA_V2[0], expected_revision=0)) == stale
    assert loader.calls == calls_before
    # A new key still needs the versions and fails before any row exists.
    with pytest.raises(SkillVersionNotFound):
        await service.build_skill_index(_build("k2"))
    assert await storage.get_operation(SCOPE, _operation_id("k2")) is None


@pytest.mark.asyncio
async def test_same_key_follower_shares_the_leaders_preflight_instead_of_loading_on_its_own() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class GatedLoader:
        def __init__(self) -> None:
            self.calls = 0

        async def load(self, ref, *, project_id):
            self.calls += 1
            if self.calls > 3:
                raise RuntimeError("repository unavailable for the follower")
            return await FakeVersionLoader().load(ref, project_id=project_id)

    class SlowEmbedAlgorithm(RecordingAlgorithm):
        async def embed(self, request: SkillManageEmbedInput):
            entered.set()
            await release.wait()
            return await super().embed(request)

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    loader = GatedLoader()
    service = SkillManageService(
        algorithm=SlowEmbedAlgorithm(),
        storage=storage,
        embedding_profile=PROFILE,
        settings=SETTINGS,
        clock=clock,
        sleep=clock.sleep,
        version_loader=loader,
    )
    leader = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await entered.wait()
    follower = asyncio.ensure_future(service.build_skill_index(_build("k1")))
    await asyncio.sleep(0)
    release.set()
    assert await follower == await leader
    assert (await leader).status is OperationStatus.COMMITTED
    assert loader.calls == 3


# ------------------------------------------------------------- bug 38


@pytest.mark.asyncio
async def test_noop_whose_finalize_never_landed_converges_to_committed_even_after_the_head_moved() -> None:
    """The no-op's result is the base generation; once that is retired, repair still proves it committed."""

    class FlakyFinalizeStorage(InMemorySkillGraphStorageAdapter):
        failures = 0

        async def record_operation_result(self, scope, operation_id, **kwargs):
            if self.failures:
                self.failures -= 1
                raise ConnectionResetError("finalize unavailable")
            return await super().record_operation_result(scope, operation_id, **kwargs)

    storage = FlakyFinalizeStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    built = await service.build_skill_index(_build("k0"))
    storage.failures = 2
    with pytest.raises(SkillManageOperationPendingRepairError):
        await service.remove_skill_from_index(_remove("r1", "absent"))
    operation_id = _operation_id("r1")
    row = await storage.get_operation(SCOPE, operation_id)
    assert row["status"] == "validating" and row["current_generation_id"] == built.generation_id

    # Another key moves the head while the no-op row is still stranded.
    clock.advance(1)
    assert (await service.remove_skill_from_index(_remove("r2", "gamma"))).result_revision == 1
    assert (await _generation_statuses(storage))[built.generation_id] == "retired"

    clock.advance(61)
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert [(a.kind, a.operation_id, a.to_status) for a in report.actions] == [
        ("operation_reconciled", operation_id, "committed")
    ]
    row = await storage.get_operation(SCOPE, operation_id)
    assert row["status"] == "committed" and row["result_revision"] == 0
    replay = await service.remove_skill_from_index(_remove("r1", "absent"))
    assert replay.status is OperationStatus.COMMITTED and replay.result_revision == 0
    assert replay.generation_id == built.generation_id


# ------------------------------------------------------------- bug 39


@pytest.mark.asyncio
async def test_repair_reclaims_records_whose_generation_row_is_gone() -> None:
    """A superseded writer's late vector/snapshot writes land after its candidate was discarded."""

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    built = await service.build_skill_index(_build("k0"))
    orphan = generation_id_from_operation(_operation_id("k-dead"), 1)
    # The candidate row was discarded, then the dead writer's writes landed.
    await storage.write_snapshot(SCOPE, orphan, nodes=[], edges=[], memberships=[])
    await storage.write_embeddings(
        SCOPE, orphan, embeddings=list(await storage.load_embeddings(SCOPE, built.generation_id))
    )
    assert orphan in await storage.list_snapshot_generation_ids(SCOPE)
    assert orphan not in await _generation_statuses(storage)

    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)

    assert [(a.kind, a.generation_id, a.reason) for a in report.actions] == [
        ("candidate_discarded", orphan, "orphan records without a generation row")
    ]
    assert orphan not in await storage.list_snapshot_generation_ids(SCOPE)
    assert await storage.load_embeddings(SCOPE, orphan) == ()
    assert len(await storage.load_embeddings(SCOPE, built.generation_id)) == 3
    assert (await _head(storage))["active_generation_id"] == built.generation_id
    again = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert again.actions == ()


@pytest.mark.asyncio
async def test_orphan_sweep_re_reads_the_row_of_every_id_the_paged_listing_missed() -> None:
    """A live candidate whose generation row was skipped by the paged listing is not an orphan."""

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0
    # A live writer: fresh lease, running row, staged candidate with its snapshot written.
    operation_id, live, _ = await _stranded_row(storage, clock, "k-live", base_revision=0)
    await storage.stage_generation(
        SCOPE, live, operation_id=operation_id, attempt_no=1, base_revision=0, target_revision=1, now=clock.iso()
    )
    await storage.write_snapshot(SCOPE, live, nodes=[], edges=[], memberships=[])
    await storage.write_embeddings(
        SCOPE,
        live,
        embeddings=list(await storage.load_embeddings(SCOPE, (await _head(storage))["active_generation_id"])),
    )

    original_list = storage.list_generations

    async def listing_that_skips_the_live_row(scope, *, statuses=None):
        # An OFFSET page boundary shifted by a concurrent delete drops one row.
        return [row for row in await original_list(scope, statuses=statuses) if row["generation_id"] != live]

    storage.list_generations = listing_that_skips_the_live_row  # type: ignore[method-assign]
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    storage.list_generations = original_list  # type: ignore[method-assign]

    assert [a.kind for a in report.actions] == ["operation_skipped"]
    assert (await _generation_statuses(storage))[live] == "staging"
    assert len(await storage.load_embeddings(SCOPE, live)) == 3


# ------------------------------------------------------------- bug 48 / 51


@pytest.mark.asyncio
async def test_cancellation_landing_after_the_lease_acquire_committed_heals_by_ttl() -> None:
    """The acquire's CAS can commit and the task be cancelled before it returns; the lease then lapses by TTL.

    It is deliberately not released on the way out (Bug 51): without a per-call
    nonce on the lease row, a same-key caller cannot tell its own abandoned
    lease from one a live writer of the same operation holds.
    """

    class CancelledAfterAcquireStorage(InMemorySkillGraphStorageAdapter):
        armed = True

        async def acquire_writer_lease(self, scope, **kwargs):
            token = await super().acquire_writer_lease(scope, **kwargs)
            if self.armed:
                self.armed = False
                raise asyncio.CancelledError
            return token

    storage = CancelledAfterAcquireStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)

    with pytest.raises(asyncio.CancelledError):
        await service.build_skill_index(_build("k1"))
    assert _renewal_tasks() == []
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] == _operation_id("k1")

    conflict = await service.build_skill_index(_build("k1"))
    assert conflict.status is OperationStatus.CONFLICT and conflict.error_retryable is True
    clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
    assert (await service.build_skill_index(_build("k1"))).status is OperationStatus.COMMITTED


@pytest.mark.asyncio
async def test_same_key_caller_failing_inside_its_acquire_does_not_release_the_live_writers_lease() -> None:
    """A holds the lease between acquire and begin; B's acquire blips. A must still publish (Bug 51)."""

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    service_a = _service(storage, RecordingAlgorithm(), clock)
    service_b = _service(storage, RecordingAlgorithm(), clock)
    a_has_lease, resume = asyncio.Event(), asyncio.Event()
    original_begin = storage.begin_initial_attempt_atomically
    original_acquire = storage.acquire_writer_lease

    async def begin_after_the_gate(*args, **kwargs):
        a_has_lease.set()
        await resume.wait()
        return await original_begin(*args, **kwargs)

    async def acquire_that_blips_for_b(scope, **kwargs):
        if resume.is_set() is False and a_has_lease.is_set():
            raise ConnectionResetError("lease row read blipped")
        return await original_acquire(scope, **kwargs)

    storage.begin_initial_attempt_atomically = begin_after_the_gate  # type: ignore[method-assign]
    writer_a = asyncio.ensure_future(service_a.build_skill_index(_build("k1")))
    await a_has_lease.wait()
    storage.acquire_writer_lease = acquire_that_blips_for_b  # type: ignore[method-assign]
    with pytest.raises(ConnectionResetError):
        await service_b.build_skill_index(_build("k1"))
    storage.acquire_writer_lease = original_acquire  # type: ignore[method-assign]

    resume.set()
    result = await writer_a
    assert result.status is OperationStatus.COMMITTED
    assert (await storage.get_operation(SCOPE, _operation_id("k1")))["status"] == "committed"


@pytest.mark.asyncio
async def test_renewal_task_ends_even_when_its_cancellation_was_consumed_inside_storage() -> None:
    """psycopg_pool can swallow a cancellation inside a checkout; the renewal loop must still notice it."""

    storage = InMemorySkillGraphStorageAdapter()
    clock = FakeClock()
    token = await storage.acquire_writer_lease(SCOPE, owner_operation_id="op-1", ttl_seconds=60.0, now=clock.iso())
    entered = asyncio.Event()
    original_renew = storage.renew_writer_lease

    async def renew_swallowing_one_cancellation(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        return await original_renew(*args, **kwargs)

    storage.renew_writer_lease = renew_swallowing_one_cancellation  # type: ignore[method-assign]
    lease = _WriterLease(
        storage=storage,
        scope=SCOPE,
        owner_operation_id="op-1",
        token=token,
        ttl_seconds=60.0,
        renew_interval_seconds=0.0,
        now=clock.iso,
        sleep=asyncio.sleep,
    )
    lease.start()
    await entered.wait()

    await asyncio.wait_for(lease.release(), timeout=2.0)
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    assert _renewal_tasks() == []


# ------------------------------------------------------------- bug 52


@pytest.mark.asyncio
async def test_repair_reports_a_discard_whose_acknowledgement_was_lost_as_discarded() -> None:
    """The candidate row goes last; a discard that landed but raised on the way back is not 'deferred'."""

    class DiscardLosesAckStorage(InMemorySkillGraphStorageAdapter):
        publish_failures = 0
        lose_ack = False

        async def publish_generation(self, scope, generation_id, **kwargs):
            if self.publish_failures:
                self.publish_failures -= 1
                raise ConnectionResetError("transaction rolled back")
            return await super().publish_generation(scope, generation_id, **kwargs)

        async def discard_generation(self, scope, generation_id):
            if self.lose_ack:
                self.lose_ack = False
                raise ConnectionResetError("vector store unavailable")
            await super().discard_generation(scope, generation_id)
            if generation_id == generation_id_from_operation(_operation_id("k1"), 1):
                raise ConnectionResetError("acknowledgement lost")

    storage = DiscardLosesAckStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0
    storage.publish_failures, storage.lose_ack = 1, True
    with pytest.raises(ConnectionResetError):
        await service.build_skill_index(_build("k1"))
    gen1 = generation_id_from_operation(_operation_id("k1"), 1)
    assert (await _generation_statuses(storage))[gen1] == "failed"

    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)

    assert [(a.kind, a.generation_id) for a in report.actions] == [("candidate_discarded", gen1)]
    assert report.warnings == ()
    assert gen1 not in await _generation_statuses(storage)


# ------------------------------------------------------------- bug 54


@pytest.mark.asyncio
async def test_commit_whose_acknowledgement_was_lost_still_runs_the_deferred_gc() -> None:
    """`record_operation_result` landing without its ack replays our own commit; the retired generation is still collected."""

    class AckLostOnceStorage(InMemorySkillGraphStorageAdapter):
        lose_ack = False

        async def record_operation_result(self, scope, operation_id, **kwargs):
            committed = await super().record_operation_result(scope, operation_id, **kwargs)
            if self.lose_ack:
                self.lose_ack = False
                raise ConnectionResetError("acknowledgement lost")
            return committed

    storage = AckLostOnceStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock, settings=_EAGER_GC)
    first = await service.build_skill_index(_build("k0"))
    clock.advance(10)
    storage.lose_ack = True
    second = await service.build_skill_index(_build("k1"))

    assert second.status is OperationStatus.COMMITTED and second.result_revision == 1
    assert (await storage.get_operation(SCOPE, second.operation_id))["status"] == "committed"
    # With keep=0 the generation retired by this publish is gone before the call returns.
    assert first.generation_id not in await _generation_statuses(storage)
    assert _renewal_tasks() == []


# ------------------------------------------------------------- bugs 55–57


@pytest.mark.asyncio
async def test_publish_outcome_unknown_follows_the_row_when_a_same_key_retry_already_settled_it() -> None:
    """A's publish stalls past the TTL, B reconciles attempt 1 and commits attempt 2, then A's publish raises."""

    clock = FakeClock()

    class StalledPublishStorage(InMemorySkillGraphStorageAdapter):
        armed = True

        async def publish_generation(self, scope, generation_id, **kwargs):
            if not self.armed:
                return await super().publish_generation(scope, generation_id, **kwargs)
            self.armed = False
            clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
            other = _service(self, RecordingAlgorithm(), clock)
            committed = await other.build_skill_index(_build("k1"))
            assert committed.status is OperationStatus.COMMITTED and committed.generation_id != generation_id
            raise RuntimeError("connection reset after the stall")

    storage = StalledPublishStorage()
    service = _service(storage, RecordingAlgorithm(), clock)
    # A is a stalled process: its renewal task must not tick while B takes over.
    service._sleep = lambda seconds: asyncio.Event().wait()

    result = await service.build_skill_index(_build("k1"))

    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 2
    assert result.status is OperationStatus.COMMITTED and result.generation_id == row["current_generation_id"]
    assert _renewal_tasks() == []


@pytest.mark.asyncio
async def test_publish_outcome_unknown_on_a_row_still_in_this_attempt_is_still_pending_repair() -> None:
    """Nobody else touched the row: the caller must not be told anything more definite than 'requires repair'."""

    class PublishRaisesStorage(InMemorySkillGraphStorageAdapter):
        async def publish_generation(self, scope, generation_id, **kwargs):
            raise RuntimeError("acknowledgement lost")

        async def mark_generation_failed(self, scope, generation_id, **kwargs):
            raise RuntimeError("store down")

        async def get_generation(self, scope, generation_id):
            raise RuntimeError("store down")

    storage = PublishRaisesStorage()
    service = _service(storage, RecordingAlgorithm(), FakeClock())

    with pytest.raises(SkillManageOperationPendingRepairError) as excinfo:
        await service.build_skill_index(_build("k1"))
    assert excinfo.value.reason == "publish_outcome_unknown"
    assert (await storage.get_operation(SCOPE, _operation_id("k1")))["status"] == "validating"


@pytest.mark.asyncio
async def test_conflict_bookkeeping_that_never_landed_is_reported_as_a_retryable_writer_conflict() -> None:
    """The fence rejected the attempt but the conflict CAS itself blipped: the row is still in flight, so is the key."""

    clock = FakeClock()

    class ConflictBlipStorage(InMemorySkillGraphStorageAdapter):
        blips = 1
        intruded = False

        async def write_embeddings(self, scope, generation_id, *, embeddings):
            await super().write_embeddings(scope, generation_id, embeddings=embeddings)
            if self.intruded:
                return
            self.intruded = True
            # Another graph writer takes the lease over after it lapsed, so publish is fenced.
            clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
            await self.acquire_writer_lease(scope, owner_operation_id="intruder", ttl_seconds=1.0, now=clock.iso())

        async def conflict_operation(self, scope, operation_id, **kwargs):
            if self.blips:
                self.blips -= 1
                raise ConnectionError("storage blip during bookkeeping")
            return await super().conflict_operation(scope, operation_id, **kwargs)

    storage = ConflictBlipStorage()
    service = _service(storage, RecordingAlgorithm(), clock)

    first = await service.build_skill_index(_build("k1"))
    assert first.status is OperationStatus.CONFLICT
    assert first.error_code == "SKILL_MANAGE_WRITER_CONFLICT" and first.error_retryable is True
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row["status"] == "validating"

    # Crash recovery settles the row as a retryable failure and the same key then commits.
    clock.advance(2)
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert [(a.kind, a.to_status) for a in report.actions if a.kind == "operation_reconciled"] == [
        ("operation_reconciled", "failed")
    ]
    assert (await storage.get_operation(SCOPE, _operation_id("k1")))["error_retryable"] is True
    clock.advance(2)
    second = await service.build_skill_index(_build("k1"))
    assert second.status is OperationStatus.COMMITTED and second.operation_id == first.operation_id


@pytest.mark.asyncio
async def test_renewal_task_stops_at_the_next_iteration_even_while_renewals_keep_failing() -> None:
    """A consumed cancellation followed by an outage must not keep ``stop()`` waiting for a renew to land."""

    class OutageStorage(InMemorySkillGraphStorageAdapter):
        swallow_once = True
        outage = 3
        calls = 0

        async def renew_writer_lease(self, scope, **kwargs):
            self.calls += 1
            if self.swallow_once:
                self.swallow_once = False
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    pass
                raise ConnectionError("dead connection after the consumed cancellation")
            if self.outage:
                self.outage -= 1
                raise ConnectionError("store still down")
            return await super().renew_writer_lease(scope, **kwargs)

    storage = OutageStorage()
    token = await storage.acquire_writer_lease(SCOPE, owner_operation_id="op", ttl_seconds=60.0)
    lease = _WriterLease(
        storage=storage,
        scope=SCOPE,
        owner_operation_id="op",
        token=token,
        ttl_seconds=60.0,
        renew_interval_seconds=20.0,
        now=lambda: "2026-09-05T00:00:00+00:00",
        sleep=lambda seconds: asyncio.sleep(0),
    )
    lease.start()
    await asyncio.sleep(0.01)
    await asyncio.wait_for(lease.stop(), timeout=2.0)
    assert storage.calls == 1
    assert _renewal_tasks() == []


# ------------------------------------------------------------- bug 58


@pytest.mark.asyncio
async def test_noop_planned_against_a_base_collected_after_the_lease_lapsed_is_a_conflict_not_a_commit() -> None:
    """A writer stalled past its TTL must not commit a 'nothing to do' against a base another writer retired."""

    clock = FakeClock()

    class StalledLoadStorage(InMemorySkillGraphStorageAdapter):
        stall_once = False

        async def load_snapshot(self, scope, generation_id):
            if self.stall_once:
                self.stall_once = False
                # W stalls between reading the head and loading the base; its lease
                # lapses, another writer publishes rev 1 and keep=0 GC collects the base.
                clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
                other = _service(self, RecordingAlgorithm(), clock, settings=_EAGER_GC)
                published = await other.upsert_skill_index(_upsert("other-key", REFS_ALPHA_V2[0]))
                assert published.status is OperationStatus.COMMITTED and published.result_revision == 1
                assert await self.get_generation(scope, generation_id) is None
            return await super().load_snapshot(scope, generation_id)

    storage = StalledLoadStorage()
    service = _service(storage, RecordingAlgorithm(), clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0

    storage.stall_once = True
    stalled = await service.remove_skill_from_index(_remove("rm-gamma", "gamma"))

    assert stalled.status is OperationStatus.CONFLICT and stalled.error_code == "SKILL_MANAGE_WRITER_CONFLICT"
    row = await storage.get_operation(SCOPE, _operation_id("rm-gamma"))
    assert row is not None and row["status"] == "conflict"
    head = await _head(storage)
    assert head["active_revision"] == 1
    assert "skill:gamma" in {
        n.graph_node_id for n in (await storage.load_snapshot(SCOPE, head["active_generation_id"])).nodes
    }
    assert _renewal_tasks() == []


# ------------------------------------------------------------- bug 59


class _StalledReconcileStorage(InMemorySkillGraphStorageAdapter):
    """A repairer that stalls past its TTL inside its reconcile, just before the settle CAS."""

    def __init__(self, clock: FakeClock) -> None:
        super().__init__()
        self.clock = clock
        self.stalled = asyncio.Event()
        self.stall_next_begin_validation = False
        self.intrude: Callable[[], Awaitable[None]] | None = None
        self.intrude_before_read: Callable[[], Awaitable[None]] | None = None
        self.die_after_reserve = False

    async def begin_validation(self, scope, operation_id, **kwargs):
        if self.stall_next_begin_validation:
            self.stall_next_begin_validation = False
            self.stalled.set()
            await asyncio.Event().wait()  # the writer process is frozen from here on
        return await super().begin_validation(scope, operation_id, **kwargs)

    async def reserve_retry_attempt(self, scope, operation_id, *, now=None):
        reserved = await super().reserve_retry_attempt(scope, operation_id, now=now)
        if self.die_after_reserve:
            raise RuntimeError("reserver dies between its two CAS steps")
        return reserved

    async def transition_operation(self, scope, operation_id, *, expected_status, changes, **kwargs):
        if self.intrude is not None and changes.get("status") in ("failed", "conflict", "committed"):
            intrude, self.intrude = self.intrude, None
            self.clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
            await intrude()
        return await super().transition_operation(
            scope, operation_id, expected_status=expected_status, changes=changes, **kwargs
        )

    async def repair_crash_recovery(self, scope, operation_id, **kwargs):
        # The other half of the window: the stall lands before the adapter's
        # own first read, so the reconcile reads the reservation itself.
        if self.intrude_before_read is not None:
            intrude, self.intrude_before_read = self.intrude_before_read, None
            self.clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
            await intrude()
        return await super().repair_crash_recovery(scope, operation_id, **kwargs)


async def _strand_a_writer(storage: _StalledReconcileStorage, clock: FakeClock, key: str) -> asyncio.Task:
    """Start a build that freezes right before ``begin_validation`` and never renews its lease again."""

    stranded = _service(storage, RecordingAlgorithm(), clock)
    stranded._sleep = lambda seconds: asyncio.Event().wait()
    storage.stall_next_begin_validation = True
    writer = asyncio.ensure_future(stranded.build_skill_index(_build(key)))
    await storage.stalled.wait()
    clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
    return writer


async def _settle(writer: asyncio.Task) -> None:
    writer.cancel()
    with contextlib.suppress(BaseException):
        await writer


@pytest.mark.asyncio
async def test_repair_reports_a_row_that_moved_during_its_reconcile_as_skipped() -> None:
    """A newer attempt's committed row is not this reconcile's outcome, and its candidate was never published."""

    clock = FakeClock()
    storage = _StalledReconcileStorage(clock)
    writer = await _strand_a_writer(storage, clock, "k1")
    operation_id = _operation_id("k1")
    retry = _service(storage, RecordingAlgorithm(), clock)

    async def same_key_retry_commits_attempt_2() -> None:
        committed = await retry.build_skill_index(_build("k1"))
        assert committed.status is OperationStatus.COMMITTED and committed.result_revision == 0

    storage.intrude = same_key_retry_commits_attempt_2
    report = await _service(storage, RecordingAlgorithm(), clock).repair_skill_index(
        project_id="p1", graph_id="g1", stale_candidate_age_seconds=0.0
    )
    await _settle(writer)

    assert [(a.kind, a.from_status, a.reason) for a in report.actions] == [
        ("operation_skipped", "running", "operation moved during the reconcile")
    ]
    assert report.warnings == ()
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "committed" and row["attempt_no"] == 2
    assert row["current_generation_id"] == generation_id_from_operation(operation_id, 2)


@pytest.mark.asyncio
async def test_stale_reconcile_cannot_settle_a_row_that_moved_into_a_retry_reservation() -> None:
    """A reservation keeps the attempt fields; a settle landing on it would end the key in a terminal conflict."""

    clock = FakeClock()
    storage = _StalledReconcileStorage(clock)
    writer = await _strand_a_writer(storage, clock, "k1")
    operation_id = _operation_id("k1")
    retry = _service(storage, RecordingAlgorithm(), clock)

    async def same_key_retry_reserves_and_dies_then_another_key_creates_the_head() -> None:
        storage.die_after_reserve = True
        with pytest.raises(RuntimeError):
            await retry.build_skill_index(_build("k1"))
        storage.die_after_reserve = False
        clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
        other = await _service(storage, RecordingAlgorithm(), clock).build_skill_index(_build("k2"))
        assert other.status is OperationStatus.COMMITTED and other.result_revision == 0

    storage.intrude = same_key_retry_reserves_and_dies_then_another_key_creates_the_head
    report = await _service(storage, RecordingAlgorithm(), clock).repair_skill_index(
        project_id="p1", graph_id="g1", stale_candidate_age_seconds=0.0
    )
    await _settle(writer)

    assert [(a.kind, a.reason) for a in report.actions if a.kind != "candidate_discarded"] == [
        ("operation_skipped", "operation moved during the reconcile")
    ]
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "pending_repair"
    assert row["pending_repair_reason"] == "retry_attempt_reservation" and row["attempt_no"] == 1
    # The same key resumes its reservation over the other key's head.
    clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
    resumed = await retry.build_skill_index(_build("k1"))
    assert resumed.status is OperationStatus.COMMITTED and resumed.result_revision == 1
    assert (await storage.get_operation(SCOPE, operation_id))["attempt_no"] == 2


@pytest.mark.asyncio
async def test_a_reconcile_that_reads_the_reservation_itself_leaves_it_alone() -> None:
    """The stall may land before the adapter's own read: a reservation is never a stranded attempt to settle."""

    clock = FakeClock()
    storage = _StalledReconcileStorage(clock)
    writer = await _strand_a_writer(storage, clock, "k1")
    operation_id = _operation_id("k1")
    retry = _service(storage, RecordingAlgorithm(), clock)

    async def same_key_retry_reserves_and_dies_then_another_key_creates_the_head() -> None:
        storage.die_after_reserve = True
        with pytest.raises(RuntimeError):
            await retry.build_skill_index(_build("k1"))
        storage.die_after_reserve = False
        clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
        other = await _service(storage, RecordingAlgorithm(), clock).build_skill_index(_build("k2"))
        assert other.status is OperationStatus.COMMITTED and other.result_revision == 0

    storage.intrude_before_read = same_key_retry_reserves_and_dies_then_another_key_creates_the_head
    report = await _service(storage, RecordingAlgorithm(), clock).repair_skill_index(
        project_id="p1", graph_id="g1", stale_candidate_age_seconds=0.0
    )
    await _settle(writer)

    assert [(a.kind, a.reason) for a in report.actions if a.kind != "candidate_discarded"] == [
        ("operation_skipped", "operation moved during the reconcile")
    ]
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "pending_repair"
    assert row["pending_repair_reason"] == "retry_attempt_reservation"
    # The only reconcile that landed is the same-key retry's own (attempt 1 -> failed).
    assert row["last_repair_action"] == "crash_recovery:failed"
    clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
    resumed = await retry.build_skill_index(_build("k1"))
    assert resumed.status is OperationStatus.COMMITTED and resumed.result_revision == 1


# ------------------------------------------------------------- bug 60


class _LeftoverCandidateStorage(InMemorySkillGraphStorageAdapter):
    """Attempt 1 fails after staging and its discard blips, leaving the candidate behind."""

    fail_embed_write = True
    discard_raises_once = True
    die_after_reserve = False
    on_reserve: Callable[[], Awaitable[None]] | None = None

    async def write_embeddings(self, scope, generation_id, *, embeddings):
        if self.fail_embed_write:
            self.fail_embed_write = False
            raise ConnectionError("vector store blip")
        return await super().write_embeddings(scope, generation_id, embeddings=embeddings)

    async def discard_generation(self, scope, generation_id):
        if self.discard_raises_once:
            self.discard_raises_once = False
            raise ConnectionError("discard blip after the row was marked failed")
        return await super().discard_generation(scope, generation_id)

    async def reserve_retry_attempt(self, scope, operation_id, *, now=None):
        reserved = await super().reserve_retry_attempt(scope, operation_id, now=now)
        if self.die_after_reserve:
            raise RuntimeError("reserver dies between its two CAS steps")
        on_reserve, self.on_reserve = self.on_reserve, None
        if on_reserve is not None:
            await on_reserve()
        return reserved


async def _failed_attempt_with_a_leftover_candidate(
    storage: _LeftoverCandidateStorage, service: SkillManageService, clock: FakeClock
) -> tuple[str, str]:
    operation_id = _operation_id("k1")
    with pytest.raises(ConnectionError):
        await service.build_skill_index(_build("k1"))
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "failed"
    assert row["error_retryable"] is True
    candidate = generation_id_from_operation(operation_id, 1)
    assert (await _generation_statuses(storage))[candidate] == "staging"
    return operation_id, candidate


@pytest.mark.asyncio
async def test_a_dead_retry_reservation_does_not_pin_its_failed_attempts_candidate() -> None:
    """The reservation names the failed attempt's candidate, which no attempt can ever publish again."""

    clock = FakeClock()
    storage = _LeftoverCandidateStorage()
    service = _service(storage, RecordingAlgorithm(), clock)
    operation_id, candidate = await _failed_attempt_with_a_leftover_candidate(storage, service, clock)

    storage.die_after_reserve = True
    with pytest.raises(RuntimeError):
        await service.build_skill_index(_build("k1"))
    storage.die_after_reserve = False
    clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)

    report = await service.repair_skill_index(project_id="p1", graph_id="g1", stale_candidate_age_seconds=0.0)
    assert [(a.kind, a.generation_id) for a in report.actions if a.kind == "candidate_discarded"] == [
        ("candidate_discarded", candidate)
    ]
    assert candidate not in await _generation_statuses(storage)
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "pending_repair"
    assert row["pending_repair_reason"] == "retry_attempt_reservation" and row["attempt_no"] == 1
    # The row is still the same key's to resume.
    resumed = await service.build_skill_index(_build("k1"))
    assert resumed.status is OperationStatus.COMMITTED and resumed.result_revision == 0


@pytest.mark.asyncio
async def test_the_sweep_between_a_live_reservers_two_cas_steps_collects_only_the_previous_candidate() -> None:
    clock = FakeClock()
    storage = _LeftoverCandidateStorage()
    service = _service(storage, RecordingAlgorithm(), clock)
    operation_id, candidate = await _failed_attempt_with_a_leftover_candidate(storage, service, clock)
    reports = []

    async def sweep_while_reserved() -> None:
        reports.append(
            await _service(storage, RecordingAlgorithm(), clock).repair_skill_index(
                project_id="p1", graph_id="g1", stale_candidate_age_seconds=0.0
            )
        )

    storage.on_reserve = sweep_while_reserved
    resumed = await service.build_skill_index(_build("k1"))

    assert resumed.status is OperationStatus.COMMITTED and resumed.result_revision == 0
    assert (await storage.get_operation(SCOPE, operation_id))["attempt_no"] == 2
    assert [(a.kind, a.generation_id) for a in reports[0].actions if a.kind == "candidate_discarded"] == [
        ("candidate_discarded", candidate)
    ]
    assert await _generation_statuses(storage) == {generation_id_from_operation(operation_id, 2): "active"}


# ---------------------------------------- cancellation inside the begin CAS (design)


@pytest.mark.asyncio
async def test_cancellation_landing_inside_the_begin_cas_is_converged_like_a_crash() -> None:
    """``cancelled`` is recorded once the attempt exists; a cancel in ``begin_*`` leaves a crash-like row.

    The lease is released at once, so the next same-key call (or the
    in-process follower) crash-recovers the row and performs the write.
    """

    clock = FakeClock()

    class CancelledAfterBeginStorage(InMemorySkillGraphStorageAdapter):
        cancel_once = True

        async def begin_initial_attempt_atomically(self, scope, operation_id, **kwargs):
            began = await super().begin_initial_attempt_atomically(scope, operation_id, **kwargs)
            if self.cancel_once:
                self.cancel_once = False
                raise asyncio.CancelledError
            return began

    storage = CancelledAfterBeginStorage()
    service = _service(storage, RecordingAlgorithm(), clock)
    operation_id = _operation_id("k1")

    with pytest.raises(asyncio.CancelledError):
        await service.build_skill_index(_build("k1"))
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "running" and row["attempt_no"] == 1
    assert (await storage.get_writer_lease(SCOPE))["owner_operation_id"] is None
    assert _renewal_tasks() == []

    committed = await service.build_skill_index(_build("k1"))
    assert committed.status is OperationStatus.COMMITTED and committed.result_revision == 0
    assert (await storage.get_operation(SCOPE, operation_id))["attempt_no"] == 2


# ------------------------------------ round-12 mutation audit: guards that had no test


@pytest.mark.asyncio
async def test_a_vector_write_that_silently_drops_rows_fails_validation() -> None:
    """The read-back count is what catches a store that acknowledged fewer vectors than it was given."""

    class DroppingVectorStorage(InMemorySkillGraphStorageAdapter):
        async def write_embeddings(self, scope, generation_id, *, embeddings):
            await super().write_embeddings(scope, generation_id, embeddings=embeddings[:-1])

    storage = DroppingVectorStorage()
    service = _service(storage, RecordingAlgorithm(), FakeClock())
    with pytest.raises(SkillManageValidationError):
        await service.build_skill_index(_build("k1"))
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["status"] == "failed" and row["error_retryable"] is False
    assert await _generation_statuses(storage) == {}
    assert await storage.get_graph_head(SCOPE) is None


@pytest.mark.asyncio
async def test_a_superseded_attempt_follows_a_row_the_newer_attempt_cancelled() -> None:
    """A fenced-out attempt whose key was since cancelled raises the typed error, like any same-key call would."""

    clock = FakeClock()

    class StalledThenCancelledStorage(InMemorySkillGraphStorageAdapter):
        armed = True

        async def begin_validation(self, scope, operation_id, **kwargs):
            if self.armed:
                self.armed = False
                # Writer A stalls past its TTL; a same-key retry takes over, opens attempt 2 and is cancelled.
                clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
                other = _service(self, RecordingAlgorithm(), clock)
                self.cancel_at_stage = True
                with pytest.raises(asyncio.CancelledError):
                    await other.build_skill_index(_build("k1"))
            return await super().begin_validation(scope, operation_id, **kwargs)

        cancel_at_stage = False

        async def stage_generation(self, scope, generation_id, **kwargs):
            await super().stage_generation(scope, generation_id, **kwargs)
            if self.cancel_at_stage:
                self.cancel_at_stage = False
                raise asyncio.CancelledError

    storage = StalledThenCancelledStorage()
    service = _service(storage, RecordingAlgorithm(), clock)
    with pytest.raises(SkillManageOperationCancelledError):
        await service.build_skill_index(_build("k1"))
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["status"] == "cancelled" and row["attempt_no"] == 2
    assert _renewal_tasks() == []
    # The cancel landed after attempt 2's stage INSERT but before it returned;
    # the candidate is discarded by the cancel bookkeeping all the same (Bug 61).
    assert await _generation_statuses(storage) == {}


@pytest.mark.asyncio
async def test_a_stale_candidate_whose_discard_fails_and_remains_is_reported_deferred() -> None:
    """Only a discard that left no row behind counts as discarded; one that raised and left the row is deferred."""

    class DiscardFailsStorage(InMemorySkillGraphStorageAdapter):
        publish_failures = 0
        discard_failures = 0

        async def publish_generation(self, scope, generation_id, **kwargs):
            if self.publish_failures:
                self.publish_failures -= 1
                raise ConnectionResetError("transaction rolled back")
            return await super().publish_generation(scope, generation_id, **kwargs)

        async def discard_generation(self, scope, generation_id):
            if self.discard_failures:
                self.discard_failures -= 1
                raise ConnectionResetError("vector store unavailable")
            await super().discard_generation(scope, generation_id)

    storage = DiscardFailsStorage()
    clock = FakeClock()
    service = _service(storage, RecordingAlgorithm(), clock)
    assert (await service.build_skill_index(_build("k0"))).result_revision == 0
    storage.publish_failures, storage.discard_failures = 1, 1
    with pytest.raises(ConnectionResetError):
        await service.build_skill_index(_build("k1"))
    gen1 = generation_id_from_operation(_operation_id("k1"), 1)
    assert (await _generation_statuses(storage))[gen1] == "failed"

    storage.discard_failures = 1
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert report.actions == ()
    assert [(w["code"], w["details"]["generation_id"]) for w in report.warnings] == [
        ("SKILL_MANAGE_REPAIR_DEFERRED", gen1)
    ]
    assert (await _generation_statuses(storage))[gen1] == "failed"
    # The next pass finishes the job.
    report = await service.repair_skill_index(project_id=SCOPE.project_id, graph_id=SCOPE.graph_id)
    assert [(a.kind, a.generation_id) for a in report.actions] == [("candidate_discarded", gen1)]
    assert gen1 not in await _generation_statuses(storage)


@pytest.mark.asyncio
async def test_dry_run_planned_against_a_base_collected_mid_load_re_plans_from_the_fresh_head() -> None:
    """A preview never reports the empty snapshot a collected generation reads back as."""

    storage = _HookedStorage()
    clock = FakeClock()
    writer = _service(storage, RecordingAlgorithm(), clock, settings=_EAGER_GC)
    built = await writer.build_skill_index(_build("k1"))
    assert built.result_revision == 0 and built.node_count == 3

    async def publish_and_collect_underneath() -> None:
        clock.advance(1)
        assert (await writer.remove_skill_from_index(_remove("r1", "gamma"))).result_revision == 1
        assert built.generation_id not in await _generation_statuses(storage)

    storage.before_load_snapshot = publish_and_collect_underneath
    preview = await writer.remove_skill_from_index(_remove(None, "beta", commit_policy="dry_run"))
    # Planned against the fresh head (alpha, beta): removing beta leaves one node, not zero.
    assert preview.status is OperationStatus.COMMITTED and preview.node_count == 1
    assert (await _head(storage))["active_revision"] == 1


@pytest.mark.asyncio
async def test_a_failure_delivered_after_the_stage_insert_landed_still_discards_the_candidate() -> None:
    """The candidate row exists as soon as the INSERT commits, whether or not the call returned (Bug 61)."""

    class StageAckLostStorage(InMemorySkillGraphStorageAdapter):
        async def stage_generation(self, scope, generation_id, **kwargs):
            await super().stage_generation(scope, generation_id, **kwargs)
            raise ConnectionResetError("acknowledgement lost")

    storage = StageAckLostStorage()
    service = _service(storage, RecordingAlgorithm(), FakeClock())
    with pytest.raises(ConnectionResetError):
        await service.build_skill_index(_build("k1"))
    row = await storage.get_operation(SCOPE, _operation_id("k1"))
    assert row is not None and row["status"] == "failed"
    assert await _generation_statuses(storage) == {}
    assert _renewal_tasks() == []


@pytest.mark.asyncio
async def test_repair_reports_a_claim_lost_to_the_writers_own_settle_as_skipped() -> None:
    """A stale writer's fenced bookkeeping landing during the takeover is not this pass's reconcile."""

    clock = FakeClock()

    class WriterSettlesDuringTakeoverStorage(InMemorySkillGraphStorageAdapter):
        writer_token: int | None = None

        async def renew_writer_lease(self, scope, **kwargs):
            token, self.writer_token = self.writer_token, None
            if token is not None:
                # The writer was slow, not dead: its own failure bookkeeping lands now.
                assert await self.fail_operation(
                    scope,
                    _operation_id("k1"),
                    error_code="SKILL_MANAGE_OPERATION_FAILED",
                    error_message="RuntimeError",
                    error_retryable=True,
                    fencing_token=token,
                    now=clock.iso(),
                )
            return await super().renew_writer_lease(scope, **kwargs)

    storage = WriterSettlesDuringTakeoverStorage()
    operation_id, generation_id, token = await _stranded_row(storage, clock, "k1", base_revision=None)
    clock.advance(SETTINGS.writer_lease_ttl_seconds + 1)
    storage.writer_token = token

    report = await _service(storage, RecordingAlgorithm(), clock).repair_skill_index(
        project_id="p1", graph_id="g1", stale_candidate_age_seconds=0.0
    )

    assert [(a.kind, a.from_status, a.reason) for a in report.actions if a.operation_id == operation_id] == [
        ("operation_skipped", "running", "crash-recovery claim lost")
    ]
    row = await storage.get_operation(SCOPE, operation_id)
    assert row is not None and row["status"] == "failed" and row.get("repair_count") in (None, 0)


@pytest.mark.asyncio
async def test_sweep_leaves_a_candidate_whose_owner_was_live_at_the_listing_alone() -> None:
    """A live writer's ready candidate is never reported discarded, even if it publishes before the re-read."""

    clock = FakeClock()

    class PublishesDuringSweepStorage(InMemorySkillGraphStorageAdapter):
        publish_on_listing: Callable[[], Awaitable[None]] | None = None

        async def list_generations(self, scope, *, statuses=None):
            listed = await super().list_generations(scope, statuses=statuses)
            hook, self.publish_on_listing = self.publish_on_listing, None
            if hook is not None:
                # The candidate was listed as ready; its writer publishes before the per-candidate re-read.
                await hook()
            return listed

    storage = PublishesDuringSweepStorage()
    operation_id, generation_id, token = await _stranded_row(storage, clock, "k1", base_revision=None)
    await storage.stage_generation(
        SCOPE,
        generation_id,
        operation_id=operation_id,
        attempt_no=1,
        base_revision=None,
        target_revision=0,
        now=clock.iso(),
    )
    await storage.write_snapshot(SCOPE, generation_id, nodes=[], edges=[], memberships=[])
    assert await storage.mark_generation_ready(
        SCOPE, generation_id, node_count=0, edge_count=0, vector_count=0, counts_hash="h", now=clock.iso()
    )
    assert await storage.begin_validation(SCOPE, operation_id, fencing_token=token, now=clock.iso())
    clock.advance(1)  # older than stale_candidate_age_seconds=0, lease (60 s) still alive

    async def writer_publishes_and_commits() -> None:
        assert await storage.publish_generation(
            SCOPE,
            generation_id,
            expected_revision=None,
            new_revision=0,
            owner_operation_id=operation_id,
            fencing_token=token,
            now=clock.iso(),
        )
        assert await storage.record_operation_result(
            SCOPE,
            operation_id,
            base_revision=None,
            result_revision=0,
            generation_id=generation_id,
            result_payload={},
            fencing_token=token,
            now=clock.iso(),
        )

    storage.publish_on_listing = writer_publishes_and_commits
    report = await _service(storage, RecordingAlgorithm(), clock).repair_skill_index(
        project_id="p1", graph_id="g1", stale_candidate_age_seconds=0.0
    )

    assert [(a.kind, a.reason) for a in report.actions] == [("operation_skipped", "writer lease is alive")]
    assert (await _head(storage))["active_generation_id"] == generation_id
    assert (await _generation_statuses(storage))[generation_id] == "active"


@pytest.mark.asyncio
async def test_await_switch_propagates_an_externally_cancelled_switch_without_spinning() -> None:
    """Cancelling the switch task itself (loop shutdown) must end the caller, not busy-loop on a cancelled task."""

    started = asyncio.Event()

    async def switch():
        started.set()
        await asyncio.Event().wait()

    outer = asyncio.ensure_future(SkillManageService._await_switch(switch()))
    await started.wait()
    inner = next(t for t in asyncio.all_tasks() if t is not outer and t is not asyncio.current_task())
    inner.cancel()
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(outer, timeout=2.0)
    assert outer.cancelled()
