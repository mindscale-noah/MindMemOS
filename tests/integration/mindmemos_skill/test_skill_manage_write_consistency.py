"""Real PG write-consistency suite: memberships, generation/operation state machines, lease, GC.

Drives the 3-skill pool of ``test_skill_manage_embedding_path`` through the
``skill_manage`` service on real PostgreSQL/pgvector and checks what each
write leaves behind physically: one membership row per skill node, exactly
one ``active`` generation with the rest ``retired``, operation rows that walk
``pending -> running -> validating -> committed`` (or ``failed`` /
``conflict``), a writer lease that is held during the write and released
after it, fencing of stale writers, no half switch on an injected publish
failure, retryable failures retried as new attempts, stranded writers
reconciled, and retired generations collected past retention.

Requires ``MINDMEMOS_SKILL_TEST_PG_DSN``. Skipped without it.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

_TESTS_MINDMEMOS = Path(__file__).parents[2] / "mindmemos_skill"
if str(_TESTS_MINDMEMOS) not in sys.path:
    sys.path.insert(0, str(_TESTS_MINDMEMOS))

from mindmemos_skill.algos.skill_manage.errors import embedding_provider_error  # noqa: E402
from mindmemos_skill.application.skill_manage.models import (  # noqa: E402
    SkillManageBuildRequest,
    SkillManageServiceSettings,
    SkillManageUpsertRequest,
    compute_skill_manage_request_hash,
)
from mindmemos_skill.infra.skill_manage import generation_id_from_operation  # noqa: E402
from test_skill_manage_embedding_path import (  # noqa: E402
    GRAPH_ID,
    POOL_ORDER,
    _make_service,
    _Pool,
    _register_pool,
    _vector_rows,
    requires_pg,  # noqa: F401
)

pytestmark = pytest.mark.integration_pgvector


def _query(dsn: str, statement: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(statement, params)
            return [dict(row) for row in cursor.fetchall()]


def _memberships(dsn: str, generation_id: str) -> dict[str, str]:
    rows = _query(
        dsn,
        "SELECT skill_id, version_id, graph_node_id FROM skill_manage_memberships WHERE generation_id = %s",
        (generation_id,),
    )
    assert all(row["graph_node_id"] == f"skill:{row['skill_id']}" for row in rows)
    return {row["skill_id"]: row["version_id"] for row in rows}


def _generations(dsn: str, project_id: str) -> dict[str, dict[str, Any]]:
    rows = _query(
        dsn,
        "SELECT generation_id, status, node_count, edge_count, vector_count, counts_hash, ready_at, activated_at, "
        "retired_at FROM skill_manage_generations WHERE _scope ->> 'project_id' = %s",
        (project_id,),
    )
    return {row["generation_id"]: row for row in rows}


def _operation(dsn: str, operation_id: str) -> dict[str, Any]:
    rows = _query(dsn, "SELECT * FROM skill_manage_operations WHERE operation_id = %s", (operation_id,))
    assert len(rows) == 1
    return rows[0]


def _lease(dsn: str, project_id: str) -> dict[str, Any] | None:
    rows = _query(
        dsn,
        "SELECT owner_operation_id, fencing_token, expires_at FROM skill_manage_writer_leases "
        "WHERE _scope ->> 'project_id' = %s AND graph_id = %s",
        (project_id, GRAPH_ID),
    )
    return rows[0] if rows else None


async def _build_upsert_remove(pool: _Pool, tmp_path: Path):
    built = await pool.build()
    plotter_v2 = await pool.publish_plotter_v2(tmp_path)
    upserted = await pool.upsert(plotter_v2)
    removed = await pool.remove("image-captioner")
    assert [r.status.value for r in (built, upserted, removed)] == ["committed"] * 3
    assert [r.result_revision for r in (built, upserted, removed)] == [0, 1, 2]
    return built, upserted, removed, plotter_v2


@pytest.mark.asyncio
async def test_write_chain_leaves_memberships_generations_and_operations_consistent(
    requires_pg: str, tmp_path: Path
) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-chain")
    try:
        built, upserted, removed, plotter_v2 = await _build_upsert_remove(pool, tmp_path)
        refs = pool.refs

        # A: one membership row per skill node in every generation; the upsert
        # repointed csv-plotter to the new version, the remove dropped image-captioner.
        assert _memberships(requires_pg, built.generation_id) == {
            refs[name].skill_id: refs[name].version_id for name in POOL_ORDER
        }
        assert _memberships(requires_pg, upserted.generation_id) == {
            refs["csv-cleaner"].skill_id: refs["csv-cleaner"].version_id,
            refs["csv-plotter"].skill_id: plotter_v2.version_id,
            refs["image-captioner"].skill_id: refs["image-captioner"].version_id,
        }
        assert _memberships(requires_pg, removed.generation_id) == {
            refs["csv-cleaner"].skill_id: refs["csv-cleaner"].version_id,
            refs["csv-plotter"].skill_id: plotter_v2.version_id,
        }
        for result in (built, upserted, removed):
            snapshot = await pool.adapter.load_snapshot(pool.scope, result.generation_id)
            assert {m.skill_id: m.version_id for m in snapshot.memberships} == _memberships(
                requires_pg, result.generation_id
            )
            assert len(snapshot.memberships) == len(snapshot.nodes)

        # B: exactly one active generation, the earlier ones retired at the
        # instant their successor activated, with real counts.
        generations = _generations(requires_pg, "proj-chain")
        assert {gid: row["status"] for gid, row in generations.items()} == {
            built.generation_id: "retired",
            upserted.generation_id: "retired",
            removed.generation_id: "active",
        }
        for result, counts in ((built, (3, 2, 3)), (upserted, (3, 2, 3)), (removed, (2, 1, 2))):
            row = generations[result.generation_id]
            assert (row["node_count"], row["edge_count"], row["vector_count"]) == counts
            assert row["counts_hash"] and row["ready_at"] < row["activated_at"]
        assert generations[built.generation_id]["retired_at"] == generations[upserted.generation_id]["activated_at"]
        assert generations[upserted.generation_id]["retired_at"] == generations[removed.generation_id]["activated_at"]
        assert generations[removed.generation_id]["retired_at"] is None

        # C: every operation row is committed with its attempt, token and timestamps.
        for token, result in enumerate((built, upserted, removed), start=1):
            row = _operation(requires_pg, result.operation_id)
            assert row["status"] == "committed"
            assert row["attempt_no"] == 1 and row["writer_fencing_token"] == token
            assert row["current_generation_id"] == result.generation_id
            assert row["current_generation_id"] == generation_id_from_operation(result.operation_id, 1)
            assert row["result_revision"] == result.result_revision
            assert row["first_started_at"] < row["validating_at"] < row["completed_at"]
            assert row["result_payload"]["vector_count"] == result.vector_count
            assert row["error_code"] is None and row["pending_repair_reason"] is None

        # D: the lease exists, is released, and its token matches the last writer.
        lease = _lease(requires_pg, "proj-chain")
        assert lease is not None and lease["owner_operation_id"] is None and lease["fencing_token"] == 3

        # E: the head names the algorithm; retention keeps both retired generations for now.
        head = await pool.head()
        assert (head["algorithm_name"], head["algorithm_version"]) == ("skill_manage", "1")
        assert head["active_generation_id"] == removed.generation_id

        # Replay of a committed operation carries the recorded counts.
        again = await pool.remove("image-captioner")
        assert again == removed
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_writer_conflict_while_another_writer_holds_the_lease(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-lease")
    try:
        now = datetime.now(UTC).isoformat()
        token = await pool.adapter.acquire_writer_lease(
            pool.scope, owner_operation_id="another-process", ttl_seconds=600.0, now=now
        )
        blocked = await pool.build()
        assert blocked.status.value == "conflict"
        assert blocked.error_code == "SKILL_MANAGE_WRITER_CONFLICT"
        assert blocked.error_retryable is True
        row = _operation(requires_pg, blocked.operation_id)
        assert row["status"] == "pending" and row["writer_fencing_token"] is None
        assert _generations(requires_pg, "proj-lease") == {}
        assert await pool.adapter.get_graph_head(pool.scope) is None

        assert await pool.adapter.release_writer_lease(
            pool.scope, owner_operation_id="another-process", token=token, now=datetime.now(UTC).isoformat()
        )
        built = await pool.build()
        assert built.status.value == "committed"
        assert built.operation_id == blocked.operation_id
        row = _operation(requires_pg, built.operation_id)
        assert row["status"] == "committed" and row["attempt_no"] == 1 and row["writer_fencing_token"] == token + 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_stale_writer_is_fenced_out_at_publish(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-fence")
    try:
        built = await pool.build()
        adapter = pool.adapter
        original_write_embeddings = adapter.write_embeddings

        async def take_over_then_write(scope, generation_id, *, embeddings):
            await original_write_embeddings(scope, generation_id, embeddings=embeddings)
            # Another writer observes the lease as expired (its clock is past
            # the TTL) and takes it over while this write is still in flight.
            later = (datetime.now(UTC) + timedelta(seconds=120)).isoformat()
            await adapter.acquire_writer_lease(scope, owner_operation_id="intruder", ttl_seconds=600.0, now=later)

        adapter.write_embeddings = take_over_then_write  # type: ignore[method-assign]
        fenced = await pool.remove("image-captioner", key="remove-fenced")
        adapter.write_embeddings = original_write_embeddings  # type: ignore[method-assign]

        assert fenced.status.value == "conflict"
        assert fenced.error_code == "SKILL_MANAGE_WRITER_CONFLICT"
        row = _operation(requires_pg, fenced.operation_id)
        assert row["status"] == "conflict" and row["error_code"] == "SKILL_MANAGE_WRITER_CONFLICT"
        loser = generation_id_from_operation(fenced.operation_id, 1)
        assert loser not in _generations(requires_pg, "proj-fence")
        assert _vector_rows(requires_pg, loser) == {}
        head = await pool.head()
        assert head["active_revision"] == 0 and head["active_generation_id"] == built.generation_id
        assert _generations(requires_pg, "proj-fence")[built.generation_id]["status"] == "active"
        lease = _lease(requires_pg, "proj-fence")
        assert lease is not None and lease["owner_operation_id"] == "intruder"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_publish_failure_after_head_swap_leaves_no_half_switch(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-inject")
    try:
        built = await pool.build()
        adapter = pool.adapter
        original_publish = adapter.publish_generation

        async def publish_then_fail(scope, generation_id, **kwargs):
            return await original_publish(scope, generation_id, fail_after_publish=True, **kwargs)

        adapter.publish_generation = publish_then_fail  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="injected publish failure"):
            await pool.remove("image-captioner", key="remove-injected")
        adapter.publish_generation = original_publish  # type: ignore[method-assign]

        # The rolled-back publish is a retryable storage failure on the row;
        # head, old active generation and its vectors are exactly as before.
        operation_id = pool.service._operation_id(pool.scope, "remove-injected")
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "failed" and row["error_code"] == "SKILL_MANAGE_STORAGE_UNAVAILABLE"
        assert row["error_message"] == "RuntimeError" and row["error_retryable"] is True
        generations = _generations(requires_pg, "proj-inject")
        assert set(generations) == {built.generation_id}
        assert generations[built.generation_id]["status"] == "active"
        assert generations[built.generation_id]["retired_at"] is None
        head = await pool.head()
        assert head["active_revision"] == 0 and head["active_generation_id"] == built.generation_id
        assert len(_vector_rows(requires_pg, built.generation_id)) == 3
        lease = _lease(requires_pg, "proj-inject")
        assert lease is not None and lease["owner_operation_id"] is None

        # The same key re-drives the remove as attempt 2 once publish works again.
        retried = await pool.remove("image-captioner", key="remove-injected")
        assert retried.status.value == "committed" and retried.result_revision == 1
        assert retried.operation_id == operation_id
        assert retried.generation_id == generation_id_from_operation(operation_id, 2)
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "committed" and row["attempt_no"] == 2 and row["error_code"] is None
        assert _generations(requires_pg, "proj-inject")[built.generation_id]["status"] == "retired"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_publish_commit_with_lost_acknowledgement_is_finalized_from_the_head(
    requires_pg: str, tmp_path: Path
) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-lost-ack")
    try:
        built = await pool.build()
        adapter = pool.adapter
        original_publish = adapter.publish_generation

        async def publish_then_lose_ack(scope, generation_id, **kwargs):
            assert await original_publish(scope, generation_id, **kwargs)
            raise ConnectionResetError("commit acknowledgement lost")

        adapter.publish_generation = publish_then_lose_ack  # type: ignore[method-assign]
        removed = await pool.remove("image-captioner", key="remove-lost-ack")
        adapter.publish_generation = original_publish  # type: ignore[method-assign]

        # The transaction committed, so the head and the candidate decide: committed.
        assert removed.status.value == "committed" and removed.result_revision == 1
        row = _operation(requires_pg, removed.operation_id)
        assert row["status"] == "committed" and row["result_revision"] == 1
        generations = _generations(requires_pg, "proj-lost-ack")
        assert generations[built.generation_id]["status"] == "retired"
        assert generations[removed.generation_id]["status"] == "active"
        head = await pool.head()
        assert head["active_revision"] == 1 and head["active_generation_id"] == removed.generation_id
        lease = _lease(requires_pg, "proj-lost-ack")
        assert lease is not None and lease["owner_operation_id"] is None
        assert await pool.remove("image-captioner", key="remove-lost-ack") == removed
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_publish_commit_still_landing_when_the_acknowledgement_is_lost_is_finalized_not_rolled_back(
    requires_pg: str, tmp_path: Path
) -> None:
    """The commit is in flight server-side when the client sees the error: nothing may be discarded.

    A plain read at that instant still shows the base head and a ``ready``
    candidate, which is exactly what a rollback looks like. The service must
    fence the candidate with a row update that queues behind the in-flight
    commit; on PostgreSQL that update re-evaluates against the committed row,
    loses, and the candidate's ``active`` status settles the outcome.
    """

    import asyncio

    from mindmemos_skill.infra.skill_manage.pg_storage import _GENERATIONS, _GRAPHS, _LEASES, _is_expired
    from mindmemos_skill.infra.skill_manage.storage import head_scope, snapshot_scope

    pool = await _register_pool(requires_pg, tmp_path, "proj-in-doubt")
    try:
        built = await pool.build()
        adapter = pool.adapter
        database = adapter._database
        original_publish = adapter.publish_generation
        writes_done = asyncio.Event()
        commit_hold = asyncio.Event()

        async def publish_with_a_slow_server_side_commit(scope, generation_id, **kwargs):
            """The publish transaction's statements, held open past the client-side error."""

            async def in_doubt_transaction() -> None:
                now = kwargs["now"]
                async with database.transaction() as tx:
                    lease = (await tx.get_records(_LEASES, head_scope(scope), (scope.graph_id,)))[0].payload
                    assert lease["owner_operation_id"] == kwargs["owner_operation_id"]
                    assert int(lease["fencing_token"]) == kwargs["fencing_token"]
                    assert not _is_expired(lease["expires_at"], now)
                    candidate = (
                        await tx.get_records(_GENERATIONS, snapshot_scope(scope, generation_id), (generation_id,))
                    )[0].payload
                    assert candidate["status"] == "ready"
                    previous = (await tx.get_records(_GRAPHS, head_scope(scope), (scope.graph_id,)))[0].payload
                    assert await tx.compare_and_swap_record(
                        _GRAPHS,
                        head_scope(scope),
                        scope.graph_id,
                        expected={"active_revision": kwargs["expected_revision"]},
                        changes={
                            "active_revision": kwargs["new_revision"],
                            "active_generation_id": generation_id,
                            "updated_at": now,
                        },
                    )
                    await tx.patch_record(
                        _GENERATIONS,
                        snapshot_scope(scope, generation_id),
                        generation_id,
                        {"status": "active", "activated_at": now},
                    )
                    previous_id = previous["active_generation_id"]
                    await tx.patch_record(
                        _GENERATIONS,
                        snapshot_scope(scope, previous_id),
                        previous_id,
                        {"status": "retired", "retired_at": now},
                    )
                    writes_done.set()
                    await commit_hold.wait()

            asyncio.ensure_future(in_doubt_transaction())
            await writes_done.wait()
            raise ConnectionResetError("commit acknowledgement lost")

        async def land_the_commit_later() -> None:
            await asyncio.sleep(0.5)
            commit_hold.set()

        adapter.publish_generation = publish_with_a_slow_server_side_commit  # type: ignore[method-assign]
        releaser = asyncio.ensure_future(land_the_commit_later())
        removed = await pool.remove("image-captioner", key="remove-in-doubt")
        await releaser
        adapter.publish_generation = original_publish  # type: ignore[method-assign]

        # Settled from the candidate once the commit landed: committed, nothing discarded.
        assert removed.status.value == "committed" and removed.result_revision == 1
        gen1 = generation_id_from_operation(removed.operation_id, 1)
        assert removed.generation_id == gen1
        row = _operation(requires_pg, removed.operation_id)
        assert row["status"] == "committed" and row["result_revision"] == 1 and row["error_code"] is None
        generations = _generations(requires_pg, "proj-in-doubt")
        assert {gid: r["status"] for gid, r in generations.items()} == {
            built.generation_id: "retired",
            gen1: "active",
        }
        assert generations[gen1]["node_count"] == 2
        head = await pool.head()
        assert head["active_revision"] == 1 and head["active_generation_id"] == gen1
        assert len(_vector_rows(requires_pg, gen1)) == 2
        assert len((await adapter.load_snapshot(pool.scope, gen1)).nodes) == 2
        lease = _lease(requires_pg, "proj-in-doubt")
        assert lease is not None and lease["owner_operation_id"] is None
        assert await pool.remove("image-captioner", key="remove-in-doubt") == removed
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_repair_during_a_commit_still_landing_converges_to_committed_instead_of_discarding(
    requires_pg: str, tmp_path: Path
) -> None:
    """The recoverer side of the in-doubt window: repair must not "prove" a rollback from a stale read.

    A stranded ``validating`` attempt whose publish transaction has executed
    every statement but not yet committed (its lease check passed before the
    lease lapsed). Repair takes the lease over and reconciles; the candidate
    fence queues behind the commit, loses, and the row converges to
    ``committed`` with every row of the now-active generation intact.
    """

    import asyncio

    from mindmemos_skill.infra.skill_manage.pg_storage import _GENERATIONS, _GRAPHS
    from mindmemos_skill.infra.skill_manage.storage import head_scope, snapshot_scope

    pool = await _register_pool(requires_pg, tmp_path, "proj-repair-in-doubt")
    try:
        built = await pool.build()
        adapter, scope = pool.adapter, pool.scope
        database = adapter._database
        gen0 = built.generation_id
        operation_id = pool.service._operation_id(scope, "build-stranded")
        gen1 = generation_id_from_operation(operation_id, 1)
        stale = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        assert await adapter.reserve_operation(
            scope, operation_id, idempotency_key="build-stranded", request_hash="h", operation_type="build", now=stale
        )
        token = await adapter.acquire_writer_lease(scope, owner_operation_id=operation_id, ttl_seconds=60.0, now=stale)
        assert await adapter.begin_initial_attempt_atomically(
            scope, operation_id, generation_id=gen1, fencing_token=token, base_revision=0, now=stale
        )
        await adapter.stage_generation(
            scope, gen1, operation_id=operation_id, attempt_no=1, base_revision=0, target_revision=1, now=stale
        )
        snapshot = await adapter.load_snapshot(scope, gen0)
        await adapter.write_snapshot(
            scope,
            gen1,
            nodes=list(snapshot.nodes),
            edges=list(snapshot.edges),
            memberships=[m.model_dump(mode="python") for m in snapshot.memberships],
        )
        await adapter.write_embeddings(scope, gen1, embeddings=list(await adapter.load_embeddings(scope, gen0)))
        assert await adapter.mark_generation_ready(
            scope,
            gen1,
            node_count=len(snapshot.nodes),
            edge_count=len(snapshot.edges),
            vector_count=len(snapshot.nodes),
            counts_hash="x",
            now=stale,
        )
        assert await adapter.begin_validation(scope, operation_id, fencing_token=token, now=stale)

        writes_done, commit_hold = asyncio.Event(), asyncio.Event()

        async def publish_with_a_slow_server_side_commit() -> None:
            now = datetime.now(UTC).isoformat()
            async with database.transaction() as tx:
                assert await tx.compare_and_swap_record(
                    _GRAPHS,
                    head_scope(scope),
                    scope.graph_id,
                    expected={"active_revision": 0},
                    changes={"active_revision": 1, "active_generation_id": gen1, "updated_at": now},
                )
                await tx.patch_record(
                    _GENERATIONS, snapshot_scope(scope, gen1), gen1, {"status": "active", "activated_at": now}
                )
                await tx.patch_record(
                    _GENERATIONS, snapshot_scope(scope, gen0), gen0, {"status": "retired", "retired_at": now}
                )
                writes_done.set()
                await commit_hold.wait()

        async def land_the_commit_later() -> None:
            await asyncio.sleep(0.5)
            commit_hold.set()

        publishing = asyncio.ensure_future(publish_with_a_slow_server_side_commit())
        await writes_done.wait()
        releaser = asyncio.ensure_future(land_the_commit_later())
        report = await pool.service.repair_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
        await releaser
        await publishing

        assert [(a.kind, a.operation_id, a.from_status, a.to_status) for a in report.actions] == [
            ("operation_reconciled", operation_id, "validating", "committed")
        ]
        assert report.warnings == ()
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "committed" and row["result_revision"] == 1
        assert row["last_repair_action"] == "crash_recovery:committed"
        generations = _generations(requires_pg, "proj-repair-in-doubt")
        assert {gid: r["status"] for gid, r in generations.items()} == {gen0: "retired", gen1: "active"}
        head = await pool.head()
        assert head["active_revision"] == 1 and head["active_generation_id"] == gen1
        assert len(_vector_rows(requires_pg, gen1)) == 3
        assert len((await adapter.load_snapshot(scope, gen1)).nodes) == 3
        lease = _lease(requires_pg, "proj-repair-in-doubt")
        assert lease is not None and lease["owner_operation_id"] is None and lease["fencing_token"] == token + 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_writer_stalled_inside_publish_cannot_activate_a_candidate_a_reconciler_discarded(
    requires_pg: str, tmp_path: Path
) -> None:
    """The generation flip inside publish is a CAS on ``ready``, so a fenced candidate rolls the switch back.

    Writer W (its clock frozen, so its lease looks valid to itself) passes
    the lease/generation/head reads of its publish transaction and then
    stalls before the head CAS. By the reconciler's clock the lease has
    lapsed: repair takes it over, fences the candidate ``ready -> failed`` and
    discards it. W resumes: its head CAS still succeeds (the revision did
    not move), so only the candidate CAS can notice — it must, and the whole
    transaction must roll back rather than point the head at deleted rows.
    """

    import asyncio
    import uuid
    from contextlib import asynccontextmanager

    from mindmemos_skill.algos.skill_manage import DefaultSkillManageAlgorithm, SkillManageConfig
    from mindmemos_skill.application.skill_manage.loader import LocalSkillVersionLoader
    from mindmemos_skill.application.skill_manage.models import SkillManageRemoveRequest
    from mindmemos_skill.application.skill_manage.service import SkillManageService
    from mindmemos_skill.infra.skill_manage.pg_storage import _GRAPHS
    from mindmemos_skill.typing.skill_manage import SkillManageRemoveSelector
    from skill_manage_embedding_stub import stub_algorithm_context
    from test_skill_manage_embedding_path import STUB_PROFILE

    pool = await _register_pool(requires_pg, tmp_path, "proj-stalled-writer")
    try:
        built = await pool.build()
        adapter, scope = pool.adapter, pool.scope
        database = adapter._database
        t0 = datetime.now(UTC)

        def make_service(clock):
            return SkillManageService(
                algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
                storage=adapter,
                embedding_profile=STUB_PROFILE,
                version_loader=LocalSkillVersionLoader(pool.app._manager.repository),
                clock=clock,
                id_generator=lambda: str(uuid.uuid4()),
            )

        writer = make_service(lambda: t0)
        reconciler = make_service(lambda: t0 + timedelta(seconds=120))
        reached, resume = asyncio.Event(), asyncio.Event()

        class StallingUnitOfWork:
            def __init__(self, inner) -> None:
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            async def compare_and_swap_record(self, table, *args, **kwargs):
                if table == _GRAPHS and not reached.is_set():
                    reached.set()
                    await resume.wait()
                return await self._inner.compare_and_swap_record(table, *args, **kwargs)

        class StallingDatabase:
            def __getattr__(self, name):
                return getattr(database, name)

            @asynccontextmanager
            async def transaction(self):
                async with database.transaction() as inner:
                    yield StallingUnitOfWork(inner)

        request = SkillManageRemoveRequest(
            project_id=pool.project_id,
            graph_id=GRAPH_ID,
            idempotency_key="remove-stalled",
            selector=SkillManageRemoveSelector(kind="family", skill_id=pool.refs["image-captioner"].skill_id),
        )
        adapter._database = StallingDatabase()  # type: ignore[assignment]
        try:
            writing = asyncio.ensure_future(writer.remove_skill_from_index(request))
            await reached.wait()
            operation_id = writer._operation_id(scope, "remove-stalled")
            gen1 = generation_id_from_operation(operation_id, 1)
            assert _operation(requires_pg, operation_id)["status"] == "validating"

            report = await reconciler.repair_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
            assert [(a.kind, a.operation_id, a.to_status) for a in report.actions] == [
                ("operation_reconciled", operation_id, "failed")
            ]
            assert gen1 not in _generations(requires_pg, "proj-stalled-writer")

            resume.set()
            outcome = await writing
        finally:
            adapter._database = database

        # The switch rolled back: the head still names the built generation with all its rows.
        head = await pool.head()
        assert head["active_revision"] == 0 and head["active_generation_id"] == built.generation_id
        generations = _generations(requires_pg, "proj-stalled-writer")
        assert {gid: r["status"] for gid, r in generations.items()} == {built.generation_id: "active"}
        assert len((await adapter.load_snapshot(scope, built.generation_id)).nodes) == 3
        assert len(_vector_rows(requires_pg, built.generation_id)) == 3
        # The writer reports the row the reconciler settled, and the same key retries as attempt 2.
        assert outcome.status.value == "failed" and outcome.error_retryable is True
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "failed" and row["error_retryable"] is True and row["attempt_no"] == 1
        retried = await writer.remove_skill_from_index(request)
        assert retried.status.value == "committed" and retried.result_revision == 1
        assert retried.generation_id == generation_id_from_operation(operation_id, 2)
        assert len((await adapter.load_snapshot(scope, retried.generation_id)).nodes) == 2
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_writer_stalled_inside_publish_cannot_activate_a_candidate_a_reconciler_fenced_but_left(
    requires_pg: str, tmp_path: Path
) -> None:
    """Same stall, but the reconciler's discard fails: the fenced ``failed`` row is still there when W resumes.

    A blind status patch would flip that row ``failed -> active`` and commit
    the head onto it; the CAS on ``ready`` is what rolls the switch back.
    """

    import asyncio
    import uuid
    from contextlib import asynccontextmanager

    from mindmemos_skill.algos.skill_manage import DefaultSkillManageAlgorithm, SkillManageConfig
    from mindmemos_skill.application.skill_manage.loader import LocalSkillVersionLoader
    from mindmemos_skill.application.skill_manage.models import SkillManageRemoveRequest
    from mindmemos_skill.application.skill_manage.service import SkillManageService
    from mindmemos_skill.infra.skill_manage.pg_storage import _GRAPHS
    from mindmemos_skill.typing.skill_manage import SkillManageRemoveSelector
    from skill_manage_embedding_stub import stub_algorithm_context
    from test_skill_manage_embedding_path import STUB_PROFILE

    pool = await _register_pool(requires_pg, tmp_path, "proj-stalled-fenced")
    try:
        built = await pool.build()
        adapter, scope = pool.adapter, pool.scope
        database = adapter._database
        t0 = datetime.now(UTC)

        def make_service(clock):
            return SkillManageService(
                algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
                storage=adapter,
                embedding_profile=STUB_PROFILE,
                version_loader=LocalSkillVersionLoader(pool.app._manager.repository),
                clock=clock,
                id_generator=lambda: str(uuid.uuid4()),
            )

        writer = make_service(lambda: t0)
        reconciler = make_service(lambda: t0 + timedelta(seconds=120))
        reached, resume = asyncio.Event(), asyncio.Event()

        class StallingUnitOfWork:
            def __init__(self, inner) -> None:
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            async def compare_and_swap_record(self, table, *args, **kwargs):
                if table == _GRAPHS and not reached.is_set():
                    reached.set()
                    await resume.wait()
                return await self._inner.compare_and_swap_record(table, *args, **kwargs)

        class StallingDatabase:
            def __getattr__(self, name):
                return getattr(database, name)

            @asynccontextmanager
            async def transaction(self):
                async with database.transaction() as inner:
                    yield StallingUnitOfWork(inner)

        original_delete = adapter._delete_generation_records

        async def delete_fails_once(scope_, generation_id, **kwargs):
            adapter._delete_generation_records = original_delete
            raise ConnectionResetError("vector store unavailable")

        request = SkillManageRemoveRequest(
            project_id=pool.project_id,
            graph_id=GRAPH_ID,
            idempotency_key="remove-stalled",
            selector=SkillManageRemoveSelector(kind="family", skill_id=pool.refs["image-captioner"].skill_id),
        )
        adapter._database = StallingDatabase()  # type: ignore[assignment]
        try:
            writing = asyncio.ensure_future(writer.remove_skill_from_index(request))
            await reached.wait()
            operation_id = writer._operation_id(scope, "remove-stalled")
            gen1 = generation_id_from_operation(operation_id, 1)
            assert _operation(requires_pg, operation_id)["status"] == "validating"

            adapter._delete_generation_records = delete_fails_once  # type: ignore[method-assign]
            with pytest.raises(ConnectionResetError):
                await reconciler.repair_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
            assert _operation(requires_pg, operation_id)["status"] == "failed"
            assert _generations(requires_pg, "proj-stalled-fenced")[gen1]["status"] == "failed"

            resume.set()
            outcome = await writing
        finally:
            adapter._database = database
            adapter._delete_generation_records = original_delete  # type: ignore[method-assign]

        # The switch rolled back: the head never moved, and W's own bookkeeping
        # discarded the fenced candidate the reconciler could not.
        head = await pool.head()
        assert head["active_revision"] == 0 and head["active_generation_id"] == built.generation_id
        generations = _generations(requires_pg, "proj-stalled-fenced")
        assert {gid: r["status"] for gid, r in generations.items()} == {built.generation_id: "active"}
        assert len((await adapter.load_snapshot(scope, built.generation_id)).nodes) == 3
        assert outcome.status.value == "failed" and outcome.error_retryable is True
        report = await reconciler.repair_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
        assert report.actions == ()
        retried = await writer.remove_skill_from_index(request)
        assert retried.status.value == "committed" and retried.result_revision == 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_gc_interrupted_after_the_tombstone_is_finished_by_the_next_pass(
    requires_pg: str, tmp_path: Path
) -> None:
    """Deletion tombstones the generation row before touching its records and is resumed if it breaks off."""

    pool = await _register_pool(requires_pg, tmp_path, "proj-tombstone")
    try:
        settings = SkillManageServiceSettings(retired_generations_to_keep=0, retired_generation_min_age_seconds=0.0)
        service, _ = _make_service(pool.app, pool.adapter, settings=settings)
        built = await service.build_skill_index(
            SkillManageBuildRequest(
                project_id=pool.project_id,
                graph_id=GRAPH_ID,
                idempotency_key="tomb-build",
                version_refs=tuple(pool.refs[name] for name in POOL_ORDER),
            )
        )
        vector_store = pool.adapter._vector_store
        original_delete = vector_store.delete_records

        async def delete_unavailable(*args, **kwargs):
            raise ConnectionResetError("vector store unavailable")

        vector_store.delete_records = delete_unavailable  # type: ignore[method-assign]
        try:
            removed = await service.upsert_skill_index(
                SkillManageUpsertRequest(
                    project_id=pool.project_id,
                    graph_id=GRAPH_ID,
                    idempotency_key="tomb-upsert",
                    version_ref=await pool.publish_plotter_v2(tmp_path),
                )
            )
        finally:
            vector_store.delete_records = original_delete  # type: ignore[method-assign]
        assert removed.status.value == "committed" and removed.result_revision == 1
        assert [w["code"] for w in removed.warnings] == ["SKILL_MANAGE_GC_DEFERRED"]
        assert removed.warnings[0]["details"] == {
            "generation_id": built.generation_id,
            "reason": "ConnectionResetError",
        }
        # The tombstone landed before any record was touched.
        generations = _generations(requires_pg, "proj-tombstone")
        assert generations[built.generation_id]["status"] == "deleted"
        assert len(_vector_rows(requires_pg, built.generation_id)) == 3
        assert len((await pool.adapter.load_snapshot(pool.scope, built.generation_id)).nodes) == 3

        # A lenient policy still finishes a tombstoned deletion.
        lenient, _ = _make_service(
            pool.app,
            pool.adapter,
            settings=SkillManageServiceSettings(
                retired_generations_to_keep=5, retired_generation_min_age_seconds=86400.0
            ),
        )
        report = await lenient.maintain_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
        assert report.removed_generation_ids == (built.generation_id,) and report.warnings == ()
        assert set(_generations(requires_pg, "proj-tombstone")) == {removed.generation_id}
        assert _vector_rows(requires_pg, built.generation_id) == {}
        assert (await pool.adapter.load_snapshot(pool.scope, built.generation_id)).nodes == ()
        assert (await pool.head())["active_generation_id"] == removed.generation_id
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_retryable_failure_is_retried_as_a_new_attempt(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-retry")
    try:
        algorithm = pool.service.algorithm
        original_embed = algorithm.embed
        failures = {"left": 1}

        async def embed_once_failing(request):
            if failures["left"]:
                failures["left"] -= 1
                raise embedding_provider_error(model=request.embedding_profile.model, task=request.task, reason="down")
            return await original_embed(request)

        algorithm.embed = embed_once_failing  # type: ignore[method-assign]
        with pytest.raises(Exception) as excinfo:
            await pool.build()
        assert excinfo.value.error_code == "SKILL_MANAGE_EMBEDDING_PROVIDER_ERROR"
        operation_id = pool.service._operation_id(pool.scope, "build-pool")
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "failed" and row["error_retryable"] is True and row["attempt_no"] == 1
        assert _generations(requires_pg, "proj-retry") == {}

        # The same key opens attempt 2 under the same operation and commits.
        built = await pool.build()
        assert built.status.value == "committed"
        assert built.operation_id == operation_id
        assert built.generation_id == generation_id_from_operation(operation_id, 2)
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "committed" and row["attempt_no"] == 2 and row["writer_fencing_token"] == 2
        assert row["error_code"] is None and row["pending_repair_reason"] is None
        assert set(_generations(requires_pg, "proj-retry")) == {built.generation_id}
        assert len(_vector_rows(requires_pg, built.generation_id)) == 3
        # Committed now: the same key replays and never re-runs.
        assert await pool.build() == built
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_transient_storage_failure_retries_same_key_after_partial_vector_write(
    requires_pg: str, tmp_path: Path
) -> None:
    import psycopg
    from psycopg.errors import SerializationFailure

    pool = await _register_pool(requires_pg, tmp_path, "proj-storage-retry")
    try:
        original_write = pool.adapter.write_embeddings
        failed = False

        async def write_then_fail_once(scope, generation_id, *, embeddings):
            nonlocal failed
            await original_write(scope, generation_id, embeddings=embeddings)
            if not failed:
                failed = True
                # Produce a real driver error after the staged vector rows landed.
                async with await psycopg.AsyncConnection.connect(requires_pg) as connection:
                    await connection.execute("DO $$ BEGIN RAISE serialization_failure; END $$")

        pool.adapter.write_embeddings = write_then_fail_once
        with pytest.raises(SerializationFailure):
            await pool.build()
        operation_id = pool.service._operation_id(pool.scope, "build-pool")
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "failed" and row["attempt_no"] == 1
        assert row["error_code"] == "SKILL_MANAGE_STORAGE_UNAVAILABLE" and row["error_retryable"] is True
        assert _generations(requires_pg, pool.project_id) == {}
        assert _vector_rows(requires_pg, generation_id_from_operation(operation_id, 1)) == {}
        assert await pool.adapter.get_graph_head(pool.scope) is None

        built = await pool.build()
        assert built.status.value == "committed" and built.result_revision == 0
        assert built.operation_id == operation_id
        assert built.generation_id == generation_id_from_operation(operation_id, 2)
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "committed" and row["attempt_no"] == 2
        assert set(_generations(requires_pg, pool.project_id)) == {built.generation_id}
        assert len(_vector_rows(requires_pg, built.generation_id)) == 3
        assert await pool.build() == built
        assert (await pool.head())["active_revision"] == 0
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_stranded_running_operation_is_recovered_by_the_next_same_key_call(
    requires_pg: str, tmp_path: Path
) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-stranded")
    try:
        adapter = pool.adapter
        request = SkillManageBuildRequest(
            project_id=pool.project_id,
            graph_id=GRAPH_ID,
            idempotency_key="build-pool",
            version_refs=tuple(pool.refs[name] for name in POOL_ORDER),
        )
        operation_id = pool.service._operation_id(pool.scope, "build-pool")
        # A writer that died after staging: running row, staged generation,
        # and a lease that expired two minutes ago.
        assert await adapter.reserve_operation(
            pool.scope,
            operation_id,
            idempotency_key="build-pool",
            request_hash=compute_skill_manage_request_hash(request),
            operation_type="build",
        )
        stale = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        token = await adapter.acquire_writer_lease(
            pool.scope, owner_operation_id=operation_id, ttl_seconds=60.0, now=stale
        )
        stranded = generation_id_from_operation(operation_id, 1)
        assert await adapter.begin_initial_attempt_atomically(
            pool.scope, operation_id, generation_id=stranded, fencing_token=token, now=stale
        )
        await adapter.stage_generation(
            pool.scope, stranded, operation_id=operation_id, attempt_no=1, base_revision=None, target_revision=0
        )

        built = await pool.service.build_skill_index(request)

        assert built.status.value == "committed"
        assert built.operation_id == operation_id
        assert built.generation_id == generation_id_from_operation(operation_id, 2)
        row = _operation(requires_pg, operation_id)
        assert row["status"] == "committed" and row["attempt_no"] == 2
        assert row["repair_count"] == 1 and row["last_repair_action"] == "crash_recovery:failed"
        assert set(_generations(requires_pg, "proj-stranded")) == {built.generation_id}
        lease = _lease(requires_pg, "proj-stranded")
        assert lease is not None and lease["owner_operation_id"] is None and lease["fencing_token"] == token + 1
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_deferred_gc_bounds_retired_generations(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-gc-chain")
    try:
        settings = SkillManageServiceSettings(retired_generations_to_keep=1, retired_generation_min_age_seconds=0.0)
        service, _ = _make_service(pool.app, pool.adapter, settings=settings)
        plotter_v1 = pool.refs["csv-plotter"]
        plotter_v2 = await pool.publish_plotter_v2(tmp_path)

        results = [
            await service.build_skill_index(
                SkillManageBuildRequest(
                    project_id=pool.project_id,
                    graph_id=GRAPH_ID,
                    idempotency_key="gc-build",
                    version_refs=tuple(pool.refs[name] for name in POOL_ORDER),
                )
            )
        ]
        for index, ref in enumerate((plotter_v2, plotter_v1, plotter_v2), start=1):
            results.append(
                await service.upsert_skill_index(
                    SkillManageUpsertRequest(
                        project_id=pool.project_id,
                        graph_id=GRAPH_ID,
                        idempotency_key=f"gc-upsert-{index}",
                        version_ref=ref,
                    )
                )
            )
        assert [r.status.value for r in results] == ["committed"] * 4
        assert [r.result_revision for r in results] == [0, 1, 2, 3]
        assert all(r.warnings == () for r in results)

        # After the fourth publish only the newest retired generation survives.
        generations = _generations(requires_pg, "proj-gc-chain")
        assert {gid: row["status"] for gid, row in generations.items()} == {
            results[2].generation_id: "retired",
            results[3].generation_id: "active",
        }
        for gone in (results[0], results[1]):
            assert _vector_rows(requires_pg, gone.generation_id) == {}
            assert await pool.mappings(gone.generation_id) == []
            assert (await pool.adapter.load_snapshot(pool.scope, gone.generation_id)).nodes == ()
        assert len(_vector_rows(requires_pg, results[3].generation_id)) == 3
        head = await pool.head()
        assert head["active_generation_id"] == results[3].generation_id and head["active_revision"] == 3
        # Operation rows outlive their generations.
        for result in results:
            assert _operation(requires_pg, result.operation_id)["status"] == "committed"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_explicit_expected_revision_is_an_optimistic_lock_on_real_pg(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-optlock")
    try:
        built = await pool.build()
        assert built.result_revision == 0
        plotter_v2 = await pool.publish_plotter_v2(tmp_path)

        # A stale expected_revision conflicts without staging a generation.
        stale = await pool.service.upsert_skill_index(
            SkillManageUpsertRequest(
                project_id=pool.project_id,
                graph_id=GRAPH_ID,
                idempotency_key="opt-stale",
                version_ref=plotter_v2,
                expected_revision=5,
            )
        )
        assert stale.status.value == "conflict"
        assert stale.error_code == "SKILL_MANAGE_REVISION_CONFLICT"
        stale_row = _operation(requires_pg, stale.operation_id)
        assert stale_row["status"] == "conflict" and stale_row["error_retryable"] is False
        assert set(_generations(requires_pg, "proj-optlock")) == {built.generation_id}
        head = await pool.head()
        assert head["active_revision"] == 0 and head["active_generation_id"] == built.generation_id

        # The matching expected_revision proceeds and advances the revision.
        matched = await pool.service.upsert_skill_index(
            SkillManageUpsertRequest(
                project_id=pool.project_id,
                graph_id=GRAPH_ID,
                idempotency_key="opt-match",
                version_ref=plotter_v2,
                expected_revision=0,
            )
        )
        assert matched.status.value == "committed"
        assert matched.base_revision == 0 and matched.result_revision == 1
        head = await pool.head()
        assert head["active_revision"] == 1 and head["active_generation_id"] == matched.generation_id
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_explicit_repair_and_maintenance_converge_a_graph_without_a_same_key_retry(
    requires_pg: str, tmp_path: Path
) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-repair")
    try:
        adapter = pool.adapter
        built = await pool.build()
        head_before = await pool.head()

        # A writer that died after staging a mutation on top of the built head:
        # running row, staged candidate and a lease that expired two minutes ago.
        request = SkillManageBuildRequest(
            project_id=pool.project_id,
            graph_id=GRAPH_ID,
            idempotency_key="build-dead",
            version_refs=tuple(pool.refs[name] for name in POOL_ORDER),
        )
        dead_id = pool.service._operation_id(pool.scope, "build-dead")
        assert await adapter.reserve_operation(
            pool.scope,
            dead_id,
            idempotency_key="build-dead",
            request_hash=compute_skill_manage_request_hash(request),
            operation_type="build",
        )
        stale = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
        token = await adapter.acquire_writer_lease(pool.scope, owner_operation_id=dead_id, ttl_seconds=60.0, now=stale)
        dead_generation = generation_id_from_operation(dead_id, 1)
        assert await adapter.begin_initial_attempt_atomically(
            pool.scope, dead_id, generation_id=dead_generation, fencing_token=token, base_revision=0, now=stale
        )
        await adapter.stage_generation(
            pool.scope,
            dead_generation,
            operation_id=dead_id,
            attempt_no=1,
            base_revision=0,
            target_revision=1,
            now=stale,
        )
        # An orphan candidate no operation row references, older than the lease TTL.
        await adapter.stage_generation(
            pool.scope,
            "orphan-gen",
            operation_id="op-vanished",
            attempt_no=1,
            base_revision=0,
            target_revision=1,
            now=stale,
        )
        assert set(_generations(requires_pg, "proj-repair")) == {built.generation_id, dead_generation, "orphan-gen"}

        report = await pool.service.repair_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)

        assert {(a.kind, a.operation_id, a.generation_id, a.to_status) for a in report.actions} == {
            ("operation_reconciled", dead_id, dead_generation, "failed"),
            ("candidate_discarded", "op-vanished", "orphan-gen", None),
        }
        assert report.warnings == ()
        row = _operation(requires_pg, dead_id)
        assert row["status"] == "failed" and row["error_retryable"] is True
        assert row["repair_count"] == 1 and row["last_repair_action"] == "crash_recovery:failed"
        # Both candidates are physically gone; the active generation and head are untouched.
        assert set(_generations(requires_pg, "proj-repair")) == {built.generation_id}
        assert _vector_rows(requires_pg, dead_generation) == {}
        assert await pool.head() == head_before
        lease = _lease(requires_pg, "proj-repair")
        assert lease is not None and lease["owner_operation_id"] is None and lease["fencing_token"] == token + 1
        # Idempotent.
        again = await pool.service.repair_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
        assert again.actions == () and again.warnings == ()

        # Explicit maintenance collects retired generations on a graph nobody
        # writes to any more: build two more revisions, then GC with keep=0.
        plotter_v2 = await pool.publish_plotter_v2(tmp_path)
        upserted = await pool.upsert(plotter_v2)
        removed = await pool.remove("image-captioner")
        settings = SkillManageServiceSettings(retired_generations_to_keep=0, retired_generation_min_age_seconds=0.0)
        service, _ = _make_service(pool.app, adapter, settings=settings)
        maintenance = await service.maintain_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
        assert set(maintenance.removed_generation_ids) == {built.generation_id, upserted.generation_id}
        assert maintenance.warnings == ()
        assert set(_generations(requires_pg, "proj-repair")) == {removed.generation_id}
        assert (await pool.head())["active_generation_id"] == removed.generation_id
        assert (
            await service.maintain_skill_index(project_id=pool.project_id, graph_id=GRAPH_ID)
        ).removed_generation_ids == ()
    finally:
        await pool.close()
