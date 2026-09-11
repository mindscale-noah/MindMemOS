"""Real PostgreSQL/pgvector storage integration suite.

Requires ``MINDMEMOS_SKILL_TEST_PG_DSN`` pointing at a dedicated, non-production
PostgreSQL 16 + pgvector >= 0.7.0 instance. Skipped without that variable.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

# The shared storage contract lives under tests/mindmemos_skill; make it importable.
_TESTS_MINDMEMOS = Path(__file__).parents[2] / "mindmemos_skill"
if str(_TESTS_MINDMEMOS) not in sys.path:
    sys.path.insert(0, str(_TESTS_MINDMEMOS))

from mindmemos_skill.infra.database import DatabaseConfig, bootstrap_database  # noqa: E402
from mindmemos_skill.infra.skill_manage import build_skill_manage_vector_tables  # noqa: E402
from mindmemos_skill.infra.skill_manage.pg_storage import PgSkillGraphStorageAdapter  # noqa: E402
from mindmemos_skill.infra.vector_store import (  # noqa: E402
    BackendConfig,
    BackendRegistry,
    BackendRequirements,
    bootstrap_vector_store,
    register_builtin_vector_stores,
)
from mindmemos_skill.persistence.tables import build_persistence_tables  # noqa: E402
from mindmemos_skill.typing import SkillGraphEdge, SkillGraphEmbedding, SkillGraphNode, SkillManageScope  # noqa: E402

pytestmark = pytest.mark.integration_pgvector

DSN = os.environ.get("MINDMEMOS_SKILL_TEST_PG_DSN")

_SKILL_MANAGE_TABLES = (
    "skill_manage_id_mappings",
    "skill_manage_chunks",
    "skill_manage_edges",
    "skill_manage_nodes",
    "skill_manage_memberships",
    "skill_manage_generations",
    "skill_manage_writer_leases",
    "skill_manage_operations",
    "skill_manage_graphs",
)


@pytest.fixture
def requires_pg() -> str:
    if not DSN:
        pytest.skip("MINDMEMOS_SKILL_TEST_PG_DSN is not set")
    # Force the public search path so this suite and the backend agree on where
    # __mindmemos_schema / __mindmemos_migrations live.
    sep = "&" if "?" in DSN else "?"
    return f"{DSN}{sep}options=-csearch_path%3Dpublic"


async def _reset_skill_manage_schema(dsn: str) -> None:
    """Drop prior skill_manage tables and their v0002 ledger/schema rows.

    This leaves a consistent v0001 database so ``bootstrap_database`` re-applies
    v0002 cleanly.
    """

    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS public.__mindmemos_schema ("
                "table_name TEXT PRIMARY KEY, schema_fingerprint TEXT NOT NULL)"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS public.__mindmemos_migrations ("
                "namespace TEXT NOT NULL, version INTEGER NOT NULL, name TEXT NOT NULL, "
                "checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL, "
                "PRIMARY KEY (namespace, version))"
            )
            for table in _SKILL_MANAGE_TABLES:
                cursor.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
            cursor.execute("DELETE FROM public.__mindmemos_schema WHERE table_name LIKE 'skill_manage_%'")
            cursor.execute(
                "DELETE FROM public.__mindmemos_migrations WHERE namespace = %s AND version = %s",
                ("mindmemos-skill", 2),
            )
        connection.commit()


@pytest.mark.asyncio
async def test_real_adapter_passes_shared_storage_contract(requires_pg: str) -> None:
    await _reset_skill_manage_schema(requires_pg)

    database = await bootstrap_database(
        DatabaseConfig(provider="postgres", options={"dsn": requires_pg}),
        build_persistence_tables(),
    )
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    vector_store = await bootstrap_vector_store(
        BackendConfig(
            provider="pgvector",
            options={"dsn": requires_pg},
            required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
        ),
        build_skill_manage_vector_tables(dimension=3072),
        registry=registry,
    )
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=vector_store)
    try:
        assert await adapter.is_ready()
        from mindmemos_skill.typing import SkillManageScope

        async def make() -> PgSkillGraphStorageAdapter:
            return adapter

        from skill_manage_storage_contract import run_storage_contract

        for case in run_storage_contract(make):
            await case(adapter)

        # The contract publishes a head in p1/g1; confirm it is visible.
        scope = SkillManageScope(project_id="p1", graph_id="g1")
        head = await adapter.get_graph_head(scope)
        assert head is not None
        assert head["active_revision"] == 0

        # A naive instant would be stored in the session time zone and read
        # back as a different instant; the backend refuses it like sqlite does.
        with pytest.raises(ValueError, match="timezone-aware"):
            await adapter.stage_generation(
                scope,
                "gen-naive",
                operation_id="op-naive",
                attempt_no=1,
                base_revision=None,
                target_revision=1,
                now="2026-09-05T10:00:00",
            )
        assert await adapter.get_generation(scope, "gen-naive") is None
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_p3_009_bootstrap_readiness_and_close_on_real_pg(requires_pg: str) -> None:
    await _reset_skill_manage_schema(requires_pg)

    database = await bootstrap_database(
        DatabaseConfig(provider="postgres", options={"dsn": requires_pg}),
        build_persistence_tables(),
    )
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    vector_store = await bootstrap_vector_store(
        BackendConfig(
            provider="pgvector",
            options={"dsn": requires_pg},
            required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
        ),
        build_skill_manage_vector_tables(dimension=3072),
        registry=registry,
    )
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=vector_store)
    assert await adapter.is_ready()
    await adapter.start()
    assert await adapter.is_ready()
    # close-once contract: double close is idempotent.
    await adapter.close()
    assert not await adapter.is_ready()
    await adapter.close()


@pytest.mark.asyncio
async def test_pooled_connections_killed_server_side_are_replaced_on_checkout(requires_pg: str) -> None:
    """After a restart/failover every idle pooled connection is dead; a checkout must not hand one out.

    Killing the pools' backends with ``pg_terminate_backend`` models the
    restart; both the structured and the vector pool must serve the next call,
    and promptly: with eight dead connections queued ahead, validating them one
    per attempt with the pool's backoff would exceed the 30 s pool timeout.
    """

    import time

    import psycopg

    await _reset_skill_manage_schema(requires_pg)
    database = await bootstrap_database(
        DatabaseConfig(provider="postgres", options={"dsn": requires_pg}),
        build_persistence_tables(),
    )
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    vector_store = await bootstrap_vector_store(
        BackendConfig(
            provider="pgvector",
            options={"dsn": requires_pg},
            required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
        ),
        build_skill_manage_vector_tables(dimension=3072),
        registry=registry,
    )
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=vector_store)
    try:
        from mindmemos_skill.typing import SkillManageScope

        scope = SkillManageScope(project_id="p-pool", graph_id="g-pool")

        async def hold(pool: object) -> None:
            async with pool.connection() as connection:  # type: ignore[attr-defined]
                await connection.execute("SELECT pg_sleep(0.2)")

        async def kill_pooled_backends() -> int:
            """Terminate every other backend of this database and wait until they are gone."""

            others = "FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()"
            with psycopg.connect(requires_pg, autocommit=True) as connection:
                killed = connection.execute(f"SELECT count(pg_terminate_backend(pid)) {others}").fetchone()[0]
                for _ in range(100):
                    if connection.execute(f"SELECT count(*) {others}").fetchone()[0] == 0:
                        break
                    await asyncio.sleep(0.02)
            # The server has closed the sockets; let the client side see it.
            await asyncio.sleep(0.05)
            return killed

        pools = (adapter._database._pool, adapter._vector_store._pool)
        # Grow both pools to eight idle connections, then kill them all.
        await asyncio.gather(*(hold(pool) for pool in pools for _ in range(8)))
        assert await kill_pooled_backends() >= 16

        started = time.monotonic()
        for _ in range(6):
            assert await adapter.get_graph_head(scope) is None
            assert await adapter.count_embeddings(scope, "gen-none") == 0
        assert time.monotonic() - started < 10.0

        # And a concurrent burst straight after another kill.
        await asyncio.gather(*(hold(pool) for pool in pools for _ in range(8)))
        assert await kill_pooled_backends() >= 16
        started = time.monotonic()
        heads = await asyncio.gather(
            *(adapter.get_graph_head(scope) for _ in range(8)),
            *(adapter.count_embeddings(scope, "gen-none") for _ in range(8)),
        )
        assert heads == [None] * 8 + [0] * 8
        assert time.monotonic() - started < 10.0
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_concurrent_first_bootstraps_all_succeed(requires_pg: str) -> None:
    """Several processes starting on an unmigrated database must all come up, not race on the catalog."""

    import psycopg

    await _reset_skill_manage_schema(requires_pg)
    databases = await asyncio.gather(
        *(
            bootstrap_database(
                DatabaseConfig(provider="postgres", options={"dsn": requires_pg}), build_persistence_tables()
            )
            for _ in range(6)
        )
    )
    try:
        with psycopg.connect(requires_pg) as connection:
            ledger = connection.execute(
                "SELECT count(*) FROM __mindmemos_migrations WHERE namespace = 'mindmemos-skill' AND version = 2"
            ).fetchone()[0]
        assert ledger == 1
        # The vector side races on its catalog row and table the same way; the
        # shared 3072-dim table is never dropped, so a throwaway table stands in.
        registry = BackendRegistry()
        register_builtin_vector_stores(registry)
        with psycopg.connect(requires_pg, autocommit=True) as connection:
            connection.execute("DROP TABLE IF EXISTS mindmemos.skill_manage_bootstrap_probe")
            connection.execute(
                "DELETE FROM mindmemos.__mindmemos_schema WHERE logical_table = 'skill_manage_bootstrap_probe'"
            )
        try:
            stores = await asyncio.gather(
                *(
                    bootstrap_vector_store(
                        BackendConfig(
                            provider="pgvector",
                            options={"dsn": requires_pg},
                            required=BackendRequirements(
                                dense_vector=True, metadata_filtering=True, batch_record_io=True
                            ),
                        ),
                        _probe_vector_tables(),
                        registry=registry,
                    )
                    for _ in range(6)
                )
            )
            for store in stores:
                await store.close()
        finally:
            with psycopg.connect(requires_pg, autocommit=True) as connection:
                connection.execute("DROP TABLE IF EXISTS mindmemos.skill_manage_bootstrap_probe")
                connection.execute(
                    "DELETE FROM mindmemos.__mindmemos_schema WHERE logical_table = 'skill_manage_bootstrap_probe'"
                )
    finally:
        for database in databases:
            await database.close()


def _probe_vector_tables():
    from mindmemos_skill.infra.vector_store import FieldSpec, FieldType, TableRegistry, TableSpec, VectorFieldSpec

    table = TableSpec(
        name="skill_manage_bootstrap_probe",
        primary_key="record_id",
        fields=(FieldSpec(name="record_id", field_type=FieldType.TEXT, nullable=False),),
        vectors=(VectorFieldSpec(name="embedding", dimensions=8, distance="cosine"),),
        scope_scoped=True,
    )
    registry = TableRegistry((table,))
    registry.freeze()
    return registry


@pytest.mark.asyncio
async def test_bootstrap_pins_the_search_path_when_the_dsn_does_not(requires_pg: str) -> None:
    """With a schema named after the role present, an unpinned DSN would resolve the catalog elsewhere on restart."""

    import psycopg

    assert DSN is not None
    await _reset_skill_manage_schema(requires_pg)
    with psycopg.connect(requires_pg, autocommit=True) as connection:
        role = connection.execute("SELECT current_user").fetchone()[0]
        connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{role}"')

    database = await bootstrap_database(
        DatabaseConfig(provider="postgres", options={"dsn": DSN}), build_persistence_tables()
    )
    try:
        async with database._pool.connection() as connection:
            assert (await (await connection.execute("SELECT current_schema()")).fetchone())[
                "current_schema"
            ] == "public"
        with psycopg.connect(requires_pg) as connection:
            owners = connection.execute(
                "SELECT schemaname FROM pg_tables WHERE tablename = 'skill_manage_graphs'"
            ).fetchall()
        assert owners == [("public",)]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_p3_009_dimension_drift_is_rejected_on_real_pg(requires_pg: str) -> None:

    await _reset_skill_manage_schema(requires_pg)
    # Bootstrap a vector table at 3072, then attempt a different dimension.
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    first = await bootstrap_vector_store(
        BackendConfig(
            provider="pgvector",
            options={"dsn": requires_pg},
            required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
        ),
        build_skill_manage_vector_tables(dimension=3072),
        registry=registry,
    )
    await first.close()

    # A vector table already fixed at 3072 must not silently accept 1536.
    with pytest.raises(ValueError, match="schema drift"):
        second = await bootstrap_vector_store(
            BackendConfig(
                provider="pgvector",
                options={"dsn": requires_pg},
                required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
            ),
            build_skill_manage_vector_tables(dimension=1536),
            registry=registry,
        )
        await second.close()


@pytest.mark.asyncio
async def test_p3_008_atomic_publish_fault_injection(requires_pg: str) -> None:
    await _reset_skill_manage_schema(requires_pg)
    database = await bootstrap_database(
        DatabaseConfig(provider="postgres", options={"dsn": requires_pg}),
        build_persistence_tables(),
    )
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    vector_store = await bootstrap_vector_store(
        BackendConfig(
            provider="pgvector",
            options={"dsn": requires_pg},
            required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
        ),
        build_skill_manage_vector_tables(dimension=3072),
        registry=registry,
    )
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=vector_store)
    scope = SkillManageScope(project_id="p8", graph_id="g8")
    try:
        await adapter.reserve_operation(scope, "op-a", idempotency_key="ka", request_hash="ha", operation_type="build")
        await adapter.stage_generation(
            scope, "g-a", operation_id="op-a", attempt_no=1, base_revision=None, target_revision=0
        )
        await adapter.write_snapshot(scope, "g-a", nodes=[], edges=[], memberships=[])
        assert await adapter.mark_generation_ready(
            scope, "g-a", node_count=0, edge_count=0, vector_count=0, counts_hash="h"
        )

        # Inject a failure AFTER the head-swap + generation-active writes but
        # before the enclosing transaction commits.
        with pytest.raises(RuntimeError, match="injected publish failure"):
            await adapter.publish_generation(
                scope, "g-a", expected_revision=None, new_revision=0, fail_after_publish=True
            )

        # Nothing half-changed: no head, and the staged generation not flipped.
        assert await adapter.get_graph_head(scope) is None
        from mindmemos_skill.infra.database import DatabaseScope, Page, RecordQuery

        gen_records, _ = await database.query_records(
            "skill_manage_generations",
            RecordQuery(
                scope=DatabaseScope(project_id="p8", graph_id="g8", generation_id="g-a"),
                page=Page(limit=100),
            ),
        )
        assert [row.payload.get("status") for row in gen_records] == ["ready"]

        # A clean retry after the failed publish succeeds.
        assert await adapter.publish_generation(scope, "g-a", expected_revision=None, new_revision=0)
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_revision"] == 0
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_vector_only_orphans_are_listed_and_reclaimable(requires_pg: str) -> None:
    """Vectors with neither a generation row nor a mapping row are still found by the orphan listing.

    A superseded writer's late ``write_embeddings`` (vector upsert, then the
    mapping upsert) landing after the reconciler deleted its vectors, or the
    writer dying between the two upserts, leaves exactly this shape.
    """

    from mindmemos_skill.infra.skill_manage.storage import snapshot_scope

    await _reset_skill_manage_schema(requires_pg)
    database = await bootstrap_database(
        DatabaseConfig(provider="postgres", options={"dsn": requires_pg}),
        build_persistence_tables(),
    )
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    vector_store = await bootstrap_vector_store(
        BackendConfig(
            provider="pgvector",
            options={"dsn": requires_pg},
            required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
        ),
        build_skill_manage_vector_tables(dimension=3072),
        registry=registry,
    )
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=vector_store)
    scope = SkillManageScope(project_id="p-vorphan", graph_id="g-vorphan")
    try:
        await adapter.stage_generation(
            scope, "gen-live", operation_id="op-live", attempt_no=1, base_revision=None, target_revision=0
        )
        live_node = SkillGraphNode(graph_node_id="n-live", node_type="skill", skill_id="sl", version_id="v1", name="L")
        await adapter.write_snapshot(scope, "gen-live", nodes=[live_node], edges=[], memberships=[])
        assert await adapter.mark_generation_ready(
            scope, "gen-live", node_count=1, edge_count=0, vector_count=0, counts_hash="h"
        )
        assert await adapter.publish_generation(scope, "gen-live", expected_revision=None, new_revision=0)
        embedding = SkillGraphEmbedding(
            graph_node_id="n-a",
            skill_id="sa",
            version_id="v1",
            content_hash="h",
            embedding_model="m",
            embedding_profile_version="1",
            vector=(1.0,) + (0.0,) * 3071,
        )
        await adapter.write_embeddings(scope, "gen-orphan", embeddings=[embedding])
        mapping_scope = snapshot_scope(scope, "gen-orphan")
        mappings = await adapter._query_structured_paged("skill_manage_id_mappings", mapping_scope)
        await database.delete_records("skill_manage_id_mappings", mapping_scope, [r.record_id for r in mappings])
        assert await adapter.count_embeddings(scope, "gen-orphan") == 1
        assert await adapter.list_snapshot_generation_ids(scope) == {"gen-live", "gen-orphan"}
        await adapter.discard_generation(scope, "gen-orphan")
        assert await adapter.count_embeddings(scope, "gen-orphan") == 0
        assert await adapter.list_snapshot_generation_ids(scope) == {"gen-live"}
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_pagination_reads_beyond_a_single_page(requires_pg: str) -> None:
    """``load_snapshot``/``load_embeddings`` must follow the cursor past ``page_size``.

    Regression for the removed ``Page(limit=10000)`` no-loop: with a tiny
    ``page_size`` this writes a snapshot larger than one page and asserts every
    node/edge/membership/vector row is read back and counted.
    """

    await _reset_skill_manage_schema(requires_pg)
    database = await bootstrap_database(
        DatabaseConfig(provider="postgres", options={"dsn": requires_pg}),
        build_persistence_tables(),
    )
    registry = BackendRegistry()
    register_builtin_vector_stores(registry)
    vector_store = await bootstrap_vector_store(
        BackendConfig(
            provider="pgvector",
            options={"dsn": requires_pg},
            required=BackendRequirements(dense_vector=True, metadata_filtering=True, batch_record_io=True),
        ),
        build_skill_manage_vector_tables(dimension=3072),
        registry=registry,
    )
    adapter = PgSkillGraphStorageAdapter(database=database, vector_store=vector_store, page_size=2)
    scope = SkillManageScope(project_id="p-page", graph_id="g-page")
    try:
        await adapter.reserve_operation(
            scope, "op-page", idempotency_key="k-page", request_hash="h-page", operation_type="build"
        )
        await adapter.stage_generation(
            scope, "g-page", operation_id="op-page", attempt_no=1, base_revision=None, target_revision=0
        )
        nodes = [
            SkillGraphNode(
                graph_node_id=f"n-{index}",
                node_type="skill",
                skill_id=f"s{index}",
                version_id="v1",
                name=f"Node {index}",
            )
            for index in range(5)
        ]
        edges = [
            SkillGraphEdge(
                edge_id=f"e-{index}",
                edge_type="dependency",
                source_id=f"n-{index}",
                target_id=f"n-{index + 1}",
                weight=1.0,
            )
            for index in range(4)
        ]
        memberships = [
            {"skill_id": f"s{index}", "version_id": "v1", "graph_node_id": f"n-{index}"} for index in range(5)
        ]
        await adapter.write_snapshot(scope, "g-page", nodes=nodes, edges=edges, memberships=memberships)
        # Embeddings must be marked with a vector to survive the read-back path.
        embeddings = [
            SkillGraphEmbedding(
                graph_node_id=f"n-{index}",
                node_type="skill",
                skill_id=f"s{index}",
                version_id="v1",
                content_hash=f"hash-{index}",
                embedding_model="stub",
                embedding_profile_version="1",
                vector=tuple(float(index + 1) for _ in range(3072)),
            )
            for index in range(5)
        ]
        await adapter.write_embeddings(scope, "g-page", embeddings=embeddings)

        snapshot = await adapter.load_snapshot(scope, "g-page")
        assert sorted(node.graph_node_id for node in snapshot.nodes) == [f"n-{index}" for index in range(5)]
        assert sorted(edge.edge_id for edge in snapshot.edges) == [f"e-{index}" for index in range(4)]
        assert sorted((m.skill_id, m.version_id, m.graph_node_id) for m in snapshot.memberships) == [
            (f"s{index}", "v1", f"n-{index}") for index in range(5)
        ]
        loaded = await adapter.load_embeddings(scope, "g-page")
        assert sorted(embedding.graph_node_id for embedding in loaded) == [f"n-{index}" for index in range(5)]
        assert await adapter.count_embeddings(scope, "g-page") == 5
    finally:
        await adapter.close()
