"""Real P5 upsert/remove generation swap on PostgreSQL/pgvector (AC-P5-001..005)."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

_TESTS_MINDMEMOS = Path(__file__).parents[2] / "mindmemos_skill"
if str(_TESTS_MINDMEMOS) not in sys.path:
    sys.path.insert(0, str(_TESTS_MINDMEMOS))

from mindmemos_skill.algos.skill_manage.algorithm import DefaultSkillManageAlgorithm  # noqa: E402
from mindmemos_skill.algos.skill_manage.config import SkillManageConfig  # noqa: E402
from mindmemos_skill.application.skill_manage.loader import LocalSkillVersionLoader  # noqa: E402
from mindmemos_skill.application.skill_manage.models import (  # noqa: E402
    SkillManageBuildRequest,
    SkillManageRemoveRequest,
    SkillManageUpsertRequest,
)
from mindmemos_skill.application.skill_manage.service import SkillManageService  # noqa: E402
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
from mindmemos_skill.management import PublishSkillRequest, RegisterSkillRequest  # noqa: E402
from mindmemos_skill.persistence.tables import build_persistence_tables  # noqa: E402
from mindmemos_skill.typing import (  # noqa: E402
    EmbeddingProfile,
    SkillGraphNode,
    SkillManageRemoveSelector,
    SkillManageScope,
    SkillVersionRef,
)
from skill_manage_embedding_stub import stub_algorithm_context  # noqa: E402

from mindmemos_skill import SkillApplication  # noqa: E402

pytestmark = pytest.mark.integration_pgvector

STUB_PROFILE = EmbeddingProfile(model="stub", dimension=3072)

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
    sep = "&" if "?" in DSN else "?"
    return f"{DSN}{sep}options=-csearch_path%3Dpublic"


async def _reset(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS public.__mindmemos_schema (table_name TEXT PRIMARY KEY, schema_fingerprint TEXT NOT NULL)"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS public.__mindmemos_migrations (namespace TEXT NOT NULL, version INTEGER NOT NULL, name TEXT NOT NULL, checksum TEXT NOT NULL, applied_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (namespace, version))"
            )
            for table in _SKILL_MANAGE_TABLES:
                cursor.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
            cursor.execute("DELETE FROM public.__mindmemos_schema WHERE table_name LIKE 'skill_manage_%'")
            cursor.execute(
                "DELETE FROM public.__mindmemos_migrations WHERE namespace = %s AND version = %s",
                ("mindmemos-skill", 2),
            )
            # The pgvector table is never dropped (its 3072 dimension is fixed
            # for the shared DB); with every structured table gone its rows are
            # orphans, and pre-publish validation counts them per generation.
            cursor.execute("SELECT to_regclass('mindmemos.skill_manage_embeddings')")
            if cursor.fetchone()[0] is not None:
                cursor.execute("DELETE FROM mindmemos.skill_manage_embeddings")
        connection.commit()


def _source(tmp_path: Path, name: str, desc: str, inputs: str, outputs: str) -> Path:
    source = tmp_path / name
    source.mkdir(parents=True, exist_ok=True)
    (source / "SKILL.md").write_text(
        f"name: {name}\ndescription: {desc}\ndomain: [data-processing]\ninputs: [{inputs}]\noutputs: [{outputs}]\n\n"
        f"# {name}\n\n{desc}\n\n## Inputs\n\n- {inputs}\n\n## Outputs\n\n- {outputs}\n",
        encoding="utf-8",
    )
    return source


@pytest.mark.asyncio
async def test_p5_upsert_then_remove_on_real_pg(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    reg1 = await app.register(
        RegisterSkillRequest(
            source_path=_source(tmp_path, "CSV Parser Skill", "Parses raw CSV text.", "raw_csv_text", "parsed_records"),
            alias="csv",
        )
    )
    skill_id = reg1.skill_id
    v1 = SkillVersionRef(skill_id=skill_id, version_id=reg1.version_id)

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
    service = SkillManageService(
        algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
        storage=adapter,
        embedding_profile=STUB_PROFILE,
        version_loader=LocalSkillVersionLoader(app._manager.repository),
        id_generator=lambda: str(uuid.uuid4()),
    )
    scope = SkillManageScope(project_id="proj-a", graph_id="g1")
    try:
        # Build with the first version.
        built = await service.build_skill_index(
            SkillManageBuildRequest(project_id="proj-a", graph_id="g1", idempotency_key="b1", version_refs=(v1,))
        )
        assert built.status.value == "committed" and built.node_count >= 1

        # --- REMOVE that skill: graph-only, canonical version untouched. ---
        removed = await service.remove_skill_from_index(
            SkillManageRemoveRequest(
                project_id="proj-a",
                graph_id="g1",
                idempotency_key="r1",
                selector=SkillManageRemoveSelector(kind="family", skill_id=skill_id),
            )
        )
        assert removed.status.value == "committed"
        assert removed.result_revision == built.result_revision + 1
        # The active generation is now empty (the old generation is retained).
        from mindmemos_skill.infra.database import DatabaseScope, Page, RecordQuery

        head = await adapter.get_graph_head(scope)
        assert head is not None
        active_gen = head["active_generation_id"]
        node_records, _ = await database.query_records(
            "skill_manage_nodes",
            RecordQuery(
                scope=DatabaseScope(project_id="proj-a", graph_id="g1", generation_id=active_gen),
                page=Page(limit=10),
            ),
        )
        assert node_records == []
        # Canonical Skill family/version still present and readable.
        assert await app.get_version("csv", reg1.version_id) is not None

        # --- already_absent: removing an already-absent family is a no-op. ---
        absent = await service.remove_skill_from_index(
            SkillManageRemoveRequest(
                project_id="proj-a",
                graph_id="g1",
                idempotency_key="r-absent",
                selector=SkillManageRemoveSelector(kind="family", skill_id=skill_id),
            )
        )
        assert absent.status.value == "committed"
        assert absent.result_revision == removed.result_revision  # no revision advance
        head_after = await adapter.get_graph_head(scope)
        assert head_after["active_revision"] == removed.result_revision

        # --- UPSERT a new same-family version: replaces the membership. ---
        reg2 = await app.publish(
            PublishSkillRequest(
                skill_ref="csv",
                content=(
                    "name: CSV Parser Skill\ndescription: Parses raw CSV text v2.\ndomain: [data-processing]\n"
                    "inputs: [raw_csv_text]\noutputs: [parsed_records]\n\n# CSV Parser Skill\n\nv2 body\n"
                ),
            )
        )
        v2 = SkillVersionRef(skill_id=skill_id, version_id=reg2.version_id)
        upserted = await service.upsert_skill_index(
            SkillManageUpsertRequest(project_id="proj-a", graph_id="g1", idempotency_key="u1", version_ref=v2)
        )
        assert upserted.status.value == "committed"
        assert upserted.result_revision == removed.result_revision + 1

        # --- GC: never removes the active generation; keeps the last two retired. ---
        removed_ids = await adapter.gc_retired_generations(scope, keep=2)
        # The active generation is untouched.
        head_final = await adapter.get_graph_head(scope)
        assert head_final["active_generation_id"] == upserted.generation_id
        # Only retired generations beyond the retention window were removed.
        assert active_gen not in removed_ids
    finally:
        await adapter.close()
        await app.close()


@pytest.mark.asyncio
async def test_p5_004_crash_recovery_repair_on_real_pg(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    reg1 = await app.register(
        RegisterSkillRequest(
            source_path=_source(tmp_path, "CSV Parser Skill", "Parses raw CSV text.", "raw_csv_text", "parsed_records"),
            alias="csv",
        )
    )
    skill_id = reg1.skill_id
    v1 = SkillVersionRef(skill_id=skill_id, version_id=reg1.version_id)

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
    scope = SkillManageScope(project_id="proj-a", graph_id="g1")
    try:
        # Simulate: a stranded running operation with an expired writer lease.
        await adapter.reserve_operation(
            scope, "op-crash", idempotency_key="kc", request_hash="hc", operation_type="build"
        )
        await adapter.begin_initial_attempt_atomically(
            scope, "op-crash", generation_id="gen-c", fencing_token=1, now="2026-09-03T00:00:00+00:00"
        )
        await adapter.stage_generation(
            scope, "gen-c", operation_id="op-crash", attempt_no=1, base_revision=None, target_revision=0
        )
        # Claim crash recovery -> pending_repair(crash_recovery).
        assert await adapter.claim_crash_recovery(
            scope, "op-crash", expected_attempt_no=1, expected_generation_id="gen-c", now="2026-09-03T00:05:00+00:00"
        )
        op = await adapter.get_operation(scope, "op-crash")
        assert op is not None and op["status"] == "pending_repair" and op["pending_repair_reason"] == "crash_recovery"

        # Reconcile: the graph still doesn't exist, so the candidate fails
        # retryable and its staged rows are discarded.
        status = await adapter.repair_crash_recovery(scope, "op-crash", attempt_no=1, generation_id="gen-c")
        assert status == "failed"
        op = await adapter.get_operation(scope, "op-crash")
        assert op["status"] == "failed" and op["error_retryable"] is True
        assert await adapter.get_generation(scope, "gen-c") is None

        # Idempotent: a second reconcile leaves the state stable.
        status2 = await adapter.repair_crash_recovery(scope, "op-crash", attempt_no=1, generation_id="gen-c")
        assert status2 in ("failed", "pending_repair")

        # Now a real build commits; a stranded validating operation whose head
        # points at its generation reconciles to committed.
        await adapter.reserve_operation(scope, "op-ok", idempotency_key="ko", request_hash="ho", operation_type="build")
        await adapter.begin_initial_attempt_atomically(
            scope, "op-ok", generation_id="gen-c", fencing_token=1, now="2026-09-03T00:10:00+00:00"
        )
        await adapter.stage_generation(
            scope, "gen-c", operation_id="op-ok", attempt_no=1, base_revision=None, target_revision=0
        )
        await adapter.write_snapshot(scope, "gen-c", nodes=[], edges=[], memberships=[])
        assert await adapter.mark_generation_ready(
            scope, "gen-c", node_count=0, edge_count=0, vector_count=0, counts_hash="h"
        )
        assert await adapter.publish_generation(scope, "gen-c", expected_revision=None, new_revision=0)
        await adapter.transition_operation(
            scope,
            "op-ok",
            expected_status=["running"],
            changes={"status": "validating"},
            now="2026-09-03T00:10:01+00:00",
        )
        status3 = await adapter.repair_crash_recovery(scope, "op-ok", attempt_no=1, generation_id="gen-c")
        assert status3 == "committed"
    finally:
        await adapter.close()
        await app.close()


@pytest.mark.asyncio
async def test_p5_003_failure_injection_preserves_old_active_generation(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    reg1 = await app.register(
        RegisterSkillRequest(
            source_path=_source(tmp_path, "CSV Parser Skill", "Parses raw CSV text.", "raw_csv_text", "parsed_records"),
            alias="csv",
        )
    )
    skill_id = reg1.skill_id
    v1 = SkillVersionRef(skill_id=skill_id, version_id=reg1.version_id)

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
    scope = SkillManageScope(project_id="proj-a", graph_id="g1")
    try:
        # Build a committed graph first.
        await adapter.reserve_operation(
            scope, "op-build", idempotency_key="kb", request_hash="hb", operation_type="build"
        )
        await adapter.begin_initial_attempt_atomically(scope, "op-build", generation_id="gen-1", fencing_token=1)
        await adapter.stage_generation(
            scope, "gen-1", operation_id="op-build", attempt_no=1, base_revision=None, target_revision=0
        )
        await adapter.write_snapshot(
            scope,
            "gen-1",
            nodes=[
                SkillGraphNode(
                    graph_node_id="n-1",
                    node_type="skill",
                    skill_id=skill_id,
                    version_id=reg1.version_id,
                    name="CSV Parser Skill",
                )
            ],
            edges=[],
            memberships=[{"skill_id": skill_id, "version_id": reg1.version_id, "graph_node_id": "n-1"}],
        )
        assert await adapter.mark_generation_ready(
            scope, "gen-1", node_count=1, edge_count=0, vector_count=0, counts_hash="h1"
        )
        assert await adapter.publish_generation(scope, "gen-1", expected_revision=None, new_revision=0)
        old_snapshot = await adapter.load_snapshot(scope, "gen-1")
        assert len(old_snapshot.nodes) == 1
        assert [(m.skill_id, m.version_id) for m in old_snapshot.memberships] == [(skill_id, reg1.version_id)]

        # A publish that fails after the head-swap rolls everything back, so the
        # old active generation is still fully readable.
        new_op = str(uuid.uuid4())
        new_gen = f"{new_op}:1"
        await adapter.reserve_operation(scope, new_op, idempotency_key="k2", request_hash="h2", operation_type="build")
        await adapter.begin_initial_attempt_atomically(scope, new_op, generation_id=new_gen, fencing_token=2)
        await adapter.stage_generation(
            scope, new_gen, operation_id=new_op, attempt_no=1, base_revision=0, target_revision=1
        )
        await adapter.write_snapshot(scope, new_gen, nodes=[], edges=[], memberships=[])
        assert await adapter.mark_generation_ready(
            scope, new_gen, node_count=0, edge_count=0, vector_count=0, counts_hash="h2"
        )
        with pytest.raises(RuntimeError, match="injected publish failure"):
            await adapter.publish_generation(
                scope, new_gen, expected_revision=0, new_revision=1, fail_after_publish=True
            )

        # No half switch: head, old active and the candidate are all unchanged.
        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_generation_id"] == "gen-1" and head["active_revision"] == 0
        still_readable = await adapter.load_snapshot(scope, "gen-1")
        assert len(still_readable.nodes) == 1
        old = await adapter.get_generation(scope, "gen-1")
        assert old is not None and old["status"] == "active" and old["retired_at"] is None
        candidate = await adapter.get_generation(scope, new_gen)
        assert candidate is not None and candidate["status"] == "ready" and candidate["activated_at"] is None
    finally:
        await adapter.close()
        await app.close()
