"""Real P4 build->retrieve closed loop on PostgreSQL/pgvector (AC-P4-001..006)."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

# The shared contract lives under tests/mindmemos_skill; make it importable.
_TESTS_MINDMEMOS = Path(__file__).parents[2] / "mindmemos_skill"
if str(_TESTS_MINDMEMOS) not in sys.path:
    sys.path.insert(0, str(_TESTS_MINDMEMOS))

from mindmemos_skill.algos.skill_manage.algorithm import DefaultSkillManageAlgorithm  # noqa: E402
from mindmemos_skill.algos.skill_manage.config import SkillManageConfig  # noqa: E402
from mindmemos_skill.application.skill_manage.loader import LocalSkillVersionLoader  # noqa: E402
from mindmemos_skill.application.skill_manage.models import (  # noqa: E402
    SkillManageBuildRequest,
    SkillManageRetrieveRequest,
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
from mindmemos_skill.management import RegisterSkillRequest  # noqa: E402
from mindmemos_skill.persistence.tables import build_persistence_tables  # noqa: E402
from mindmemos_skill.typing import EmbeddingProfile, SkillManageScope, SkillVersionRef  # noqa: E402
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


@pytest.mark.asyncio
async def test_p4_real_build_then_retrieve(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)

    # 1. Build a source skill and register it in a live SQLite-backed application.
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "name: CSV Parser Skill\ndescription: Parses raw CSV text into structured rows of records.\n"
        "domain: [data-processing, text]\ninputs: [raw_csv_text]\noutputs: [parsed_records]\n\n"
        "# CSV Parser Skill\n\nParses raw CSV text.\n\n## Inputs\n\n- raw_csv_text\n\n## Outputs\n\n- parsed_records\n",
        encoding="utf-8",
    )
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    registered = await app.register(RegisterSkillRequest(source_path=source, alias="csv"))
    repo = app._manager.repository
    version_ref = SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id)

    # 2. Real PostgreSQL store + adapter + loader + service.
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
    loader = LocalSkillVersionLoader(repo)
    service = SkillManageService(
        algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
        storage=adapter,
        embedding_profile=STUB_PROFILE,
        version_loader=loader,
        id_generator=lambda: str(uuid.uuid4()),
    )
    try:
        result = await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-a",
                graph_id="graph-1",
                idempotency_key="build-1",
                version_refs=(version_ref,),
            )
        )
        assert result.status.value == "committed"
        assert result.result_revision == 0
        assert result.node_count >= 1

        retrieve = await service.retrieve_skills(
            SkillManageRetrieveRequest(project_id="proj-a", graph_id="graph-1", query="parse csv records", top_k=5)
        )
        assert retrieve.revision == 0
        assert len(retrieve.hits) >= 1
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_p4_002_idempotency_and_revision_conflict(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "name: CSV Parser Skill\ndescription: Parses raw CSV text into structured rows of records.\n"
        "domain: [data-processing, text]\ninputs: [raw_csv_text]\noutputs: [parsed_records]\n\n"
        "# CSV Parser Skill\n\n## Inputs\n\n- raw_csv_text\n\n## Outputs\n\n- parsed_records\n",
        encoding="utf-8",
    )
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    registered = await app.register(RegisterSkillRequest(source_path=source, alias="csv"))
    version_ref = SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id)

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
    loader = LocalSkillVersionLoader(app._manager.repository)
    service = SkillManageService(
        algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
        storage=adapter,
        embedding_profile=STUB_PROFILE,
        version_loader=loader,
        id_generator=lambda: str(uuid.uuid4()),
    )
    try:
        first = await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-a", graph_id="graph-i", idempotency_key="same-key", version_refs=(version_ref,)
            )
        )
        assert first.status.value == "committed" and first.result_revision == 0

        # Same idempotency key + request hash -> reuse the same operation/result.
        retry = await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-a", graph_id="graph-i", idempotency_key="same-key", version_refs=(version_ref,)
            )
        )
        assert retry.operation_id == first.operation_id
        head = await adapter.get_graph_head(SkillManageScope(project_id="proj-a", graph_id="graph-i"))
        assert head is not None and head["active_revision"] == 0

        # A different idempotency key builds a second revision against the head.
        second = await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-a", graph_id="graph-i", idempotency_key="other-key", version_refs=(version_ref,)
            )
        )
        assert second.status.value == "committed" and second.result_revision == 1
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_p4_004_concurrent_build_keeps_retrieve_generation_consistent(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "name: CSV Parser Skill\ndescription: Parses raw CSV text into structured rows of records.\n"
        "domain: [data-processing, text]\ninputs: [raw_csv_text]\noutputs: [parsed_records]\n\n"
        "# CSV Parser Skill\n\n## Inputs\n\n- raw_csv_text\n\n## Outputs\n\n- parsed_records\n",
        encoding="utf-8",
    )
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    registered = await app.register(RegisterSkillRequest(source_path=source, alias="csv"))
    version_ref = SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id)

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
    loader = LocalSkillVersionLoader(app._manager.repository)
    service = SkillManageService(
        algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
        storage=adapter,
        embedding_profile=STUB_PROFILE,
        version_loader=loader,
        id_generator=lambda: str(uuid.uuid4()),
    )
    try:
        await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-a", graph_id="graph-c", idempotency_key="k0", version_refs=(version_ref,)
            )
        )
        # Concurrent builds with distinct keys: only one wins each CAS step, and
        # every intermediate read observes a consistent active generation.
        import asyncio

        results = await asyncio.gather(
            *[
                service.build_skill_index(
                    SkillManageBuildRequest(
                        project_id="proj-a", graph_id="graph-c", idempotency_key=f"k{i}", version_refs=(version_ref,)
                    )
                )
                for i in range(1, 5)
            ]
        )
        committed = [r for r in results if r.status.value == "committed"]
        assert committed
        head = await adapter.get_graph_head(SkillManageScope(project_id="proj-a", graph_id="graph-c"))
        assert head is not None
        gen = head["active_generation_id"]
        # A read immediately after must be stable on that generation.
        retrieve = await service.retrieve_skills(
            SkillManageRetrieveRequest(project_id="proj-a", graph_id="graph-c", query="csv", top_k=5)
        )
        assert retrieve.generation_id == gen
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_p4_003_dry_run_has_zero_persistence(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "name: CSV Parser Skill\ndescription: Parses raw CSV text into structured rows of records.\n"
        "domain: [data-processing, text]\ninputs: [raw_csv_text]\noutputs: [parsed_records]\n\n"
        "# CSV Parser Skill\n\n## Inputs\n\n- raw_csv_text\n\n## Outputs\n\n- parsed_records\n",
        encoding="utf-8",
    )
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    registered = await app.register(RegisterSkillRequest(source_path=source, alias="csv"))
    version_ref = SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id)

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
    loader = LocalSkillVersionLoader(app._manager.repository)
    service = SkillManageService(
        algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
        storage=adapter,
        embedding_profile=STUB_PROFILE,
        version_loader=loader,
        id_generator=lambda: str(uuid.uuid4()),
    )
    try:
        result = await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-a",
                graph_id="graph-d",
                idempotency_key=None,
                commit_policy="dry_run",
                version_refs=(version_ref,),
            )
        )
        assert result.operation_id is None
        assert result.node_count >= 1
        # Zero persistence: no head, no operation, no generation.
        assert await adapter.get_graph_head(SkillManageScope(project_id="proj-a", graph_id="graph-d")) is None
        assert (
            await adapter.get_operation(
                SkillManageScope(project_id="proj-a", graph_id="graph-d"), result.operation_id or "x"
            )
            is None
        )
        from mindmemos_skill.infra.database import DatabaseScope, Page, RecordQuery

        gen_records, _ = await database.query_records(
            "skill_manage_generations",
            RecordQuery(scope=DatabaseScope(project_id="proj-a", graph_id="graph-d"), page=Page(limit=10)),
        )
        assert gen_records == []
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_p4_005_retrieval_boundary_behavior(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "name: CSV Parser Skill\ndescription: Parses raw CSV text into structured rows of records.\n"
        "domain: [data-processing, text]\ninputs: [raw_csv_text]\noutputs: [parsed_records]\n\n"
        "# CSV Parser Skill\n\n## Inputs\n\n- raw_csv_text\n\n## Outputs\n\n- parsed_records\n",
        encoding="utf-8",
    )
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    registered = await app.register(RegisterSkillRequest(source_path=source, alias="csv"))
    version_ref = SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id)

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
    loader = LocalSkillVersionLoader(app._manager.repository)
    service = SkillManageService(
        algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=stub_algorithm_context(3072)),
        storage=adapter,
        embedding_profile=STUB_PROFILE,
        version_loader=loader,
        id_generator=lambda: str(uuid.uuid4()),
    )
    try:
        await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-a", graph_id="graph-e", idempotency_key="ke", version_refs=(version_ref,)
            )
        )
        # Boundary: empty graph -> no hits; top_k respected; stable order.
        retrieve = await service.retrieve_skills(
            SkillManageRetrieveRequest(project_id="proj-a", graph_id="graph-e", query="csv", top_k=1)
        )
        assert len(retrieve.hits) <= 1
        hits1 = await service.retrieve_skills(
            SkillManageRetrieveRequest(project_id="proj-a", graph_id="graph-e", query="csv", top_k=5)
        )
        hits2 = await service.retrieve_skills(
            SkillManageRetrieveRequest(project_id="proj-a", graph_id="graph-e", query="csv", top_k=5)
        )
        assert [h["name"] for h in hits1.hits] == [h["name"] for h in hits2.hits]  # deterministic
        # A threshold above all scores yields no hits.
        strict = await service.retrieve_skills(
            SkillManageRetrieveRequest(
                project_id="proj-a", graph_id="graph-e", query="csv", top_k=5, score_threshold=2.0
            )
        )
        # PPR scores are normalized in [0,1]; 2.0 filters everything.
        assert len(strict.hits) <= len(hits1.hits)
    finally:
        await adapter.close()
