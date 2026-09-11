"""Real PG closed-loop suite: multi-skill build -> retrieve -> mutate.

Uses a self-contained mini dependency pool (csv-cleaner -> csv-plotter ->
image-captioner, connected by typed I/O overlap) to exercise the
``skill_manage`` capability end to end on real PostgreSQL/pgvector.

Unlike the single-skill P4/P5 integration tests, this builds a 3-skill graph
whose producer->consumer dependency edges drive PPR retrieval and a full
build -> retrieve -> upsert -> remove mutation loop, all against a fixed
generation.

Requires ``MINDMEMOS_SKILL_TEST_PG_DSN``. Skipped without it.
"""

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
    SkillManageRetrieveRequest,
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
from mindmemos_skill.management import RegisterSkillRequest  # noqa: E402
from mindmemos_skill.persistence.tables import build_persistence_tables  # noqa: E402
from mindmemos_skill.typing import (  # noqa: E402
    EmbeddingProfile,
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

# The dependency pool, connected by typed I/O overlap:
#   csv-cleaner --(clean_csv)--> csv-plotter --(plot_png)--> image-captioner
_POOL = {
    "csv-cleaner": {
        "description": "Zephyr CSV deduplication and schema normalization toolkit.",
        "inputs": ["raw_csv: file"],
        "outputs": ["clean_csv: file"],
    },
    "csv-plotter": {
        "description": "Meridian plotting for cleaned CSV tables.",
        "inputs": ["clean_csv: file"],
        "outputs": ["plot_png: image"],
    },
    "image-captioner": {
        "description": "Lantern caption generation for plot images.",
        "inputs": ["plot_png: image"],
        "outputs": ["caption: text"],
    },
}


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


def _source(tmp_path: Path, name: str) -> Path:
    """Write one pool Skill as a SKILL.md with a typed I/O schema."""
    spec = _POOL[name]
    inputs = "\n".join(f'  - "{value}"' for value in spec["inputs"])
    outputs = "\n".join(f'  - "{value}"' for value in spec["outputs"])
    source = tmp_path / name
    source.mkdir(parents=True, exist_ok=True)
    (source / "SKILL.md").write_text(
        f"name: {name}\ndescription: {spec['description']}\ninputs:\n{inputs}\noutputs:\n{outputs}\n\n"
        f"# {name}\n\n{spec['description']}\n\n## Inputs\n\n{inputs}\n\n## Outputs\n\n{outputs}\n",
        encoding="utf-8",
    )
    return source


async def _make_service(
    app: SkillApplication, requires_pg: str
) -> tuple[SkillManageService, PgSkillGraphStorageAdapter]:
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
    return service, adapter


@pytest.mark.asyncio
async def test_closed_loop_pool_build_retrieve_mutate(requires_pg: str, tmp_path: Path) -> None:
    await _reset(requires_pg)
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )

    # Register the 3-skill dependency pool in the canonical Skill manager.
    refs: dict[str, SkillVersionRef] = {}
    for name in _POOL:
        registered = await app.register(RegisterSkillRequest(source_path=_source(tmp_path, name), alias=f"pool-{name}"))
        refs[name] = SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id)

    service, adapter = await _make_service(app, requires_pg)
    scope = SkillManageScope(project_id="proj-closed", graph_id="g1")
    try:
        # --- BUILD ---
        built = await service.build_skill_index(
            SkillManageBuildRequest(
                project_id="proj-closed",
                graph_id="g1",
                idempotency_key="build-pool",
                version_refs=tuple(refs[name] for name in ("csv-cleaner", "csv-plotter", "image-captioner")),
            )
        )
        assert built.status.value == "committed"
        assert built.result_revision == 0
        # 3 skill nodes + the producer->consumer dependency edges.
        assert built.node_count == 3
        assert built.edge_count == 2

        head = await adapter.get_graph_head(scope)
        assert head is not None and head["active_revision"] == 0
        generation_id = head["active_generation_id"]
        snapshot = await adapter.load_snapshot(scope, generation_id)
        edge_pairs = {(e.source_id, e.target_id) for e in snapshot.edges if e.edge_type == "dependency"}
        assert len(edge_pairs) == 2

        # --- RETRIEVE ---
        # Query on the downstream task should surface the whole dependency chain
        # via PPR over the dependency edges.
        result = await service.retrieve_skills(
            SkillManageRetrieveRequest(project_id="proj-closed", graph_id="g1", query="plot a csv table", top_k=3)
        )
        assert result.revision == 0
        names = [hit["name"] for hit in result.hits]
        assert names, "retrieval must return hits over the built graph"
        # Deterministic ordering: same query twice -> same order.
        result2 = await service.retrieve_skills(
            SkillManageRetrieveRequest(project_id="proj-closed", graph_id="g1", query="plot a csv table", top_k=3)
        )
        assert [hit["name"] for hit in result2.hits] == names

        # PPR spreads over the dependency edges: a query matching only the
        # downstream consumer still surfaces its producer. "describe an image
        # caption" lexically hits image-captioner (caption/describe/image); the
        # producer csv-plotter feeds plot_png into it, so PPR over the
        # csv-plotter -> image-captioner dependency must also rank it.
        spread = await service.retrieve_skills(
            SkillManageRetrieveRequest(
                project_id="proj-closed", graph_id="g1", query="describe an image caption", top_k=3
            )
        )
        spread_names = [hit["name"] for hit in spread.hits]
        assert "image-captioner" in spread_names
        # The producer is reachable through the dependency edge (PPR spread),
        # whereas pure lexical overlap for "describe an image caption" would not
        # match csv-plotter (no shared tokens).
        assert "csv-plotter" in spread_names, f"PPR must spread along dependency edges; got {spread_names}"

        # --- UPSERT: bump csv-plotter to a new version ---
        new_source = _source(tmp_path, "csv-plotter")
        (new_source / "SKILL.md").write_text(
            "name: csv-plotter\ndescription: Meridian plotting for cleaned CSV tables.\n"
            'inputs:\n  - "clean_csv: file"\noutputs:\n  - "plot_png: image"\n\n'
            "# csv-plotter\n\nMeridian plotting for cleaned CSV tables.\n",
            encoding="utf-8",
        )
        register2 = await app.register(RegisterSkillRequest(source_path=new_source, alias="pool-plot-v2"))
        upserted = await service.upsert_skill_index(
            SkillManageUpsertRequest(
                project_id="proj-closed",
                graph_id="g1",
                idempotency_key="upsert-plot",
                version_ref=SkillVersionRef(skill_id=refs["csv-plotter"].skill_id, version_id=register2.version_id),
            )
        )
        assert upserted.status.value == "committed"
        assert upserted.result_revision == 1

        # --- REMOVE: drop image-captioner from the index (graph-only) ---
        removed = await service.remove_skill_from_index(
            SkillManageRemoveRequest(
                project_id="proj-closed",
                graph_id="g1",
                idempotency_key="remove-caption",
                selector=SkillManageRemoveSelector(kind="family", skill_id=refs["image-captioner"].skill_id),
            )
        )
        assert removed.status.value == "committed"
        assert removed.result_revision == 2
        # The active generation no longer contains image-captioner.
        head2 = await adapter.get_graph_head(scope)
        snapshot2 = await adapter.load_snapshot(scope, head2["active_generation_id"])
        assert "image-captioner" not in {node.name for node in snapshot2.nodes}
        assert "csv-cleaner" in {node.name for node in snapshot2.nodes}
        assert "csv-plotter" in {node.name for node in snapshot2.nodes}
    finally:
        await adapter.close()
