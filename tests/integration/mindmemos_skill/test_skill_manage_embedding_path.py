"""Real PG embedding-path suite: vectors written, KNN-seeded PPR, reuse, GC.

Drives the same 3-skill dependency pool as ``test_skill_manage_closed_loop``
(csv-cleaner -> csv-plotter -> image-captioner) through the ``skill_manage``
service on real PostgreSQL/pgvector and checks the physical embedding layer:
one ``mindmemos.skill_manage_embeddings`` row per skill node with its
``content_hash``, the ``skill_manage_id_mappings`` bridge, KNN seeding of PPR,
vector reuse across upsert/remove, same-key replay, legacy generations without
vectors, profile mismatch and generation GC.

Requires ``MINDMEMOS_SKILL_TEST_PG_DSN``. Skipped without it.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

_TESTS_MINDMEMOS = Path(__file__).parents[2] / "mindmemos_skill"
if str(_TESTS_MINDMEMOS) not in sys.path:
    sys.path.insert(0, str(_TESTS_MINDMEMOS))

from mindmemos_skill.algos.skill_manage.algorithm import DefaultSkillManageAlgorithm  # noqa: E402
from mindmemos_skill.algos.skill_manage.config import SkillManageConfig  # noqa: E402
from mindmemos_skill.algos.skill_manage.embedding import embedding_text  # noqa: E402
from mindmemos_skill.application.skill_manage.loader import LocalSkillVersionLoader  # noqa: E402
from mindmemos_skill.application.skill_manage.models import (  # noqa: E402
    SkillManageBuildRequest,
    SkillManageOperationResult,
    SkillManageRemoveRequest,
    SkillManageRetrieveRequest,
    SkillManageRetrieveResult,
    SkillManageServiceSettings,
    SkillManageUpsertRequest,
)
from mindmemos_skill.application.skill_manage.service import SkillManageService  # noqa: E402
from mindmemos_skill.errors import SkillManageProfileMismatchError  # noqa: E402
from mindmemos_skill.infra.database import DatabaseConfig, Page, RecordQuery, bootstrap_database  # noqa: E402
from mindmemos_skill.infra.skill_manage import (  # noqa: E402
    build_skill_manage_vector_tables,
    generation_id_from_operation,
    snapshot_scope,
)
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
    SkillGraphEdge,
    SkillGraphNode,
    SkillManageRemoveSelector,
    SkillManageScope,
    SkillVersionRef,
)
from skill_manage_embedding_stub import ContentHashEmbedder, stub_algorithm_context  # noqa: E402

from mindmemos_skill import SkillApplication  # noqa: E402

pytestmark = pytest.mark.integration_pgvector

STUB_PROFILE = EmbeddingProfile(model="stub", dimension=3072)
GRAPH_ID = "g1"
POOL_ORDER = ("csv-cleaner", "csv-plotter", "image-captioner")

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

_PLOTTER_V2_DESCRIPTION = "Meridian chart rendering for cleaned CSV tables with axis labels."


@pytest.fixture
def requires_pg() -> str:
    if not DSN:
        pytest.skip("MINDMEMOS_SKILL_TEST_PG_DSN is not set")
    sep = "&" if "?" in DSN else "?"
    return f"{DSN}{sep}options=-csearch_path%3Dpublic"


async def _reset(dsn: str, project_id: str) -> None:
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
            # The pgvector table lives in the ``mindmemos`` schema and is never
            # dropped (its 3072 dimension is fixed for the shared DB), so clear
            # this test's project scope instead.
            cursor.execute("SELECT to_regclass('mindmemos.skill_manage_embeddings')")
            if cursor.fetchone()[0] is not None:
                cursor.execute(
                    "DELETE FROM mindmemos.skill_manage_embeddings WHERE _scope ->> 'project_id' = %s",
                    (project_id,),
                )
        connection.commit()


def _source(tmp_path: Path, name: str, description: str | None = None) -> Path:
    """Write one pool Skill as a SKILL.md with a typed I/O schema."""
    spec = _POOL[name]
    description = description or spec["description"]
    inputs = "\n".join(f'  - "{value}"' for value in spec["inputs"])
    outputs = "\n".join(f'  - "{value}"' for value in spec["outputs"])
    source = tmp_path / name
    source.mkdir(parents=True, exist_ok=True)
    (source / "SKILL.md").write_text(
        f"name: {name}\ndescription: {description}\ninputs:\n{inputs}\noutputs:\n{outputs}\n\n"
        f"# {name}\n\n{description}\n\n## Inputs\n\n{inputs}\n\n## Outputs\n\n{outputs}\n",
        encoding="utf-8",
    )
    return source


async def _make_adapter(requires_pg: str) -> PgSkillGraphStorageAdapter:
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
    return PgSkillGraphStorageAdapter(database=database, vector_store=vector_store)


def _make_service(
    app: SkillApplication,
    adapter: PgSkillGraphStorageAdapter,
    *,
    profile: EmbeddingProfile = STUB_PROFILE,
    settings: SkillManageServiceSettings | None = None,
) -> tuple[SkillManageService, ContentHashEmbedder]:
    context = stub_algorithm_context(3072)
    service = SkillManageService(
        algorithm=DefaultSkillManageAlgorithm(config=SkillManageConfig(), context=context),
        storage=adapter,
        embedding_profile=profile,
        settings=settings,
        version_loader=LocalSkillVersionLoader(app._manager.repository),
        id_generator=lambda: str(uuid.uuid4()),
    )
    return service, context.embedding_models["embedding"]


class _Pool:
    """One registered 3-skill pool plus the service/adapter bound to its project."""

    def __init__(
        self,
        *,
        app: SkillApplication,
        service: SkillManageService,
        adapter: PgSkillGraphStorageAdapter,
        embedder: ContentHashEmbedder,
        refs: dict[str, SkillVersionRef],
        project_id: str,
    ) -> None:
        self.app = app
        self.service = service
        self.adapter = adapter
        self.embedder = embedder
        self.refs = refs
        self.project_id = project_id
        self.scope = SkillManageScope(project_id=project_id, graph_id=GRAPH_ID)

    async def build(self) -> SkillManageOperationResult:
        return await self.service.build_skill_index(
            SkillManageBuildRequest(
                project_id=self.project_id,
                graph_id=GRAPH_ID,
                idempotency_key="build-pool",
                version_refs=tuple(self.refs[name] for name in POOL_ORDER),
            )
        )

    async def retrieve(self, query: str, **overrides: Any) -> SkillManageRetrieveResult:
        return await self.service.retrieve_skills(
            SkillManageRetrieveRequest(project_id=self.project_id, graph_id=GRAPH_ID, query=query, top_k=3, **overrides)
        )

    async def publish_plotter_v2(self, tmp_path: Path) -> SkillVersionRef:
        """Publish a csv-plotter version whose description (and thus embedding text) changed."""
        source = _source(tmp_path / "v2", "csv-plotter", _PLOTTER_V2_DESCRIPTION)
        published = await self.app.publish(
            PublishSkillRequest(skill_ref=self.refs["csv-plotter"].skill_id, source_path=source)
        )
        return SkillVersionRef(skill_id=self.refs["csv-plotter"].skill_id, version_id=published.version_id)

    async def upsert(self, ref: SkillVersionRef, *, key: str = "upsert-plot") -> SkillManageOperationResult:
        return await self.service.upsert_skill_index(
            SkillManageUpsertRequest(
                project_id=self.project_id, graph_id=GRAPH_ID, idempotency_key=key, version_ref=ref
            )
        )

    async def remove(self, name: str, *, key: str = "remove-caption") -> SkillManageOperationResult:
        return await self.service.remove_skill_from_index(
            SkillManageRemoveRequest(
                project_id=self.project_id,
                graph_id=GRAPH_ID,
                idempotency_key=key,
                selector=SkillManageRemoveSelector(kind="family", skill_id=self.refs[name].skill_id),
            )
        )

    async def head(self) -> dict[str, Any]:
        head = await self.adapter.get_graph_head(self.scope)
        assert head is not None
        return head

    async def mappings(self, generation_id: str) -> list[dict[str, Any]]:
        records, _ = await self.adapter._database.query_records(
            "skill_manage_id_mappings",
            RecordQuery(scope=snapshot_scope(self.scope, generation_id), page=Page(limit=100)),
        )
        return [dict(record.payload) for record in records]

    async def close(self) -> None:
        await self.adapter.close()
        await self.app.close()


async def _register_pool(requires_pg: str, tmp_path: Path, project_id: str) -> _Pool:
    await _reset(requires_pg, project_id)
    app = await SkillApplication.from_config(
        {"local": {"root_dir": str(tmp_path / "skill"), "database": {"provider": "sqlite", "path": "state.db"}}}
    )
    refs: dict[str, SkillVersionRef] = {}
    for name in POOL_ORDER:
        registered = await app.register(RegisterSkillRequest(source_path=_source(tmp_path, name), alias=f"pool-{name}"))
        refs[name] = SkillVersionRef(skill_id=registered.skill_id, version_id=registered.version_id)
    adapter = await _make_adapter(requires_pg)
    service, embedder = _make_service(app, adapter)
    return _Pool(app=app, service=service, adapter=adapter, embedder=embedder, refs=refs, project_id=project_id)


def _vector_rows(dsn: str, generation_id: str) -> dict[str, dict[str, Any]]:
    """Read the physical pgvector rows of one generation, keyed by graph_node_id."""
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT graph_node_id, skill_id, version_id, node_type, content_hash, embedding_model, "
                "embedding_profile_version, vector_record_id, vector_dims(embedding) AS dims "
                "FROM mindmemos.skill_manage_embeddings WHERE generation_id = %s",
                (generation_id,),
            )
            rows = cursor.fetchall()
    by_node = {row["graph_node_id"]: dict(row) for row in rows}
    assert len(by_node) == len(rows), "one vector row per graph_node_id"
    return by_node


def _by_name(hits) -> dict[str, dict[str, Any]]:
    return {hit["name"]: hit for hit in hits}


@pytest.mark.asyncio
async def test_build_writes_one_vector_per_skill_node(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-build")
    try:
        built = await pool.build()
        assert built.status.value == "committed"
        assert built.result_revision == 0
        assert built.node_count == 3
        assert (built.vector_count, built.inserted, built.reused) == (3, 3, 0)

        head = await pool.head()
        assert head["active_generation_id"] == built.generation_id
        assert head["embedding_profile"] == STUB_PROFILE.model_dump(mode="json")

        snapshot = await pool.adapter.load_snapshot(pool.scope, built.generation_id)
        rows = _vector_rows(requires_pg, built.generation_id)
        assert len(rows) == len(snapshot.nodes) == 3
        for node in snapshot.nodes:
            row = rows[node.graph_node_id]
            assert row["dims"] == 3072
            expected_hash = hashlib.sha256(embedding_text(node).encode("utf-8")).hexdigest()
            assert re.fullmatch(r"[0-9a-f]{64}", row["content_hash"])
            assert row["content_hash"] == expected_hash == node.content_hash
            assert row["embedding_model"] == "stub"
            assert row["embedding_profile_version"] == "1"
            assert row["node_type"] == "skill"
            assert (row["skill_id"], row["version_id"]) == (node.skill_id, node.version_id)

        loaded = await pool.adapter.load_embeddings(pool.scope, built.generation_id)
        assert {embedding.graph_node_id for embedding in loaded} == set(rows)
        assert all(len(embedding.vector) == 3072 for embedding in loaded)

        mappings = await pool.mappings(built.generation_id)
        assert len(mappings) == 3
        assert {mapping["graph_node_id"] for mapping in mappings} == set(rows)
        assert {mapping["vector_record_id"] for mapping in mappings} == {
            row["vector_record_id"] for row in rows.values()
        }
        assert all(mapping["mapping_type"] == "node" for mapping in mappings)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_retrieve_seeds_ppr_from_knn(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-retrieve")
    try:
        built = await pool.build()
        assert built.status.value == "committed"
        head = await pool.head()

        expectations = {
            "plot a chart from a cleaned table": ("csv-plotter", 1),
            "describe an image with a caption": ("image-captioner", 1),
            # csv-plotter is the PPR hub of the chain, so csv-cleaner may rank second.
            "clean a messy csv file": ("csv-cleaner", 2),
        }
        snapshot = await pool.adapter.load_snapshot(pool.scope, built.generation_id)
        name_by_node = {node.graph_node_id: node.name for node in snapshot.nodes}
        for query, (expected_name, within) in expectations.items():
            result = await pool.retrieve(query)
            assert result.revision == 0 == head["active_revision"]
            assert result.generation_id == built.generation_id
            hits = list(result.hits)
            assert len(hits) == 3
            names = [hit["name"] for hit in hits]
            assert expected_name in names[:within], f"{query!r}: {names}"

            # All three nodes are within seed_top_k with no threshold, so each
            # hit carries its own KNN similarity and shares the PPR mass.
            knn = await pool.adapter.search_embeddings(
                pool.scope, built.generation_id, query_vector=tuple(pool.embedder.embed_one(query)), top_k=3
            )
            knn_score = {name_by_node[hit.graph_node_id]: hit.score for hit in knn}
            assert set(knn_score) == set(names)
            for hit in hits:
                ref = pool.refs[hit["name"]]
                assert hit["skill_id"] == ref.skill_id
                assert hit["version_id"] == ref.version_id
                assert hit["skill_id"] != hit["version_id"]
                assert isinstance(hit["vector_score"], float)
                assert -1.0 <= hit["vector_score"] <= 1.0
                assert hit["vector_score"] == knn_score[hit["name"]]
            assert any(hit["ppr_score"] > 0 for hit in hits)
            assert sum(hit["ppr_score"] for hit in hits) == pytest.approx(1.0, abs=1e-6)

            again = await pool.retrieve(query)
            assert list(again.hits) == hits
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_search_embeddings_orders_by_cosine_score(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-search")
    try:
        built = await pool.build()
        assert built.status.value == "committed"
        snapshot = await pool.adapter.load_snapshot(pool.scope, built.generation_id)
        name_by_node = {node.graph_node_id: node.name for node in snapshot.nodes}
        query_vector = tuple(pool.embedder.embed_one("plot a chart from a cleaned table"))

        hits = await pool.adapter.search_embeddings(pool.scope, built.generation_id, query_vector=query_vector, top_k=3)
        assert len(hits) == 3
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 <= score <= 1.0 for score in scores)
        assert name_by_node[hits[0].graph_node_id] == "csv-plotter"
        assert hits[0].skill_id == pool.refs["csv-plotter"].skill_id
        assert hits[0].version_id == pool.refs["csv-plotter"].version_id

        limited = await pool.adapter.search_embeddings(
            pool.scope, built.generation_id, query_vector=query_vector, top_k=1
        )
        assert [hit.graph_node_id for hit in limited] == [hits[0].graph_node_id]

        nothing = await pool.adapter.search_embeddings(
            pool.scope, built.generation_id, query_vector=query_vector, top_k=3, score_threshold=2.0
        )
        assert nothing == ()

        # The service pushes the same threshold down: no seeds, PPR mass stays zero.
        unseeded = await pool.retrieve("plot a chart from a cleaned table", score_threshold=2.0)
        assert len(unseeded.hits) == 3
        assert all(hit["vector_score"] is None and hit["ppr_score"] == 0.0 for hit in unseeded.hits)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_upsert_reuses_unchanged_vectors(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-upsert")
    try:
        built = await pool.build()
        assert built.status.value == "committed"
        rows0 = _vector_rows(requires_pg, built.generation_id)
        embed_calls_before = len(pool.embedder.calls)

        plotter_v2 = await pool.publish_plotter_v2(tmp_path)
        assert plotter_v2.version_id != pool.refs["csv-plotter"].version_id
        upserted = await pool.upsert(plotter_v2)
        assert upserted.status.value == "committed"
        assert upserted.result_revision == 1
        assert (upserted.vector_count, upserted.inserted, upserted.reused) == (3, 1, 2)
        # Exactly one embedding call for the one changed node.
        build_calls = [
            call for call in pool.embedder.calls[embed_calls_before:] if call["task"] == "skill_manage.build"
        ]
        assert [call["count"] for call in build_calls] == [1]

        rows1 = _vector_rows(requires_pg, upserted.generation_id)
        assert set(rows1) == set(rows0)
        plotter_node = f"skill:{pool.refs['csv-plotter'].skill_id}"
        assert rows1[plotter_node]["content_hash"] != rows0[plotter_node]["content_hash"]
        assert rows1[plotter_node]["version_id"] == plotter_v2.version_id
        for node_id in set(rows0) - {plotter_node}:
            assert rows1[node_id]["content_hash"] == rows0[node_id]["content_hash"]
            assert rows1[node_id]["version_id"] == rows0[node_id]["version_id"]
        # Generation-qualified identity: the old generation's rows are untouched.
        assert _vector_rows(requires_pg, built.generation_id) == rows0

        result = await pool.retrieve("plot a chart from a cleaned table")
        assert result.revision == 1
        assert result.generation_id == upserted.generation_id
        hit = _by_name(result.hits)["csv-plotter"]
        assert hit["version_id"] == plotter_v2.version_id
        assert hit["skill_id"] == plotter_v2.skill_id
        assert result.hits[0]["name"] == "csv-plotter"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_remove_drops_vectors_and_mappings(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-remove")
    try:
        built = await pool.build()
        assert built.status.value == "committed"
        upserted = await pool.upsert(await pool.publish_plotter_v2(tmp_path))
        assert upserted.result_revision == 1
        rows1 = _vector_rows(requires_pg, upserted.generation_id)

        removed = await pool.remove("image-captioner")
        assert removed.status.value == "committed"
        assert removed.result_revision == 2
        assert (removed.vector_count, removed.inserted, removed.reused) == (2, 0, 2)

        rows2 = _vector_rows(requires_pg, removed.generation_id)
        captioner_node = f"skill:{pool.refs['image-captioner'].skill_id}"
        assert set(rows2) == set(rows1) - {captioner_node}
        assert len(rows2) == 2
        for node_id, row in rows2.items():
            assert row["content_hash"] == rows1[node_id]["content_hash"]
        mappings = await pool.mappings(removed.generation_id)
        assert len(mappings) == 2
        assert {mapping["vector_record_id"] for mapping in mappings} == {
            row["vector_record_id"] for row in rows2.values()
        }

        head = await pool.head()
        assert head["active_revision"] == 2
        assert head["active_generation_id"] == removed.generation_id

        result = await pool.retrieve("describe an image with a caption")
        assert result.revision == 2
        assert len(result.hits) == 2
        assert {hit["name"] for hit in result.hits} == {"csv-cleaner", "csv-plotter"}
        assert all(isinstance(hit["vector_score"], float) for hit in result.hits)
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_same_key_retry_replays_recorded_result(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-replay")
    try:
        built = await pool.build()
        assert built.status.value == "committed"
        plotter_v2 = await pool.publish_plotter_v2(tmp_path)
        upserted = await pool.upsert(plotter_v2)
        removed = await pool.remove("image-captioner")
        assert (upserted.result_revision, removed.result_revision) == (1, 2)

        # Same key + same request: replay of the recorded outcome, no new generation.
        upsert_again = await pool.upsert(plotter_v2)
        assert upsert_again.status.value == "committed"
        assert upsert_again.operation_id == upserted.operation_id
        assert upsert_again.result_revision == 1
        assert upsert_again.generation_id == upserted.generation_id

        remove_again = await pool.remove("image-captioner")
        assert remove_again.status.value == "committed"
        assert remove_again.operation_id == removed.operation_id
        assert remove_again.result_revision == 2
        assert remove_again.generation_id == removed.generation_id

        head = await pool.head()
        assert head["active_revision"] == 2
        assert head["active_generation_id"] == removed.generation_id

        # Same key + different request: idempotency conflict, still no new generation.
        conflict = await pool.remove("csv-cleaner", key="remove-caption")
        assert conflict.status.value == "conflict"
        assert conflict.error_code == "SKILL_MANAGE_IDEMPOTENCY_CONFLICT"
        assert conflict.operation_id == removed.operation_id
        assert (await pool.head())["active_revision"] == 2

        operation = await pool.service.get_skill_manage_operation(
            removed.operation_id, project_id=pool.project_id, graph_id=GRAPH_ID
        )
        assert operation is not None
        assert operation["result_revision"] == 2
        assert operation["current_generation_id"] == removed.generation_id
        assert operation["idempotency_key"] == "remove-caption"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_legacy_generation_without_vectors_degrades_to_lexical(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-legacy")
    try:
        legacy_scope = SkillManageScope(project_id=pool.project_id, graph_id="legacy")
        adapter = pool.adapter
        assert await adapter.reserve_operation(
            legacy_scope, "op-legacy", idempotency_key="legacy", request_hash="h-legacy", operation_type="build"
        )
        await adapter.stage_generation(
            legacy_scope, "gen-legacy", operation_id="op-legacy", attempt_no=1, base_revision=None, target_revision=0
        )
        nodes = [
            SkillGraphNode(
                graph_node_id=f"skill:{pool.refs[name].skill_id}",
                node_type="skill",
                skill_id=pool.refs[name].skill_id,
                version_id=pool.refs[name].version_id,
                name=name,
                description=_POOL[name]["description"],
                inputs=tuple(_POOL[name]["inputs"]),
                outputs=tuple(_POOL[name]["outputs"]),
            )
            for name in POOL_ORDER
        ]
        edges = [
            SkillGraphEdge(
                edge_id="edge:0:csv-cleaner:csv-plotter:dependency",
                edge_type="dependency",
                source_id=nodes[0].graph_node_id,
                target_id=nodes[1].graph_node_id,
                weight=1.0,
            )
        ]
        await adapter.write_snapshot(legacy_scope, "gen-legacy", nodes=nodes, edges=edges, memberships=[])
        assert await adapter.mark_generation_ready(
            legacy_scope, "gen-legacy", node_count=len(nodes), edge_count=len(edges), vector_count=0, counts_hash="h"
        )
        assert await adapter.publish_generation(legacy_scope, "gen-legacy", expected_revision=None, new_revision=0)
        head = await adapter.get_graph_head(legacy_scope)
        assert head is not None and head["embedding_profile"] == {}
        assert await adapter.load_embeddings(legacy_scope, "gen-legacy") == ()

        query_calls = len([call for call in pool.embedder.calls if call["task"] == "skill_manage.query"])
        result = await pool.service.retrieve_skills(
            SkillManageRetrieveRequest(
                project_id=pool.project_id, graph_id="legacy", query="plot a chart from a cleaned table", top_k=3
            )
        )
        assert result.revision == 0
        assert result.generation_id == "gen-legacy"
        assert len(result.hits) == 3
        assert all(hit["ppr_score"] == 0.0 and hit["vector_score"] is None for hit in result.hits)
        assert result.hits[0]["name"] == "csv-plotter"
        # The vector stage is skipped entirely: the query is never embedded.
        assert len([call for call in pool.embedder.calls if call["task"] == "skill_manage.query"]) == query_calls
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_retrieve_rejects_mismatched_profile(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-profile")
    try:
        built = await pool.build()
        assert built.status.value == "committed"

        other, _ = _make_service(pool.app, pool.adapter, profile=EmbeddingProfile(model="other", dimension=3072))
        with pytest.raises(SkillManageProfileMismatchError) as excinfo:
            await other.retrieve_skills(
                SkillManageRetrieveRequest(project_id=pool.project_id, graph_id=GRAPH_ID, query="plot a chart")
            )
        assert excinfo.value.expected["model"] == "other"
        assert excinfo.value.actual == STUB_PROFILE.model_dump(mode="json")

        # A mismatched profile on a write request is rejected before any I/O.
        with pytest.raises(SkillManageProfileMismatchError):
            await pool.service.upsert_skill_index(
                SkillManageUpsertRequest(
                    project_id=pool.project_id,
                    graph_id=GRAPH_ID,
                    idempotency_key="upsert-other-profile",
                    version_ref=pool.refs["csv-plotter"],
                    embedding_profile=EmbeddingProfile(model="other", dimension=3072),
                )
            )
        assert (await pool.head())["active_revision"] == 0

        # The matching service still retrieves against the same head.
        result = await pool.retrieve("plot a chart from a cleaned table")
        assert result.hits[0]["name"] == "csv-plotter"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_gc_deletes_retired_generation_vectors(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-gc")
    try:
        built = await pool.build()
        upserted = await pool.upsert(await pool.publish_plotter_v2(tmp_path))
        assert upserted.result_revision == 1
        gen0, gen1 = built.generation_id, upserted.generation_id
        assert len(_vector_rows(requires_pg, gen0)) == 3
        assert len(_vector_rows(requires_pg, gen1)) == 3
        rows1 = _vector_rows(requires_pg, gen1)

        # The publish of gen1 retired gen0 in the same transaction.
        retired = await pool.adapter.get_generation(pool.scope, gen0)
        assert retired is not None and retired["status"] == "retired" and retired["retired_at"] is not None
        # keep=1 must protect the newest retired generation; the default
        # minimum age protects it too; keep=0 with no age floor collects it.
        assert await pool.adapter.gc_retired_generations(pool.scope, keep=1) == []
        assert await pool.adapter.gc_retired_generations(pool.scope, keep=0, min_age_seconds=86400) == []
        assert len(_vector_rows(requires_pg, gen0)) == 3
        assert await pool.adapter.gc_retired_generations(pool.scope, keep=0) == [gen0]

        assert _vector_rows(requires_pg, gen0) == {}
        assert await pool.mappings(gen0) == []
        gone = await pool.adapter.load_snapshot(pool.scope, gen0)
        assert gone.nodes == () and gone.edges == ()
        assert await pool.adapter.get_generation(pool.scope, gen0) is None

        assert _vector_rows(requires_pg, gen1) == rows1
        assert len(await pool.mappings(gen1)) == 3
        head = await pool.head()
        assert head["active_generation_id"] == gen1
        intact = await pool.adapter.load_snapshot(pool.scope, gen1)
        assert len(intact.nodes) == 3
        result = await pool.retrieve("plot a chart from a cleaned table")
        assert result.revision == 1 and result.hits[0]["name"] == "csv-plotter"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_lost_head_cas_records_failure_and_discards_staged_rows(requires_pg: str, tmp_path: Path) -> None:
    pool = await _register_pool(requires_pg, tmp_path, "proj-emb-conflict")
    try:
        built = await pool.build()
        adapter = pool.adapter
        original_publish = adapter.publish_generation
        losses = {"left": 1}

        async def lose_once(scope, generation_id, *, expected_revision, new_revision, **kwargs):
            if losses["left"] and expected_revision is not None:
                losses["left"] -= 1
                return False
            return await original_publish(
                scope, generation_id, expected_revision=expected_revision, new_revision=new_revision, **kwargs
            )

        adapter.publish_generation = lose_once  # type: ignore[method-assign]
        lost = await pool.remove("image-captioner", key="remove-lost")
        assert lost.status.value == "conflict"
        assert lost.error_code == "SKILL_MANAGE_REVISION_CONFLICT"
        assert lost.error_retryable is False
        loser_generation = generation_id_from_operation(lost.operation_id, 1)

        # The loser's staged generation, vectors and mappings are gone; the head is untouched.
        assert _vector_rows(requires_pg, loser_generation) == {}
        assert await pool.mappings(loser_generation) == []
        assert await adapter.get_generation(pool.scope, loser_generation) is None
        head = await pool.head()
        assert head["active_revision"] == 0 and head["active_generation_id"] == built.generation_id

        operation = await pool.service.get_skill_manage_operation(
            lost.operation_id, project_id=pool.project_id, graph_id=GRAPH_ID
        )
        assert operation is not None
        assert operation["status"] == "conflict"
        assert operation["error_code"] == "SKILL_MANAGE_REVISION_CONFLICT"
        assert operation["error_retryable"] is False
        lease = await adapter.get_writer_lease(pool.scope)
        assert lease is not None and lease["owner_operation_id"] is None

        # conflict is terminal: the same key replays it, a new key re-drives the remove.
        replay = await pool.remove("image-captioner", key="remove-lost")
        assert replay.status.value == "conflict"
        assert replay.error_code == "SKILL_MANAGE_REVISION_CONFLICT"

        retried = await pool.remove("image-captioner", key="remove-retry")
        assert retried.status.value == "committed" and retried.result_revision == 1
        assert len(_vector_rows(requires_pg, retried.generation_id)) == 2
    finally:
        await pool.close()
