"""Graph schema integrity, backend declarations, and portable round trips."""

from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from mindmemos_skill.common.time_utils import utcnow
from mindmemos_skill.errors import GraphConflictError
from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence import (
    SkillGraphEdgeRecord,
    SkillGraphNodeRecord,
    SkillGraphRecord,
    SkillGraphVersionRecord,
    SkillVersionRecord,
    build_persistence_tables,
    create_persistence,
    ensure_collection,
)
from mindmemos_skill.persistence.constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_VERSIONS_TABLE,
)
from mindmemos_skill.persistence.db_services.skill_management.service import GraphQueryService
from mindmemos_skill.skill import content_hash, skill_hash
from sqlalchemy import create_mock_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable


@pytest.fixture
async def database(tmp_path):
    db = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'graphs.db'}"))
    store = create_persistence(db)
    (await ensure_collection(store))
    try:
        yield db
    finally:
        (await db.close())


async def add_graph(db, project="p1", graph="g1"):
    record = SkillGraphRecord(project_id=project, graph_id=graph, name="Graph")
    (await db.insert(SKILL_GRAPHS_TABLE, [record.model_dump()]))


async def add_version(db, project="p1", graph="g1"):
    record = SkillGraphVersionRecord(project_id=project, graph_id=graph, graph_version_id=uuid4())
    (await db.insert(SKILL_GRAPH_VERSIONS_TABLE, [record.model_dump()]))
    return record.graph_version_id


async def add_skill(db, version="skill-v1", project="p1"):
    record = SkillVersionRecord(
        project_id=project,
        version_id=version,
        skill_id="skill1",
        name="Skill",
        content="content",
        description="Instructions",
        standard_format=False,
        skill_hash=skill_hash("content", {}),
        content_hash=content_hash("content"),
    )
    (await db.insert(SKILL_VERSIONS_TABLE, [record.model_dump()]))


async def test_graph_resolve_rejects_ambiguous_family_and_version_prefixes(database):
    """Graph and snapshot prefixes must identify exactly one project-owned row."""
    for suffix in ("a", "b"):
        graph_id = f"g-{suffix}"
        snapshot = UUID(f"a{suffix}" + "0" * 30)
        (
            await database.insert(
                SKILL_GRAPHS_TABLE,
                [
                    SkillGraphRecord(
                        project_id="p1", graph_id=graph_id, public_id=f"skg-aa{suffix}", alias=graph_id, name=graph_id
                    ).model_dump()
                ],
            )
        )
        (
            await database.insert(
                SKILL_GRAPH_VERSIONS_TABLE,
                [
                    SkillGraphVersionRecord(
                        project_id="p1",
                        graph_id=graph_id,
                        graph_version_id=snapshot,
                        public_version_id=f"skgv-{snapshot.hex}",
                    ).model_dump()
                ],
            )
        )
    resolver = GraphQueryService(database, "p1")
    assert (await resolver.resolve("g-a"))["graph_id"] == "skg-aaa"
    assert (await resolver.resolve("skg-aab"))["graph_id"] == "skg-aab"
    with pytest.raises(GraphConflictError, match="Ambiguous"):
        (await resolver.resolve("skg-aa"))
    with pytest.raises(GraphConflictError, match="Ambiguous"):
        (await resolver.resolve("skgv-a"))


def node(version, **changes):
    return SkillGraphNodeRecord(
        **(
            {
                "project_id": "p1",
                "graph_id": "g1",
                "graph_version_id": version,
                "node_id": "skill:1",
                "skill_version_id": "skill-v1",
                "name": "Skill",
                "embedding": [1.0, 2.0, 3.0],
                "embedding_input_hash": "a" * 64,
                "inputs": ["document"],
                "metadata": {"nested": [1, True, None]},
            }
            | changes
        )
    )


async def test_snapshot_nodes_edges_and_dynamic_vectors_round_trip(database):
    (await add_graph(database))
    (await add_skill(database))
    v1 = await add_version(database)
    first = node(v1)
    scene = node(v1, node_id="scene:1", node_type="scene", skill_version_id=None, owner_node_id="skill:1")
    (await database.insert(SKILL_GRAPH_NODES_TABLE, [first.model_dump(), scene.model_dump()]))
    edge = SkillGraphEdgeRecord(
        project_id="p1",
        graph_id="g1",
        graph_version_id=v1,
        edge_id=uuid4(),
        source_node_id="skill:1",
        target_node_id="scene:1",
        weight=0.5,
    )
    (await database.insert(SKILL_GRAPH_EDGES_TABLE, [edge.model_dump()]))
    assert (
        SkillGraphNodeRecord.model_validate(
            (await database.select(SKILL_GRAPH_NODES_TABLE, filters={"node_id": "skill:1"}))[0]
        )
        == first
    )
    assert SkillGraphEdgeRecord.model_validate((await database.select(SKILL_GRAPH_EDGES_TABLE))[0]) == edge
    v2 = await add_version(database)
    different_dimension = node(v2, embedding=[1.0, 0.5])
    (await database.insert(SKILL_GRAPH_NODES_TABLE, [different_dimension.model_dump()]))
    assert (await database.select(SKILL_GRAPH_NODES_TABLE, filters={"graph_version_id": v2}))[0]["embedding"] == [
        1.0,
        0.5,
    ]


async def test_project_and_snapshot_foreign_keys(database):
    (await add_graph(database))
    (await add_graph(database, project="p2"))
    (await add_skill(database))
    v1 = await add_version(database)
    v2 = await add_version(database, project="p2")
    with pytest.raises(IntegrityError):
        (await database.insert(SKILL_GRAPH_NODES_TABLE, [node(v2, project_id="p2").model_dump()]))
    (await database.insert(SKILL_GRAPH_NODES_TABLE, [node(v1).model_dump()]))
    other = await add_version(database)
    with pytest.raises(IntegrityError):
        (
            await database.insert(
                SKILL_GRAPH_NODES_TABLE,
                [
                    node(
                        other, node_id="scene:1", node_type="scene", skill_version_id=None, owner_node_id="skill:1"
                    ).model_dump()
                ],
            )
        )
    with pytest.raises(IntegrityError):
        (
            await database.insert(
                SKILL_GRAPH_EDGES_TABLE,
                [
                    SkillGraphEdgeRecord(
                        project_id="p1",
                        graph_id="g1",
                        graph_version_id=other,
                        edge_id=uuid4(),
                        source_node_id="skill:1",
                        target_node_id="missing",
                    ).model_dump()
                ],
            )
        )
    with pytest.raises(IntegrityError):
        (await database.delete(SKILL_VERSIONS_TABLE, filters={"version_id": "skill-v1"}))


async def test_uniqueness_prevents_duplicate_skill_and_edge(database):
    (await add_graph(database))
    (await add_skill(database))
    version = await add_version(database)
    (await database.insert(SKILL_GRAPH_NODES_TABLE, [node(version).model_dump()]))
    with pytest.raises(IntegrityError):
        (await database.insert(SKILL_GRAPH_NODES_TABLE, [node(version, node_id="another").model_dump()]))
    (
        await database.insert(
            SKILL_GRAPH_NODES_TABLE,
            [
                node(
                    version, node_id="scene:1", node_type="scene", skill_version_id=None, owner_node_id="skill:1"
                ).model_dump()
            ],
        )
    )
    values = dict(
        project_id="p1", graph_id="g1", graph_version_id=version, source_node_id="skill:1", target_node_id="scene:1"
    )
    (await database.insert(SKILL_GRAPH_EDGES_TABLE, [SkillGraphEdgeRecord(**values, edge_id=uuid4()).model_dump()]))
    with pytest.raises(IntegrityError):
        (await database.insert(SKILL_GRAPH_EDGES_TABLE, [SkillGraphEdgeRecord(**values, edge_id=uuid4()).model_dump()]))


async def test_snapshot_idempotency_and_current_pointer_constraints(database):
    (await add_graph(database))
    (await add_graph(database, graph="g2"))
    version = await add_version(database)
    assert (
        await database.update(
            SKILL_GRAPH_VERSIONS_TABLE, {"request_hash": "b" * 64}, filters={"graph_version_id": version}
        )
    ) == 1
    with pytest.raises(IntegrityError):
        (
            await database.insert(
                SKILL_GRAPH_VERSIONS_TABLE,
                [
                    SkillGraphVersionRecord(
                        project_id="p1", graph_id="g1", graph_version_id=uuid4(), request_hash="b" * 64
                    ).model_dump()
                ],
            )
        )
    graph = database.tables.get(SKILL_GRAPHS_TABLE).table
    async with database.engine.begin() as connection:
        (await connection.execute(graph.update().where(graph.c.graph_id == "g1").values(current_version_id=version)))
    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            (
                await connection.execute(
                    graph.update().where(graph.c.graph_id == "g2").values(current_version_id=version)
                )
            )
    with pytest.raises(IntegrityError):
        (
            await database.delete(
                SKILL_GRAPH_VERSIONS_TABLE, filters={"project_id": "p1", "graph_id": "g1", "graph_version_id": version}
            )
        )


async def test_ready_constraints(database):
    (await add_graph(database))
    version = await add_version(database)
    table = database.tables.get(SKILL_GRAPH_VERSIONS_TABLE).table
    with pytest.raises(IntegrityError):
        async with database.engine.begin() as connection:
            (await connection.execute(table.update().values(status="ready")))
    for changes in (
        {"status": "ready"},
        {"vector_count": 1},
        {"snapshot_hash": "not-a-hash"},
        {"created_at": utcnow(), "ready_at": utcnow() - timedelta(days=1)},
    ):
        with pytest.raises(ValueError):
            SkillGraphVersionRecord(project_id="p1", graph_id="g1", graph_version_id=uuid4(), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"embedding": [0.0, 0.0]},
        {"embedding": [float("inf")]},
        {"embedding": []},
        {"embedding": [True]},
        {"embedding_input_hash": None},
        {"owner_node_id": "other"},
        {"node_type": "scene", "skill_version_id": None, "owner_node_id": "skill:1"},
        {"inputs": [None]},
        {"metadata": []},
    ],
)
def test_invalid_node_values(changes):
    with pytest.raises(ValueError):
        node(uuid4(), **changes)


def test_postgres_ddl_types_and_circular_fk_creation():
    registry = build_persistence_tables()
    nodes = registry.get(SKILL_GRAPH_NODES_TABLE).table
    ddl = str(CreateTable(nodes).compile(dialect=postgresql.dialect()))
    assert "embedding VECTOR" in ddl
    assert "inputs TEXT[]" in ddl
    assert "metadata JSONB" in ddl
    assert f"REFERENCES {SKILL_VERSIONS_TABLE} (project_id, version_id)" in ddl
    statements = []
    engine = create_mock_engine(
        "postgresql://", lambda sql, *args, **kwargs: statements.append(str(sql.compile(dialect=postgresql.dialect())))
    )
    nodes.metadata.create_all(engine)
    assert any(
        f"ALTER TABLE {SKILL_GRAPHS_TABLE} ADD CONSTRAINT skill_graphs_current_version_fk" in sql for sql in statements
    )
    assert all("resource_locks" not in sql and "skill_graph_operations" not in sql for sql in statements)
