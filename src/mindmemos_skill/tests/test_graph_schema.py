"""Graph schema integrity, backend declarations, and portable round trips."""

from datetime import timedelta
from uuid import uuid4

import pytest
from mindmemos_skill.common.time_utils import utcnow
from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence import (
    SkillGraphEdgeRecord,
    SkillGraphNodeRecord,
    SkillGraphRecord,
    SkillGraphRevisionRecord,
    SkillGraphVersionRecord,
    SkillVersionRecord,
    build_persistence_tables,
    create_persistence,
    ensure_collection,
)
from mindmemos_skill.persistence.constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_REVISIONS_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_VERSIONS_TABLE,
)
from sqlalchemy import create_mock_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable


@pytest.fixture
def database(tmp_path):
    db = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'graphs.db'}"))
    store = create_persistence(db, include_remote=False)
    ensure_collection(store)
    try:
        yield db
    finally:
        db.close()


def add_graph(db, project="p1", graph="g1"):
    record = SkillGraphRecord(project_id=project, graph_id=graph, name="Graph")
    db.insert(SKILL_GRAPHS_TABLE, [record.model_dump()])


def add_version(db, project="p1", graph="g1"):
    record = SkillGraphVersionRecord(project_id=project, graph_id=graph, graph_version_id=uuid4())
    db.insert(SKILL_GRAPH_VERSIONS_TABLE, [record.model_dump()])
    return record.graph_version_id


def add_skill(db, version="skill-v1", project="p1"):
    record = SkillVersionRecord(
        project_id=project, version_id=version, skill_id="skill1", name="Skill", bundle="content", content_hash="hash"
    )
    db.insert(SKILL_VERSIONS_TABLE, [record.model_dump()])


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


def test_snapshot_nodes_edges_and_dynamic_vectors_round_trip(database):
    add_graph(database)
    add_skill(database)
    v1 = add_version(database)
    first = node(v1)
    scene = node(v1, node_id="scene:1", node_type="scene", skill_version_id=None, owner_node_id="skill:1")
    database.insert(SKILL_GRAPH_NODES_TABLE, [first.model_dump(), scene.model_dump()])
    edge = SkillGraphEdgeRecord(
        project_id="p1",
        graph_id="g1",
        graph_version_id=v1,
        edge_id=uuid4(),
        source_node_id="skill:1",
        target_node_id="scene:1",
        weight=0.5,
    )
    database.insert(SKILL_GRAPH_EDGES_TABLE, [edge.model_dump()])
    assert (
        SkillGraphNodeRecord.model_validate(database.select(SKILL_GRAPH_NODES_TABLE, filters={"node_id": "skill:1"})[0])
        == first
    )
    assert SkillGraphEdgeRecord.model_validate(database.select(SKILL_GRAPH_EDGES_TABLE)[0]) == edge
    v2 = add_version(database)
    different_dimension = node(v2, embedding=[1.0, 0.5])
    database.insert(SKILL_GRAPH_NODES_TABLE, [different_dimension.model_dump()])
    assert database.select(SKILL_GRAPH_NODES_TABLE, filters={"graph_version_id": v2})[0]["embedding"] == [1.0, 0.5]


def test_project_and_snapshot_foreign_keys(database):
    add_graph(database)
    add_graph(database, project="p2")
    add_skill(database)
    v1 = add_version(database)
    v2 = add_version(database, project="p2")
    with pytest.raises(IntegrityError):
        database.insert(SKILL_GRAPH_NODES_TABLE, [node(v2, project_id="p2").model_dump()])
    database.insert(SKILL_GRAPH_NODES_TABLE, [node(v1).model_dump()])
    other = add_version(database)
    with pytest.raises(IntegrityError):
        database.insert(
            SKILL_GRAPH_NODES_TABLE,
            [
                node(
                    other, node_id="scene:1", node_type="scene", skill_version_id=None, owner_node_id="skill:1"
                ).model_dump()
            ],
        )
    with pytest.raises(IntegrityError):
        database.insert(
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
    with pytest.raises(IntegrityError):
        database.delete(SKILL_VERSIONS_TABLE, filters={"version_id": "skill-v1"})


def test_uniqueness_prevents_duplicate_skill_and_edge(database):
    add_graph(database)
    add_skill(database)
    version = add_version(database)
    database.insert(SKILL_GRAPH_NODES_TABLE, [node(version).model_dump()])
    with pytest.raises(IntegrityError):
        database.insert(SKILL_GRAPH_NODES_TABLE, [node(version, node_id="another").model_dump()])
    database.insert(
        SKILL_GRAPH_NODES_TABLE,
        [
            node(
                version, node_id="scene:1", node_type="scene", skill_version_id=None, owner_node_id="skill:1"
            ).model_dump()
        ],
    )
    values = dict(
        project_id="p1", graph_id="g1", graph_version_id=version, source_node_id="skill:1", target_node_id="scene:1"
    )
    database.insert(SKILL_GRAPH_EDGES_TABLE, [SkillGraphEdgeRecord(**values, edge_id=uuid4()).model_dump()])
    with pytest.raises(IntegrityError):
        database.insert(SKILL_GRAPH_EDGES_TABLE, [SkillGraphEdgeRecord(**values, edge_id=uuid4()).model_dump()])


def test_revision_idempotency_and_current_pointer_constraints(database):
    add_graph(database)
    add_graph(database, graph="g2")
    version = add_version(database)
    revision = SkillGraphRevisionRecord(
        project_id="p1",
        graph_id="g1",
        revision=1,
        graph_version_id=version,
        idempotency_key="request1",
        request_hash="b" * 64,
        action="build",
        actor_id="actor",
    )
    database.insert(SKILL_GRAPH_REVISIONS_TABLE, [revision.model_dump()])
    with pytest.raises(IntegrityError):
        database.insert(SKILL_GRAPH_REVISIONS_TABLE, [revision.model_copy(update={"revision": 2}).model_dump()])
    with pytest.raises(IntegrityError):
        database.insert(SKILL_GRAPH_REVISIONS_TABLE, [revision.model_copy(update={"graph_id": "g2"}).model_dump()])
    graph = database.tables.get(SKILL_GRAPHS_TABLE).table
    with database.engine.begin() as connection:
        connection.execute(graph.update().where(graph.c.graph_id == "g1").values(current_revision=1))
    with pytest.raises(IntegrityError):
        with database.engine.begin() as connection:
            connection.execute(graph.update().where(graph.c.graph_id == "g2").values(current_revision=1))
    with pytest.raises(IntegrityError):
        database.delete(SKILL_GRAPH_REVISIONS_TABLE, filters={"project_id": "p1", "graph_id": "g1", "revision": 1})
    assert database.update(SKILL_GRAPH_REVISIONS_TABLE, {"request_hash": "c" * 64}, filters={"revision": 1}) == 1


def test_ready_constraints_and_restore_references(database):
    add_graph(database)
    version = add_version(database)
    table = database.tables.get(SKILL_GRAPH_VERSIONS_TABLE).table
    with pytest.raises(IntegrityError):
        with database.engine.begin() as connection:
            connection.execute(table.update().values(status="ready"))
    revision = SkillGraphRevisionRecord(
        project_id="p1",
        graph_id="g1",
        revision=2,
        graph_version_id=version,
        idempotency_key="restore1",
        request_hash="a" * 64,
        action="restore",
        actor_id="a",
        restored_from_revision=1,
    )
    with pytest.raises(IntegrityError):
        database.insert(SKILL_GRAPH_REVISIONS_TABLE, [revision.model_dump()])
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
    registry = build_persistence_tables(include_remote=False)
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
        f"ALTER TABLE {SKILL_GRAPHS_TABLE} ADD CONSTRAINT skill_graphs_current_revision_fk" in sql for sql in statements
    )
    assert all("resource_locks" not in sql and "skill_graph_operations" not in sql for sql in statements)
