"""Behavioral contracts shared by real SQLite and optional PostgreSQL."""

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from mindmemos_skill.infra import Infrastructure
from mindmemos_skill.infra.database import (
    Filter,
    PostgresConfig,
    PostgresDatabase,
    SQLiteConfig,
    SQLiteDatabase,
    TableRegistry,
    TableSpec,
    Vector,
)
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint, text
from sqlalchemy.exc import IntegrityError

POSTGRES_TEST_URL = "MINDMEMOS_TEST_POSTGRES_URL"
"""Opt-in URL of a disposable PostgreSQL database with vector installed."""


class Record(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=30)
    status: str = "active"
    rank: int = Field(default=0, ge=0)
    note: str | None = None
    embedding: list[float] | None = None


def registry(suffix=""):
    metadata = MetaData()
    table = Table(
        f"infra_contract{suffix}",
        metadata,
        Column("id", String, primary_key=True),
        Column("name", String(30), nullable=False),
        Column("status", String, nullable=False),
        Column("rank", Integer, nullable=False),
        Column("note", String),
        Column("embedding", Vector(3)),
        UniqueConstraint("name"),
    )
    return TableRegistry([TableSpec(name="items", table=table, insert_model=Record)])


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path):
    tables = registry("_" + uuid4().hex[:12])
    if request.param == "postgres":
        url = os.environ.get(POSTGRES_TEST_URL)
        if not url:
            pytest.skip(f"Set {POSTGRES_TEST_URL} to run real PostgreSQL contracts")
        database = PostgresDatabase(PostgresConfig(url=url), tables=tables)
    else:
        database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'store.db'}"), tables=tables)
    database.create_tables()
    try:
        yield database
    finally:
        with database.engine.begin() as connection:
            tables.get("items").table.drop(connection)
        database.close()


def seed(db):
    db.insert(
        "items",
        [
            {"id": "a", "name": "alpha", "embedding": [1.0, 0.0, 0.0]},
            {"id": "b", "name": "beta", "embedding": [0.0, 1.0, 0.0], "rank": 2},
            {"id": "c", "name": "gamma", "embedding": [-1.0, 0.0, 0.0], "status": "hidden"},
        ],
    )


def test_crud_defaults_projection_and_filters(db):
    seed(db)
    assert db.get("items", "missing") is None
    assert db.get("items", "a")["status"] == "active"
    assert db.count("items", filters={"status": "active"}) == 2
    assert db.select("items", fields=["id"], order_by=["-rank"], limit=1) == [{"id": "b"}]
    assert db.select("items", fields=["id"], offset=2) == [{"id": "c"}]
    assert db.count("items", filters=[Filter(field="rank", op="ge", value=1)]) == 1
    assert db.count("items", filters=[Filter(field="id", op="in", value=["a", "c"])]) == 2
    assert db.count("items", filters=[Filter(field="note", op="is_null")]) == 3
    assert db.update("items", {"note": "kept"}, filters={"id": "a"}) == 1
    assert db.update_many("items", [{"id": "a", "rank": 3}, {"id": "missing", "rank": 1}]) == 1
    assert db.get("items", "a")["note"] == "kept"
    assert db.update("items", {"note": None}, filters={"id": "a"}) == 1
    assert db.get("items", "a")["note"] is None
    assert db.delete("items", filters={"id": "b"}) == 1
    assert db.count("items") == 2


def test_vector_search_filter_before_top_k(db):
    seed(db)
    hits = db.search("items", vector_field="embedding", query_vector=[1.0, 0.0, 0.0], fields=["id"], top_k=3)
    assert [h.record["id"] for h in hits] == ["a", "b", "c"]
    assert [h.distance for h in hits] == pytest.approx([0.0, 1.0, 2.0], abs=1e-6)
    filtered = db.search(
        "items", vector_field="embedding", query_vector=[1.0, 0.0, 0.0], filters={"status": "hidden"}, top_k=1
    )
    assert filtered[0].record["id"] == "c"
    db.update_many("items", [{"id": "c", "embedding": [1.0, 0.0, 0.0]}])
    assert db.get("items", "c")["embedding"] == pytest.approx([1.0, 0.0, 0.0])
    db.insert("items", [{"id": "n", "name": "no_vector"}])
    assert len(db.search("items", vector_field="embedding", query_vector=[1.0, 0.0, 0.0])) == 3


@pytest.mark.parametrize("vector", [[0.0, 0.0, 0.0], [1.0, 2.0], [float("nan"), 0.0, 0.0], [1e100, 0.0, 0.0]])
def test_invalid_vectors_rejected_before_write(db, vector):
    with pytest.raises(ValueError):
        db.insert("items", [{"id": "a", "name": "alpha", "embedding": vector}])
    with pytest.raises(ValueError):
        db.search("items", vector_field="embedding", query_vector=vector)
    assert db.count("items") == 0


def test_validation_entire_batch_and_partial_fields(db):
    with pytest.raises(ValueError, match="record 1.*rank"):
        db.insert("items", [{"id": "a", "name": "alpha"}, {"id": "b", "name": "beta", "rank": -1}])
    assert db.count("items") == 0
    seed(db)
    with pytest.raises(ValueError, match="record 1"):
        db.update_many("items", [{"id": "a", "rank": 3}, {"id": "b", "name": ""}])
    assert db.get("items", "a")["rank"] == 0
    with pytest.raises(ValueError, match="Unknown input fields"):
        db.insert("items", [{"id": "d", "name": "delta", "typo": 1}])
    with pytest.raises(ValueError):
        db.update("items", {"name": None}, filters={"id": "a"})
    with pytest.raises(ValueError):
        db.update("items", {"rank": "3"}, filters={"id": "a"})
    assert db.update("items", {"id": "renamed"}, filters={"id": "a"}) == 1
    assert db.get("items", "a") is None
    assert db.get("items", "renamed")["name"] == "alpha"


def test_atomic_batch_and_transaction_rollback(db):
    with pytest.raises(IntegrityError):
        db.insert("items", [{"id": "a", "name": "same"}, {"id": "b", "name": "same"}])
    assert db.count("items") == 0
    with pytest.raises(RuntimeError, match="abort"):
        with db.transaction() as tx:
            tx.insert("items", [{"id": "a", "name": "alpha"}])
            assert tx.get("items", "a") is not None
            raise RuntimeError("abort")
    assert db.count("items") == 0
    with pytest.raises(RuntimeError, match="closed"):
        tx.count("items")
    with db.transaction() as tx:
        tx.insert("items", [{"id": "a", "name": "alpha"}])
        with pytest.raises(IntegrityError):
            tx.insert("items", [{"id": "b", "name": "beta"}, {"id": "c", "name": "alpha"}])
        assert tx.count("items") == 1
    assert db.count("items") == 1


def test_upsert_unique_constraint_and_explicit_fields(db):
    seed(db)
    assert db.upsert("items", [{"id": "a", "name": "new", "rank": 5}], update_fields=["name"]) == 1
    assert db.get("items", "a")["rank"] == 0
    assert db.get("items", "a")["name"] == "new"
    assert db.upsert("items", [{"id": "a", "name": "ignored"}]) == 0
    assert (
        db.upsert("items", [{"id": "new", "name": "new", "rank": 9}], conflict_fields=["name"], update_fields=["rank"])
        == 1
    )
    assert db.get("items", "a")["rank"] == 9
    assert db.get("items", "new") is None
    with pytest.raises(ValueError):
        db.upsert("items", [{"id": "z", "name": "z"}], conflict_fields=["status"])


def test_unknown_query_fields_and_invalid_operands(db):
    for operation in [
        lambda: db.count("unknown"),
        lambda: db.select("items", fields=["typo"]),
        lambda: db.select("items", filters={"rank": "1"}),
        lambda: db.select("items", filters=[Filter(field="rank", op="gt")]),
        lambda: db.select("items", limit=-1),
        lambda: db.search("items", vector_field="name", query_vector=[1.0, 0.0, 0.0]),
        lambda: db.update("items", {"typo": "a"}, filters={}),
    ]:
        with pytest.raises(ValueError):
            operation()


def test_registry_validation_and_isolation():
    tables = registry()
    with pytest.raises(ValueError, match="Duplicate"):
        tables.register(tables.get("items"))
    assert TableRegistry().values() == ()
    with pytest.raises(ValueError, match="primary key"):
        TableSpec(name="bad", table=Table("bad", MetaData(), Column("id", String)), insert_model=Record)
    with pytest.raises(ValueError, match="absent"):
        TableSpec(
            name="bad", table=Table("bad", MetaData(), Column("id", String, primary_key=True)), insert_model=Record
        )


def test_default_backend_registration_is_lazy(tmp_path):
    path = tmp_path / "nested" / "store.db"
    database = Infrastructure.build("sqlite", {"url": f"sqlite:///{path}"})
    assert not path.parent.exists()
    database.tables.register(registry().get("items"))
    database.create_tables()
    assert path.exists()
    database.close()
    with pytest.raises(RuntimeError, match="closed"):
        database.count("items")


def test_sqlite_connections_and_concurrent_writes(tmp_path):
    database = SQLiteDatabase(
        SQLiteConfig(url=f"sqlite:///{tmp_path / 'concurrent.db'}", timeout=10), tables=registry()
    )
    database.create_tables()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: database.insert("items", [{"id": str(i), "name": f"row{i}"}]), range(20)))
        assert database.count("items") == 20
        with database.engine.connect() as a, database.engine.connect() as b:
            assert a.scalar(text("SELECT vec_version()"))
            assert b.scalar(text("PRAGMA foreign_keys")) == 1
            assert b.scalar(text("PRAGMA journal_mode")) == "wal"
    finally:
        database.close()


def test_composite_key_server_defaults_and_foreign_keys(tmp_path):
    class Parent(BaseModel):
        id: int | None = None
        name: str

    class Child(BaseModel):
        parent_id: int
        key: str
        value: int = 0

    from sqlalchemy import ForeignKey

    metadata = MetaData()
    parent = Table("parents", metadata, Column("id", Integer, primary_key=True), Column("name", String, nullable=False))
    child = Table(
        "children",
        metadata,
        Column("parent_id", ForeignKey("parents.id"), primary_key=True),
        Column("key", String, primary_key=True),
        Column("value", Integer, nullable=False),
    )
    tables = TableRegistry(
        [
            TableSpec(name="children", table=child, insert_model=Child),
            TableSpec(name="parents", table=parent, insert_model=Parent),
        ]
    )
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'relations.db'}"), tables=tables)
    try:
        database.create_tables()
        database.insert("parents", [{"name": "p"}])
        assert database.get("parents", 1)["name"] == "p"
        database.insert("children", [{"parent_id": 1, "key": "k"}])
        database.update_many("children", [{"parent_id": 1, "key": "k", "value": 9}])
        assert database.get("children", {"parent_id": 1, "key": "k"})["value"] == 9
        with pytest.raises(IntegrityError):
            database.insert("children", [{"parent_id": 999, "key": "k"}])
        assert database.update("parents", {"name": "new"}, filters={"id": 1}) == 1
        assert database.get("parents", 1)["name"] == "new"
    finally:
        database.close()


def test_postgres_compilation_and_codec_without_server():
    pytest.importorskip("psycopg")
    pytest.importorskip("pgvector")
    from sqlalchemy.schema import CreateTable

    tables = registry()
    database = PostgresDatabase(
        PostgresConfig(url="postgresql+psycopg://unused:unused@localhost/unused"), tables=tables
    )
    try:
        table = tables.get("items").table
        dialect = database.engine.dialect
        assert "VECTOR(3)" in str(CreateTable(table).compile(dialect=dialect))
        expression = database._distance(table.c.embedding, [1.0, 0.0, 0.0])
        assert "<=>" in str(expression.compile(dialect=dialect))
        vector_type = table.c.embedding.type.dialect_impl(dialect)
        stored = vector_type.bind_processor(dialect)([1.0, 0.0, 0.0])
        assert vector_type.result_processor(dialect, None)(stored) == [1.0, 0.0, 0.0]
    finally:
        database.close()


def test_service_shutdown_closes_database(tmp_path):
    from fastapi.testclient import TestClient
    from mindmemos_skill.api.entries.fastapi import create_app

    from mindmemos_skill import ServiceConfig, create_service

    service = create_service(
        ServiceConfig.model_validate(
            {
                "backends": {
                    "db": {
                        "module": "infra",
                        "class_name": "sqlite",
                        "config": {"url": f"sqlite:///{tmp_path / 'service.db'}"},
                    }
                }
            }
        )
    )
    database = service.backends["db"]
    database.tables.register(registry().get("items"))
    database.create_tables()
    with TestClient(create_app(service)) as client:
        assert client.get("/health").status_code == 200
        assert database.count("items") == 0
    with pytest.raises(RuntimeError, match="closed"):
        database.count("items")
