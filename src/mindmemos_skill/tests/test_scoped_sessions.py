"""Security boundaries for project sessions across service and infra layers."""

import asyncio

import pytest
from fastapi.testclient import TestClient
from mindmemos_skill.api.auth import SkillHttpIdentity
from mindmemos_skill.api.entries.fastapi import create_app
from mindmemos_skill.api.typing import ListRequest
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase, TableSpec, Vector
from pydantic import BaseModel
from sqlalchemy import Column, MetaData, Table, Text

from mindmemos_skill import FastAPIConfig, HttpAuthConfig, ServiceConfig, create_service


class Row(BaseModel):
    """Small scalar and vector record used to verify every scoped operation."""

    project_id: str
    """Project owning the row."""

    item_id: str
    """Project-local row identity."""

    agent_id: str | None = None
    """Optional agent owner."""

    embedding: list[float] | None = None
    """Optional two-dimensional search vector."""


class UnscopedRow(BaseModel):
    """Record intentionally lacking a project ownership field."""

    item_id: str
    """Identity of an unscoped record."""


@pytest.fixture
async def database(tmp_path):
    """Build an isolated SQLite resource with project and agent dimensions."""
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'scoped.db'}"))
    table = Table(
        "scope_test_rows",
        MetaData(),
        Column("project_id", Text, primary_key=True),
        Column("item_id", Text, primary_key=True),
        Column("agent_id", Text, nullable=True),
        Column("embedding", Vector(2), nullable=True),
    )
    database.tables.register(TableSpec(name="scope_test_rows", table=table, insert_model=Row))
    unscoped = Table("unscoped_test_rows", MetaData(), Column("item_id", Text, primary_key=True))
    database.tables.register(TableSpec(name="unscoped_test_rows", table=unscoped, insert_model=UnscopedRow))
    (await database.create_tables())
    try:
        yield database
    finally:
        (await database.close())


async def test_scoped_database_guards_reads_writes_and_search(database):
    """No operation can read or alter a second project's row."""
    table = "scope_test_rows"
    (await database.insert(table, [{"project_id": "other", "item_id": "same", "embedding": [1.0, 0.0]}]))
    with database.scoped(Scope(fields={"project_id": "mine"})) as scoped:
        (await scoped.insert(table, [{"item_id": "same", "embedding": [0.0, 1.0]}]))
        assert (await scoped.count(table)) == 1
        assert (await scoped.select_distinct(table, "item_id")) == ["same"]
        assert (await scoped.select_distinct(table, "project_id")) == ["mine"]
        assert (await scoped.get(table, {"project_id": "mine", "item_id": "same"}))["project_id"] == "mine"
        assert (await scoped.search(table, vector_field="embedding", query_vector=[0.0, 1.0]))[0].record[
            "project_id"
        ] == "mine"
        assert (await scoped.update(table, {"agent_id": "a"}, filters={})) == 1
        assert (await scoped.update_many(table, [{"project_id": "mine", "item_id": "same", "agent_id": "b"}])) == 1
        async with scoped.transaction() as transaction:
            assert (await transaction.count(table)) == 1
            assert (await transaction.upsert(table, [{"item_id": "new"}])) == 1
        with pytest.raises(ValueError, match="conflicts"):
            (await scoped.select(table, filters={"project_id": "other"}))
        with pytest.raises(ValueError, match="conflicts"):
            (await scoped.insert(table, [{"project_id": "other", "item_id": "wrong"}]))
        with pytest.raises(ValueError, match="Scope fields cannot be updated"):
            (await scoped.update(table, {"project_id": "other"}, filters={}))
        assert (await scoped.delete(table, filters={})) == 2
    assert (await database.count(table, filters={"project_id": "other"})) == 1
    with pytest.raises(RuntimeError, match="closed"):
        (await scoped.count(table))


async def test_missing_scope_dimension_fails_closed(database):
    """A requested field cannot be silently dropped from a resource."""
    table = "scope_test_rows"
    with database.scoped(Scope(fields={"project_id": "mine", "user_id": "u"})) as scoped:
        with pytest.raises(ValueError, match="user_id"):
            (await scoped.count(table))
    with database.scoped(Scope(fields={"project_id": "mine", "dataset_id": "d"})) as scoped:
        with pytest.raises(ValueError, match="dataset_id"):
            (await scoped.count(table))
    with database.scoped(Scope(fields={"project_id": "mine"})) as scoped:
        with pytest.raises(ValueError, match="project_id"):
            (await scoped.count("unscoped_test_rows"))


async def test_agent_scope_narrows_rows_within_a_project(database):
    """Optional agent ownership is applied when explicitly requested."""
    table = "scope_test_rows"
    with database.scoped(Scope(fields={"project_id": "mine", "agent_id": "a"})) as scoped:
        (await scoped.insert(table, [{"item_id": "a"}]))
        assert (await scoped.get(table, {"project_id": "mine", "item_id": "a"}))["agent_id"] == "a"
    with database.scoped(Scope(fields={"project_id": "mine", "agent_id": "b"})) as scoped:
        (await scoped.insert(table, [{"item_id": "b"}]))
        assert (await scoped.count(table)) == 1
        assert (await scoped.get(table, {"project_id": "mine", "item_id": "a"})) is None
        assert (await scoped.delete(table, filters={})) == 1
    assert (await database.count(table, filters={"project_id": "mine"})) == 1


async def test_infra_accepts_any_registered_scope_field(database):
    """Infra can isolate a table without assigning meaning to its column names."""
    (await database.insert("unscoped_test_rows", [{"item_id": "one"}, {"item_id": "two"}]))
    supplied = {"item_id": "one"}
    scope = Scope(fields=supplied)
    supplied["item_id"] = "two"
    with database.scoped(scope) as scoped:
        assert (await scoped.count("unscoped_test_rows")) == 1
        assert (await scoped.get("unscoped_test_rows", "one")) == {"item_id": "one"}
        with pytest.raises(ValueError, match="conflicts"):
            (await scoped.get("unscoped_test_rows", "two"))
        with pytest.raises(ValueError, match="conflicts"):
            (await scoped.delete("unscoped_test_rows", filters={"item_id": "two"}))
    with database.scoped({"item_id": "two"}) as scoped:
        assert (await scoped.count("unscoped_test_rows")) == 1
    with pytest.raises(ValueError, match="unique"):
        Scope(fields=(("item_id", "one"), ("item_id", "two")))


@pytest.mark.asyncio
async def test_cloud_service_requires_authorization_and_closes_graph_capability(tmp_path):
    """The service rejects missing identity and does not leak a graph query handle."""
    config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'cloud.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
            "manage_pipeline": {"class_name": "graph_manage"},
        }
    )

    def authorize(principal, project_id):
        if principal == "extra":
            return Scope(fields={"project_id": project_id, "agent_id": "agent"})
        if principal == "unknown":
            return Scope(fields={"project_id": project_id, "dataset_id": "dataset"})
        if principal != project_id:
            raise PermissionError("Project access denied")
        return Scope(fields={"project_id": project_id})

    service = await create_service(config, project_authorizer=authorize)
    try:
        with pytest.raises(PermissionError, match="authorization is required"):
            await service.list_skills(ListRequest(project_id="one"))
        with service.request_principal("one"):
            assert await service.list_skills(ListRequest(project_id="one")) == []
            assert await service.list_graph("one") == []
            with pytest.raises(PermissionError, match="access denied"):
                await service.list_skills(ListRequest(project_id="two"))
        with service.request_principal("extra"):
            with pytest.raises(ValueError, match="agent_id"):
                await service.list_skills(ListRequest(project_id="one"))
        with service.request_principal("unknown"):
            with pytest.raises(PermissionError, match="scope fields"):
                await service.list_skills(ListRequest(project_id="one"))
        with pytest.raises(PermissionError, match="authorization is required"):
            await service.list_graph("one")
        with pytest.raises(RuntimeError, match="session is required"):
            service.repository.persistence.skills
        with TestClient(
            create_app(
                service,
                api_config=FastAPIConfig(auth=HttpAuthConfig(mode="gateway_jwt", gateway_jwt_secret="test")),
                identity_from_bearer=lambda token: SkillHttpIdentity(
                    project_id=token, scopes={"skill:read"}, principal=token
                ),
            )
        ) as client:
            assert client.post("/v1/skills/list", json={}).status_code == 401
            assert client.post("/v1/skills/list", json={}, headers={"Authorization": "Bearer one"}).status_code == 200
            assert (
                client.post(
                    "/v1/skills/list", json={"project_id": "one"}, headers={"Authorization": "Bearer one"}
                ).status_code
                == 422
            )
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_principal_context_does_not_outlive_request(tmp_path):
    """A child task inheriting context cannot use a principal after request exit."""
    config = ServiceConfig.model_validate(
        {
            "home": tmp_path / "home",
            "persistence": {
                "database": {"class_name": "sqlite", "config": {"url": f"sqlite:///{tmp_path / 'cloud.db'}"}}
            },
            "repository_pipeline": {"class_name": "repository"},
        }
    )
    service = await create_service(
        config, project_authorizer=lambda principal, project_id: Scope(fields={"project_id": project_id})
    )
    ready = asyncio.Event()

    async def late_query():
        await ready.wait()
        with pytest.raises(PermissionError, match="authorization is required"):
            await service.list_skills(ListRequest(project_id="one"))

    try:
        with service.request_principal("one"):
            task = asyncio.create_task(late_query())
        ready.set()
        await task
    finally:
        await service.close()
