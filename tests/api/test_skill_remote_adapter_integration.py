"""End-to-end Skill v2 cloud protocol through Application and SDK HTTP."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from mindmemos.api.deps import get_request_context
from mindmemos.api.schemas import AuthContext
from mindmemos.api.services import get_skill_service
from mindmemos.api.services.skill_service import SkillService
from mindmemos.api.skill_routes import router
from mindmemos.errors import ApiError
from mindmemos.infra.db import SkillRelationalRepository, build_cloud_skill_tables
from mindmemos_sdk.composition import build_skill_remote_port
from mindmemos_sdk.config import HttpConnectionConfig
from mindmemos_sdk.connections import HttpConnection
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.infra.database import DatabaseConfig, bootstrap_database
from mindmemos_skill.management import PublishSkillRequest, RegisterSkillRequest

from mindmemos_skill import SkillApplication


@pytest.mark.asyncio
async def test_application_push_sync_and_status_cas_against_relational_cloud(tmp_path: Path) -> None:
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": ":memory:"}),
        build_cloud_skill_tables(),
    )
    repository = SkillRelationalRepository(database)
    service = SkillService(repository=repository)
    app = FastAPI()

    @app.exception_handler(ApiError)
    async def _handle_api_error(_request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    app.include_router(router)
    app.dependency_overrides[get_skill_service] = lambda: service
    app.dependency_overrides[get_request_context] = lambda: AuthContext(
        request_id="request",
        account_id="account",
        project_id="project",
        api_key_uuid="key",
        memory_algorithm="schema",
        scopes=["skills:read", "skills:write"],
    )
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    transport = AsyncHttpTransport(base_url="http://test", api_key="test-key", client=http_client)
    connection = HttpConnection(
        HttpConnectionConfig(base_url="http://test", api_key="test-key"),
        transport=transport,
        owns_transport=False,
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text('name: demo\nversion: "1.0.0"\n\nRoot\n', encoding="utf-8")
    (source / "scripts").mkdir()
    (source / "scripts" / "check.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "references").mkdir()
    (source / "references" / "private.md").write_text("user private\n", encoding="utf-8")
    application = await SkillApplication.from_config(
        {
            "local": {
                "root_dir": str(tmp_path / "local-skill"),
                "database": {"provider": "sqlite", "path": "state.db"},
            }
        },
        remote=build_skill_remote_port(connection),
    )
    try:
        registered = await application.register(
            RegisterSkillRequest(
                source_path=source,
                alias="demo",
                runtime_type="virtual_components",
                runtime_schema_version=1,
                runtime_metadata={
                    "components": [
                        {
                            "component_id": "root",
                            "name": "Root guidance",
                            "description": "root procedure",
                            "content": "Root\n",
                        }
                    ]
                },
            )
        )
        pushed_root = await application.push("demo")
        cloud_bundle = await repository.get_bundle("project", pushed_root.version_id)
        await application.sync("demo")
        child = await application.publish(
            PublishSkillRequest(
                skill_ref="demo",
                content='name: demo\nversion: "1.1.0"\n\nChild\n',
            )
        )
        pushed_child = await application.push("demo", child.version_id)
        status = await http_client.post(
            f"/v1/skills/versions/{child.version_id}/status",
            json={"status": "published", "expected_revision": 0},
        )
        latest = await repository.latest_available_version("project", pushed_root.cloud_skill_id)
    finally:
        await application.close()
        await http_client.aclose()
        await database.close()

    assert pushed_root.version_id == registered.version_id
    assert {item.path: item.content for item in cloud_bundle.files} == {
        "SKILL.md": 'name: demo\nversion: "1.0.0"\n\nRoot\n'
    }
    assert "print('ok')" not in cloud_bundle.canonical_json()
    assert "user private" not in cloud_bundle.canonical_json()
    assert pushed_child.version_id == child.version_id
    assert status.status_code == 200
    assert latest.version_id == child.version_id
    assert latest.status.value == "published"
    assert latest.runtime_type == "virtual_components"
    assert latest.runtime_metadata["components"][0]["component_id"] == "root"
