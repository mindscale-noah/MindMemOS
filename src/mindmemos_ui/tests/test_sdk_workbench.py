"""Exercise browser routes through the SDK and a real Skill HTTP service."""

from contextlib import asynccontextmanager

import httpx
import pytest
import pytest_asyncio
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.skills import AsyncSkillClient
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.api.entries.fastapi import create_app as create_skill_app
from mindmemos_ui.app import create_app as create_ui_app

from mindmemos_skill import SkillConfig, create_service


@pytest_asyncio.fixture
async def workbench(tmp_path, monkeypatch):
    """Connect the UI's SDK transport to an isolated project-owned Skill API."""
    service = await create_service(
        SkillConfig.model_validate(
            {
                "service": {
                    "home": tmp_path / "skill",
                    "persistence": {"database": {"class_name": "sqlite"}},
                    "repository_pipeline": {"class_name": "repository"},
                    "manage_pipeline": {"class_name": "graph_manage"},
                },
                "fastapi": {"auth": {"mode": "none", "default_project_id": "project-a"}},
            }
        )
    )
    skill_app = create_skill_app(service)
    manager = ConfigManager(tmp_path / "sdk")
    config = manager.load_or_default()
    config.service_type = "custom"
    config.memory_service.base_url = "http://127.0.0.1"
    config.memory_service.auth.api_key = "test-key"
    manager.save(config)

    @asynccontextmanager
    async def sdk_client(_manager):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=skill_app), base_url="http://127.0.0.1"
        ) as remote:
            async with AsyncHttpTransport(base_url="http://127.0.0.1", api_key="test-key", client=remote) as transport:
                yield AsyncSkillClient(transport)

    monkeypatch.setattr("mindmemos_ui.sdk_workbench.skill_client", sdk_client)
    ui = await create_ui_app(manager)
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=ui), base_url="http://127.0.0.1") as browser:
            token = (await browser.get("/api/session")).json()["token"]
            browser.headers["x-workbench-token"] = token
            yield browser, manager
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_skill_lifecycle_uses_sdk_and_preserves_auxiliary_files(workbench, tmp_path):
    browser, _ = workbench
    source = tmp_path / "source"
    (source / "assets").mkdir(parents=True)
    (source / "SKILL.md").write_text("---\nname: demo\ndescription: Example\n---\nOriginal", encoding="utf-8")
    (source / "assets" / "data.bin").write_bytes(b"\x00original")

    preview = await browser.post("/api/register/preview", json={"path": str(source)})
    assert preview.status_code == 200, preview.text
    assert preview.json()["name"] == "demo"
    assert preview.json()["files"] == ["SKILL.md", "assets/data.bin"]
    assert preview.json()["service_url"] == "http://127.0.0.1"
    result = await browser.post("/api/register/confirm", json={"plan_id": preview.json()["plan_id"]})
    assert result.status_code == 200, result.text
    skill_id = result.json()["skill"]["skill_id"]
    first = result.json()["version"]["version_id"]

    listed = await browser.get("/api/skills")
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["skill_id"] == skill_id
    assert listed.json()[0]["active_version_id"] == first
    content = await browser.get("/api/content", params={"source": "local", "skill_id": skill_id, "version_id": first})
    assert content.json()["files"]["SKILL.md"].endswith("Original")
    assert content.json()["binary_files"] == ["assets/data.bin"]

    (source / "SKILL.md").write_text("---\nname: demo\ndescription: Example\n---\nUpdated", encoding="utf-8")
    (source / "assets" / "data.bin").write_bytes(b"\x00updated")
    committed = await browser.post(
        "/api/commit",
        json={"skill_id": skill_id, "path": str(source), "base_version_id": first, "version_bump": "patch"},
    )
    assert committed.status_code == 200, committed.text
    second = committed.json()["version"]["version_id"]
    versions = (await browser.get("/api/versions", params={"skill_id": skill_id})).json()
    assert [row["version_label"] for row in versions] == ["v1.0.0", "v1.0.1"]
    diff = await browser.post(
        "/api/diff",
        json={
            "before": {"source": "local", "skill_id": skill_id, "version_id": first},
            "after": {"source": "local", "skill_id": skill_id, "version_id": second},
        },
    )
    assert diff.status_code == 200, diff.text
    assert {row["path"] for row in diff.json()["files"]} == {"SKILL.md", "assets/data.bin"}

    assembled = await browser.post(
        "/api/assemble",
        json={
            "versions": [
                {"source": "local", "skill_id": skill_id, "version_id": first},
                {"source": "local", "skill_id": skill_id, "version_id": second},
            ],
            "instruction": "Review the change",
        },
    )
    assert assembled.status_code == 200, assembled.text
    assert {item["path"] for item in assembled.json()["comparison"]["files"]} == {
        "SKILL.md",
        "assets/data.bin",
    }
    assert next(item for item in assembled.json()["comparison"]["files"] if item["path"] == "assets/data.bin")["binary"]

    target = tmp_path / "exported"
    export = await browser.post(
        "/api/export/preview",
        json={"version": {"source": "local", "skill_id": skill_id, "version_id": second}, "directory": str(target)},
    )
    assert export.status_code == 200, export.text
    confirmed = await browser.post("/api/export/confirm", json={"plan_id": export.json()["plan_id"]})
    assert confirmed.status_code == 200, confirmed.text
    assert (target / "assets" / "data.bin").read_bytes() == b"\x00updated"
    assert (target / "SKILL.md").read_text(encoding="utf-8").endswith("Updated")

    switch = await browser.post("/api/active/preview", json={"skill_id": skill_id, "version_id": first})
    assert switch.status_code == 200, switch.text
    activated = await browser.post("/api/active/confirm", json={"plan_id": switch.json()["plan_id"]})
    assert activated.status_code == 200, activated.text
    assert (await browser.get(f"/api/skills/{skill_id}")).json()["skill"]["active_version_id"] == first
    assert (await browser.get("/api/graphs")).json() == []
    assert (await browser.delete(f"/api/skills/{skill_id}")).json() == {"ok": True}
    assert target.exists()


@pytest.mark.asyncio
async def test_sdk_workbench_requires_configuration(tmp_path):
    """A new UI home does not silently open a private local Skill database."""
    ui = await create_ui_app(ConfigManager(tmp_path / "empty"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=ui), base_url="http://127.0.0.1") as browser:
        response = await browser.get("/api/skills")
        assert response.status_code == 503
        assert "Configure the Skill service" in response.json()["detail"]
