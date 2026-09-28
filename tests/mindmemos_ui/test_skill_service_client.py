"""Algorithm HTTP attempts are never replayed, and experiments retain status logs."""

import httpx
import pytest
from mindmemos_sdk.config.connections import SkillConnectionConfig
from mindmemos_sdk.skills.service_client import SkillServiceClient


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_single_attempt_and_experiment_log(tmp_path, failure):
    requests = []

    def handle(request):
        requests.append(request)
        assert "authorization" not in request.headers
        assert request.extensions["timeout"]["read"] == 21600
        if failure:
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(200, json={"result": "test-output"})

    config = SkillConnectionConfig(experiment_dir="实验")
    client = SkillServiceClient(tmp_path, config=config)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=config.base_url, timeout=config.timeout_seconds, transport=httpx.MockTransport(handle)
    )
    async with client:
        if failure:
            with pytest.raises(httpx.HTTPStatusError):
                await client.train({"tasks": []})
        else:
            assert await client.train({"tasks": []}) == {"result": "test-output"}
    assert len(requests) == 1
    directories = list((tmp_path / "实验").iterdir())
    assert len(directories) == 1
    assert "retry=false" in (directories[0] / "experiment.log").read_text()
    assert (directories[0] / "output.json").exists() is not failure


@pytest.mark.asyncio
async def test_local_service_assigns_stable_project_without_credentials(tmp_path):
    from fastapi.testclient import TestClient
    from mindmemos_skill.api import create_service
    from mindmemos_skill.api.entries.fastapi import create_app

    service = await create_service(home=tmp_path)
    identity = service.local_project_id
    await service.initialize_storage()
    with TestClient(create_app(service)) as http:
        assert http.post("/skills/list", json={}).status_code == 200
    restored = await create_service(home=tmp_path)
    try:
        assert restored.local_project_id == identity
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_startup_passes_home_and_empty_scope(tmp_path, monkeypatch):
    import sys
    from unittest.mock import AsyncMock

    import uvicorn
    from mindmemos_skill.api.entries import fastapi as entry

    service = AsyncMock()
    factory = AsyncMock(return_value=service)
    server = AsyncMock()
    monkeypatch.setattr(entry, "create_service", factory)
    monkeypatch.setattr(entry, "create_app", lambda value, **kwargs: "test-app")
    monkeypatch.setattr(uvicorn, "Config", lambda *args, **kwargs: None)
    monkeypatch.setattr(uvicorn, "Server", lambda config: server)
    monkeypatch.setattr(sys, "argv", ["dev-skill", "--home", str(tmp_path), "--project-id", ""])
    await entry._main()
    factory.assert_awaited_once_with(None, home=tmp_path, project_id=None)
    service.initialize_storage.assert_not_awaited()
    server.serve.assert_awaited_once()
