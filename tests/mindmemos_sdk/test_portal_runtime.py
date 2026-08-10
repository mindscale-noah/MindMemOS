"""Phase 6 portal configuration and lifecycle contracts."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from mindmemos_sdk.config import (
    ConfigManager,
    ConfigValidationError,
    HttpConnectionConfig,
    SDKConfigCompilerV2,
    SDKPortalConfigV2,
)
from mindmemos_sdk.connections import HttpConnection
from mindmemos_sdk.memory import AsyncMemoryClient
from mindmemos_sdk.memory.backends import AsyncMemoryBackend
from mindmemos_sdk.memory.core import MemoryDefaults
from mindmemos_sdk.transport import AsyncHttpTransport
from mindmemos_skill.management import ExportSkillRequest, PublishSkillRequest, RegisterSkillRequest

from mindmemos_sdk import (
    AsyncMindMemOSClient,
    SDKPortalRuntime,
    SkillCapabilityUnavailableError,
    SkillRemoteError,
)


def _portal(tmp_path: Path, **profile_overrides):
    profile = {
        "default_connection": "mindmemos_main",
        "connections": {
            "mindmemos_main": {
                "type": "http",
                "base_url": "https://api.test",
                "api_key": "test-key",
            }
        },
        "skill": {
            "remote": {"connection": "mindmemos_main"},
            "application": {
                "local": {
                    "root_dir": str(tmp_path / "skill"),
                    "database": {"provider": "sqlite", "path": "state.db"},
                }
            }
        },
    }
    profile.update(profile_overrides)
    return {"version": 2, "active_profile": "default", "profiles": {"default": profile}}


def test_portal_routes_memory_and_skill_to_the_same_default_connection(tmp_path: Path) -> None:
    compiled = SDKConfigCompilerV2().compile(_portal(tmp_path))

    assert compiled.profile.memory_connection == "mindmemos_main"
    assert compiled.profile.skill_connection == "mindmemos_main"
    assert compiled.profile.default_connection == "mindmemos_main"


@pytest.mark.parametrize("omit_remote", [True, False])
def test_portal_compiles_omitted_or_null_skill_remote_as_local_only(tmp_path: Path, omit_remote: bool) -> None:
    config = _portal(tmp_path)
    skill = config["profiles"]["default"]["skill"]
    if omit_remote:
        skill.pop("remote")
    else:
        skill["remote"] = {"connection": None}

    compiled = SDKConfigCompilerV2().compile(config)

    assert compiled.profile.memory_connection == "mindmemos_main"
    assert compiled.profile.skill_connection is None


def test_portal_rejects_unknown_explicit_skill_remote(tmp_path: Path) -> None:
    config = _portal(tmp_path)
    config["profiles"]["default"]["skill"]["remote"] = {"connection": "missing"}

    with pytest.raises(ConfigValidationError, match="skill.remote connection does not exist"):
        SDKConfigCompilerV2().compile(config)


def test_portal_allows_explicit_skill_connection_split(tmp_path: Path) -> None:
    config = _portal(tmp_path)
    profile = config["profiles"]["default"]
    profile["connections"]["skill_remote"] = {
        "type": "http",
        "base_url": "https://skill.test",
    }
    profile["skill"]["remote"] = {"connection": "skill_remote"}

    compiled = SDKConfigCompilerV2().compile(config)

    assert compiled.profile.memory_connection == "mindmemos_main"
    assert compiled.profile.skill_connection == "skill_remote"


def test_portal_rejects_unknown_route_connection(tmp_path: Path) -> None:
    config = _portal(tmp_path, memory={"connection": "missing"})

    with pytest.raises(ConfigValidationError, match="memory connection does not exist"):
        SDKConfigCompilerV2().compile(config)


def test_config_manager_round_trips_config_yaml_without_touching_legacy_config(tmp_path: Path) -> None:
    manager = ConfigManager(config_dir=tmp_path / "config")
    portal = SDKPortalConfigV2.model_validate(_portal(tmp_path))

    manager.save_portal(portal)

    assert manager.load_portal() == portal
    assert manager.compile_portal().profile.memory_connection == "mindmemos_main"
    assert manager.portal_path.name == "config.yaml"
    assert not manager.config_path.exists()
    assert "test-key" in manager.portal_path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_portal_runtime_owns_application_and_async_facade(tmp_path: Path) -> None:
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    transport = AsyncHttpTransport(base_url="https://api.test", api_key="test-key", client=async_client)
    connection = HttpConnection(
        HttpConnectionConfig(base_url="https://api.test", api_key="test-key"),
        transport=transport,
        owns_transport=False,
    )
    runtime = SDKPortalRuntime(_portal(tmp_path), connections={"mindmemos_main": connection})
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text('name: demo\nversion: "1.0.0"\n\nBody\n', encoding="utf-8")

    try:
        await runtime.start()
        registered = await runtime.skills.register(RegisterSkillRequest(source_path=source, alias="demo"))

        assert runtime.skills.application is runtime.application
        assert registered.skill_id == (await runtime.skills.get_skill("demo")).skill.skill_id
        assert (await runtime.skills.get_management_overview()).skills[0].skill.skill_id == registered.skill_id
        assert (await runtime.skills.get_management_detail("demo")).latest_version.version_id == registered.version_id
        assert "push" in runtime.application.capabilities
    finally:
        await runtime.aclose()
        await async_client.aclose()

    with pytest.raises(RuntimeError, match="not started"):
        _ = runtime.skills


@pytest.mark.asyncio
async def test_local_only_runtime_requires_only_memory_connection_and_keeps_local_capabilities(tmp_path: Path) -> None:
    config = _portal(tmp_path)
    config["profiles"]["default"]["skill"]["remote"] = {"connection": None}
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    transport = AsyncHttpTransport(base_url="https://api.test", api_key="test-key", client=async_client)
    memory_connection = HttpConnection(
        HttpConnectionConfig(base_url="https://api.test", api_key="test-key"),
        transport=transport,
        owns_transport=False,
    )
    runtime = SDKPortalRuntime(config, connections={"mindmemos_main": memory_connection})
    source = tmp_path / "local-source"
    source.mkdir()
    (source / "SKILL.md").write_text('name: local\nversion: "1.0.0"\n\nBody\n', encoding="utf-8")

    try:
        await runtime.start()
        registered = await runtime.skills.register(RegisterSkillRequest(source_path=source))
        published = await runtime.skills.publish(
            PublishSkillRequest(
                skill_ref=registered.skill_id,
                content='name: local\nversion: "1.1.0"\n\nUpdated\n',
            )
        )
        diff = await runtime.skills.diff(
            registered.skill_id,
            from_version_id=registered.version_id,
            to_version_id=published.version_id,
        )
        exported = await runtime.skills.export(
            ExportSkillRequest(
                skill_ref=registered.skill_id,
                version_id=published.version_id,
                target_path=tmp_path / "exported",
            )
        )
        capabilities = runtime.application.capabilities

        assert "register" in capabilities
        assert not {"push", "pull", "sync", "promote"}.intersection(capabilities)
        assert [skill.skill_id for skill in await runtime.skills.list_skills()] == [registered.skill_id]
        assert "Updated" in diff.diff
        assert exported.version_id == published.version_id
        assert (await runtime.skills.get_skill(registered.skill_id)).skill.skill_id == registered.skill_id
        with pytest.raises(SkillCapabilityUnavailableError, match="remote push"):
            await runtime.skills.push(registered.skill_id)
    finally:
        await runtime.aclose()
        await async_client.aclose()


@pytest.mark.asyncio
async def test_configured_remote_failure_does_not_change_capabilities_or_application_lifecycle(tmp_path: Path) -> None:
    state = {"available": False, "calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if not state["available"]:
            raise httpx.ConnectError("offline", request=request)
        body = json.loads(request.content)
        version = {
            **body["version"],
            "cloud_skill_id": "cloud-1",
            "received_at": body["version"]["created_at"],
        }
        return httpx.Response(
            200,
            json={
                "code": "ok",
                "data": {"version": version},
            },
        )

    config = _portal(tmp_path)
    config["profiles"]["default"]["connections"]["mindmemos_main"]["max_retries"] = 0
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(
        base_url="https://api.test",
        api_key="test-key",
        max_retries=0,
        client=async_client,
    )
    connection = HttpConnection(
        HttpConnectionConfig(base_url="https://api.test", api_key="test-key", max_retries=0),
        transport=transport,
        owns_transport=False,
    )
    runtime = SDKPortalRuntime(config, connections={"mindmemos_main": connection})
    source = tmp_path / "remote-source"
    source.mkdir()
    (source / "SKILL.md").write_text('name: remote\nversion: "1.0.0"\n\nBody\n', encoding="utf-8")

    try:
        await runtime.start()
        assert state["calls"] == 0
        registered = await runtime.skills.register(RegisterSkillRequest(source_path=source))
        before = runtime.application.capabilities

        with pytest.raises(SkillRemoteError) as failure:
            await runtime.skills.push(registered.skill_id)
        assert failure.value.error_code == "remote_unavailable"
        assert failure.value.retryable is True
        assert runtime.application.capabilities == before
        assert (await runtime.skills.get_skill(registered.skill_id)).skill.skill_id == registered.skill_id

        state["available"] = True
        pushed = await runtime.skills.push(registered.skill_id)
        assert pushed.version_id == registered.version_id
        assert runtime.application.capabilities == before
    finally:
        await runtime.aclose()
        await async_client.aclose()


@pytest.mark.asyncio
async def test_async_root_client_delegates_the_portal_lifecycle(tmp_path: Path) -> None:
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(500)))
    transport = AsyncHttpTransport(base_url="https://api.test", api_key="test-key", client=async_client)
    connection = HttpConnection(
        HttpConnectionConfig(base_url="https://api.test", api_key="test-key"),
        transport=transport,
        owns_transport=False,
    )

    client = AsyncMindMemOSClient(config=_portal(tmp_path), connections={"mindmemos_main": connection})
    try:
        await client.start()
        assert client.skills.application is client.runtime.application
        assert client.memory is client.runtime.memory
    finally:
        await client.aclose()
        await async_client.aclose()


@pytest.mark.asyncio
async def test_async_memory_auto_context_delegates_to_skill_application() -> None:
    class Backend(AsyncMemoryBackend):
        request = None

        async def execute(self, request):
            self.request = request
            return "added"

    class Application:
        calls = []

        async def resolve_skill_context(self, messages):
            self.calls.append(messages)
            return [
                {
                    "name": "demo",
                    "content_hash": "hash",
                    "base_version_id": "version-1",
                    "usage": "injected",
                }
            ]

    backend = Backend()
    application = Application()
    client = AsyncMemoryClient(
        backend,
        memory_defaults=MemoryDefaults(user_id="user-1", add_auto_skill_context=True),
        skill_application=application,
    )
    messages = [{"role": "user", "content": "remember"}]

    result = await client.add(messages)

    assert result == "added"
    assert application.calls == [messages]
    assert backend.request.body["skill_context"] == [
        {
            "name": "demo",
            "content_hash": "hash",
            "base_version_id": "version-1",
            "usage": "injected",
        }
    ]
