"""SDK projection tests for the transport-neutral Skill v2 remote port."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest
from mindmemos_sdk.config import HttpConnectionConfig
from mindmemos_sdk.connections import HttpConnection
from mindmemos_sdk.skills import HttpSkillRemoteAdapter
from mindmemos_sdk.transport import AsyncHttpTransport

from mindmemos_skill import (
    RemoteEvolveRequest,
    RemotePushRequest,
    RemoteSyncItem,
    RemoteSyncRequest,
    RemoteTrajectoryPullRequest,
    RemoteTrajectoryReportRequest,
    SkillBundle,
    SkillRemoteRequestError,
    SkillVersionCore,
)


def _connection(handler, *, api_key: str | None = "mk_test"):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = AsyncHttpTransport(
        base_url="https://api.test",
        api_key=api_key,
        max_retries=0,
        client=client,
    )
    return (
        HttpConnection(
            HttpConnectionConfig(base_url="https://unused.test", api_key="unused"),
            transport=transport,
            owns_transport=False,
        ),
        client,
    )


def _version(*, version_id: str = "v1", cloud_skill_id: str | None = None) -> SkillVersionCore:
    bundle = SkillBundle.from_files({"SKILL.md": "demo"})
    now = datetime(2026, 8, 7, tzinfo=UTC)
    return SkillVersionCore(
        version_id=version_id,
        cloud_skill_id=cloud_skill_id,
        name="demo",
        content_hash=bundle.content_hash,
        version_label="1.0.0",
        status="draft",
        origin="local",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (401, "remote_unauthorized", False),
        (403, "remote_forbidden", False),
        (404, "remote_not_found", False),
        (409, "remote_conflict", False),
        (429, "remote_rate_limited", True),
        (503, "remote_server_error", True),
    ],
)
async def test_http_adapter_maps_stable_errors(status, code, retryable):
    connection, client = _connection(
        lambda _request: httpx.Response(
            status,
            json={"code": "provider-code", "message": "private", "request_id": "req"},
        )
    )
    with pytest.raises(SkillRemoteRequestError) as failure:
        await HttpSkillRemoteAdapter(connection).pull_versions("family")
    assert failure.value.error_code == code
    assert failure.value.retryable is retryable
    assert "private" not in str(failure.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_projects_version_and_sync_without_head_pointers():
    calls = []
    version = _version()
    bundle = SkillBundle.from_files({"SKILL.md": "demo"})

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("/register"):
            response_version = {**body["version"], "cloud_skill_id": "family", "received_at": "2026-08-07T00:00:01Z"}
            data = {"version": response_version}
        elif request.url.path.endswith("/versions"):
            data = {"versions": [{**version.model_dump(mode="json"), "cloud_skill_id": "family"}], "next_cursor": None}
        elif request.url.path.endswith("/content"):
            data = {"version": {**version.model_dump(mode="json"), "cloud_skill_id": "family"}, "bundle": bundle.model_dump(mode="json")}
        else:
            data = {"items": [{"cloud_skill_id": "family", "versions": []}]}
        return httpx.Response(200, json={"code": "ok", "data": data})

    connection, client = _connection(handler)
    adapter = HttpSkillRemoteAdapter(connection)
    pushed = await adapter.push_version(RemotePushRequest(operation_id="op", version=version, bundle=bundle))
    assert pushed.cloud_skill_id == "family"
    assert (await adapter.pull_versions("family")).versions[0].version_id == "v1"
    assert (await adapter.pull_content("family", "v1")).bundle == bundle
    synced = await adapter.sync(
        RemoteSyncRequest(items=[RemoteSyncItem(cloud_skill_id="family", known_version_revisions={"v1": 0})])
    )
    assert synced.items[0].cloud_skill_id == "family"
    assert all("published_head_id" not in json.dumps(call) for call in calls)
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_exposes_trajectory_and_evolve_routes():
    paths = []
    version = _version(cloud_skill_id="family")

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/trajectories") and request.method == "POST":
            data = {"items": [{"trajectory_id": "t1", "status": "queued", "error_code": None}]}
        elif request.url.path.endswith("/trajectories"):
            data = {"items": [], "returned_count": 0, "next_cursor": None, "has_more": False}
        elif request.url.path.endswith("/evolve"):
            data = {
                "operation_id": "evolve",
                "evolution_run_id": "run",
                "cloud_skill_id": "family",
                "base_version_id": "v1",
                "status": "queued",
                "candidate_version_ids": [],
                "selected_version_id": None,
            }
        else:
            data = {"version": version.model_dump(mode="json")}
        return httpx.Response(200, json={"code": "ok", "data": data})

    connection, client = _connection(handler)
    adapter = HttpSkillRemoteAdapter(connection)
    report = RemoteTrajectoryReportRequest.model_construct(operation_id="report", mode="async", items=[])
    assert (await adapter.report_trajectories(report)).items[0].status == "queued"
    assert (await adapter.pull_trajectories(RemoteTrajectoryPullRequest(cloud_skill_id="family"))).items == []
    assert (
        await adapter.evolve(
            RemoteEvolveRequest(
                operation_id="evolve",
                cloud_skill_id="family",
                base_version_id="v1",
                algorithm="fake",
                mode="async",
            )
        )
    ).status == "queued"
    assert paths == [
        "/v1/skills/trajectories",
        "/v1/skills/trajectories",
        "/v1/skills/evolve",
    ]
    await client.aclose()
