"""The pointer-based synchronous cloud client was removed by Skill protocol v2."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import mindmemos_sdk.skills as skills
from mindmemos_sdk.skills import PushVersionRequest, SkillCloudClient
from mindmemos_sdk.transport import HttpTransport

from mindmemos_skill import SkillBundle, SkillVersionCore


def test_public_sdk_uses_the_shared_v2_remote_adapter_only() -> None:
    assert hasattr(skills, "HttpSkillRemoteAdapter")
    assert hasattr(skills, "SkillCloudClient")
    assert not hasattr(skills, "PromoteCloudRequest")
    assert not hasattr(skills, "PromoteCloudResult")


def test_compatibility_cloud_client_pushes_nested_v2_contract() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.update(body)
        stored = dict(body["version"])
        stored["cloud_skill_id"] = "cloud-1"
        stored["received_at"] = "2026-08-07T00:00:01Z"
        return httpx.Response(200, json={"code": "ok", "data": {"version": stored}})

    transport = HttpTransport(
        base_url="https://api.test",
        api_key="mk_test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    bundle = SkillBundle.from_files({"SKILL.md": "# Demo\n"})
    result = SkillCloudClient(transport).push_version(
        PushVersionRequest(
            operation_id="op-1",
            name="demo",
            version_id="version-1",
            parent_version_ids=["parent-a", "parent-b"],
            content=bundle.canonical_json(),
            expected_content_hash=bundle.content_hash,
            version_label="1.0.0",
            created_at="2026-08-07T00:00:00Z",
        )
    )

    assert set(captured) == {"operation_id", "version", "bundle"}
    assert captured["version"]["parent_version_ids"] == ["parent-a", "parent-b"]
    assert result.cloud_skill_id == "cloud-1"
    assert result.status.value == "draft"


def test_compatibility_cloud_client_reads_v2_content_and_evolves_from_explicit_latest() -> None:
    now = datetime(2026, 8, 7, tzinfo=UTC)
    bundle = SkillBundle.from_files({"SKILL.md": "# Demo\n"})
    version = SkillVersionCore(
        version_id="version-1",
        cloud_skill_id="cloud-1",
        name="demo",
        content_hash=bundle.content_hash,
        version_label="1.0.0",
        status="draft",
        origin="local",
        created_at=now,
        updated_at=now,
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path.endswith("/content"):
            data = {"version": version.model_dump(mode="json"), "bundle": bundle.model_dump(mode="json")}
        elif request.method == "GET":
            data = {"cloud_skill_id": "cloud-1", "name": "demo", "latest_version": version.model_dump(mode="json")}
        else:
            data = {
                "operation_id": "op-evolve",
                "cloud_skill_id": "cloud-1",
                "status": "succeeded",
                "candidate_version_ids": ["version-2"],
                "selected_version_id": "version-2",
            }
        return httpx.Response(200, json={"code": "ok", "data": data})

    client = SkillCloudClient(
        HttpTransport(
            base_url="https://api.test",
            api_key="mk_test",
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    )

    content = client.get_content("cloud-1", "version-1")
    evolved = client.evolve("cloud-1")

    assert content.version.parent_version_ids == []
    assert content.content == bundle.canonical_json()
    assert evolved.new_version_id == "version-2"
    assert requests[-1][2]["base_version_id"] == "version-1"
