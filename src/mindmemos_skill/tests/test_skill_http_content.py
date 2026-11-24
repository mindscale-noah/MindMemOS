"""Pin structured HTTP content, SDK hashes, and scoped history cursors."""

import base64

import pytest
from fastapi.testclient import TestClient
from mindmemos_sdk.skills import content_hash, decode_resources, skill_hash
from mindmemos_skill.api.entries.fastapi import create_app

from mindmemos_skill import create_service

from .service_config import repository_config


@pytest.fixture
async def client(tmp_path):
    service = await create_service(repository_config(tmp_path, "test"))
    with TestClient(create_app(service)) as client:
        yield client


def source(text="Exact\r\n内容\r", **values):
    return {"content": text, "name": "Test", "description": "Test source", "standard_format": False, **values}


def test_history_returns_submission_message(client):
    """Expose persisted submission text through the public history response."""
    message = "Validate totals.\nKeep source evidence."
    registered = client.post("/v1/skills/register", json={"skill": source(commit_message=message)}).json()["data"]
    response = client.post("/v1/skills/history", json={"skill_id": registered["skill_id"]})
    assert response.status_code == 200
    assert response.json()["data"]["items"][0]["commit_message"] == message


def test_sdk_hashes_and_http_roundtrip(client):
    data = source(resources={"a.bin": base64.b64encode(b"\x00\xff").decode()})
    response = client.post("/v1/skills/register", json={"skill": data})
    assert response.status_code == 201, response.text
    first = response.json()["data"]
    assert first["content_hash"] == content_hash(data["content"])
    assert first["skill_hash"] == skill_hash(data["content"], decode_resources(data["resources"]))
    shown = client.post("/v1/skills/show", json={"skill_id": first["skill_id"]}).json()["data"]
    assert shown["version"]["content"] == data["content"]
    assert shown["version"]["resources"] == data["resources"]
    assert shown["skill"]["active_skill_version_id"] == first["skill_version_id"]
    assert "bundle" not in shown["version"]
    second = client.post(
        "/v1/skills/commit", json={"skill_id": first["skill_id"], "skill": source("New"), "active": False}
    ).json()["data"]
    assert second["active_skill_version_id"] == first["skill_version_id"]
    resolved = client.post("/v1/skills/version-resolve", json={"skill_hash": second["skill_hash"]})
    assert resolved.status_code == 200 and resolved.json()["data"]["is_active"] is False
    activated = client.post(
        "/v1/skills/activate",
        json={
            "skill_id": first["skill_id"],
            "skill_version_id": second["skill_version_id"],
            "expected_active_version_id": first["skill_version_id"],
        },
    )
    assert activated.status_code == 200, activated.text
    stale = client.post(
        "/v1/skills/activate",
        json={
            "skill_id": first["skill_id"],
            "skill_version_id": first["skill_version_id"],
            "expected_active_version_id": first["skill_version_id"],
        },
    )
    assert stale.status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"source": "/etc/passwd"},
        {"skill": {"path": "/etc/passwd"}},
        {"bundle": {"SKILL.md": "old"}},
        {"skill": source(resources={"../escape": "YQ=="})},
        {"skill": source(resources={"bad": "%%%"})},
    ],
)
def test_http_rejects_paths_old_shapes_and_invalid_resources(client, body):
    response = client.post("/v1/skills/register", json=body)
    assert response.status_code == 422, response.text


def test_openapi_only_exposes_structured_registration(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]
    register = schema["RegisterHttpRequest"]
    assert register["additionalProperties"] is False
    assert set(register["properties"]) == {"skill", "alias"}
    structured = schema["StructuredSkillInfo"]
    assert structured["additionalProperties"] is False
    assert "content" in structured["properties"]
    assert not {"path", "bundle", "source"} & structured["properties"].keys()
    assert "limit" not in schema["ListHttpRequest"].get("required", [])
    assert {"type": "null"} in schema["ListHttpRequest"]["properties"]["limit"]["anyOf"]


def test_history_cursor_is_stable_and_family_bound(client):
    first = client.post("/v1/skills/register", json={"skill": source()}).json()["data"]
    other = client.post("/v1/skills/register", json={"skill": source("Other")}).json()["data"]
    for number in range(3):
        assert (
            client.post(
                "/v1/skills/commit", json={"skill_id": first["skill_id"], "skill": source(str(number))}
            ).status_code
            == 201
        )
    page = client.post("/v1/skills/history", json={"skill_id": first["skill_id"], "limit": 2}).json()["data"]
    cursor = page["next_cursor"]
    assert cursor and page["has_more"]
    client.post("/v1/skills/commit", json={"skill_id": first["skill_id"], "skill": source("Later")})
    rest = client.post("/v1/skills/history", json={"skill_id": first["skill_id"], "cursor": cursor}).json()["data"]
    ids = [item["skill_version_id"] for item in page["items"] + rest["items"]]
    assert len(ids) == len(set(ids)) == 4
    assert rest["has_more"] is False
    assert client.post("/v1/skills/history", json={"skill_id": other["skill_id"], "cursor": cursor}).status_code == 422
    assert client.post("/v1/skills/history", json={"skill_id": first["skill_id"], "cursor": "bad"}).status_code == 422
