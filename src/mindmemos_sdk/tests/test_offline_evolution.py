"""Verify single-version and collection requests share the evolution endpoint."""

import asyncio
import json

import httpx
import pytest
from mindmemos_sdk.skills import AsyncSkillClient, EvolveRequest, SkillClient
from mindmemos_sdk.skills.core import SkillCore, SkillRoute
from mindmemos_sdk.transport import AsyncHttpTransport, HttpTransport


def test_single_version_wire_payload_remains_original():
    request = SkillCore.prepare(SkillRoute.EVOLVE, EvolveRequest(skill_version_id="skv-a", auto_accept=False))
    assert request.path == "/v1/skills/algos/evolve"
    assert request.body == {"skill_version_id": "skv-a", "auto_accept": False}


@pytest.mark.parametrize("version_id", ["skv-a", "", "  skv-a  ", "  "])
@pytest.mark.parametrize("auto_accept", [None, False, True])
def test_existing_single_version_validation_and_wire_payload(version_id, auto_accept):
    """Preserve SDK string handling and leave identifier normalization to the HTTP boundary."""
    payload = {"skill_version_id": version_id}
    if auto_accept is not None:
        payload["auto_accept"] = auto_accept
    value = EvolveRequest(**payload)
    assert value.auto_accept is (True if auto_accept is None else auto_accept)
    assert SkillCore.prepare(SkillRoute.EVOLVE, value).body == payload


def test_collection_uses_the_same_endpoint_with_explicit_pipeline_and_permissions():
    request = SkillCore.prepare(
        SkillRoute.EVOLVE,
        EvolveRequest(pipeline="offline", skill_version_ids=["skv-a"], editable_skills=[], auto_accept=False),
    )
    assert request.path == "/v1/skills/algos/evolve"
    assert request.body == {
        "pipeline": "offline",
        "skill_version_ids": ["skv-a"],
        "editable_skills": [],
        "auto_accept": False,
    }


@pytest.mark.parametrize(
    "selection",
    [
        {},
        {"skill_version_id": "skv-a", "skill_version_ids": ["skv-b"]},
        {"skill_version_ids": []},
        {"skill_version_ids": ["skv-a", "skv-a"]},
        {"skill_version_ids": [""]},
        {"skill_version_id": "skv-a", "pipeline": "  "},
    ],
)
def test_evolution_rejects_missing_ambiguous_and_duplicate_inputs(selection):
    with pytest.raises(ValueError):
        EvolveRequest(**selection)


@pytest.mark.parametrize(
    "selection", [{"skill_version_id": "skv-a"}, {"pipeline": "offline", "skill_version_ids": ["skv-a", "skv-b"]}]
)
def test_sync_and_async_evolution_share_the_same_transport_contract(selection):
    """Keep both clients on the original endpoint for either supported input shape."""
    received = []

    def handler(request):
        received.append((request.url.path, json.loads(request.content)))
        return httpx.Response(202, json={"code": "accepted", "data": {"task_id": "evolution-task"}})

    request = EvolveRequest(**selection, auto_accept=False)
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = SkillClient(HttpTransport(base_url="http://test", api_key="key", client=http))
        assert client.evolve(request) == {"task_id": "evolution-task"}

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = AsyncSkillClient(AsyncHttpTransport(base_url="http://test", api_key="key", client=http))
            assert await client.evolve(request) == {"task_id": "evolution-task"}

    asyncio.run(run())
    assert received[0] == received[1]
    assert received[0] == ("/v1/skills/algos/evolve", {**selection, "auto_accept": False})
