"""Verify the list-based evolution request across SDK clients."""

import asyncio
import json

import httpx
import pytest
from mindmemos_sdk.skills import AsyncSkillClient, EvolveRequest, SkillClient
from mindmemos_sdk.skills.core import SkillCore, SkillRoute
from mindmemos_sdk.transport import AsyncHttpTransport, HttpTransport


def test_one_version_is_sent_as_a_list():
    request = SkillCore.prepare(SkillRoute.EVOLVE, EvolveRequest(skill_version_ids=["skv-a"], auto_accept=False))
    assert request.path == "/v1/skills/algos/evolve"
    assert request.body == {"skill_version_ids": ["skv-a"], "auto_accept": False}


def test_evolution_normalizes_identities_and_defaults_to_auto_accept():
    value = EvolveRequest(skill_version_ids=[" skv-a "], editable_skills=[" sk-a "], pipeline=" offline ")
    assert value.auto_accept is True
    assert SkillCore.prepare(SkillRoute.EVOLVE, value).body == {
        "skill_version_ids": ["skv-a"],
        "editable_skills": ["sk-a"],
        "pipeline": "offline",
    }


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
        {"skill_version_id": "skv-a"},
        {"skill_version_id": "skv-a", "skill_version_ids": ["skv-b"]},
        {"skill_version_ids": "skv-a"},
        {"skill_version_ids": []},
        {"skill_version_ids": ["skv-a", " skv-a "]},
        {"skill_version_ids": [" "]},
        {"skill_version_ids": ["skv-a"], "pipeline": "  "},
    ],
)
def test_evolution_rejects_missing_ambiguous_and_duplicate_inputs(selection):
    with pytest.raises(ValueError):
        EvolveRequest(**selection)


@pytest.mark.parametrize(
    "selection", [{"skill_version_ids": ["skv-a"]}, {"pipeline": "offline", "skill_version_ids": ["skv-a", "skv-b"]}]
)
def test_sync_and_async_evolution_share_the_same_transport_contract(selection):
    """Send the same list payload from either SDK client."""
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
