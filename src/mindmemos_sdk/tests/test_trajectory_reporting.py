"""Exercise unified delivery through the public SDK and CLI."""

import io
import json

import httpx
import pytest
from mindmemos_sdk.config import SDKConfig
from mindmemos_sdk.transport import HttpTransport

from mindmemos_sdk import MindMemOSClient, TrajectoryEvent, cli


def event(*, versions: list[str]) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_id="test-session/1",
        source="test",
        session_id="test-session",
        turn_id="1",
        skill_version_ids=versions,
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "run", "id": "call-1"}]},
            {"role": "tool", "content": "Done", "tool_call_id": "call-1"},
        ],
    )


@pytest.fixture
def delivery():
    calls = []

    def route(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), request.headers["Authorization"], body))
        if request.url.host == "memory.test":
            if body["metadata"]["event_id"] == "timeout":
                raise httpx.ReadTimeout("Response lost")
            return httpx.Response(200, json={"code": "queued", "request_id": "memory-receipt"})
        return httpx.Response(
            201,
            json={
                "code": "ok",
                "data": {"trajectory_id": "shared-attempt", "skill_version_ids": body["skill_version_ids"]},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(route)) as http:
        with MindMemOSClient(
            config=SDKConfig(),
            user_id="alice",
            agent_id="assistant",
            transport=HttpTransport(base_url="https://memory.test", api_key="memory-key", client=http),
            skill_transport=HttpTransport(base_url="https://skill.test", api_key="skill-key", client=http),
        ) as client:
            yield client, calls


@pytest.mark.parametrize("versions", [["version-a", "version-b", "version-a"], []])
def test_one_call_delivers_one_skill_trajectory(delivery, versions):
    client, calls = delivery
    report = client.report_trajectory(event(versions=versions))
    assert report.memory.status == "queued"
    assert report.memory.request_id == "memory-receipt"
    assert report.skill.status == "stored"
    assert report.skill.trajectory_id == "shared-attempt"
    memory, skill = calls
    assert memory[:2] == ("https://memory.test/v1/memory/add", "Bearer memory-key")
    assert skill[:2] == ("https://skill.test/v1/skills/trajectories/report", "Bearer skill-key")
    assert memory[2]["messages"] == [{"role": "user", "content": "Hello"}, {"role": "tool", "content": "Done"}]
    assert memory[2]["metadata"] == {"event_id": "test-session/1", "source": "test", "turn_id": "1"}
    assert skill[2]["messages"][1]["tool_calls"][0]["id"] == skill[2]["messages"][2]["tool_call_id"]
    assert skill[2]["skill_version_ids"] == list(dict.fromkeys(versions))
    assert skill[2]["event_id"] == report.event_id
    assert memory[2]["user_id"] == skill[2]["user_id"] == "alice"


def test_cli_retains_successful_receipt_after_timeout(delivery, monkeypatch, capsys):
    client, calls = delivery
    payload = event(versions=["version-a", "version-b"]).model_copy(update={"event_id": "timeout"})
    monkeypatch.setattr(cli.ConfigManager, "load", lambda _: SDKConfig())
    monkeypatch.setattr(cli, "MindMemOSClient", lambda **_: client)
    monkeypatch.setattr("sys.stdin", io.StringIO(payload.model_dump_json()))

    assert cli.main(["trajectory", "report", "--file", "-", "--json"]) == 1
    receipts = json.loads(capsys.readouterr().out)
    assert receipts["memory"]["status"] == "unknown"
    assert receipts["skill"]["trajectory_id"] == "shared-attempt"
    assert len(calls) == 2

    monkeypatch.setattr("sys.stdin", io.StringIO(payload.model_dump_json()))
    assert cli.main(["trajectory", "report", "--file", "-", "--json", "--skill-only"]) == 0
    assert "memory" not in json.loads(capsys.readouterr().out)
    assert [url for url, _, _ in calls].count("https://memory.test/v1/memory/add") == 1
