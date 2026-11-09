"""Verify custom OpenClaw model configuration and isolated native execution."""

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.agents import Agent, AgentDefinition, OpenClawAgent, OpenClawConfig
from mindmemos_skill.agents.buitin_agents.cli import run_cli
from mindmemos_skill.agents.buitin_agents.openclaw import agent as module
from pydantic import ValidationError


def native_result():
    """Return a native local success envelope with cached-token usage."""
    return {
        "payloads": [{"text": "answer"}],
        "meta": {"agentMeta": {"usage": {"input": 4, "output": 2, "cacheRead": 3, "total": 9}}},
    }


@pytest.mark.parametrize("api", ["openai-completions", "openai-responses", "anthropic-messages"])
async def test_model_switching_and_concurrent_isolation(monkeypatch, tmp_path, api):
    """Keep concurrent models, credentials, workspaces and session traces separate."""
    states = []
    sessions = []

    async def fake_run(command, workspace, config, overrides):
        state = Path(overrides[module.OPENCLAW_STATE_DIR_ENV])
        states.append(state)
        native = json.loads(Path(overrides[module.OPENCLAW_CONFIG_PATH_ENV]).read_text())
        provider = native["models"]["providers"]["mindmemos"]
        assert provider["models"][0]["id"] == config.model
        assert provider["api"] == api
        assert provider["baseUrl"] == config.base_url
        assert provider["apiKey"] == "${MINDMEMOS_OPENCLAW_API_KEY}"
        assert overrides[module.OPENCLAW_API_KEY_ENV] == config.api_key
        assert config.api_key not in json.dumps(native)
        assert config.api_key not in str(command)
        assert native["agents"]["defaults"]["workspace"] == str(workspace)
        assert native["agents"]["defaults"]["model"]["primary"] == f"mindmemos/{config.model}"
        assert native["skills"]["load"]["extraDirs"] == [str(workspace / ".agents/skills")]
        assert command[1:4] == ["agent", "--local", "--json"]
        assert command[command.index("--thinking") + 1] == "medium"
        assert command[-1] == "System instructions:\ninstructions\n\nUser:\ntask"
        session_id = command[command.index("--session-id") + 1]
        sessions.append(session_id)
        transcript = state / "agents/main/sessions" / f"{session_id}.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_text(json.dumps({"type": "message", "message": {"role": "toolResult", "content": "done"}}))
        await asyncio.sleep(0)
        return json.dumps(native_result())

    monkeypatch.setattr(module, "run_cli", fake_run)
    agents = [
        Agent.build(
            "openclaw",
            {
                "model": model,
                "api_key": f"secret-{model}",
                "base_url": f"https://{model}.invalid/v1",
                "api": api,
                "reasoning_effort": "medium",
            },
        )
        for model in ("first", "second")
    ]
    results = await asyncio.gather(
        *(agent.run(system="instructions", user="task", workspace=tmp_path) for agent in agents)
    )
    assert len(set(states)) == len(set(sessions)) == 2
    assert all(not state.exists() for state in states)
    for result in results:
        assert result.output == "answer"
        assert result.usage["total_tokens"] == 9
        assert result.usage["input_tokens"] == 4
        assert result.trajectory[2]["message"]["role"] == "toolResult"


@pytest.mark.parametrize(
    "payload",
    [
        "invalid",
        "[]",
        "{}",
        json.dumps({"payloads": [], "meta": {}}),
        json.dumps({"payloads": [{"text": "error", "isError": True}], "meta": {}}),
        json.dumps({"payloads": [{"text": "partial"}], "meta": {"aborted": True}}),
        json.dumps({"payloads": [{"text": "partial"}], "meta": {"error": {"kind": "retry_limit"}}}),
        json.dumps({"payloads": [{"text": "partial"}], "meta": {"stopReason": "tool_calls"}}),
    ],
)
async def test_invalid_results_and_cleanup(monkeypatch, tmp_path, payload):
    """Do not turn malformed, empty or failed native runs into successful results."""
    paths = []

    async def fake_run(command, workspace, config, overrides):
        paths.append(Path(overrides[module.OPENCLAW_STATE_DIR_ENV]))
        return payload

    monkeypatch.setattr(module, "run_cli", fake_run)
    agent = Agent.build("openclaw", {"model": "custom", "api_key": "secret", "base_url": "https://example.invalid"})
    with pytest.raises(RuntimeError, match="OpenClaw"):
        await agent.run(system="", user="task", workspace=tmp_path)
    assert all(not path.exists() for path in paths)


def test_config_registration_and_validation():
    """Resolve the backend through public exports and validate the model protocol."""
    assert Agent.resolve("openclaw") is OpenClawAgent
    definition = AgentDefinition(class_name="openclaw", config={"model": "custom"})
    assert isinstance(definition.config, OpenClawConfig)
    for options in ({}, {"model": ""}, {"model": "custom", "api": "invalid"}, {"model": "custom", "context_window": 0}):
        with pytest.raises(ValidationError):
            Agent.build("openclaw", options)


async def test_missing_connection_fails_before_process(monkeypatch, tmp_path):
    """Require explicit credentials and endpoint before native execution."""
    process = AsyncMock()
    monkeypatch.setattr(module, "run_cli", process)
    with pytest.raises(ValueError, match="api_key"):
        await Agent.build("openclaw", {"model": "custom"}).run(system="", user="task", workspace=tmp_path)
    process.assert_not_called()


@pytest.mark.skipif(shutil.which("openclaw") is None, reason="OpenClaw CLI is not installed")
async def test_generated_config_validates_with_installed_cli(monkeypatch, tmp_path):
    """Ask the installed CLI to validate generated config without a model call."""

    async def validate_config(command, workspace, config, overrides):
        await run_cli([command[0], "config", "validate", "--json"], workspace, config, overrides)
        return json.dumps(native_result())

    monkeypatch.setattr(module, "run_cli", validate_config)
    agent = Agent.build(
        "openclaw", {"model": "custom", "api_key": "offline-test", "base_url": "https://example.invalid/v1"}
    )
    assert (await agent.run(system="", user="task", workspace=tmp_path)).output == "answer"


@pytest.mark.skipif(shutil.which("openclaw") is None, reason="OpenClaw CLI is not installed")
async def test_installed_cli_calls_custom_local_endpoint(tmp_path):
    """Run the actual harness against an offline SSE stub, without a paid model call."""
    requests = []

    async def respond(reader, writer):
        header = (await reader.readuntil(b"\r\n\r\n")).decode()
        headers = dict(line.split(": ", 1) for line in header.split("\r\n")[1:] if ": " in line)
        length = int(next(value for key, value in headers.items() if key.lower() == "content-length"))
        body = json.loads(await reader.readexactly(length))
        requests.append((header, body))
        chunks = [
            {
                "id": "offline",
                "object": "chat.completion.chunk",
                "model": "custom-offline",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "offline answer"}, "finish_reason": None}
                ],
            },
            {
                "id": "offline",
                "object": "chat.completion.chunk",
                "model": "custom-offline",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        ]
        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n" + payload.encode()
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        agent = Agent.build(
            "openclaw",
            {
                "model": "custom-offline",
                "api_key": "offline-secret",
                "base_url": f"http://127.0.0.1:{port}/v1",
                "timeout_s": 45,
            },
        )
        result = await agent.run(system="Return plain text.", user="Say offline answer.", workspace=tmp_path)
    assert result.output == "offline answer"
    assert result.usage["total_tokens"] == 12
    assert requests and all(body["model"] == "custom-offline" for _, body in requests)
    assert all("Bearer offline-secret" in header for header, _ in requests)


@pytest.mark.parametrize("failure", [RuntimeError("process failed"), asyncio.CancelledError()])
async def test_process_failure_cleans_state(monkeypatch, tmp_path, failure):
    """Clean private state when process execution fails or the caller cancels."""
    states = []

    async def fail_run(command, workspace, config, overrides):
        states.append(Path(overrides[module.OPENCLAW_STATE_DIR_ENV]))
        raise failure

    monkeypatch.setattr(module, "run_cli", fail_run)
    agent = Agent.build("openclaw", {"model": "custom", "api_key": "secret", "base_url": "https://example.invalid"})
    with pytest.raises(type(failure)):
        await agent.run(system="", user="task", workspace=tmp_path)
    assert states and all(not state.exists() for state in states)


@pytest.mark.parametrize(
    "usage, expected",
    [(None, {}), ({"input": 3, "output": 2}, {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})],
)
async def test_usage_preserves_unknown_and_normalizes_counts(monkeypatch, tmp_path, usage, expected):
    """Keep absent measurements unknown and normalize available native counts."""
    response = native_result()
    response["meta"]["agentMeta"]["usage"] = usage
    monkeypatch.setattr(module, "run_cli", AsyncMock(return_value=json.dumps(response)))
    agent = Agent.build("openclaw", {"model": "custom", "api_key": "secret", "base_url": "https://example.invalid"})
    result = await agent.run(system="", user="task", workspace=tmp_path)
    if usage is None:
        assert result.usage == {}
    else:
        assert expected.items() <= result.usage.items()
