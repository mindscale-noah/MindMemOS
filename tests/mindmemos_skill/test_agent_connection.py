"""Verify shared connection precedence, secret handling, and transport forwarding."""

import json

import httpx
import openai
import pytest
from mindmemos_skill.agents import Agent
from mindmemos_skill.agents.buitin_agents.claude import agent as claude_module
from mindmemos_skill.agents.buitin_agents.codex import agent as codex_module
from mindmemos_skill.agents.buitin_agents.deepseek_harness import agent as dsh_module
from mindmemos_skill.agents.constants import AGENT_API_KEY_ENV, AGENT_BASE_URL_ENV
from mindmemos_skill.envs.builtin_envs.livemath.config import LiveMathConfig
from pydantic import ValidationError


@pytest.mark.parametrize("name", ["openai_chat", "codex", "claude", "deepseek_harness"])
@pytest.mark.parametrize("override", ["neither", "key", "url", "both", "null"])
def test_connection_precedence_and_json_safety(monkeypatch, name, override):
    """Resolve each field independently and keep credentials out of JSON snapshots."""
    monkeypatch.setenv(AGENT_API_KEY_ENV, "environment-secret")
    monkeypatch.setenv(AGENT_BASE_URL_ENV, "https://environment.invalid/v1")
    options = {"model": "test-model"}
    if override in {"key", "both"}:
        options["api_key"] = "explicit-secret"
    if override in {"url", "both"}:
        options["base_url"] = "https://explicit.invalid/v1"
    if override == "null":
        options.update(api_key=None, base_url=None)
    config = Agent.build(name, options).config
    expected_key = options.get("api_key") or "environment-secret"
    assert config.api_key.get_secret_value() == expected_key
    assert config.base_url == (options.get("base_url") or "https://environment.invalid/v1")
    assert expected_key not in repr(config)
    assert expected_key not in config.model_dump_json()
    restored = type(config).model_validate(config.model_dump())
    assert restored.api_key.get_secret_value() == expected_key
    monkeypatch.setenv(AGENT_BASE_URL_ENV, "https://later.invalid/v1")
    assert config.base_url != "https://later.invalid/v1"
    if name == "openai_chat":
        nested = LiveMathConfig(agent=config)
        assert expected_key not in nested.model_dump_json()
        assert nested.model_dump(mode="json")["agent"]["base_url"] == config.base_url


@pytest.mark.parametrize("name", ["openai_chat", "codex", "claude", "deepseek_harness"])
def test_connection_missing_and_invalid(monkeypatch, name):
    """Allow omitted defaults but reject explicit blanks and obsolete config fields."""
    monkeypatch.delenv(AGENT_API_KEY_ENV, raising=False)
    monkeypatch.delenv(AGENT_BASE_URL_ENV, raising=False)
    config = Agent.build(name, {"model": "test"}).config
    assert config.api_key is None and config.base_url is None
    for invalid in [{"api_key": ""}, {"base_url": ""}, {"api_key_env": "OLD"}, {"base_url_env": "OLD"}]:
        with pytest.raises(ValidationError):
            Agent.build(name, {"model": "test", **invalid})


@pytest.mark.parametrize("name", ["openai_chat", "codex", "claude", "deepseek_harness"])
@pytest.mark.parametrize("source", ["config", "environment"])
def test_connection_reaches_transport(monkeypatch, tmp_path, name, source):
    """Send the selected values to the SDK or CLI without modifying native process env."""
    monkeypatch.setenv(AGENT_API_KEY_ENV, "environment-secret")
    monkeypatch.setenv(AGENT_BASE_URL_ENV, "https://environment.invalid/v1")
    options = {"model": "test-model"}
    if source == "config":
        options.update(api_key="explicit-secret", base_url="https://explicit.invalid/v1")
    key = "explicit-secret" if source == "config" else "environment-secret"
    url = f"https://{'explicit' if source == 'config' else 'environment'}.invalid/v1"
    agent = Agent.build(name, options)
    calls = []

    def respond(request):
        calls.append(request)
        assert str(request.url) == url + "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {key}"
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
            },
        )

    def run_cli(command, workspace, config, overrides, prompt=None):
        calls.append(command)
        assert all(key not in arg for arg in command)
        if name == "codex":
            assert overrides["CODEX_API_KEY"] == key
            assert f'model_providers.mindmemos.base_url="{url}"' in command
            return "\n".join(
                map(
                    json.dumps,
                    [
                        {"type": "item.completed", "item": {"type": "agent_message", "text": "OK"}},
                        {"type": "turn.completed"},
                    ],
                )
            )
        if name == "claude":
            assert overrides["ANTHROPIC_AUTH_TOKEN"] == key
            assert overrides["ANTHROPIC_BASE_URL"] == url
            assert command[command.index("--setting-sources") + 1] == ""
            return json.dumps({"type": "result", "subtype": "success", "result": "OK"})
        assert overrides["DEEPSEEK_API_KEY"] == key
        assert overrides["DEEPSEEK_BASE_URL"] == url
        return "OK"

    if name == "openai_chat":
        monkeypatch.setattr(
            openai,
            "DefaultHttpxClient",
            lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs),
        )
        result = agent.run(system="test", user="test")
    else:
        module = {"codex": codex_module, "claude": claude_module, "deepseek_harness": dsh_module}[name]
        monkeypatch.setattr(module, "run_cli", run_cli)
        if name == "deepseek_harness":
            monkeypatch.setattr(dsh_module, "read_session_events", lambda root: (root, []))
        result = agent.run(system="test", user="test", workspace=tmp_path)
    assert result.output == "OK" and len(calls) == 1
