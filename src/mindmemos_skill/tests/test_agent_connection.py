"""Verify explicit connection settings, secret handling, and transport forwarding."""

import json

import httpx
import openai
import pytest
from mindmemos_skill.agents import Agent
from mindmemos_skill.agents.buitin_agents.claude import agent as claude_module
from mindmemos_skill.agents.buitin_agents.codex import agent as codex_module
from mindmemos_skill.agents.buitin_agents.deepseek_harness import agent as dsh_module
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.livemath.config import LiveMathConfig
from pydantic import ValidationError

CONNECTION_ENV = {
    "MINDMEMOS_AGENT_API_KEY": "environment-secret",
    "MINDMEMOS_AGENT_BASE_URL": "https://environment.invalid/v1",
    "OPENAI_API_KEY": "native-secret",
    "OPENAI_BASE_URL": "https://native.invalid/v1",
}
"""Ambient connection variables that must never supply model configuration."""


@pytest.mark.parametrize("name", ["openai_chat", "codex", "claude", "deepseek_harness"])
@pytest.mark.parametrize("override", ["neither", "key", "url", "both", "null"])
def test_connection_ignores_environment_and_preserves_json_configuration(monkeypatch, name, override):
    """Use only explicit fields and preserve credentials in JSON snapshots."""
    for variable, value in CONNECTION_ENV.items():
        monkeypatch.setenv(variable, value)
    options = {"model": "test-model"}
    if override in {"key", "both"}:
        options["api_key"] = "explicit-secret"
    if override in {"url", "both"}:
        options["base_url"] = "https://explicit.invalid/v1"
    if override == "null":
        options.update(api_key=None, base_url=None)
    config = Agent.build(name, options).config
    expected_key = options.get("api_key")
    assert (config.api_key if config.api_key else None) == expected_key
    assert config.base_url == options.get("base_url")
    if expected_key:
        assert expected_key in repr(config)
        assert expected_key in config.model_dump_json()
    restored = type(config).model_validate(config.model_dump())
    assert restored.api_key == config.api_key
    if name == "openai_chat":
        nested = LiveMathConfig(agent=config)
        if expected_key:
            assert expected_key in nested.model_dump_json()
        assert nested.model_dump(mode="json")["agent"]["base_url"] == config.base_url


@pytest.mark.parametrize("name", ["openai_chat", "codex", "claude", "deepseek_harness"])
def test_connection_missing_and_invalid(name):
    """Allow omitted defaults but reject explicit blanks and obsolete config fields."""
    config = Agent.build(name, {"model": "test"}).config
    assert config.api_key is None and config.base_url is None
    for invalid in [{"api_key": ""}, {"base_url": ""}, {"api_key_env": "OLD"}, {"base_url_env": "OLD"}]:
        with pytest.raises(ValidationError):
            Agent.build(name, {"model": "test", **invalid})


@pytest.mark.parametrize("name", ["openai_chat", "codex", "claude", "deepseek_harness"])
def test_connection_reaches_transport(monkeypatch, tmp_path, name):
    """Send the selected values to the SDK or CLI without modifying native process env."""
    for variable, value in CONNECTION_ENV.items():
        monkeypatch.setenv(variable, value)
    key = "explicit-secret"
    url = "https://explicit.invalid/v1"
    options = {"model": "test-model", "api_key": key, "base_url": url}
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


@pytest.mark.parametrize("name", ["openai_chat", "codex", "claude", "deepseek_harness"])
@pytest.mark.parametrize("missing", ["api_key", "base_url"])
def test_model_call_requires_explicit_connection(monkeypatch, tmp_path, name, missing):
    """Fail before network or process execution even when ambient credentials exist."""
    for variable, value in CONNECTION_ENV.items():
        monkeypatch.setenv(variable, value)
    options = {"model": "test", "api_key": "explicit-secret", "base_url": "https://explicit.invalid/v1"}
    del options[missing]
    agent = Agent.build(name, options)
    with pytest.raises(ValueError, match=f"Provide {missing} in the agent config"):
        if name == "openai_chat":
            agent.run(system="", user="task")
        else:
            agent.run(system="", user="task", workspace=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name",
    [
        "searchqa",
        "livemath",
        "officeqa",
        "alfworld_bounded_history",
        "spreadsheetbench_tool_calling",
        "spreadsheetbench_text_action",
    ],
)
def test_environment_models_use_explicit_connection_config(monkeypatch, name):
    """Apply the same config-only connection contract to every built-in chat environment."""
    for variable, value in CONNECTION_ENV.items():
        monkeypatch.setenv(variable, value)
    options = {"agent": {"model": "test"}}
    if name == "alfworld_bounded_history":
        options["simulator"] = {}
    config_type = Environment.resolve(name).config_type
    missing = config_type.model_validate(options)
    assert missing.agent.api_key is None and missing.agent.base_url is None
    options["agent"].update(api_key="configured-secret", base_url="https://configured.invalid/v1")
    configured = config_type.model_validate(options)
    assert configured.agent.api_key == "configured-secret"
    assert configured.agent.base_url == "https://configured.invalid/v1"
