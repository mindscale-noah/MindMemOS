"""Verify protocol switching, observations, and workspace tool boundaries."""

import json
import subprocess

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.agents.buitin_agents.openai_chat.callbacks import (
    native_observation,
    parse_native_tools,
    parse_text_action,
    text_format_error,
    text_observation,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.tools import run_tool
from mindmemos_skill.infra.execution import ExecutionConfig


class Model:
    """Return scripted model events without external calls."""

    config = OpenAIChatConfig(model="test")

    def __init__(self, event):
        self.event = event
        self.tools = None

    def run_messages(self, *, messages, tools=None):
        self.tools = tools
        return AgentResult(output=self.event.get("content", ""), trajectory=[*messages, self.event])


@pytest.mark.parametrize("protocol", ["openai_tool", "text_action"])
def test_protocol_decodes_calls_and_preserves_observations(protocol):
    action = {"name": "write", "arguments": {"path": "a.txt", "content": "hello"}}
    event = {"role": "assistant", "content": "Action:\n" + json.dumps(action)}
    if protocol == "openai_tool":
        event = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "write", "arguments": json.dumps(action["arguments"])},
                }
            ],
        }
    model = Model(event)
    agent = OpenAIChatAgent(
        model.config,
        model=model,
        parse_response=parse_text_action if protocol == "text_action" else parse_native_tools,
        format_error=text_format_error,
        observation=text_observation if protocol == "text_action" else native_observation,
        native_tools=protocol == "openai_tool",
    )
    result = agent.run(messages=[], tools=[{"type": "function"}])
    call = result.output.calls[0]
    assert call.name == "write" and call.arguments == action["arguments"]
    detail = "Exact\n\n✓✗ 'text'"
    message = agent.observation(call, detail)
    if protocol == "openai_tool":
        assert model.tools and message["tool_call_id"] == "call1"
        assert message["content"] == detail and message["role"] == "tool"
    else:
        assert model.tools is None
        assert message == {"role": "user", "content": "Observation: " + detail}


def test_native_multiple_calls_and_invalid_arguments():
    event = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": str(i), "function": {"name": "read", "arguments": args}}
            for i, args in enumerate(['{"path":"x"}', "[]", "{"])
        ],
    }
    model = Model(event)
    turn = (
        OpenAIChatAgent(model.config, model=model, parse_response=parse_native_tools).run(messages=[], tools=[]).output
    )
    assert [call.call_id for call in turn.calls] == ["0", "1", "2"]
    assert not turn.calls[0].error
    assert turn.calls[1].error and turn.calls[2].error


def test_workspace_tools_and_access(tmp_path):
    config = ExecutionConfig()
    assert (
        run_tool("write", {"path": "test.txt", "content": "unique"}, tmp_path, config)
        == f"Successfully wrote 6 characters to {tmp_path / 'test.txt'}"
    )
    assert run_tool("read", {"path": str(tmp_path / "test.txt")}, tmp_path, config) == "unique"
    assert (
        run_tool("edit", {"path": "test.txt", "original_text": "unique", "replacement_text": "done"}, tmp_path, config)
        == f"Successfully edited {tmp_path / 'test.txt'}"
    )
    assert run_tool("read", {"path": "../private"}, tmp_path, config).startswith("Error: access denied")
    (tmp_path / "link").symlink_to(tmp_path.parent)
    assert run_tool("read", {"path": "link/private"}, tmp_path, config).startswith("Error: access denied")
    assert "matched 0 times" in run_tool("edit", {"path": "test.txt", "original_text": "missing"}, tmp_path, config)


def test_shell_sequence_output_and_timeout(tmp_path, monkeypatch):
    calls = []

    def execute(command, workspace, config):
        calls.append((command, config.timeout_s))
        return subprocess.CompletedProcess(command, 3, "out\n", "err\n")

    monkeypatch.setattr("mindmemos_skill.envs.builtin_envs.spreadsheetbench.tools.execute", execute)
    result = run_tool(
        "shell", {"commands": ["pip install xyz", "python a.py"], "timeout_ms": 5000}, tmp_path, ExecutionConfig()
    )
    assert (
        result
        == "Command: pip install xyz\nError: installing packages is disabled\n\nCommand: python a.py\nstdout:\nout\n\nstderr:\nerr\n\nexit_code: 3"
    )
    assert calls == [([*ExecutionConfig().shell_command, "python a.py"], 5)]


def test_custom_callbacks_preserve_trace_usage_and_error_feedback():
    """Use caller callbacks without registration or replacing transport failures."""
    from mindmemos_skill.agents.typing import ToolTurn

    model = Model({"role": "assistant", "content": "custom payload"})
    seen = []

    def parse(result, config):
        seen.append((result.output, config))
        raise ValueError("missing command")

    agent = OpenAIChatAgent(
        model.config,
        model=model,
        parse_response=parse,
        format_error=lambda error: f"Custom correction: {error}",
        observation=lambda call, content: {"role": "user", "content": content},
        native_tools=False,
    )
    result = agent.run(system="instructions", user="task", tools=[{"type": "function"}])
    assert result.output == ToolTurn(text="custom payload", error="Custom correction: missing command")
    assert seen == [("custom payload", model.config)]
    assert result.trajectory[0] == {"role": "system", "content": "instructions"}
    assert model.tools is None

    def transport_failure(**kwargs):
        raise ValueError("provider failure")

    model.run_messages = transport_failure
    with pytest.raises(ValueError, match="provider failure"):
        agent.run(messages=[])


def test_text_format_error_and_completion():
    """Keep the released correction text and completion behavior."""
    from mindmemos_skill.agents.buitin_agents.openai_chat.callbacks import FORMAT_ERROR

    model = Model({"role": "assistant", "content": "Action: invalid"})
    agent = OpenAIChatAgent(
        model.config, model=model, parse_response=parse_text_action, format_error=text_format_error, native_tools=False
    )
    assert agent.run(messages=[]).output.error == FORMAT_ERROR
    model.event["content"] = "ACTION: TASK_COMPLETE"
    turn = agent.run(messages=[]).output
    assert not turn.calls and not turn.error


def test_native_empty_arguments_policy():
    """Preserve the configured permissive policy for malformed native arguments."""
    model = Model({"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "read", "arguments": "["}}]})
    model.config = model.config.model_copy(update={"invalid_tool_arguments": "empty"})
    agent = OpenAIChatAgent(model.config, model=model, parse_response=parse_native_tools)
    call = agent.run(messages=[]).output.calls[0]
    assert call.arguments == {} and not call.error and call.call_id == "c"


def test_plain_chat_and_conflicting_message_inputs():
    """Plain chat needs no callbacks and conflicting histories are rejected."""
    model = Model({"role": "assistant", "content": "answer"})
    agent = OpenAIChatAgent(model.config, model=model)
    assert agent.run(system="s", user="u").output == "answer"
    with pytest.raises(ValueError, match="cannot be combined"):
        agent.run(system="s", messages=[])
    with pytest.raises(ValueError, match="both system and user"):
        agent.run(user="u")
