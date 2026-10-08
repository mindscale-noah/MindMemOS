"""Exercise native harness contracts without calling external providers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from mindmemos_skill.agents import Agent, AgentResult
from mindmemos_skill.agents.buitin_agents.claude import agent as claude_module
from mindmemos_skill.agents.buitin_agents.cli import parse_json_lines, run_cli
from mindmemos_skill.agents.buitin_agents.codex import agent as codex_module
from mindmemos_skill.agents.buitin_agents.config import CLIConfig
from mindmemos_skill.agents.buitin_agents.deepseek_harness import agent as dsh_module
from pydantic import ValidationError


@pytest.mark.parametrize("name", ["codex", "claude", "deepseek_harness"])
def test_registration_and_config(name):
    """Build with current registry and reject legacy or unsupported options."""
    assert isinstance(Agent.build(name, {}), Agent)
    if name == "deepseek_harness":
        assert Agent.build(name, {}).config.provider == "deepseek-official"
    with pytest.raises(ValidationError):
        Agent.build(name, {"temperature": 0.5})


def test_codex_native_result(monkeypatch, tmp_path):
    """Forward stdin, credentials, model settings, and preserve native events."""
    stream = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 3, "cached_input_tokens": 1}},
    ]

    def fake_run(command, workspace, config, overrides, prompt):
        assert workspace == tmp_path and prompt == "task"
        assert command[-1] == "-" and "--ephemeral" in command
        assert 'developer_instructions="instructions"' in command
        assert overrides == {"CODEX_API_KEY": "secret"}
        return "\n".join(map(json.dumps, stream))

    monkeypatch.setattr(codex_module, "run_cli", fake_run)
    result = Agent.build("codex", {"model": "model", "api_key": "secret", "base_url": "https://example.com/v1"}).run(
        system="instructions", user="task", workspace=tmp_path
    )
    assert isinstance(result, AgentResult)
    assert result.output == "done" and result.trajectory[2:] == stream
    assert result.usage["total_tokens"] == 5
    assert "secret" not in result.model_dump_json()


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "turn.failed", "error": "failure"}],
        [{"type": "thread.started"}],
        [{"type": "turn.completed"}],
    ],
)
def test_codex_requires_completed_nonempty_output(monkeypatch, tmp_path, events):
    """Never treat a partial stream or provider failure as success."""
    monkeypatch.setattr(codex_module, "run_cli", lambda *args: "\n".join(map(json.dumps, events)))
    with pytest.raises(RuntimeError):
        Agent.build("codex", {"api_key": "secret", "base_url": "https://example.com/v1"}).run(
            system="", user="task", workspace=tmp_path
        )


@pytest.mark.parametrize("failure", [False, True])
def test_claude_terminal_and_environment(monkeypatch, tmp_path, failure):
    """Honor native turn limits and terminal errors."""

    def fake_run(command, workspace, config, overrides, prompt):
        assert command[command.index("--max-turns") + 1] == "4"
        assert command[command.index("--setting-sources") + 1] == ""
        assert overrides["ANTHROPIC_AUTH_TOKEN"] == "secret"
        assert overrides["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "100"
        return json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": failure,
                "result": "answer",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            }
        )

    monkeypatch.setattr(claude_module, "run_cli", fake_run)
    agent = Agent.build(
        "claude",
        {
            "max_turns": 4,
            "max_completion_tokens": 100,
            "api_key": "secret",
            "base_url": "https://example.com/v1",
        },
    )
    if failure:
        with pytest.raises(RuntimeError):
            agent.run(system="instructions", user="task", workspace=tmp_path)
    else:
        result = agent.run(system="instructions", user="task", workspace=tmp_path)
        assert result.output == "answer" and result.usage["total_tokens"] == 6


@pytest.mark.parametrize("version", [0, 2])
@pytest.mark.parametrize("reason", ["completed", "cancelled"])
def test_deepseek_sessions_and_isolation(monkeypatch, tmp_path, reason, version):
    """Read only the fresh main session and reject unfinished native runs."""
    roots = []

    def fake_run(command, workspace, config, overrides):
        patch = yaml.safe_load(Path(command[command.index("--patch") + 1]).read_text())
        root = Path(patch[0]["config"]["root"])
        roots.append(root)
        root.mkdir()
        assert patch[1]["config"]["customSkillDirs"] == [str(tmp_path / ".dsh/skills")]
        assert command[-1] == "System instructions:\ninstructions\n\nUser:\ntask"
        rows = [{"type": "session", "version": version}, {"type": "turn/end", "data": {"reason": {"kind": reason}}}]
        (root / "session-main.jsonl").write_text("\n".join(map(json.dumps, rows)))
        return "answer\n"

    monkeypatch.setattr(dsh_module, "run_cli", fake_run)
    agent = Agent.build("deepseek_harness", {"api_key": "secret", "base_url": "https://example.com/v1"})
    if reason == "cancelled":
        with pytest.raises(RuntimeError):
            agent.run(system="instructions", user="task", workspace=tmp_path)
    else:
        for _ in range(2):
            result = agent.run(system="instructions", user="task", workspace=tmp_path)
            assert result.output == "answer" and result.usage == {}
        assert roots[0] != roots[1]


@pytest.mark.parametrize("text", ["", "[]", "broken"])
def test_invalid_jsonl(text):
    """Require a nonempty JSON object stream."""
    with pytest.raises(RuntimeError):
        parse_json_lines(text, cli_name="test")


def test_process_stdin_exit_and_timeout(tmp_path):
    """Exercise actual process IO, exit errors, and timeout cleanup."""
    config = CLIConfig(timeout_s=5)
    output = run_cli([sys.executable, "-c", "import sys; print(sys.stdin.read())"], tmp_path, config, {}, "你好")
    assert output.strip() == "你好"
    with pytest.raises(RuntimeError, match="status 2"):
        run_cli([sys.executable, "-c", "raise SystemExit(2)"], tmp_path, config, {})
    with pytest.raises(subprocess.TimeoutExpired):
        run_cli([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, CLIConfig(timeout_s=0.1), {})


def test_cli_process_excludes_ambient_model_credentials(monkeypatch, tmp_path):
    """Pass configured credentials while removing competing parent model settings."""
    from mindmemos_skill.agents.buitin_agents.constants import NATIVE_CONNECTION_ENV

    for variable in NATIVE_CONNECTION_ENV:
        monkeypatch.setenv(variable, "ambient-value")
    overrides = {"CODEX_API_KEY": "configured-secret"}
    script = (
        "import os, json; print(json.dumps({name: os.environ.get(name) for name in "
        + repr(NATIVE_CONNECTION_ENV)
        + "}))"
    )
    result = json.loads(run_cli([sys.executable, "-c", script], tmp_path, CLIConfig(), overrides))
    assert result == {variable: overrides.get(variable) for variable in NATIVE_CONNECTION_ENV}
