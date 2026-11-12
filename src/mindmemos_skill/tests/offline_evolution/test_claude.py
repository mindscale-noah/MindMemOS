"""Verify offline scopes through the ordinary Claude Agent and real subprocess execution."""

import json
import os
import subprocess
import sys
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from mindmemos_skill.agents.buitin_agents.claude import ClaudeAgent, ClaudeConfig
from mindmemos_skill.agents.buitin_agents.claude import agent as claude_module
from mindmemos_skill.algos.evolve.offline_evolution.agents import claude as offline_module
from mindmemos_skill.algos.evolve.offline_evolution.agents import session_identity
from mindmemos_skill.algos.evolve.offline_evolution.agents.claude import (
    _deny_rules,
    _path_rule,
    run_claude,
    session_arguments,
)
from mindmemos_skill.algos.evolve.offline_evolution.agents.typing import SessionOptions
from mindmemos_skill.infra.execution import executor


def options(tmp_path, **changes):
    """Create independent native scopes for one invocation."""
    read, write, host = (tmp_path / name for name in ("inputs", "workspace", ".offline-evolution"))
    for path in (read, write, host):
        path.mkdir(exist_ok=True)
    return SessionOptions(
        config_dir=host / ".claude",
        read_dirs=(read,),
        write_dirs=(write,),
        deny_paths=(host,),
        log_path=host / "events.jsonl",
        **changes,
    )


def config(**changes):
    """Select a local executable with explicit fake model credentials."""
    return ClaudeConfig(cli_path=sys.executable, api_key="test-key", base_url="https://example.invalid", **changes)


@pytest.fixture
def native(tmp_path, monkeypatch):
    """Replace only the native executable while retaining the shared CLI and executor."""
    script = tmp_path / "native.py"
    calls = []
    original = offline_module.run_cli

    async def invoke(command, workspace, settings, overrides, prompt=None, **execution):
        calls.append(command)
        return await original([sys.executable, str(script)], workspace, settings, overrides, prompt, **execution)

    monkeypatch.setattr(claude_module, "run_cli", invoke)
    monkeypatch.setattr(offline_module, "run_cli", invoke)
    return script, calls


def test_scopes_resume_and_host_denial(tmp_path):
    request = options(tmp_path, session_id="known", readonly_dirs=(tmp_path / "workspace" / "keep",))
    command = session_arguments(request, tmp_path)
    assert command[command.index("--resume") + 1] == "known"
    assert _path_rule("Read", request.read_dirs[0]) in command
    assert _path_rule("Write", request.read_dirs[0]) not in command
    settings = json.loads(command[command.index("--settings") + 1])
    assert "Bash(*.offline-evolution*)" in settings["permissions"]["deny"]
    assert _path_rule("Write", request.readonly_dirs[0]) in settings["permissions"]["deny"]
    assert _path_rule("Read", request.readonly_dirs[0]) not in settings["permissions"]["deny"]
    assert _deny_rules(request, tmp_path) == _deny_rules(request, tmp_path.resolve())


async def test_existing_agent_handles_large_events_and_persistent_native_arguments(tmp_path, native):
    script, calls = native
    script.write_text(
        "import json\n"
        "print(json.dumps({'type':'system','session_id':'native-session'}), flush=True)\n"
        "print(json.dumps({'type':'assistant','payload':'x'*200000}), flush=True)\n"
        "print(json.dumps({'type':'result','session_id':'native-session','subtype':'success',"
        "'result':'done','usage':{'input_tokens':3,'cache_creation_input_tokens':7,'cache_read_input_tokens':11,'output_tokens':2},'total_cost_usd':0.25}), flush=True)\n"
    )
    request = options(tmp_path)
    settings = config(model="test-model", extra_args=["--max-budget-usd", "1"])
    result = await run_claude(settings, user="prompt", workspace=tmp_path, session=request)
    assert result.output == "done" and session_identity(result) == "native-session"
    assert result.usage["total_tokens"] == 5
    assert result.usage["cache_creation_input_tokens"] == 7
    assert result.usage["cache_read_input_tokens"] == 11
    assert request.log_path.stat().st_size > 200000
    command = calls[0]
    assert "--no-session-persistence" not in command
    assert command.count("--model") == 1
    assert "--max-budget-usd" in command
    UUID(command[command.index("--session-id") + 1])
    assert settings.extra_args == ["--max-budget-usd", "1"]
    await run_claude(
        settings, user="feedback", workspace=tmp_path, session=options(tmp_path, session_id="native-session")
    )
    assert calls[1][calls[1].index("--resume") + 1] == "native-session"
    assert "--no-session-persistence" not in calls[1]


async def test_provider_error_on_stdout_survives_native_nonzero_exit(tmp_path, native):
    """Retain the actual API error when Claude exits with empty stderr."""
    script, _ = native
    script.write_text(
        "import json, sys\n"
        "print(json.dumps({'type':'system','session_id':'failed-session'}), flush=True)\n"
        "print(json.dumps({'type':'result','is_error':True,"
        "'result':'API Error: 400 tool_calls maximum 128, got 129'}), flush=True)\n"
        "sys.exit(1)\n"
    )
    request = options(tmp_path)
    with pytest.raises(RuntimeError, match="tool_calls maximum 128, got 129"):
        await run_claude(config(), user="prompt", workspace=tmp_path, session=request)
    assert "failed-session" in request.log_path.read_text()


async def test_child_environment_uses_explicit_connection_and_drops_ambient_aliases(tmp_path, monkeypatch, native):
    script, _ = native
    script.write_text(
        "import json,os\n"
        "names=['ANTHROPIC_DEFAULT_OPUS_MODEL','ANTHROPIC_API_KEY','ANTHROPIC_AUTH_TOKEN',"
        "'CLAUDECODE','CLAUDE_CODE_USE_POWERSHELL_TOOL','CLAUDE_CONFIG_DIR','CLAUDE_CODE_DISABLE_AUTO_MEMORY']\n"
        "print(json.dumps({'type':'result','result':json.dumps({name:os.getenv(name) for name in names})}))\n"
    )
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "ambient-alias")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_USE_POWERSHELL_TOOL", "1")
    request = options(tmp_path)
    result = await run_claude(config(), user="prompt", workspace=tmp_path, session=request)
    env = json.loads(result.output)
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] is None and env["ANTHROPIC_API_KEY"] is None
    assert env["CLAUDECODE"] is None and env["CLAUDE_CODE_USE_POWERSHELL_TOOL"] is None
    assert env["ANTHROPIC_AUTH_TOKEN"] == "test-key"
    assert env["CLAUDE_CONFIG_DIR"] == str(request.config_dir)
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert os.environ["ANTHROPIC_API_KEY"] == "ambient-key"


@pytest.mark.parametrize(
    "argument", ["--no-session-persistence", "--permission-mode=bypassPermissions", "--allowed-tools"]
)
async def test_conflicting_offline_preset_arguments_are_rejected_at_the_algorithm_boundary(tmp_path, argument):
    with pytest.raises(ValueError, match="conflict with offline"):
        await run_claude(config(extra_args=[argument]), user="prompt", workspace=tmp_path, session=options(tmp_path))


async def test_ordinary_agent_keeps_native_extra_arguments_and_transient_default(tmp_path, native):
    script, calls = native
    script.write_text("import json\nprint(json.dumps({'type':'result','result':'done'}))\n")
    result = await ClaudeAgent(config(extra_args=["--allowed-tools", "Read"])).run(
        system="normal task", user="prompt", workspace=tmp_path
    )
    assert result.output == "done"
    assert "--no-session-persistence" in calls[0]
    assert calls[0][calls[0].index("--allowed-tools") + 1] == "Read"
    assert not (tmp_path / ".offline-evolution").exists()


async def test_timeout_preserves_partial_events_in_the_shared_executor(tmp_path, native):
    script, _ = native
    script.write_text(
        "import json,time\nprint(json.dumps({'type':'system','session_id':'interrupted'}),flush=True)\ntime.sleep(30)\n"
    )
    request = options(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        await run_claude(config(timeout_s=0.5), user="prompt", workspace=tmp_path, session=request)
    assert "interrupted" in request.log_path.read_text()


async def test_log_initialization_failure_does_not_start_a_child(tmp_path, monkeypatch, native):
    request = options(tmp_path).model_copy(update={"log_path": tmp_path / "directory"})
    request.log_path.mkdir()
    launch = AsyncMock()
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", launch)
    with pytest.raises(IsADirectoryError):
        await run_claude(config(), user="prompt", workspace=tmp_path, session=request)
    launch.assert_not_called()


async def test_platform_claude_renews_project_token_for_each_native_request(tmp_path, native, monkeypatch):
    """Use the ordinary offline subprocess path with a session-local loopback relay."""
    from time import time

    import httpx
    import jwt
    from mindmemos_skill.infra.model_access import anthropic, platform

    signing_key = "only-on-host-test-gateway-signing-key"
    monkeypatch.setenv(platform.PLATFORM_MODEL_SECRET_ENV, signing_key)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "ambient-oauth")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    now = int(time())
    clock = iter((now, now, now + 61))
    monkeypatch.setattr(platform, "time", lambda: next(clock))
    tokens = []
    original_client = httpx.AsyncClient

    def provider(request):
        assert request.url == "https://platform.invalid/internal/v1/skill-models/v1/messages?beta=true"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "x-api-key" not in request.headers
        jwt_value = request.headers["authorization"].removeprefix("Bearer ")
        tokens.append(
            jwt.decode(
                jwt_value,
                signing_key,
                algorithms=["HS256"],
                audience="platform-skill-model",
                options={"verify_iat": False},
            )
        )
        return httpx.Response(
            200, text="event: message_stop\ndata: {}\n\n", headers={"Content-Type": "text/event-stream"}
        )

    monkeypatch.setattr(
        anthropic.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(provider), **kwargs),
    )
    script, calls = native
    script.write_text(
        """
import json
import os
import urllib.error
import urllib.request

for name in ('MINDMEMOS_SKILL_MODEL_GATEWAY_SECRET', 'CLAUDE_CODE_OAUTH_TOKEN',
             'CLAUDE_CODE_USE_BEDROCK', 'ANTHROPIC_API_KEY'):
    assert name not in os.environ

base = os.environ['ANTHROPIC_BASE_URL']
assert base.startswith('http://127.0.0.1:')
def send(token):
    request = urllib.request.Request(
        base + '/v1/messages?beta=true', data=b'{"stream":true}', headers={
            'Authorization': 'Bearer ' + token, 'anthropic-version': '2023-06-01',
        },
    )
    return urllib.request.urlopen(request)

try:
    send('unrelated-session')
except urllib.error.HTTPError as exc:
    assert exc.code == 401
else:
    raise AssertionError('accepted unrelated credentials')

for _ in range(2):
    with send(os.environ['ANTHROPIC_AUTH_TOKEN']) as response:
        assert b'event: message_stop' in response.read()
print(json.dumps({'type': 'result', 'subtype': 'success', 'result': base}))
"""
    )
    settings = ClaudeConfig(
        transport="platform_gateway", cli_path=sys.executable, api_key="unused", base_url="https://unused.invalid"
    )
    gateway = platform.PlatformModelGateway("https://platform.invalid/internal/v1/skill-models", signing_key)
    with platform.bind_platform_models(gateway, "authorized-project"):
        result = await run_claude(settings, user="prompt", workspace=tmp_path, session=options(tmp_path))
    assert calls[0][calls[0].index("--model") + 1] == "platform-chat"
    assert [claims["project_id"] for claims in tokens] == ["authorized-project"] * 2
    assert [claims["exp"] for claims in tokens] == [now + 60, now + 121]
    async with original_client(trust_env=False) as client:
        with pytest.raises(httpx.ConnectError):
            await client.post(f"{result.output}/v1/messages")
