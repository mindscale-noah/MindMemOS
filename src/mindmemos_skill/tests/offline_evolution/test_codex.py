"""Exercise Codex stage adaptation without external providers or sandbox bypasses."""

import json
import os
import subprocess
import sys
import tomllib

import pytest
from mindmemos_skill.agents import Agent
from mindmemos_skill.agents.buitin_agents.codex import agent as codex_module
from mindmemos_skill.agents.buitin_agents.codex.agent import CodexAgent
from mindmemos_skill.agents.buitin_agents.codex.config import CodexConfig
from mindmemos_skill.algos.evolve.offline_evolution.agents import codex as offline_module
from mindmemos_skill.algos.evolve.offline_evolution.agents import session_identity
from mindmemos_skill.algos.evolve.offline_evolution.agents.codex import (
    CODEX_HOME_ENV,
    PERMISSION_PROFILE,
    run_codex,
    session_arguments,
)
from mindmemos_skill.algos.evolve.offline_evolution.agents.typing import SessionOptions
from mindmemos_skill.algos.evolve.offline_evolution.config import EvolutionOptions, OfflineEvolutionConfig
from mindmemos_skill.algos.evolve.offline_evolution.engine import OfflineEvolution
from mindmemos_skill.algos.evolve.offline_evolution.typing import RunLayout

from .helpers import EVIDENCE_SCRIPT, make_inputs


def options(root, **changes):
    """Create distinct readable, writable and private native scopes."""
    read, write, host = (root / name for name in ("inputs", "workspace", ".offline-evolution"))
    for path in (read, write, host):
        path.mkdir(parents=True, exist_ok=True)
    return SessionOptions(
        config_dir=host / ".codex",
        read_dirs=(read,),
        write_dirs=(write,),
        deny_paths=(host,),
        log_path=host / "events.jsonl",
        **changes,
    )


def config(**changes):
    """Use explicit fake connection settings for a locally substituted executable."""
    return CodexConfig(api_key="codex-test-key", base_url="https://example.invalid/v1", **changes)


def settings(command):
    """Parse the actual native overrides using the standard TOML parser."""
    return tomllib.loads("\n".join(command[index + 1] for index, value in enumerate(command) if value == "--config"))


@pytest.fixture
def native(tmp_path, monkeypatch):
    """Replace only the executable while retaining shared command construction and execution."""
    script = tmp_path / "codex_stub.py"
    calls = []
    original = offline_module.run_cli

    async def invoke(command, workspace, selected, overrides, prompt=None, **execution):
        calls.append((command, overrides, execution))
        return await original(
            [sys.executable, str(script), *command[1:]], workspace, selected, overrides, prompt, **execution
        )

    monkeypatch.setattr(codex_module, "run_cli", invoke)
    monkeypatch.setattr(offline_module, "run_cli", invoke)
    return script, calls


def test_native_profile_separates_write_roots_readonly_members_and_host(tmp_path):
    readonly = tmp_path / "workspace/readonly"
    readonly.mkdir(parents=True)
    request = options(tmp_path, readonly_dirs=(readonly,))
    scratch = tmp_path / ".codex-tmp"
    command = session_arguments(request, scratch)
    value = settings(command)
    profile = value["permissions"][PERMISSION_PROFILE]
    filesystem = profile["filesystem"]
    assert filesystem[str(request.read_dirs[0])] == "read"
    assert filesystem[str(request.write_dirs[0])] == "write"
    assert filesystem[str(readonly)] == "read"
    assert filesystem[str(request.deny_paths[0])] == "deny"
    assert profile["network"]["enabled"] is False
    assert value["shell_environment_policy"]["inherit"] == "none"
    assert "CODEX_API_KEY" not in value["shell_environment_policy"]["set"]
    assert "--ignore-user-config" in command and "--ignore-rules" in command
    assert value["features"]["multi_agent"] is False
    assert value["project_doc_max_bytes"] == 0 and value["web_search"] == "disabled"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--ephemeral"],
        ["--sandbox", "workspace-write"],
        ["-sdanger-full-access"],
        ["--dangerously-bypass-approvals-and-sandbox"],
        ["resume", "other-thread"],
        ["-c", 'default_permissions=":danger-full-access"'],
        ["--config=shell_environment_policy.inherit=all"],
        ["-cpermissions={}"],
    ],
)
async def test_managed_policy_cannot_be_overridden(tmp_path, arguments):
    with pytest.raises(ValueError, match="offline|managed"):
        await run_codex(config(extra_args=arguments), user="stage", workspace=tmp_path, session=options(tmp_path))


async def test_shared_codex_persists_resumes_and_normalizes_cache_usage(tmp_path, native):
    script, calls = native
    script.write_text("""import json, sys
sys.stdin.read()
print(json.dumps({"type":"thread.started","thread_id":"native-thread"}), flush=True)
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"done"}}), flush=True)
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":6,"output_tokens":4}}), flush=True)
""")
    request = options(tmp_path)
    original_home = os.environ.get(CODEX_HOME_ENV)
    selected = config(model="test-model", extra_args=["--config", "model_max_output_tokens=1000"])
    result = await run_codex(selected, user="initial", workspace=tmp_path, session=request)
    assert result.output == "done" and session_identity(result) == "native-thread"
    assert result.usage["input_tokens"] == 4 and result.usage["cache_read_input_tokens"] == 6
    assert result.usage["total_tokens"] == 14
    assert result.trajectory[-1]["usage"]["input_tokens"] == 10
    assert len(request.log_path.read_text().splitlines()) == 3
    assert calls[0][1][CODEX_HOME_ENV] == str(request.config_dir)
    assert calls[0][1]["CODEX_API_KEY"] == "codex-test-key"
    assert "--ephemeral" not in calls[0][0] and "--sandbox" not in calls[0][0]
    assert settings(calls[0][0])["default_permissions"] == PERMISSION_PROFILE
    assert selected.extra_args == ["--config", "model_max_output_tokens=1000"]
    await run_codex(
        selected,
        user="feedback",
        workspace=tmp_path,
        session=request.model_copy(update={"session_id": "native-thread"}),
    )
    assert calls[1][0][:4] == ["codex", "exec", "resume", "native-thread"]
    assert os.environ.get(CODEX_HOME_ENV) == original_home


async def test_timeout_keeps_partial_thread_events(tmp_path, native):
    script, _ = native
    script.write_text("""import json, time
print(json.dumps({"type":"thread.started","thread_id":"interrupted-thread"}), flush=True)
time.sleep(10)
""")
    request = options(tmp_path)
    with pytest.raises(subprocess.TimeoutExpired):
        await run_codex(config(timeout_s=0.15), user="stage", workspace=tmp_path, session=request)
    assert "interrupted-thread" in request.log_path.read_text()


async def test_offline_invocation_does_not_change_ordinary_codex_behavior(tmp_path, native):
    """Keep native command defaults, token fields and registration unchanged across offline calls."""
    script, calls = native
    script.write_text("""import json
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"done"}}))
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":6,"output_tokens":4}}))
""")
    selected = config(model="ordinary-model", reasoning_effort="high")
    agent = Agent.from_config(selected)
    before = await agent.run(system="instructions", user="task", workspace=tmp_path)
    await run_codex(selected, user="offline", workspace=tmp_path, session=options(tmp_path))
    after = await agent.run(system="instructions", user="task", workspace=tmp_path)
    assert before == after and before.usage["input_tokens"] == 10
    assert calls[0] == calls[2]
    ordinary, overrides, execution = calls[2]
    assert "--ephemeral" in ordinary
    assert ordinary[ordinary.index("--sandbox") + 1] == "workspace-write"
    assert CODEX_HOME_ENV not in overrides and execution == {}
    assert "default_permissions" not in settings(ordinary)
    assert Agent.resolve("codex") is CodexAgent
    assert selected.extra_args == []


@pytest.mark.parametrize("scenario", ["plain", "feedback", "interrupted"])
async def test_full_codex_flow_and_internal_preflight_reuse_shared_pipeline(tmp_path, native, scenario):
    script, calls = native
    script.write_text(
        """import json, pathlib, subprocess, sys, tomllib
prompt = sys.stdin.read()
args = sys.argv[1:]
config = tomllib.loads("\\n".join(args[i+1] for i, arg in enumerate(args) if arg == "--config"))
root = pathlib.Path.cwd()
evidence = "collection.json" in prompt
stage = "evidence" if evidence else "change"
print(json.dumps({"type":"thread.started","thread_id":"thread-"+stage}), flush=True)
if SCENARIO == "interrupted" and evidence and "resume" not in args:
    sys.exit(1)
if evidence:
    workspace = root / "workspace/evidence"
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts/collect.py").write_text(COLLECTOR)
    (workspace / "collection.json").write_text('{"collector":"scripts/collect.py"}')
else:
    workspace = root / "workspace/change"
    forms = workspace / "skills/pdf/forms.md"
    forms.write_text(forms.read_text().replace("python scripts", "python3 scripts"))
    findings = ["F99"] if SCENARIO == "feedback" and "resume" not in args else ["F1","F2"]
    (workspace / "changes.json").write_text(json.dumps({"changes":[{"id":"C1","files":["pdf/forms.md"],"findings":findings,"rationale":"Use the documented working interpreter."}]}))
preview = subprocess.run([sys.executable,"-m","mindmemos_skill.algos.evolve.offline_evolution.check",stage,"--run",str(root)], env=config["shell_environment_policy"]["set"], text=True, capture_output=True)
expected = 1 if SCENARIO == "feedback" and not evidence and "resume" not in args else 0
assert preview.returncode == expected, (preview.stdout, preview.stderr)
print(json.dumps({"type":"item.completed","item":{"type":"agent_message","text":"done"}}), flush=True)
print(json.dumps({"type":"turn.completed","usage":{"input_tokens":10,"cached_input_tokens":6,"output_tokens":4}}), flush=True)
""".replace(
            "prompt = sys.stdin.read()",
            "SCENARIO = " + repr(scenario) + "\nCOLLECTOR = " + repr(EVIDENCE_SCRIPT) + "\nprompt = sys.stdin.read()",
        )
    )
    skills, trajectories = make_inputs(tmp_path)
    algorithm = OfflineEvolution(
        OfflineEvolutionConfig(agent=config(timeout_s=30)), EvolutionOptions(allow_create_skills=False)
    )
    if scenario == "interrupted":
        with pytest.raises(RuntimeError):
            await algorithm.evolve(skills=skills, trajectories=trajectories, work_dir=tmp_path / "run")
        result = await algorithm.resume(work_dir=tmp_path / "run")
        assert calls[1][0][:4] == ["codex", "exec", "resume", "thread-evidence"]
    else:
        result = await algorithm.evolve(skills=skills, trajectories=trajectories, work_dir=tmp_path / "run")
    assert result.outcome == "updated" and len(calls) == (2 if scenario == "plain" else 3)
    assert "python3 scripts" in (result.skills / "pdf/forms.md").read_text()
    assert result.state.evidence.session_id == "thread-evidence" and result.state.change.session_id == "thread-change"
    assert result.state.evidence.tokens.input == 4
    assert result.state.change.tokens.cache_read == (12 if scenario == "feedback" else 6)
    assert result.state.change_gate.feedback_rounds == (1 if scenario == "feedback" else 0)
    if scenario == "feedback":
        assert calls[2][0][:4] == ["codex", "exec", "resume", "thread-change"]
    layout = RunLayout(root=tmp_path / "run")
    preview = json.loads(layout.preflight_path.read_text())
    assert "agent" not in preview and "codex-test-key" not in layout.preflight_path.read_text()
    assert "codex-test-key" not in layout.state_path.read_text()
    for command, _, _ in calls:
        assert settings(command)["permissions"][PERMISSION_PROFILE]["filesystem"][str(layout.host)] == "deny"
