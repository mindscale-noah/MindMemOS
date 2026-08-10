from __future__ import annotations

import builtins
import subprocess
import sys
import textwrap
from datetime import UTC, datetime

import pytest
from mindmemos_skill.agents import (
    Agent,
    AgentConfig,
    AgentExecutionRequest,
    ClaudeAgentConfig,
    ClaudeSDKAgentConfig,
    OpenClawAgentConfig,
    get_agent,
    list_agents,
)
from mindmemos_skill.agents.claude import ClaudeAgent, ClaudeSDKAgent, ClaudeSkillRuntime
from mindmemos_skill.registry import get_agent as registry_get_agent
from mindmemos_skill.registry import list_agents as registry_list_agents
from mindmemos_skill.typing import (
    AgentType,
    Environment,
    ExecutionInfo,
    Rollout,
    Skill,
    SkillInjectionMode,
    SkillUsageType,
    Task,
    Trajectory,
    TrajectoryStatus,
)
from pydantic import ValidationError


def make_skill(*, content_hash: str = "sha256:demo") -> Skill:
    return Skill(
        skill_id="skill-1",
        version_id="version-1",
        version_label="1.0.0",
        content_hash=content_hash,
        name="demo",
        blob={"SKILL.md": "instructions"},
        resources={"script.py": "first"},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )


def make_request(**updates: object) -> AgentExecutionRequest:
    data: dict[str, object] = {
        "trajectory_id": "trajectory-1",
        "task": Task(task_id="task-1", instruction="solve it"),
        "rollout": Rollout(rollout_id="rollout-1"),
    }
    data.update(updates)
    return AgentExecutionRequest.model_validate(data)


def test_get_agent_builds_cli_agent_from_common_and_specific_config() -> None:
    config = ClaudeAgentConfig(
        model="claude-sonnet",
        max_turns=8,
        cli_path="/opt/bin/claude",
        timeout_seconds=45,
        dangerously_skip_permissions=False,
    )

    agent = get_agent(agent_type=AgentType.CLAUDE, config=config)

    assert isinstance(agent, ClaudeAgent)
    assert agent.config == config


def test_get_agent_builds_sdk_agent_without_importing_optional_sdk() -> None:
    agent = get_agent(
        agent_type=AgentType.CLAUDE_SDK,
        config={"model": "claude-opus", "max_turns": 4, "permission_mode": "default"},
    )

    assert isinstance(agent, ClaudeSDKAgent)
    assert agent.config == ClaudeSDKAgentConfig(
        model="claude-opus",
        max_turns=4,
        permission_mode="default",
    )


def test_get_agent_rejects_config_for_a_different_agent_type() -> None:
    with pytest.raises(ValidationError):
        get_agent(
            agent_type=AgentType.CLAUDE,
            config={"permission_mode": "bypassPermissions"},
        )


def test_list_agents_uses_agent_type_values() -> None:
    assert list_agents() == [
        AgentType.CLAUDE.value,
        AgentType.CLAUDE_SDK.value,
        AgentType.OPENCLAW.value,
        AgentType.REACT.value,
    ]
    assert get_agent is registry_get_agent
    assert list_agents is registry_list_agents


def test_get_agent_builds_openclaw_agent_config() -> None:
    from mindmemos_skill.agents.openclaw import OpenClawAgent

    agent = get_agent(
        agent_type=AgentType.OPENCLAW,
        config={"model": "openai/gpt-5", "agent_id": "main", "timeout_seconds": 30},
    )

    assert isinstance(agent, OpenClawAgent)
    assert agent.config == OpenClawAgentConfig(
        model="openai/gpt-5",
        agent_id="main",
        timeout_seconds=30,
    )


def test_agent_without_mounted_skill_runtime_rejects_injection() -> None:
    class AgentWithoutSkillRuntime(Agent[AgentConfig]):
        async def execute(self, request: AgentExecutionRequest) -> Trajectory:
            raise NotImplementedError

    agent = AgentWithoutSkillRuntime({})

    assert agent.supported_skill_injection_modes == frozenset()
    with pytest.raises(ValueError, match="no Skill injection mode configured"):
        agent.inject_skills([])


def test_agent_family_limits_supported_skill_injection_modes() -> None:
    claude = ClaudeAgent({})

    assert claude.supported_skill_injection_modes == frozenset({SkillInjectionMode.FILESYSTEM})
    assert isinstance(claude.get_skill_runtime(), ClaudeSkillRuntime)
    with claude.inject_skills([]) as injection:
        assert injection.mode is SkillInjectionMode.FILESYSTEM
        assert injection.tools == []
    with pytest.raises(ValueError, match="does not support 'tool'"):
        claude.inject_skills([], mode=SkillInjectionMode.TOOL)


def test_skill_binding_uses_persisted_version_identity_and_hash() -> None:
    skill = make_skill(content_hash="sha256:canonical")
    request = make_request(skills=[skill])
    now = datetime.now(UTC)
    trajectory = Trajectory(
        trajectory_id=request.trajectory_id,
        task=request.task,
        rollout=request.rollout,
        environment=request.environment,
        injected_skills=request.skills,
        events=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "Skill", "arguments": '{"skill":"demo"}'},
                    }
                ],
            }
        ],
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
        ),
    )

    binding = ClaudeAgent({}).bind_skills(trajectory)[0]

    assert binding.skill_id == "skill-1"
    assert binding.version_id == "version-1"
    assert binding.version_label == "1.0.0"
    assert binding.content_hash == "sha256:canonical"
    assert binding.usage is SkillUsageType.INJECTED


def test_skill_workspace_materializes_the_persisted_bundle_without_rewriting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "injected"
    workspace.mkdir()
    monkeypatch.setattr(
        "mindmemos_skill.agents.claude.skill_runtime.tempfile.mkdtemp",
        lambda **_: str(workspace),
    )
    skill = make_skill()
    skill.resources["references/guide.md"] = "guide"
    agent = ClaudeAgent({})

    with agent.inject_skills([skill]) as result:
        skill_dir = workspace / ".claude" / "skills" / "demo"
        assert result.mode is SkillInjectionMode.FILESYSTEM
        assert result.workspace == str(workspace)
        assert (skill_dir / "SKILL.md").read_text() == "instructions\n"
        assert (skill_dir / "script.py").read_text() == "first"
        assert (skill_dir / "references" / "guide.md").read_text() == "guide"

    assert not workspace.exists()


@pytest.mark.asyncio
async def test_cli_agent_execution_uses_construction_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"type":"result","session_id":"session-1","num_turns":2}\n', b""

    async def fake_create_subprocess_exec(*command: str, **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        return FakeProcess()

    monkeypatch.setattr("mindmemos_skill.agents.claude.cli.shutil.which", lambda _: "/resolved/claude")
    monkeypatch.setattr(
        "mindmemos_skill.agents.claude.cli.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    config = ClaudeAgentConfig(
        model="claude-sonnet",
        max_turns=6,
        timeout_seconds=10,
        dangerously_skip_permissions=False,
    )
    agent = get_agent(agent_type=AgentType.CLAUDE, config=config)

    result = await agent.execute(
        make_request(
            environment=Environment(running_dir="/workspace"),
            options={"max_turns": 3},
        )
    )

    assert captured["command"] == (
        "/resolved/claude",
        "-p",
        "solve it",
        "--model",
        "claude-sonnet",
        "--max-turns",
        "3",
        "--output-format",
        "stream-json",
        "--verbose",
    )
    assert captured["cwd"] == "/workspace"
    assert result.trajectory_id == "trajectory-1"
    assert result.rollout.rollout_id == "rollout-1"
    assert result.agent.model == "claude-sonnet"
    assert result.agent.max_turns == 3
    assert result.agent.skill_injection_mode is SkillInjectionMode.FILESYSTEM
    assert result.agent.timeout_seconds == 10
    assert result.agent.config == {"dangerously_skip_permissions": False}
    assert result.to_record().trajectory_id == "trajectory-1"


@pytest.mark.asyncio
async def test_sdk_agent_returns_failed_result_when_optional_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def reject_claude_sdk_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "claude_agent_sdk" or name.startswith("claude_agent_sdk."):
            exc = ModuleNotFoundError("No module named 'claude_agent_sdk'")
            exc.name = "claude_agent_sdk"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_claude_sdk_import)
    agent = get_agent(agent_type=AgentType.CLAUDE_SDK, config={})

    result = await agent.execute(make_request())

    assert result.execution.status is TrajectoryStatus.FAILED
    assert result.execution.n_turn == 0
    assert result.execution.error_info is not None
    assert "SkillCapabilityUnavailableError" in result.execution.error_info
    assert "mindmemos-skill[claude-sdk]" in result.execution.error_info


def test_listing_agents_does_not_import_claude_agent_sdk() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys


        class RejectClaudeSDKImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "claude_agent_sdk" or fullname.startswith("claude_agent_sdk."):
                    raise AssertionError(f"unexpected eager import: {fullname}")
                return None


        sys.meta_path.insert(0, RejectClaudeSDKImport())

        from mindmemos_skill.agents import get_agent, list_agents
        from mindmemos_skill.typing import AgentType

        assert AgentType.CLAUDE_SDK.value in list_agents()
        get_agent(agent_type=AgentType.CLAUDE_SDK, config={})
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)
