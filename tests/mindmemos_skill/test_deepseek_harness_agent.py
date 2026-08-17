from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mindmemos_skill.agents import DeepSeekHarnessAgentConfig, get_agent, list_agents
from mindmemos_skill.agents.deepseek_harness import (
    DeepSeekHarnessAgent,
    DeepSeekHarnessSkillRuntime,
)
from mindmemos_skill.registry import ComponentType, get_component
from mindmemos_skill.typing import (
    AgentExecutionRequest,
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
    compute_skill_content_hash,
)
from pydantic import ValidationError


def make_skill(*, name: str = "demo", content: str = 'name: demo\nversion: "1.0.0"\n\nRoot\n') -> Skill:
    return Skill(
        skill_id=f"skill-{name}",
        version_id=f"version-{name}-1",
        version_label="1.0.0",
        content_hash=compute_skill_content_hash({"SKILL.md": content}),
        name=name,
        blob={"SKILL.md": content},
        resources={"references/guide.md": "Guide\n"},
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )


def make_request(
    *,
    skills: list[Skill] | None = None,
    running_dir: str | None = None,
    options: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> AgentExecutionRequest:
    return AgentExecutionRequest(
        trajectory_id="trajectory-dsh-1",
        task=Task(
            task_id="task-dsh-1",
            instruction="Use the demo Skill",
            system_prompt=system_prompt,
        ),
        rollout=Rollout(rollout_id="rollout-dsh-1"),
        environment=Environment(running_dir=running_dir),
        skills=skills or [],
        options=options or {},
    )


def make_trajectory(skill: Skill, events: list[dict[str, Any]]) -> Trajectory:
    now = datetime.now(UTC)
    return Trajectory(
        trajectory_id="trajectory-dsh-bind",
        task=Task(task_id="task-dsh-bind", instruction="Use the demo Skill"),
        rollout=Rollout(rollout_id="rollout-dsh-bind"),
        injected_skills=[skill],
        events=events,
        execution=ExecutionInfo(status=TrajectoryStatus.SUCCEEDED, started_at=now, finished_at=now),
    )


class FakeProcess:
    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._stdout is NotImplemented:
            raise asyncio.TimeoutError("timed out")
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        return None


def test_get_agent_builds_deepseek_harness_agent() -> None:
    agent = get_agent(
        agent_type=AgentType.DEEPSEEK_HARNESS,
        config={"model": "deepseek-v4-flash", "timeout_seconds": 45, "cli_path": "/opt/bin/dsh"},
    )

    assert isinstance(agent, DeepSeekHarnessAgent)
    assert agent.config == DeepSeekHarnessAgentConfig(
        model="deepseek-v4-flash",
        timeout_seconds=45,
        cli_path="/opt/bin/dsh",
    )
    assert agent.supported_skill_injection_modes == frozenset(
        {SkillInjectionMode.FILESYSTEM, SkillInjectionMode.SYSTEM_PROMPT}
    )


def test_list_agents_includes_deepseek_harness() -> None:
    assert AgentType.DEEPSEEK_HARNESS.value in list_agents()
    spec = get_component(type=ComponentType.AGENT, name=AgentType.DEEPSEEK_HARNESS.value)
    assert "filesystem" in spec.requirements.supported_skill_injection_modes
    assert "system_prompt" in spec.requirements.supported_skill_injection_modes


def test_config_rejects_unknown_fields_and_unsupported_modes() -> None:
    with pytest.raises(ValidationError):
        DeepSeekHarnessAgentConfig(unknown_field=True)
    with pytest.raises(ValueError, match="does not support 'tool'"):
        DeepSeekHarnessAgent({"skill_injection_mode": SkillInjectionMode.TOOL})
    assert DeepSeekHarnessAgentConfig(skill_injection_mode=SkillInjectionMode.SYSTEM_PROMPT).skill_injection_mode is (
        SkillInjectionMode.SYSTEM_PROMPT
    )


def test_runtime_materializes_workspace_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    injection_root = tmp_path / "injection"
    injection_root.mkdir()
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.skill_runtime.tempfile.mkdtemp",
        lambda **_: str(injection_root),
    )

    runtime = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM)
    with runtime.inject([make_skill()]) as injection:
        skill_root = injection_root / ".dsh" / "skills" / "demo"
        assert injection.workspace == str(injection_root)
        assert injection.skill_names == {"demo"}
        materialized = (skill_root / "SKILL.md").read_text(encoding="utf-8")
        assert materialized.endswith("Root\n")
        assert "name: demo" in materialized
        assert "description:" in materialized
        assert (skill_root / "references" / "guide.md").read_text(encoding="utf-8") == "Guide\n"

    assert not injection_root.exists()


def test_runtime_slugs_skill_names_and_adds_missing_description(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    injection_root = tmp_path / "injection"
    injection_root.mkdir()
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.skill_runtime.tempfile.mkdtemp",
        lambda **_: str(injection_root),
    )

    runtime = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM)
    skill = make_skill(name="My Skill!", content="plain body without frontmatter\n")
    with runtime.inject([skill]) as injection:
        materialized = (injection_root / ".dsh" / "skills" / "my-skill" / "SKILL.md").read_text(encoding="utf-8")
        assert materialized.startswith("---\n")
        assert "name: my-skill" in materialized
        assert 'description: "MindMemOS Skill: My Skill!"' in materialized
        assert materialized.endswith("plain body without frontmatter\n")
        assert injection.skill_names == {"My Skill!"}


def test_runtime_rejects_duplicate_directories_after_slugging() -> None:
    runtime = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM)
    first = make_skill(name="My Skill!", content="one\n")
    colliding = make_skill(name="my-skill", content="two\n")
    with pytest.raises(ValueError, match="duplicate"):
        with runtime.inject([first, colliding]):
            pass


def test_runtime_rejects_escaping_skill_paths() -> None:
    skill = make_skill()
    skill.resources = {"../escape.txt": "evil\n"}
    runtime = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM)
    with pytest.raises(ValueError, match="escapes"):
        with runtime.inject([skill]):
            pass


def test_runtime_binds_used_and_unused_skills() -> None:
    skill = make_skill()
    used = make_trajectory(
        skill,
        [
            {"role": "user", "content": "Use the demo Skill"},
            {"role": "assistant", "content": "I read `.dsh/skills/demo/SKILL.md` and applied it."},
        ],
    )
    [binding] = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM).bind(used)
    assert binding.name == "demo"
    assert binding.usage is SkillUsageType.INJECTED
    assert binding.version_id == skill.version_id

    unused = make_trajectory(
        skill,
        [
            {"role": "user", "content": "Do something"},
            {"role": "assistant", "content": "Done without any skill."},
        ],
    )
    [binding] = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM).bind(unused)
    assert binding.usage is SkillUsageType.UNUSED


def test_runtime_does_not_bind_bare_skill_names_from_task_or_answer() -> None:
    skill = make_skill()
    trajectory = make_trajectory(
        skill,
        [
            {"role": "user", "content": "Use the demo Skill"},
            {"role": "assistant", "content": "I used demo guidance."},
        ],
    )

    [binding] = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM).bind(trajectory)

    assert binding.usage is SkillUsageType.UNUSED


def test_runtime_binds_skill_content_blocks() -> None:
    skill = make_skill()
    trajectory = make_trajectory(
        skill,
        [
            {"role": "assistant", "content": "Loaded <skill_content name=\"demo\">..."},
        ],
    )
    [binding] = DeepSeekHarnessSkillRuntime(SkillInjectionMode.FILESYSTEM).bind(trajectory)
    assert binding.usage is SkillUsageType.INJECTED


@pytest.mark.asyncio
async def test_agent_executes_headless_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        return FakeProcess(returncode=0, stdout=b"The demo skill worked.\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    agent = DeepSeekHarnessAgent({"timeout_seconds": 20})
    trajectory = await agent.execute(make_request())

    assert captured["command"] == ("/resolved/dsh", "--profile", "headless", "User:\nUse the demo Skill")
    assert trajectory.execution.status is TrajectoryStatus.SUCCEEDED
    assert trajectory.execution.n_turn == 1
    assert trajectory.agent.agent_type is AgentType.DEEPSEEK_HARNESS
    assert trajectory.events[-1] == {"role": "assistant", "content": "The demo skill worked."}
    assert trajectory.metadata["profile"] == "headless"


@pytest.mark.asyncio
async def test_agent_initializes_private_profile_once(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []
    homes: list[str] = []

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        commands.append(command)
        homes.append(kwargs["env"]["DSH_HOME"])
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    agent = DeepSeekHarnessAgent({})

    await agent.execute(make_request())
    await agent.execute(make_request())

    assert sum(command[-1] == "--dump-config" for command in commands) == 1
    assert len(set(homes)) == 1


@pytest.mark.asyncio
async def test_agent_places_user_prompt_after_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    await DeepSeekHarnessAgent({}).execute(make_request(system_prompt="Follow the benchmark format."))

    assert captured["command"][-1] == (
        "System instructions:\nFollow the benchmark format.\n\nUser:\nUse the demo Skill"
    )


@pytest.mark.asyncio
async def test_agent_execute_appends_system_prompt_skill_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    skill = make_skill(content="name: demo\n\nUse the injected instructions.\n")
    trajectory = await DeepSeekHarnessAgent({}).execute(
        make_request(
            skills=[skill],
            system_prompt="Follow the benchmark format.",
            options={"skill_injection_mode": SkillInjectionMode.SYSTEM_PROMPT.value},
        )
    )

    assert captured["command"][-1] == (
        "System instructions:\n"
        "Follow the benchmark format.\n\nname: demo\n\nUse the injected instructions.\n\n"
        "User:\nUse the demo Skill"
    )
    assert trajectory.events[0]["role"] == "system"
    assert "Use the injected instructions." in trajectory.events[0]["content"]
    [binding] = trajectory.skill_bindings
    assert binding.usage is SkillUsageType.INJECTED


@pytest.mark.asyncio
async def test_agent_keeps_task_workspace_isolates_skills_and_bridges_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    task_workspace = tmp_path / "task-workspace"
    task_workspace.mkdir()

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        if command[-1] == "--dump-config":
            return FakeProcess(returncode=0, stdout=b"config\n")
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        captured["command"] = command
        patch_path = command[command.index("--patch") + 1]
        captured["patch"] = Path(patch_path).read_text(encoding="utf-8")
        skills_root = Path(captured["patch"].split("customSkillDirs:\n      - ", 1)[1].splitlines()[0])
        captured["skills_root"] = str(skills_root)
        captured["skill_md"] = (skills_root / "demo" / "SKILL.md").read_text(encoding="utf-8")
        captured["dsh_home"] = kwargs["env"]["DSH_HOME"]
        assert Path(captured["dsh_home"]).is_dir()
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("DSH_HOME", "/user-controlled/dsh-home")

    agent = DeepSeekHarnessAgent({})
    skill = make_skill()
    trajectory = await agent.execute(make_request(skills=[skill], running_dir=str(task_workspace)))

    assert captured["cwd"] == str(task_workspace)
    assert "name: demo" in captured["skill_md"]
    assert "skill-filesystem" in captured["patch"]
    assert f"customSkillDirs:\n      - {captured['skills_root']}" in captured["patch"]
    assert Path(captured["skills_root"]).is_relative_to(task_workspace)
    assert "includeDefaultRoots: false" in captured["patch"]
    assert "watch: false" in captured["patch"]
    assert captured["env"]["DEEPSEEK_API_KEY"] == "sk-openai"
    assert captured["env"]["DEEPSEEK_BASE_URL"] == "https://gateway.example/v1"
    assert captured["env"]["DSH_HOME"] != "/user-controlled/dsh-home"
    assert Path(captured["dsh_home"]).exists()
    assert not Path(captured["skills_root"]).exists()
    assert trajectory.execution.status is TrajectoryStatus.SUCCEEDED
    agent._dsh_home.cleanup()
    assert not Path(captured["dsh_home"]).exists()


@pytest.mark.asyncio
async def test_agent_uses_running_dir_without_skills(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["cwd"] = kwargs["cwd"]
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    agent = DeepSeekHarnessAgent({})
    await agent.execute(make_request(running_dir="/env/running"))
    assert captured["cwd"] == "/env/running"


@pytest.mark.asyncio
async def test_agent_forwards_model_patch_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    patch_content: dict[str, str] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        if command[-1] == "--dump-config":
            return FakeProcess(returncode=0, stdout=b"config\n")
        captured["command"] = command
        patch_path = command[command.index("--patch") + 1]
        patch_content["body"] = Path(patch_path).read_text(encoding="utf-8")
        captured["patch_path"] = patch_path
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    agent = DeepSeekHarnessAgent({"model": "deepseek-v4-flash", "provider": "deepseek-official"})
    trajectory = await agent.execute(make_request())

    assert captured["command"][0] == "/resolved/dsh"
    assert captured["command"][2] == "headless"
    assert "provider: deepseek-official" in patch_content["body"]
    assert "model: deepseek-v4-flash" in patch_content["body"]
    assert not os.path.exists(captured["patch_path"])
    assert trajectory.metadata["model"] == "deepseek-v4-flash"
    assert trajectory.metadata["provider"] == "deepseek-official"


@pytest.mark.asyncio
async def test_agent_uses_cached_npx_binary_when_dsh_is_not_on_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    cache_bin = tmp_path / "_npx" / "cache" / "node_modules" / ".bin" / "dsh"
    cache_bin.parent.mkdir(parents=True)
    cache_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    cache_bin.chmod(0o755)

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        return FakeProcess(returncode=0, stdout=b"ok\n")

    def fake_which(executable: str) -> str | None:
        return None

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", fake_which)
    monkeypatch.setenv("npm_config_cache", str(tmp_path))
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    trajectory = await DeepSeekHarnessAgent({}).execute(make_request())

    assert captured["command"][0] == str(cache_bin)
    assert trajectory.execution.status is TrajectoryStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_agent_falls_back_to_npx_launcher_when_cache_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        return FakeProcess(returncode=0, stdout=b"ok\n")

    def fake_which(executable: str) -> str | None:
        return "/usr/bin/npx" if executable == "npx" else None

    monkeypatch.setenv("npm_config_cache", str(tmp_path))
    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", fake_which)
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    trajectory = await DeepSeekHarnessAgent({}).execute(make_request())

    assert captured["command"][:2] == ("/usr/bin/npx", "@deepseek-ai/dsh")
    assert trajectory.execution.status is TrajectoryStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_agent_omits_patch_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    agent = DeepSeekHarnessAgent({})
    await agent.execute(make_request())
    assert "--patch" not in captured["command"]


@pytest.mark.asyncio
async def test_agent_rejects_empty_success_output(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[tuple[str, ...]] = []

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        commands.append(command)
        return FakeProcess(returncode=0, stdout=b"")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    trajectory = await DeepSeekHarnessAgent({}).execute(make_request())

    assert trajectory.execution.status is TrajectoryStatus.FAILED
    assert trajectory.execution.error_info == "dsh exited successfully without a final assistant message"
    assert trajectory.metadata["headless_attempts"] == 2
    assert sum(command[-1] != "--dump-config" for command in commands) == 2


@pytest.mark.asyncio
async def test_agent_records_failure_with_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        return FakeProcess(returncode=1, stderr=b"dsh: AUTH: missing credentials\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    trajectory = await DeepSeekHarnessAgent({}).execute(make_request())

    assert trajectory.execution.status is TrajectoryStatus.FAILED
    assert trajectory.execution.n_turn == 0
    assert trajectory.execution.error_info == "dsh: AUTH: missing credentials"


@pytest.mark.asyncio
async def test_agent_handles_missing_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: None)
    monkeypatch.setenv("npm_config_cache", str(tmp_path))
    trajectory = await DeepSeekHarnessAgent({}).execute(make_request())

    assert trajectory.execution.status is TrajectoryStatus.FAILED
    assert "not found" in (trajectory.execution.error_info or "")


@pytest.mark.asyncio
async def test_agent_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        return FakeProcess(stdout=NotImplemented)  # type: ignore[arg-type]

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    trajectory = await DeepSeekHarnessAgent({"timeout_seconds": 0.01}).execute(make_request())

    assert trajectory.execution.status is TrajectoryStatus.FAILED
    assert "timed out" in (trajectory.execution.error_info or "")


def test_runtime_system_prompt_mode_exposes_suffix_and_binds() -> None:
    runtime = DeepSeekHarnessSkillRuntime(SkillInjectionMode.SYSTEM_PROMPT)
    skill = make_skill(content="name: demo\n\nUse the root instructions.\n")
    with runtime.inject([skill]) as injection:
        assert injection.workspace is None
        assert injection.skill_names == {"demo"}
        assert injection.system_prompt_suffix == "name: demo\n\nUse the root instructions."
        assert (injection.system_prompt_suffix or "").endswith("Use the root instructions.")

    used = make_trajectory(
        skill,
        [
            {"role": "system", "content": f"Base instructions.\n\n{skill.content}"},
            {"role": "user", "content": "Solve it."},
            {"role": "assistant", "content": "Applied the demo skill."},
        ],
    )
    [binding] = runtime.bind(used)
    assert binding.usage is SkillUsageType.INJECTED
    assert binding.injection_mode is SkillInjectionMode.SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_agent_respond_runs_headless_and_returns_chat_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        return FakeProcess(returncode=0, stdout=b"The answer is B.\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    agent = DeepSeekHarnessAgent({"model": "deepseek-v4-flash"})
    request = make_request(running_dir="/env/running")
    response = await agent.respond(
        request,
        [
            {"role": "system", "content": "You are a math solver."},
            {"role": "user", "content": "What is 2+2? Answer A/B/C/D."},
        ],
    )

    assert response.content == "The answer is B."
    assert response.model == "deepseek-v4-flash"
    task_text = captured["command"][-1]
    assert task_text == (
        "System instructions:\nYou are a math solver.\n\n"
        "User:\nWhat is 2+2? Answer A/B/C/D."
    )
    assert "--patch" in captured["command"]


@pytest.mark.asyncio
async def test_agent_respond_accepts_system_prompt_injection_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        return FakeProcess(returncode=0, stdout=b"ok\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    # LiveMath forces skill_injection_mode=system_prompt through request options.
    request = make_request(
        running_dir="/env/running",
        options={"skill_injection_mode": SkillInjectionMode.SYSTEM_PROMPT.value},
    )
    response = await DeepSeekHarnessAgent({"model": "deepseek-v4-flash"}).respond(
        request,
        [{"role": "user", "content": "Solve it."}],
    )
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_agent_respond_rejects_environment_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")

    with pytest.raises(TypeError, match="does not support environment-provided tools"):
        await DeepSeekHarnessAgent({}).respond(
            make_request(),
            [{"role": "user", "content": "Solve it."}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )


@pytest.mark.asyncio
async def test_agent_respond_rejects_role_histories_it_cannot_preserve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")

    with pytest.raises(TypeError, match="received role 'assistant'"):
        await DeepSeekHarnessAgent({}).respond(
            make_request(),
            [
                {"role": "user", "content": "First turn."},
                {"role": "assistant", "content": "Previous answer."},
                {"role": "user", "content": "Second turn."},
            ],
        )


@pytest.mark.asyncio
async def test_agent_respond_raises_on_headless_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        return FakeProcess(returncode=1, stderr=b"dsh: AUTH: missing credentials\n")

    monkeypatch.setattr("mindmemos_skill.agents.deepseek_harness.agent.shutil.which", lambda _: "/resolved/dsh")
    monkeypatch.setattr(
        "mindmemos_skill.agents.deepseek_harness.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    with pytest.raises(RuntimeError, match="AUTH"):
        await DeepSeekHarnessAgent({}).respond(
            make_request(),
            [{"role": "user", "content": "Solve it."}],
        )
