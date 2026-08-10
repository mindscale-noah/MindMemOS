from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mindmemos_skill.agents.openclaw import OpenClawAgent, OpenClawSkillRuntime
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


def make_skill(content: str = 'name: demo\nversion: "1.0.0"\n\nRoot\n') -> Skill:
    return Skill(
        skill_id="skill-openclaw-1",
        version_id="version-openclaw-1",
        version_label="1.0.0",
        content_hash=compute_skill_content_hash({"SKILL.md": content}),
        name="demo",
        blob={"SKILL.md": content},
        resources={
            "references/guide.md": "Guide\n",
            "scripts/check.py": "print('ok')\n",
        },
        created_at=datetime(2026, 8, 5, tzinfo=UTC),
    )


def make_trajectory(skill: Skill, events: list[dict[str, Any]]) -> Trajectory:
    now = datetime.now(UTC)
    return Trajectory(
        trajectory_id="trajectory-openclaw-1",
        task=Task(task_id="task-openclaw-1", instruction="Use the demo Skill"),
        rollout=Rollout(rollout_id="rollout-openclaw-1"),
        injected_skills=[skill],
        events=events,
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
        ),
    )


def test_openclaw_runtime_materializes_workspace_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    injection_root = tmp_path / "injection"
    injection_root.mkdir()
    monkeypatch.setattr(
        "mindmemos_skill.agents.openclaw.skill_runtime.tempfile.mkdtemp",
        lambda **_: str(injection_root),
    )

    runtime = OpenClawSkillRuntime(SkillInjectionMode.FILESYSTEM)
    with runtime.inject([make_skill()]) as injection:
        skill_root = injection_root / "skills" / "demo"
        assert injection.workspace == str(injection_root)
        assert injection.skill_names == {"demo"}
        assert (skill_root / "SKILL.md").read_text(encoding="utf-8").endswith("Root\n")
        assert (skill_root / "scripts" / "check.py").read_text(encoding="utf-8") == "print('ok')\n"
        assert (skill_root / "references" / "guide.md").read_text(encoding="utf-8") == "Guide\n"

    assert not injection_root.exists()


def test_openclaw_runtime_binds_modified_content_directly() -> None:
    skill = make_skill()
    path = "/workspace/skills/demo/SKILL.md"
    trajectory = make_trajectory(
        skill,
        [
            {
                "role": "assistant",
                "content": f'[tool_call] read({{"path":"{path}"}})',
                "tool_call_id": "read-1",
            },
            {
                "role": "tool",
                "name": "read",
                "tool_call_id": "read-1",
                "content": skill.content,
                "is_error": False,
            },
            {
                "role": "assistant",
                "content": (
                    f'[tool_call] edit({{"path":"{path}","edits":[{{"oldText":"Root","newText":"Improved"}}]}})'
                ),
                "tool_call_id": "edit-1",
            },
            {
                "role": "tool",
                "name": "edit",
                "tool_call_id": "edit-1",
                "content": "Updated file.",
                "is_error": False,
            },
        ],
    )

    [binding] = OpenClawSkillRuntime(SkillInjectionMode.FILESYSTEM).bind(trajectory)

    expected = skill.content.replace("Root", "Improved")
    assert binding.name == "demo"
    assert binding.skill_id == skill.skill_id
    assert binding.base_version_id == skill.version_id
    assert binding.version_id is None
    assert binding.version_label == "1.0.0"
    assert binding.content_hash == compute_skill_content_hash({"SKILL.md": expected})
    assert binding.usage is SkillUsageType.MODIFIED
    assert binding.injection_mode is SkillInjectionMode.FILESYSTEM


def test_openclaw_runtime_returns_unresolved_binding_for_external_skill() -> None:
    content = 'name: external\nversion: "0.2.0"\n\nInstructions\n'
    trajectory = make_trajectory(
        make_skill(),
        [
            {
                "role": "assistant",
                "content": '[tool_call] read({"path":"/external/skills/external/SKILL.md"})',
            },
            {"role": "tool", "content": content},
        ],
    )

    bindings = OpenClawSkillRuntime(SkillInjectionMode.FILESYSTEM).bind(trajectory)

    external = next(binding for binding in bindings if binding.name == "external")
    unused = next(binding for binding in bindings if binding.name == "demo")
    assert external.skill_id is None
    assert external.version_id is None
    assert external.content_hash == compute_skill_content_hash({"SKILL.md": content})
    assert external.usage is SkillUsageType.INJECTED
    assert unused.usage is SkillUsageType.UNUSED


def test_openclaw_runtime_ignores_failed_edit() -> None:
    skill = make_skill()
    path = "/workspace/skills/demo/SKILL.md"
    trajectory = make_trajectory(
        skill,
        [
            {
                "role": "assistant",
                "content": f'[tool_call] edit({{"path":"{path}","content":"invalid"}})',
                "tool_call_id": "edit-failed",
            },
            {
                "role": "tool",
                "name": "edit",
                "tool_call_id": "edit-failed",
                "content": "permission denied",
                "is_error": True,
            },
        ],
    )

    [binding] = OpenClawSkillRuntime(SkillInjectionMode.FILESYSTEM).bind(trajectory)

    assert binding.version_id == skill.version_id
    assert binding.usage is SkillUsageType.UNUSED


@pytest.mark.asyncio
async def test_openclaw_agent_runs_cli_reads_native_transcript_and_binds_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_config = tmp_path / "source-openclaw.json"
    source_config.write_text(
        json.dumps({"models": {"providers": {"test": {"models": []}}}}),
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    running_dir = tmp_path / "task-workspace"
    running_dir.mkdir()
    skill = make_skill()
    session_file = tmp_path / "session.jsonl"
    session_events = [
        {"type": "session", "id": "session-native", "cwd": str(running_dir)},
        {
            "type": "message",
            "message": {"role": "user", "content": "Use the demo Skill"},
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "read-native",
                        "name": "read",
                        "arguments": {"path": str(running_dir / "skills" / "demo" / "SKILL.md")},
                    }
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "read-native",
                "isError": False,
                "content": [{"type": "text", "text": skill.content}],
            },
        },
        {
            "type": "message",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        },
    ]
    session_file.write_text(
        "".join(json.dumps(event) + "\n" for event in session_events),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}

    class FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                json.dumps(
                    {
                        "payloads": [{"text": "done"}],
                        "meta": {
                            "transport": "embedded",
                            "agentMeta": {
                                "sessionId": "session-native",
                                "sessionFile": str(session_file),
                            },
                        },
                    }
                ).encode(),
                b"",
            )

    async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> FakeProcess:
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        captured["environment"] = kwargs["env"]
        overlay_path = Path(kwargs["env"]["OPENCLAW_CONFIG_PATH"])
        captured["overlay"] = json.loads(overlay_path.read_text(encoding="utf-8"))
        return FakeProcess()

    monkeypatch.setattr("mindmemos_skill.agents.openclaw.agent.shutil.which", lambda _: "/resolved/openclaw")
    monkeypatch.setattr(
        "mindmemos_skill.agents.openclaw.agent.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    agent = OpenClawAgent(
        {
            "model": "test/model",
            "config_path": source_config,
            "state_dir": state_dir,
            "timeout_seconds": 20,
            "thinking": "medium",
        }
    )
    request = AgentExecutionRequest(
        trajectory_id="trajectory-openclaw-cli",
        task=Task(task_id="task-openclaw-cli", instruction="Use the demo Skill"),
        rollout=Rollout(rollout_id="rollout-openclaw-cli"),
        environment=Environment(running_dir=str(running_dir)),
        skills=[skill],
    )

    trajectory = await agent.execute(request)

    assert captured["cwd"] == str(running_dir)
    assert captured["environment"]["OPENCLAW_STATE_DIR"] == str(state_dir)
    assert captured["overlay"]["agents"]["defaults"]["workspace"] == str(running_dir)
    assert captured["overlay"]["agents"]["defaults"]["skills"] == ["demo"]
    assert captured["overlay"]["skills"]["allowBundled"] == []
    assert captured["command"] == (
        "/resolved/openclaw",
        "--no-color",
        "agent",
        "--local",
        "--json",
        "--agent",
        "main",
        "--session-id",
        "trajectory-openclaw-cli",
        "--timeout",
        "20",
        "--message",
        f"Working directory: {running_dir}\n\nUse the demo Skill",
        "--model",
        "test/model",
        "--thinking",
        "medium",
    )
    assert trajectory.agent.agent_type is AgentType.OPENCLAW
    assert trajectory.execution.status is TrajectoryStatus.SUCCEEDED
    assert trajectory.execution.n_turn == 2
    assert trajectory.metadata["session_id"] == "session-native"
    assert trajectory.metadata["transport"] == "embedded"
    assert trajectory.events[1]["content"].startswith("[tool_call] read(")
    assert trajectory.skill_bindings[0].version_id == skill.version_id
    assert trajectory.skill_bindings[0].usage is SkillUsageType.INJECTED
