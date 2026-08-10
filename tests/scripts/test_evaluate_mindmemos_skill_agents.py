from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path

from mindmemos_skill.typing import (
    AgentProfile,
    AgentType,
    Environment,
    ExecutionInfo,
    Rollout,
    SkillBinding,
    SkillInjectionMode,
    SkillUsageType,
    Task,
    Trajectory,
    TrajectoryStatus,
)

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_mindmemos_skill_agents.py"
SPEC = importlib.util.spec_from_file_location("evaluate_mindmemos_skill_agents", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT)

SKILL_NAME = SCRIPT.SKILL_NAME
_assess_trajectory = SCRIPT._assess_trajectory
_build_skill = SCRIPT._build_skill
_load_env_file = SCRIPT._load_env_file


def test_load_env_file_handles_comments_export_and_quotes(tmp_path) -> None:
    env_file = tmp_path / ".agent.env"
    env_file.write_text(
        "# provider\nexport OPENAI_BASE_URL='https://example.test/v1'\nOPENAI_API_KEY=secret\n",
        encoding="utf-8",
    )

    assert _load_env_file(env_file) == {
        "OPENAI_BASE_URL": "https://example.test/v1",
        "OPENAI_API_KEY": "secret",
    }


def _trajectory(*, expected_token: str, final_answer: str) -> Trajectory:
    now = datetime.now(UTC)
    skill = _build_skill(run_id="test", expected_token=expected_token)
    return Trajectory(
        trajectory_id="trajectory-test",
        task=Task(task_id="task-test", instruction="use the skill"),
        rollout=Rollout(rollout_id="rollout-test"),
        environment=Environment(running_dir="/tmp"),
        agent=AgentProfile(
            agent_type=AgentType.REACT,
            skill_injection_mode=SkillInjectionMode.TOOL,
        ),
        injected_skills=[skill],
        events=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "skill", "arguments": f'{{"name":"{SKILL_NAME}"}}'},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "skill",
                "tool_call_id": "call-1",
                "content": "Result of 'skill' delivered in the following user message.",
            },
            {
                "role": "user",
                "content": (
                    f"Loaded skill '{SKILL_NAME}'.\n"
                    "Skill directory (absolute path): /tmp/skills/mindmemos-runtime-eval\n"
                    "Reference files live under that directory; read or run them with the read/shell tools as needed.\n\n"
                    f"----- {SKILL_NAME}/SKILL.md -----\nUse the private token."
                ),
            },
            {"role": "assistant", "content": final_answer},
        ],
        skill_bindings=[
            SkillBinding(
                name=SKILL_NAME,
                content_hash=skill.content_hash,
                skill_id=skill.skill_id,
                version_id=skill.version_id,
                version_label=skill.version_label,
                usage=SkillUsageType.INJECTED,
                injection_mode=SkillInjectionMode.TOOL,
            )
        ],
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=2,
        ),
    )


def test_assessment_passes_only_with_native_discovery_adherence_and_round_trip() -> None:
    token = "MINDMEMOS_SKILL_EVAL_PASS_TEST"

    passed = _assess_trajectory(
        _trajectory(expected_token=token, final_answer=token),
        expected_token=token,
        mode=SkillInjectionMode.TOOL,
    )
    ignored = _assess_trajectory(
        _trajectory(expected_token=token, final_answer="I did not follow it"),
        expected_token=token,
        mode=SkillInjectionMode.TOOL,
    )

    assert passed["passed"] is True
    assert passed["checks"] == {
        "agent_ran": True,
        "skill_injected": True,
        "native_skill_discovery": True,
        "skill_applied_to_task": True,
        "trajectory_produced_and_round_trips": True,
    }
    assert ignored["passed"] is False
    assert ignored["checks"]["skill_applied_to_task"] is False
