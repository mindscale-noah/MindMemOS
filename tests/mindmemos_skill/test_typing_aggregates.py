from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest
from mindmemos_skill.contracts import SkillBundle
from mindmemos_skill.persistence import TrajectoryRecord
from mindmemos_skill.typing import (
    AgentProfile,
    AgentType,
    AlgorithmIdentity,
    AlgorithmLog,
    AlgorithmStep,
    Environment,
    ExecutionInfo,
    Reward,
    Rollout,
    RolloutType,
    Skill,
    SkillBinding,
    SkillUsageType,
    SkillVersionOrigin,
    SkillVersionStatus,
    Task,
    Trajectory,
    TrajectoryStatus,
)
from pydantic import ValidationError


def test_skill_aggregates_version_identity_content_and_lineage() -> None:
    created_at = datetime(2026, 8, 4, tzinfo=UTC)
    content_hash = SkillBundle.from_files({"SKILL.md": "# Spreadsheet Skill"}).content_hash
    skill = Skill(
        skill_id="skill-1",
        version_id="version-2",
        parent_version_ids=["version-1"],
        version_label="1.1.0",
        content_hash=content_hash,
        status=SkillVersionStatus.PUBLISHED,
        origin=SkillVersionOrigin.EVOLUTION,
        name="spreadsheet",
        description="Edit and validate workbooks",
        alias="sheet",
        blob={"SKILL.md": "# Spreadsheet Skill"},
        resources={
            "references/format.md": "format guide",
            "scripts/validate.py": "print('ok')",
        },
        created_at=created_at,
    )

    assert skill.blob == {"SKILL.md": "# Spreadsheet Skill\n"}
    assert skill.resources["scripts/validate.py"] == "print('ok')"
    assert skill.content == "# Spreadsheet Skill\n"
    assert skill.parent_version_ids == ["version-1"]
    assert Skill.from_record(skill.to_record()) == skill

    with pytest.raises(ValidationError, match="cannot be its own parent"):
        Skill.model_validate(
            {
                **skill.model_dump(),
                "parent_version_ids": ["version-2"],
            }
        )


def test_trajectory_groups_flat_record_fields_by_business_concept() -> None:
    started_at = datetime(2026, 8, 4, tzinfo=UTC)
    skill = Skill(
        skill_id="skill-1",
        version_id="version-2",
        version_label="1.1.0",
        content_hash="sha256:abc",
        name="spreadsheet",
        blob={"SKILL.md": "# Spreadsheet Skill"},
        created_at=started_at,
    )
    binding = SkillBinding(
        name="spreadsheet",
        content_hash="sha256:abc",
        skill_id="skill-1",
        version_id="version-2",
        version_label="1.1.0",
        usage=SkillUsageType.INJECTED,
    )
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        task=Task(
            task_id="task-1",
            instruction="Update the workbook",
            tags=["spreadsheet"],
            metadata={"dataset": "SpreadsheetBench"},
        ),
        rollout=Rollout(rollout_id="rollout-1", attempt_no=1, rollout_type=RolloutType.EVALUATE),
        environment=Environment(running_dir="/tmp/workbook", metadata={"sandbox": "local"}),
        agent=AgentProfile(
            agent_type=AgentType.CODEX,
            provider="openai",
            model="gpt-5",
            base_url="https://api.openai.com/v1",
            temperature=0.2,
            max_retries=2,
            timeout_seconds=30,
            config={"sandbox": "workspace-write"},
        ),
        injected_skills=[skill],
        events=[{"sequence_no": 0, "event_type": "user_input"}],
        skill_bindings=[binding],
        reward=Reward(score=1.0, detail="passed", metadata={"metric": "accuracy"}),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=2),
            n_turn=3,
        ),
    )

    assert trajectory.task.task_id == "task-1"
    assert trajectory.rollout.attempt_no == 1
    assert trajectory.agent.model == "gpt-5"
    assert trajectory.agent.config == {"sandbox": "workspace-write"}
    assert trajectory.reward is not None and trajectory.reward.score == 1.0
    assert trajectory.execution.duration_s == 2.0
    assert trajectory.skill_bindings[0].version_id == "version-2"

    restored = Trajectory.model_validate_json(trajectory.model_dump_json())
    assert restored == trajectory
    assert restored.execution.duration_s == 2.0
    assert Trajectory.from_record(trajectory.to_record()) == trajectory


def test_agent_profile_promotes_common_config_and_hashes_api_key_for_json() -> None:
    profile = AgentProfile.from_config(
        agent_type=AgentType.CODEX,
        config={
            "provider": "openai",
            "model": "gpt-5",
            "api_base": "https://gateway.example.com/v1",
            "api_key": "sk-private-value",
            "temperature": 0.2,
            "max_tokens": 4096,
            "max_completion_tokens": 2048,
            "num_retries": 3,
            "timeout": 45,
            "sandbox": "workspace-write",
        },
    )

    assert profile.base_url == "https://gateway.example.com/v1"
    assert profile.max_tokens == 4096
    assert profile.max_completion_tokens == 2048
    assert profile.max_retries == 3
    assert profile.timeout_seconds == 45
    assert profile.config == {"sandbox": "workspace-write"}
    assert "sk-private-value" not in repr(profile)

    dumped = profile.model_dump()
    assert dumped["api_key"] == f"sha256:{hashlib.sha256(b'sk-private-value').hexdigest()}"
    assert "sk-private-value" not in dumped["api_key"]

    restored = AgentProfile.model_validate(dumped)
    assert restored.api_key is not None
    assert restored.api_key.get_secret_value() == dumped["api_key"]
    assert restored.model_dump(mode="json")["api_key"] == dumped["api_key"]


def test_agent_profile_reads_legacy_flat_config_snapshot() -> None:
    profile = AgentProfile.from_serialized(
        {
            "model": "claude-sonnet",
            "max_turns": 4,
            "timeout_seconds": 30,
            "dangerously_skip_permissions": True,
        },
        agent_type=AgentType.CLAUDE,
    )

    assert profile.agent_type == AgentType.CLAUDE
    assert profile.model == "claude-sonnet"
    assert profile.max_turns == 4
    assert profile.timeout_seconds == 30
    assert profile.config == {"dangerously_skip_permissions": True}


def test_trajectory_record_stores_only_agent_api_key_digest() -> None:
    started_at = datetime(2026, 8, 4, tzinfo=UTC)
    trajectory = Trajectory(
        trajectory_id="trajectory-secret",
        task=Task(task_id="task-secret", instruction="Run securely"),
        rollout=Rollout(rollout_id="rollout-secret"),
        agent=AgentProfile(
            agent_type=AgentType.CODEX,
            provider="openai",
            model="gpt-5",
            api_key="sk-private-value",
        ),
        execution=ExecutionInfo(started_at=started_at),
    )

    record = trajectory.to_record()
    assert record.agent_profile["api_key"].startswith("sha256:")
    assert "sk-private-value" not in str(record.agent_profile)
    restored = Trajectory.from_record(record)
    assert restored.agent.api_key is not None
    assert restored.agent.api_key.get_secret_value() == record.agent_profile["api_key"]


def test_persistence_trajectory_keeps_algorithm_aggregates_as_json_columns() -> None:
    record = TrajectoryRecord(
        trajectory_id="trajectory-1",
        trajectory_hash="0" * 64,
        task_id="task-1",
        task_instruction="Update the workbook",
        rollout_id="rollout-1",
        injected_skills=[],
        trajectory=[{"sequence_no": 0, "event_type": "user_input"}],
        skill_bindings=[],
    )

    dumped = record.model_dump(mode="json")
    assert dumped["task_id"] == "task-1"
    assert dumped["trajectory"][0]["event_type"] == "user_input"


def test_algorithm_log_groups_identity_and_step_without_inventing_run_state() -> None:
    created_at = datetime(2026, 8, 4, tzinfo=UTC)
    log = AlgorithmLog(
        log_id="log-1",
        algorithm=AlgorithmIdentity(name="trace-summary", version="v1"),
        step=AlgorithmStep(
            component_name="candidate-generator",
            name="propose_patch",
            status="succeeded",
            payload={
                "input_trajectory_ids": ["trajectory-1", "trajectory-2"],
                "candidate_version_id": "version-4",
            },
            created_at=created_at,
        ),
    )

    assert log.algorithm.name == "trace-summary"
    assert log.step.name == "propose_patch"
    assert log.step.payload["candidate_version_id"] == "version-4"
    assert AlgorithmLog.model_validate_json(log.model_dump_json()) == log
    assert AlgorithmLog.from_record(log.to_record()) == log
