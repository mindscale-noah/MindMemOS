from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from mindmemos_skill.contracts import SkillBundle
from mindmemos_skill.persistence import (
    AlgorithmLogRecord,
    LLMCallRecord,
    RolloutType,
    SkillRecord,
    SkillRemoteOperationRecord,
    SkillSyncStateRecord,
    TrajectoryRecord,
)
from pydantic import ValidationError


def test_one_skill_record_is_one_version_with_inline_content() -> None:
    bundle = SkillBundle.from_files({"SKILL.md": "# Spreadsheet Skill"})
    version = SkillRecord(
        version_id="version-3",
        skill_id="skill-1",
        name="spreadsheet",
        parent_version_ids=["version-1", "version-2"],
        content_hash=bundle.content_hash,
        bundle=bundle.canonical_json(),
        resources='{"references/guide.md":"guide"}',
        local_snapshot_hash=bundle.content_hash,
        version_label="1.2.0",
    )

    assert version.parent_version_ids == ["version-1", "version-2"]
    assert '"SKILL.md":"# Spreadsheet Skill\\n"' in version.blob

    with pytest.raises(ValidationError, match="must be unique and ordered"):
        duplicate_bundle = SkillBundle.from_files({"SKILL.md": "# Spreadsheet Skill"})
        SkillRecord(
            version_id="version-3",
            skill_id="skill-1",
            name="spreadsheet",
            parent_version_ids=["version-1", "version-1"],
            content_hash=duplicate_bundle.content_hash,
            bundle=duplicate_bundle.canonical_json(),
            local_snapshot_hash=duplicate_bundle.content_hash,
            version_label="1.2.0",
        )

    with pytest.raises(ValidationError):
        SkillRecord(
            version_id="version-4",
            skill_id="skill-1",
            name="spreadsheet",
            content_hash=bundle.content_hash,
            bundle="# Plain Markdown is not the persistence format",
            local_snapshot_hash=bundle.content_hash,
            version_label="1.3.0",
        )


def test_trajectory_is_one_flat_row_with_json_columns() -> None:
    started = datetime(2026, 8, 3, tzinfo=UTC)
    trajectory = TrajectoryRecord(
        trajectory_id="trajectory-1",
        trajectory_hash="sha256:trajectory-1",
        task_id="task-1",
        task_instruction="Update the workbook",
        rollout_id="rollout-1",
        rollout_type=RolloutType.EVALUATE,
        agent_profile={"provider": "openai", "model": "gpt-5", "temperature": 0.2},
        started_at=started,
        finished_at=started + timedelta(seconds=2),
        trajectory=[
            {"sequence_no": 0, "event_type": "user_input"},
            {"sequence_no": 1, "event_type": "tool_call", "tool_name": "spreadsheet"},
        ],
        skill_bindings=[
            {
                "name": "spreadsheet",
                "content_hash": "hash",
                "skill_id": "skill-1",
                "version_id": "version-3",
            },
        ],
        reward_score=1.0,
        reward_detail="passed",
        reward_metadata={"metric": "accuracy"},
    )

    assert trajectory.task_id == "task-1"
    assert trajectory.rollout_type == RolloutType.EVALUATE
    assert trajectory.skill_bindings[0]["version_id"] == "version-3"
    assert trajectory.reward_score == 1.0

    restored = TrajectoryRecord.model_validate_json(trajectory.model_dump_json())
    assert restored == trajectory


def test_trajectory_retry_keeps_rollout_id_and_appends_attempt() -> None:
    failed = TrajectoryRecord(
        trajectory_id="trajectory-1",
        trajectory_hash="sha256:trajectory-1",
        task_id="task-1",
        task_instruction="Update the workbook",
        rollout_id="rollout-1",
        attempt_no=0,
        rollout_type=RolloutType.TRAIN,
    )
    retried = TrajectoryRecord(
        trajectory_id="trajectory-2",
        trajectory_hash="sha256:trajectory-2",
        task_id="task-1",
        task_instruction="Update the workbook",
        rollout_id="rollout-1",
        attempt_no=1,
        rollout_type=RolloutType.TRAIN,
    )

    assert retried.rollout_id == failed.rollout_id
    assert retried.attempt_no > failed.attempt_no
    assert retried.trajectory_id != failed.trajectory_id


def test_algorithm_log_accepts_component_specific_json_payload() -> None:
    log = AlgorithmLogRecord(
        log_id="log-1",
        algorithm_name="trace-summary",
        algorithm_version="v1",
        component_name="candidate-generator",
        step_name="propose_patch",
        status="succeeded",
        payload={
            "input_trajectory_ids": ["trajectory-1", "trajectory-2"],
            "candidate_version_id": "version-4",
            "metrics": {"accepted": True},
        },
    )

    restored = AlgorithmLogRecord.model_validate_json(log.model_dump_json())
    assert restored == log
    assert restored.payload["candidate_version_id"] == "version-4"

    with pytest.raises(ValidationError):
        AlgorithmLogRecord(
            log_id="log-2",
            algorithm_name="trace-summary",
            component_name="candidate-generator",
            step_name="invalid_payload",
            payload={"not_json": object()},
        )


def test_llm_call_record_keeps_flat_tokens_and_full_response_usage() -> None:
    started = datetime(2026, 8, 10, tzinfo=UTC)
    call = LLMCallRecord(
        call_id="call-1",
        run_id="run-1",
        task="skill_grpo.patch",
        call_type="chat",
        request={"model": "chat", "messages": [{"role": "user", "content": "patch"}]},
        response={
            "content": "done",
            "usage": {"prompt_tokens_details": {"cached_tokens": 4}},
        },
        model="gpt-test",
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
        status="succeeded",
        started_at=started,
        finished_at=started + timedelta(milliseconds=25),
        latency_ms=25.0,
    )

    assert LLMCallRecord.model_validate_json(call.model_dump_json()) == call
    assert call.response["usage"]["prompt_tokens_details"] == {"cached_tokens": 4}


def test_persistence_exports_canonical_fact_and_control_records() -> None:
    from mindmemos_skill.persistence import models

    record_names = {
        name
        for name, value in vars(models).items()
        if isinstance(value, type) and name.endswith("Record") and value.__module__ == models.__name__
    }

    assert record_names == {
        "AlgorithmLogRecord",
        "LLMCallRecord",
        "SkillFamilyStateRecord",
        "SkillRecord",
        "SkillRemoteOperationRecord",
        "SkillSyncStateRecord",
        "TrajectoryRecord",
    }


def test_persistence_defines_six_physical_row_models() -> None:
    from mindmemos_skill.persistence import models

    row_model_names = {
        value.__name__
        for value in vars(models).values()
        if isinstance(value, type)
        and issubclass(value, models.PersistenceModel)
        and value is not models.PersistenceModel
    }

    assert row_model_names == {
        "AlgorithmLogRecord",
        "LLMCallRecord",
        "SkillRecord",
        "SkillRemoteOperationRecord",
        "SkillSyncStateRecord",
        "TrajectoryRecord",
    }


def test_sync_state_and_remote_operation_are_independent_records() -> None:
    state = SkillSyncStateRecord(
        skill_id="skill-1",
        trajectory_pull_cursor="cursor-1",
    )
    operation = SkillRemoteOperationRecord(
        operation_id="push-1",
        operation_type="push_version",
        skill_id="skill-1",
        version_id="version-1",
        request_hash="sha256:request-1",
        status="pending",
    )

    assert state.trajectory_pull_cursor == "cursor-1"
    assert operation.operation_id == "push-1"

    with pytest.raises(ValidationError):
        SkillRemoteOperationRecord(
            operation_id="",
            operation_type="push_version",
            request_hash="sha256:request-1",
            status="pending",
        )


def test_persistence_layer_does_not_import_business_typing() -> None:
    from mindmemos_skill import persistence

    persistence_dir = Path(persistence.__file__).parent
    forbidden_imports: list[str] = []
    for source_path in persistence_dir.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module and node.module.startswith("typing"):
                forbidden_imports.append(f"{source_path.name}:{node.lineno}:{node.module}")

    assert forbidden_imports == []
