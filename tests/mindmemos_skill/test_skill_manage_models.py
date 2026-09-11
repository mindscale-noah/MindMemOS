"""Domain model round-trips for the P4 operation/generation envelope."""

from __future__ import annotations

from datetime import UTC, datetime

from mindmemos_skill.application.skill_manage.models import (
    SkillGraphGeneration,
    SkillManageOperation,
    SkillManageOperationResult,
)
from mindmemos_skill.application.skill_manage.state_machine import (
    GenerationStatus,
    OperationStatus,
    OperationType,
)


def _now() -> datetime:
    return datetime(2026, 9, 3, tzinfo=UTC)


def test_operation_model_round_trips_full_envelope() -> None:
    operation = SkillManageOperation(
        operation_id="op-1",
        idempotency_key="key-1",
        request_hash="hash-1",
        operation_type=OperationType.BUILD,
        status=OperationStatus.RUNNING,
        attempt_no=2,
        current_generation_id="gen-2",
        created_at=_now(),
        updated_at=_now(),
        first_started_at=_now(),
        attempt_started_at=_now(),
        repair_count=1,
    )

    assert operation.operation_id == "op-1"
    assert operation.attempt_no == 2
    assert operation.repair_count == 1
    assert operation.status is OperationStatus.RUNNING


def test_generation_model_round_trips() -> None:
    generation = SkillGraphGeneration(
        generation_id="gen-1",
        operation_id="op-1",
        attempt_no=1,
        status=GenerationStatus.STAGING,
        node_count=3,
        edge_count=2,
        created_at=_now(),
    )

    assert generation.status is GenerationStatus.STAGING
    assert generation.node_count == 3


def test_operation_result_is_secret_free_shape() -> None:
    result = SkillManageOperationResult(
        operation_id="op-1",
        status=OperationStatus.COMMITTED,
        base_revision=0,
        result_revision=1,
        generation_id="gen-1",
        node_count=6,
        edge_count=3,
    )

    payload = result.model_dump(mode="json")
    assert payload["operation_id"] == "op-1"
    assert payload["status"] == "committed"
    assert payload["result_revision"] == 1
