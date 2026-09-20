"""AC-P3-005 (state machine portion): frozen operation/generation transitions."""

from __future__ import annotations

import pytest
from mindmemos_skill.application.skill_manage.state_machine import (
    GENERATION_TRANSITIONS,
    OPERATION_TRANSITIONS,
    TERMINAL_OPERATION_STATUSES,
    GenerationStatus,
    OperationStatus,
    can_transition,
    can_transition_generation,
    require_transition,
)


def test_operation_states_are_frozen_and_exhaustive() -> None:
    assert {status.value for status in OperationStatus} == {
        "pending",
        "running",
        "validating",
        "committed",
        "failed",
        "conflict",
        "cancelled",
        "pending_repair",
    }
    assert TERMINAL_OPERATION_STATUSES == {
        OperationStatus.COMMITTED,
        OperationStatus.CONFLICT,
        OperationStatus.CANCELLED,
    }


def test_allowed_operation_transitions() -> None:
    assert can_transition(OperationStatus.PENDING, OperationStatus.RUNNING)
    assert can_transition(OperationStatus.RUNNING, OperationStatus.VALIDATING)
    assert can_transition(OperationStatus.VALIDATING, OperationStatus.COMMITTED)
    assert can_transition(OperationStatus.FAILED, OperationStatus.PENDING_REPAIR)
    assert can_transition(OperationStatus.PENDING_REPAIR, OperationStatus.RUNNING)
    assert can_transition(OperationStatus.PENDING_REPAIR, OperationStatus.COMMITTED)


def test_forbidden_operation_transitions() -> None:
    assert not can_transition(OperationStatus.FAILED, OperationStatus.RUNNING)
    assert not can_transition(OperationStatus.COMMITTED, OperationStatus.RUNNING)
    assert not can_transition(OperationStatus.COMMITTED, OperationStatus.CANCELLED)
    assert not can_transition(OperationStatus.CONFLICT, OperationStatus.RUNNING)
    assert not can_transition(OperationStatus.CANCELLED, OperationStatus.RUNNING)


def test_require_transition_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="invalid operation transition"):
        require_transition(OperationStatus.COMMITTED, OperationStatus.RUNNING)


def test_generation_transitions() -> None:
    assert can_transition_generation(GenerationStatus.STAGING, GenerationStatus.READY)
    assert can_transition_generation(GenerationStatus.READY, GenerationStatus.ACTIVE)
    assert can_transition_generation(GenerationStatus.ACTIVE, GenerationStatus.RETIRED)
    assert can_transition_generation(GenerationStatus.RETIRED, GenerationStatus.ACTIVE)  # privileged rollback
    assert can_transition_generation(GenerationStatus.FAILED, GenerationStatus.DELETED)
    assert not can_transition_generation(GenerationStatus.ACTIVE, GenerationStatus.DELETED)
    assert not can_transition_generation(GenerationStatus.DELETED, GenerationStatus.ACTIVE)


def test_terminal_statuses_have_no_outgoing_edges() -> None:
    for status in TERMINAL_OPERATION_STATUSES:
        assert OPERATION_TRANSITIONS[status] == frozenset()


def test_generation_deleted_is_terminal() -> None:
    assert GENERATION_TRANSITIONS[GenerationStatus.DELETED] == frozenset()
