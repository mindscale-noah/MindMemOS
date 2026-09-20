"""Frozen operation/generation state machines for ``skill_manage``.

These enums and transition tables are the P0-frozen oracle: every transition in
the failure matrix must pass through them, and any transition not listed here is
rejected. This module is pure and has no storage dependency.
"""

from __future__ import annotations

from enum import StrEnum


class OperationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMMITTED = "committed"
    FAILED = "failed"
    CONFLICT = "conflict"
    CANCELLED = "cancelled"
    PENDING_REPAIR = "pending_repair"


class OperationType(StrEnum):
    BUILD = "build"
    UPSERT = "upsert"
    REMOVE = "remove"


class PendingRepairReason(StrEnum):
    RETRY_ATTEMPT_RESERVATION = "retry_attempt_reservation"
    PUBLISH_OUTCOME_UNKNOWN = "publish_outcome_unknown"
    FINALIZE_UNAVAILABLE = "finalize_unavailable"
    CRASH_RECOVERY = "crash_recovery"


class GenerationStatus(StrEnum):
    STAGING = "staging"
    READY = "ready"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"
    DELETED = "deleted"


# Frozen allowed transitions. retryable-failed -> pending_repair is the only
# way out of failed; cancelled and committed/conflict are terminal.
OPERATION_TRANSITIONS: dict[OperationStatus, frozenset[OperationStatus]] = {
    OperationStatus.PENDING: frozenset({OperationStatus.RUNNING, OperationStatus.FAILED, OperationStatus.CANCELLED}),
    OperationStatus.RUNNING: frozenset(
        {
            OperationStatus.VALIDATING,
            OperationStatus.FAILED,
            OperationStatus.CONFLICT,
            OperationStatus.CANCELLED,
            OperationStatus.PENDING_REPAIR,
        }
    ),
    OperationStatus.VALIDATING: frozenset(
        {
            OperationStatus.COMMITTED,
            OperationStatus.FAILED,
            OperationStatus.CONFLICT,
            OperationStatus.CANCELLED,
            OperationStatus.PENDING_REPAIR,
        }
    ),
    OperationStatus.FAILED: frozenset({OperationStatus.PENDING_REPAIR}),
    OperationStatus.PENDING_REPAIR: frozenset(
        {
            OperationStatus.RUNNING,
            OperationStatus.COMMITTED,
            OperationStatus.FAILED,
            OperationStatus.CONFLICT,
        }
    ),
    OperationStatus.COMMITTED: frozenset(),
    OperationStatus.CONFLICT: frozenset(),
    OperationStatus.CANCELLED: frozenset(),
}

TERMINAL_OPERATION_STATUSES = frozenset(
    {
        OperationStatus.COMMITTED,
        OperationStatus.CONFLICT,
        OperationStatus.CANCELLED,
    }
)

GENERATION_TRANSITIONS: dict[GenerationStatus, frozenset[GenerationStatus]] = {
    GenerationStatus.STAGING: frozenset({GenerationStatus.READY, GenerationStatus.FAILED}),
    GenerationStatus.READY: frozenset({GenerationStatus.ACTIVE, GenerationStatus.FAILED}),
    GenerationStatus.ACTIVE: frozenset({GenerationStatus.RETIRED}),
    GenerationStatus.RETIRED: frozenset({GenerationStatus.ACTIVE, GenerationStatus.DELETED}),
    GenerationStatus.FAILED: frozenset({GenerationStatus.DELETED}),
    GenerationStatus.DELETED: frozenset(),
}


def can_transition(current: OperationStatus, target: OperationStatus) -> bool:
    return target in OPERATION_TRANSITIONS[current]


def can_transition_generation(current: GenerationStatus, target: GenerationStatus) -> bool:
    return target in GENERATION_TRANSITIONS[current]


def require_transition(current: OperationStatus, target: OperationStatus) -> None:
    if not can_transition(current, target):
        raise ValueError(f"invalid operation transition: {current.value} -> {target.value}")


__all__ = [
    "GENERATION_TRANSITIONS",
    "OPERATION_TRANSITIONS",
    "TERMINAL_OPERATION_STATUSES",
    "GenerationStatus",
    "OperationStatus",
    "OperationType",
    "PendingRepairReason",
    "can_transition",
    "can_transition_generation",
    "require_transition",
]
