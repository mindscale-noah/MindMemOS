"""Typed error envelope for the ``skill_manage`` algorithm layer.

The envelope mirrors the P0-frozen public error contract: a stable
``error_code``, a safe human-readable ``message``, a ``retryable`` flag, and a
bounded, JSON-serializable ``details`` mapping. It must never carry a
traceback, DSN, SQL, provider raw response, or full private Skill content.
"""

from __future__ import annotations

from typing import Any


class SkillManageError(Exception):
    """Domain error raised by the ``skill_manage`` algorithm core."""

    def __init__(
        self,
        error_code: str,
        message: str = "",
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> None:
        super().__init__(message or error_code)
        self.error_code = error_code
        self.message = message or error_code
        self.retryable = retryable
        self.details = dict(details or {})
        self.operation_id = operation_id

    def to_dict(self) -> dict[str, Any]:
        envelope: dict[str, Any] = {
            "error_code": self.error_code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }
        if self.operation_id is not None:
            envelope["operation_id"] = self.operation_id
        return envelope


def invalid_skill_document(*, version_id: str, reason: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_INVALID_SKILL_DOCUMENT",
        f"invalid Skill document {version_id!r}: {reason}",
        details={"version_id": version_id, "reason": reason},
    )


def input_too_large(*, field: str, limit: int, actual: int) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_INPUT_TOO_LARGE",
        f"{field} exceeds the configured limit of {limit} characters",
        details={"field": field, "limit": limit, "actual": actual},
    )


def unsupported_retrieval_mode(mode: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_UNSUPPORTED_RETRIEVAL_MODE",
        f"unsupported retrieval mode: {mode!r}",
        details={"mode": mode},
    )


def invalid_graph_weight(reason: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_INVALID_GRAPH_WEIGHT",
        f"invalid graph weight: {reason}",
        details={"reason": reason},
    )


def ppr_not_converged(max_iter: int) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_PPR_NOT_CONVERGED",
        f"personalized PageRank did not converge within {max_iter} iterations",
        details={"max_iter": max_iter},
    )


def configuration_error(reason: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_CONFIGURATION_ERROR",
        f"skill_manage is misconfigured: {reason}",
        details={"reason": reason},
    )


def embedding_provider_error(*, model: str, task: str, reason: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_EMBEDDING_PROVIDER_ERROR",
        f"embedding provider call failed (task={task}, model={model}): {reason}",
        retryable=True,
        details={"model": model, "task": task, "reason": reason},
    )


def embedding_profile_mismatch(*, expected: dict[str, Any], actual: dict[str, Any]) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_EMBEDDING_PROFILE_MISMATCH",
        "embedding profile does not match the profile this generation or algorithm uses",
        details={"expected": expected, "actual": actual},
    )


def embedding_dimension_mismatch(*, expected: int, actual: int, model: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_EMBEDDING_DIMENSION_MISMATCH",
        f"embedding model {model!r} returned {actual} dimensions, expected {expected}",
        details={"expected": expected, "actual": actual, "model": model},
    )


def embedding_count_mismatch(*, expected: int, actual: int, model: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_EMBEDDING_COUNT_MISMATCH",
        f"embedding model {model!r} returned {actual} vectors for {expected} inputs",
        retryable=True,
        details={"expected": expected, "actual": actual, "model": model},
    )


def embedding_non_finite(*, index: int, model: str) -> SkillManageError:
    return SkillManageError(
        "SKILL_MANAGE_EMBEDDING_NON_FINITE",
        f"embedding model {model!r} returned a non-finite value in vector {index}",
        retryable=True,
        details={"index": index, "model": model},
    )


__all__ = [
    "SkillManageError",
    "configuration_error",
    "embedding_count_mismatch",
    "embedding_dimension_mismatch",
    "embedding_non_finite",
    "embedding_profile_mismatch",
    "embedding_provider_error",
    "input_too_large",
    "invalid_graph_weight",
    "invalid_skill_document",
    "ppr_not_converged",
    "unsupported_retrieval_mode",
]
