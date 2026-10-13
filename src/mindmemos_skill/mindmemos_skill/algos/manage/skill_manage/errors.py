"""Stable parsing errors preserved from the source Skill management algorithm."""

from enum import StrEnum


class SceneGraphFailure(StrEnum):
    """Stable failure categories for the Scene Graph contract."""

    INVALID_INPUT = "SCENE_GRAPH_INVALID_INPUT"
    """Invalid source, bindings, typed records or configuration."""

    INSUFFICIENT_SOURCE = "SCENE_GRAPH_INSUFFICIENT_SOURCE"
    """The document cannot justify a Scene; do not invent one or silently disable it."""

    MODEL_FAILED = "SCENE_GRAPH_MODEL_FAILED"
    """Extraction, linking, rewriting or embedding failed; no implicit fallback."""

    PROFILE_MISMATCH = "SCENE_GRAPH_PROFILE_MISMATCH"
    """Stored schema, template or embedding profile cannot satisfy the request."""


class SceneGraphError(ValueError):
    """Expose only a stable category, never model output or exception context."""

    def __init__(self, code: SceneGraphFailure):
        """Create a safe public failure.

        Args:
            code: Validated failure category; raw exception text is not accepted.
        """
        self.code = SceneGraphFailure(code)
        super().__init__(self.code.value)

    def to_dict(self) -> dict[str, str]:
        """Return a payload-free public failure envelope."""
        return {"code": self.code.value}


class SkillManageError(ValueError):
    """Carry a stable error code and bounded parsing context."""

    def __init__(self, error_code: str, message: str, *, details: dict):
        """Record the public algorithm failure.

        Args:
            error_code: Machine-readable source-compatible error category.
            message: Explanation suitable for the caller.
            details: Structured parsing context without source document contents.
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = dict(details)
        self.retryable = False

    def to_dict(self) -> dict:
        """Return the source-compatible serializable error envelope."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }


def invalid_skill_document(*, version_id: str, reason: str) -> SkillManageError:
    """Describe an invalid source document without inventing its identity.

    Args:
        version_id: Optional caller-provided source identity used by the parser.
        reason: Explanation of the violated document requirement.
    """
    return SkillManageError(
        "SKILL_MANAGE_INVALID_SKILL_DOCUMENT",
        f"Invalid Skill document {version_id!r}: {reason}",
        details={"version_id": version_id, "reason": reason},
    )


def input_too_large(*, field: str, limit: int, actual: int) -> SkillManageError:
    """Reject oversize canonical content rather than silently truncating it.

    Args:
        field: Input field being checked.
        limit: Maximum supported character count.
        actual: Observed character count.
    """
    return SkillManageError(
        "SKILL_MANAGE_INPUT_TOO_LARGE",
        f"{field} exceeds the configured limit of {limit} characters",
        details={"field": field, "limit": limit, "actual": actual},
    )
