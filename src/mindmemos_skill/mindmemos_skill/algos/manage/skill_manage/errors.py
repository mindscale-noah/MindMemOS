"""Stable parsing errors preserved from the source Skill management algorithm."""


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
