"""Controlled errors with original causes kept outside public payloads."""

from .typing import ContentLimitDetails, ErrorCategory, ErrorCode, ErrorInfo, StageDetails


class MindMemosSkillError(Exception):
    """Base exception recognized by all demonstration application boundaries."""

    def __init__(self, info: ErrorInfo):
        """Store public failure data while preserving normal exception semantics.

        Args:
            info: Validated public failure created by an owning error definition.
        """
        self.info = info
        super().__init__(info.message)


class InputTooLargeError(MindMemosSkillError):
    """The input exceeds its character budget."""

    def __init__(self, *, limit: int, actual: int):
        """Describe a size violation without retaining source content.

        Args:
            limit: Maximum accepted number of characters.
            actual: Observed number of characters; must exceed the limit.
        """
        super().__init__(
            ErrorInfo(
                code=ErrorCode.INPUT_TOO_LARGE,
                category=ErrorCategory.VALIDATION,
                message="Skill content exceeds the configured character limit.",
                details=ContentLimitDetails(limit=limit, actual=actual),
            )
        )


class ModelTimeoutError(MindMemosSkillError):
    """A model request timed out without exposing provider diagnostics."""

    def __init__(self, *, stage: str):
        """Identify the affected stage and the temporary failure.

        Args:
            stage: Controlled algorithm stage name.
        """
        super().__init__(
            ErrorInfo(
                code=ErrorCode.MODEL_TIMEOUT,
                category=ErrorCategory.DEPENDENCY,
                message="The model request timed out.",
                details=StageDetails(stage=stage),
                retryable=True,
            )
        )


class InvalidPatchError(MindMemosSkillError):
    """A generated patch violates the algorithm protocol."""

    def __init__(self):
        """Report invalid generated output without retaining the response."""
        super().__init__(
            ErrorInfo(
                code=ErrorCode.INVALID_PATCH,
                category=ErrorCategory.ALGORITHM,
                message="The model response does not satisfy the patch protocol.",
                details=StageDetails(stage="patch_generation"),
            )
        )


def public_error(exc: Exception) -> ErrorInfo:
    """Normalize a failure only at an application boundary.

    Args:
        exc: A caught exception; cancellation and process exits must propagate.

    Returns:
        Existing controlled data or a generic internal failure without raw text.
    """
    if isinstance(exc, MindMemosSkillError):
        return exc.info
    return ErrorInfo(
        code=ErrorCode.INTERNAL,
        category=ErrorCategory.INTERNAL,
        message="An internal error occurred. Contact support with the request or task ID.",
    )
