"""Public error contracts for the standalone architecture demonstration."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ErrorCode(StrEnum):
    """Stable failure identities understood by every demonstration adapter."""

    INPUT_TOO_LARGE = "SKILL_MANAGE_INPUT_TOO_LARGE"
    """The source document exceeds the accepted character limit."""

    MODEL_TIMEOUT = "dependency.model_timeout"
    """The model provider did not respond before the deadline."""

    INVALID_PATCH = "algorithm.invalid_patch"
    """The model response does not satisfy the patch protocol."""

    INTERNAL = "internal_error"
    """An unexpected implementation failure reached the application boundary."""


class ErrorCategory(StrEnum):
    """Broad failure groups used for reporting, rather than HTTP status selection."""

    VALIDATION = "validation"
    """The caller must correct the input."""

    DEPENDENCY = "dependency"
    """An external service failed to complete its operation."""

    ALGORITHM = "algorithm"
    """An algorithm result violates its execution protocol."""

    INTERNAL = "internal"
    """The application encountered an unclassified implementation failure."""


class ContentLimitDetails(BaseModel):
    """Public input size information without document contents."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject arbitrary diagnostic fields and prevent field reassignment."""

    field: str = Field(default="content", pattern="^content$")
    """The document field whose character count was checked."""

    limit: int = Field(gt=0)
    """Maximum accepted number of characters."""

    actual: int = Field(ge=0)
    """Observed document character count."""

    @model_validator(mode="after")
    def _check_exceeded_limit(self) -> "ContentLimitDetails":
        """Require an actual limit violation before reporting this failure."""
        if self.actual <= self.limit:
            raise ValueError("actual must exceed limit")
        return self


class StageDetails(BaseModel):
    """Public algorithm stage identity without model responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Reject arbitrary diagnostic fields and prevent field reassignment."""

    stage: str = Field(pattern="^(scene_extraction|patch_generation)$")
    """One of the two supported demonstration stages."""


class ErrorInfo(BaseModel):
    """Transport-independent public failure data."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    """Keep the public contract explicit and prevent field reassignment."""

    code: ErrorCode
    """Stable identifier used by callers to handle this failure."""

    category: ErrorCategory
    """Broad group used to aggregate failures."""

    message: str
    """Controlled public explanation, independent of the original exception text."""

    details: ContentLimitDetails | StageDetails | None = None
    """Validated public context; absent when no context is appropriate."""

    retryable: bool = False
    """Whether this failure is temporary; does not authorize replaying an operation."""


class ErrorEnvelope(BaseModel):
    """HTTP failure body with request correlation outside the error itself."""

    code: str = "error"
    """Outer response outcome, matching the existing Skill HTTP envelope."""

    request_id: str
    """Identifier supplied by the request boundary."""

    error: ErrorInfo
    """The same public failure used by other adapters."""


class HttpErrorResponse(BaseModel):
    """A simulated HTTP response; no server is required."""

    status_code: int
    """Status chosen exclusively by the HTTP adapter."""

    body: ErrorEnvelope
    """Structured response suitable for JSON serialization."""


class FailedTask(BaseModel):
    """A simulated terminal task record."""

    task_id: str
    """Identifier supplied by the task executor."""

    status: str = "failed"
    """Terminal execution outcome, independent of task-query HTTP success."""

    error: ErrorInfo
    """Persistable public failure instead of only an exception class name."""
