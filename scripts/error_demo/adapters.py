"""HTTP, CLI, task and retry adapters sharing one public error contract."""

from collections.abc import Awaitable, Callable

from .errors import public_error
from .typing import ErrorCode, ErrorEnvelope, ErrorInfo, FailedTask, HttpErrorResponse

HTTP_STATUS = {
    ErrorCode.INPUT_TOO_LARGE: 422,
    ErrorCode.MODEL_TIMEOUT: 504,
    ErrorCode.INVALID_PATCH: 502,
    ErrorCode.INTERNAL: 500,
}
"""HTTP-specific status mapping; domain errors do not depend on HTTP."""


def http_response(info: ErrorInfo, *, request_id: str) -> HttpErrorResponse:
    """Render a controlled error as a simulated HTTP response.

    Args:
        info: Public failure shared with the other adapters.
        request_id: Correlation identifier owned by the request boundary.

    Returns:
        Status and JSON-serializable response body.
    """
    return HttpErrorResponse(
        status_code=HTTP_STATUS[info.code],
        body=ErrorEnvelope(request_id=request_id, error=info),
    )


def cli_text(info: ErrorInfo, *, request_id: str) -> str:
    """Render the explicit CLI format without changing exception string behavior.

    Args:
        info: Public failure shared with the other adapters.
        request_id: Correlation identifier to help locate diagnostics.

    Returns:
        Human-readable text suitable for standard error.
    """
    return f"Error [{info.code}]: {info.message}\nrequest_id={request_id}"


async def execute_task(operation: Callable[[], Awaitable[None]], *, task_id: str) -> FailedTask | None:
    """Demonstrate failure persistence while allowing cancellation to propagate.

    Args:
        operation: Simulated work with no network calls or persistent side effects.
        task_id: Correlation identifier owned by the task executor.

    Returns:
        A failed record, or None when the simulated work completes successfully.
    """
    try:
        await operation()
    except Exception as exc:
        return FailedTask(task_id=task_id, error=public_error(exc))
    return None


def can_retry(info: ErrorInfo, *, idempotent: bool, attempt: int, max_attempts: int) -> bool:
    """Gate retry eligibility without executing or scheduling a retry.

    Args:
        info: Failure describing whether its cause is temporary.
        idempotent: Whether this operation is safe to repeat.
        attempt: Current attempt number, starting at one.
        max_attempts: Maximum total attempts, including the initial attempt.

    Returns:
        True only for temporary failures with safe replay and remaining budget.
    """
    if attempt < 1 or max_attempts < 1:
        raise ValueError("Attempt counts must be positive")
    return info.retryable and idempotent and attempt < max_attempts
