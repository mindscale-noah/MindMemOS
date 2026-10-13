"""Verify public failure boundaries and lifecycle behavior in the architecture demo."""

import asyncio

import pytest
from pydantic import ValidationError

from scripts.error_demo.adapters import can_retry, cli_text, execute_task, http_response
from scripts.error_demo.demo import _cancelled, _internal_bug, _invalid_patch, _model_timeout, _oversized_input
from scripts.error_demo.errors import public_error
from scripts.error_demo.typing import ContentLimitDetails, ErrorCode, StageDetails


@pytest.mark.parametrize(
    ("operation", "code", "status", "cause_type"),
    [
        (_oversized_input, ErrorCode.INPUT_TOO_LARGE, 422, None),
        (_model_timeout, ErrorCode.MODEL_TIMEOUT, 504, TimeoutError),
        (_invalid_patch, ErrorCode.INVALID_PATCH, 502, ValueError),
        (_internal_bug, ErrorCode.INTERNAL, 500, None),
    ],
)
async def test_http_and_task_preserve_the_same_public_failure(operation, code, status, cause_type):
    """Keep error identity and data intact while the original cause stays diagnostic."""
    with pytest.raises(Exception) as captured:
        await operation()
    exc = captured.value
    info = public_error(exc)
    response = http_response(info, request_id="req-test")
    task = await execute_task(operation, task_id="task-test")
    assert info.code == code
    assert response.status_code == status
    assert response.body.error == task.error == info
    assert (type(exc.__cause__) if exc.__cause__ is not None else None) is cause_type
    public_output = response.body.model_dump_json() + task.model_dump_json() + cli_text(info, request_id="req-test")
    assert "Provider diagnostics" not in public_output
    assert "Raw model response" not in public_output
    assert "Private implementation state" not in public_output


async def test_temporary_failure_requires_safe_replay_and_remaining_budget():
    """A retryable timeout alone must not authorize repeating work."""
    with pytest.raises(Exception) as captured:
        await _model_timeout()
    info = public_error(captured.value)
    assert can_retry(info, idempotent=True, attempt=1, max_attempts=3)
    assert not can_retry(info, idempotent=False, attempt=1, max_attempts=3)
    assert not can_retry(info, idempotent=True, attempt=3, max_attempts=3)
    with pytest.raises(Exception) as invalid_patch:
        await _invalid_patch()
    assert not can_retry(public_error(invalid_patch.value), idempotent=True, attempt=1, max_attempts=3)


async def test_cancellation_does_not_become_a_failed_record():
    """Let the lifecycle owner distinguish cancellation from ordinary failure."""
    with pytest.raises(asyncio.CancelledError):
        await execute_task(_cancelled, task_id="task-test")


async def test_success_does_not_create_a_failed_record():
    """A completed operation must not be reported as a generic internal failure."""

    async def complete():
        return None

    assert await execute_task(complete, task_id="task-test") is None


def test_public_details_reject_raw_diagnostics_and_invalid_limits():
    """Prevent arbitrary private fields from being attached to public error details."""
    with pytest.raises(ValidationError):
        StageDetails(stage="scene_extraction", model_response="private")
    with pytest.raises(ValidationError):
        ContentLimitDetails(limit=100, actual=99)
