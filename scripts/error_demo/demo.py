"""Run with .venv/bin/python -m scripts.error_demo.demo from the repository root."""

import asyncio
import json
from collections.abc import Awaitable, Callable

from .adapters import can_retry, cli_text, execute_task, http_response
from .errors import InputTooLargeError, InvalidPatchError, ModelTimeoutError, public_error


async def _oversized_input() -> None:
    """Simulate explicit input validation before starting model work."""
    raise InputTooLargeError(limit=100000, actual=120000)


async def _model_timeout() -> None:
    """Translate a provider failure at the dependency boundary, retaining cause."""
    try:
        raise TimeoutError("Provider diagnostics that must stay outside public responses")
    except TimeoutError as exc:
        raise ModelTimeoutError(stage="scene_extraction") from exc


async def _invalid_patch() -> None:
    """Simulate an algorithm protocol failure rather than a transport timeout."""
    try:
        raise ValueError("Raw model response that must stay outside public responses")
    except ValueError as exc:
        raise InvalidPatchError() from exc


async def _internal_bug() -> None:
    """Simulate a ValueError that is not a caller input validation failure."""
    raise ValueError("Private implementation state that must stay outside public responses")


async def _cancelled() -> None:
    """Simulate cancellation as a lifecycle event rather than an ordinary failure."""
    raise asyncio.CancelledError()


async def _show_failure(name: str, operation: Callable[[], Awaitable[None]]) -> None:
    """Show the same deterministic failure through each application adapter."""
    request_id = f"req-{name}"
    try:
        await operation()
    except Exception as exc:
        info = public_error(exc)
        print(f"\n=== {name} ===")
        print(f"Python print(exc): {exc}")
        response = http_response(info, request_id=request_id)
        print(f"HTTP {response.status_code}:")
        print(response.body.model_dump_json(indent=2, exclude_none=True))
        print("CLI text:")
        print(cli_text(info, request_id=request_id))
        print("CLI JSON:")
        print(info.model_dump_json(exclude_none=True))
        print("Diagnostic event (public fields only; original exception remains available):")
        cause_type = type(exc.__cause__).__name__ if exc.__cause__ is not None else "none"
        print(
            f"request.failed | request_id={request_id} code={info.code} category={info.category} "
            f"exception_type={type(exc).__name__} cause_type={cause_type}"
        )
        print("Retry eligibility (no retry is executed):")
        print(
            json.dumps(
                {
                    "temporary_and_idempotent": can_retry(info, idempotent=True, attempt=1, max_attempts=3),
                    "unsafe_to_replay": can_retry(info, idempotent=False, attempt=1, max_attempts=3),
                    "budget_exhausted": can_retry(info, idempotent=True, attempt=3, max_attempts=3),
                }
            )
        )
    # Replay only these deterministic, side-effect-free fixtures to illustrate a worker.
    failed = await execute_task(operation, task_id=f"task-{name}")
    if failed is not None:
        print("Task record (task lookup itself would return HTTP 200):")
        print(failed.model_dump_json(indent=2, exclude_none=True))


async def main() -> None:
    """Display four failures and one propagated cancellation without external I/O."""
    for name, operation in (
        ("input-too-large", _oversized_input),
        ("model-timeout", _model_timeout),
        ("invalid-patch", _invalid_patch),
        ("internal-bug", _internal_bug),
    ):
        await _show_failure(name, operation)
    print("\n=== cancellation ===")
    try:
        await execute_task(_cancelled, task_id="task-cancelled")
    except asyncio.CancelledError:
        print("CancelledError propagated to the lifecycle owner; no failed ErrorInfo was created.")


if __name__ == "__main__":
    asyncio.run(main())
