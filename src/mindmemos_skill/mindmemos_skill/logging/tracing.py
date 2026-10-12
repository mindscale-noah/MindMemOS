"""Monitor decorated functions through standard logs and OpenTelemetry spans."""

import asyncio
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter
from typing import Any, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .config import get_logger
from .context import _context, _reported_error, log_context
from .sanitization import safe_text, structured_text


def get_tracer() -> trace.Tracer:
    """Return the service-bound tracer or the process provider's tracer."""
    return _tracer.get() or trace.get_tracer("mindmemos_skill")


def log_failure(logger, message: str, *, body: dict[str, Any]) -> None:
    """Log an active exception once across nested stage and request boundaries.

    Args:
        logger: Standard Python logger receiving the diagnostic.
        message: Stable event name.
        body: JSON-compatible outcome fields such as stage and duration_ms.
    """
    import sys

    error = sys.exception()
    if error is not None and _reported_error.get() is error:
        logger.error(message, extra={"event_body": body})
    else:
        logger.exception(message, extra={"event_body": body})
        _reported_error.set(error)


@contextmanager
def stage(name: str) -> Iterator[trace.Span]:
    """Time a bounded stage and record its outcome through ordinary logging.

    Args:
        name: Stable stage name chosen by the algorithm or execution boundary.

    Yields:
        Active child span; exceptions are recorded automatically by OTel.
    """
    if not name:
        raise ValueError("Stage name must be nonempty")
    started = perf_counter()
    logger = get_logger(__name__)
    cancelled = None
    with log_context(), get_tracer().start_as_current_span(name) as span:
        for key, value in _context.get().items():
            if value is not None:
                span.set_attribute(key, value)
        span.set_attribute("stage", name)
        try:
            yield span
        except asyncio.CancelledError as exc:
            cancelled = exc
            span.set_attribute("status", "cancelled")
            logger.info(
                "stage.cancelled",
                extra={
                    "event_body": {
                        "stage": name,
                        "status": "cancelled",
                        "duration_ms": (perf_counter() - started) * 1000,
                    }
                },
            )
        except BaseException as exc:
            log_failure(
                logger,
                "stage.failed",
                body={
                    "stage": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "duration_ms": (perf_counter() - started) * 1000,
                },
            )
            raise
        else:
            span.set_status(Status(StatusCode.OK))
            logger.info(
                "stage.completed",
                extra={
                    "event_body": {
                        "stage": name,
                        "status": "succeeded",
                        "duration_ms": (perf_counter() - started) * 1000,
                    }
                },
            )
        finally:
            span.set_attribute("duration_ms", (perf_counter() - started) * 1000)
    if cancelled is not None:
        raise cancelled


F = TypeVar("F", bound=Callable[..., Any])
_tracer: ContextVar[trace.Tracer | None] = ContextVar("skill_logging_tracer", default=None)


@contextmanager
def bind_tracer(tracer: trace.Tracer) -> Iterator[None]:
    """Bind the owning service's tracer without replacing the global provider.

    Args:
        tracer: Service-owned tracer used by public calls and nested stages.

    Yields:
        Control while this execution context uses the supplied tracer.
    """
    token = _tracer.set(tracer)
    try:
        yield
    finally:
        _tracer.reset(token)


def traced(
    name: str | None = None,
    *,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[F], F]:
    """Record function outcome, duration and a child span automatically.

    Args:
        name: Span name; defaults to the qualified function name.
        record_args: Include sanitized arguments, excluding self and cls.
        record_result: Include the sanitized return value.

    Returns:
        A decorator preserving synchronous or asynchronous call semantics.
        Named correlation arguments (request_id, project_id, task_id, attempt,
        run_id, service_name) bind automatically; other fields inherit the caller.

    Raises:
        TypeError: A generator function is supplied; its iteration is not a call.
    """

    def decorator(func: F) -> F:
        if inspect.isgeneratorfunction(func) or inspect.isasyncgenfunction(func):
            raise TypeError("traced does not support generator functions")
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        signature = inspect.signature(func)
        logger = get_logger(func.__module__)

        @contextmanager
        def monitor(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Iterator[dict[str, Any]]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            fields = {
                key: value
                for key, value in bound.arguments.items()
                if key in {"request_id", "project_id", "task_id", "attempt", "run_id", "service_name"}
                and isinstance(value, (str, int))
            }
            body: dict[str, Any] = {"function": span_name}
            if record_args:
                body["arguments"] = structured_text(
                    {key: value for key, value in bound.arguments.items() if key not in {"self", "cls"}}
                )
            tracer = get_tracer()
            with (
                log_context(**fields),
                tracer.start_as_current_span(span_name, record_exception=False, set_status_on_exception=False) as span,
            ):
                started = perf_counter()
                try:
                    yield body
                except asyncio.CancelledError:
                    body.update(status="cancelled", duration_ms=(perf_counter() - started) * 1000)
                    logger.info("traced call cancelled", extra={"event_body": body})
                    raise
                except BaseException as exc:
                    body.update(
                        status="failed", duration_ms=(perf_counter() - started) * 1000, error_type=type(exc).__name__
                    )
                    span.set_status(Status(StatusCode.ERROR, safe_text(str(exc))))
                    span.add_event(
                        "exception", {"exception.type": type(exc).__name__, "exception.message": safe_text(str(exc))}
                    )
                    log_failure(logger, "traced call failed", body=body)
                    raise
                else:
                    body.update(status="succeeded", duration_ms=(perf_counter() - started) * 1000)
                    span.set_status(Status(StatusCode.OK))
                    logger.info("traced call", extra={"event_body": body})
                finally:
                    span.set_attribute("duration_ms", (perf_counter() - started) * 1000)
                    if record_args:
                        span.set_attribute("arguments", body["arguments"])
                    if "result" in body:
                        span.set_attribute("result", body["result"])

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with monitor(args, kwargs) as body:
                    result = await func(*args, **kwargs)
                    if record_result:
                        body["result"] = structured_text(result)
                    return result

            return cast(F, async_wrapper)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with monitor(args, kwargs) as body:
                result = func(*args, **kwargs)
                if record_result:
                    body["result"] = structured_text(result)
                return result

        return cast(F, sync_wrapper)

    return decorator
