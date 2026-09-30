"""Function logging decorators for algorithm operations."""

import inspect
import logging
import traceback
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Any, TypeVar, cast

from .config import get_logger
from .otel import debug_fields

F = TypeVar("F", bound=Callable[..., Any])
_MAX_TRACEBACK_CHARS = 32768


def _log_input(logger: logging.Logger, name: str, signature: inspect.Signature, args: tuple, kwargs: dict) -> None:
    """Record passed arguments at DEBUG without retaining method receivers."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    try:
        arguments = signature.bind_partial(*args, **kwargs).arguments
    except TypeError:
        arguments = {"args": args, "kwargs": kwargs}
    visible = {key: value for key, value in arguments.items() if key not in {"self", "cls"}}
    logger.debug("function.input", extra={"event_body": {"function": name, **debug_fields(visible)}})


def _log_success(logger: logging.Logger, name: str, started: float, result: Any) -> None:
    """Record successful duration and the result at DEBUG."""
    duration_ms = round((perf_counter() - started) * 1000, 3)
    logger.info("function.call", extra={"event_body": {"function": name, "ok": True, "duration_ms": duration_ms}})
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("function.output", extra={"event_body": {"function": name, **debug_fields({"result": result})}})


def _log_failure(logger: logging.Logger, name: str, started: float, error: BaseException) -> None:
    """Record failed duration, error type, and a bounded traceback."""
    duration_ms = round((perf_counter() - started) * 1000, 3)
    stack = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    if len(stack) > _MAX_TRACEBACK_CHARS:
        stack = stack[:_MAX_TRACEBACK_CHARS] + "...[truncated]"
    logger.exception(
        "function.call",
        extra={
            "event_body": {
                "function": name,
                "ok": False,
                "duration_ms": duration_ms,
                "error_type": type(error).__name__,
                "traceback": stack,
            }
        },
    )


def log_function(function: F) -> F:
    """Log inputs, result, duration, and failures for a sync or async function.

    Inputs and outputs are DEBUG events; completion and failure are INFO and
    ERROR events. The OTel bridge correlates these events when a training run
    is active. Generator functions are unsupported because execution continues
    after a generator object is returned.

    Args:
        function: Synchronous or asynchronous function to instrument.

    Returns:
        Wrapped function with its metadata and calling behavior preserved.

    Raises:
        TypeError: If the target is a generator or async generator function.
    """
    if inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(function):
        raise TypeError("log_function does not support generator functions")
    signature = inspect.signature(function)
    name = f"{function.__module__}.{function.__qualname__}"
    logger = get_logger(function.__module__)

    if inspect.iscoroutinefunction(function):

        @wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Run one asynchronous invocation and record its outcome."""
            _log_input(logger, name, signature, args, kwargs)
            started = perf_counter()
            try:
                result = await function(*args, **kwargs)
            except BaseException as error:
                _log_failure(logger, name, started, error)
                raise
            _log_success(logger, name, started, result)
            return result

        return cast(F, async_wrapper)

    @wraps(function)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Run one synchronous invocation and record its outcome."""
        _log_input(logger, name, signature, args, kwargs)
        started = perf_counter()
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            _log_failure(logger, name, started, error)
            raise
        _log_success(logger, name, started, result)
        return result

    return cast(F, sync_wrapper)
