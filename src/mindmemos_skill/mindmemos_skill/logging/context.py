"""Context-local correlation shared by all Python logging destinations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_context_depth: ContextVar[int] = ContextVar("skill_log_context_depth", default=0)
_reported_error: ContextVar[BaseException | None] = ContextVar("skill_reported_error", default=None)

_context: ContextVar[dict[str, str | int | None]] = ContextVar("skill_log_context", default={})


@contextmanager
def log_context(**fields: str | int | None) -> Iterator[None]:
    """Bind correlation fields for this execution and its child asyncio tasks.

    Args:
        **fields: Request, project, task, attempt, run, service, tree, node or group identity;
            None clears an inherited value within this scope.

    Yields:
        Control with the supplied context merged into the enclosing scope.

    Raises:
        ValueError: An unsupported correlation field was supplied.
    """
    if fields.keys() - {
        "request_id",
        "project_id",
        "task_id",
        "attempt",
        "run_id",
        "service_name",
        "tree_id",
        "parent_id",
        "node_id",
        "group_id",
    }:
        raise ValueError("Unsupported logging context field")
    depth = _context_depth.get()
    depth_token = _context_depth.set(depth + 1)
    error_token = _reported_error.set(None) if depth == 0 else None
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)
        _context_depth.reset(depth_token)
        if error_token is not None:
            _reported_error.reset(error_token)


def capture_trace_context() -> dict[str, str]:
    """Capture W3C propagation headers from the current submission span."""
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier


def current_request_id() -> str | None:
    """Return the request identity bound at the HTTP or Python service boundary."""
    return _context.get().get("request_id")
