"""Ordered asynchronous persistence of standard logs within a task scope."""

import asyncio
import logging
import sys
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .config import attach_capture_handler, detach_capture_handler
from .context import log_context
from .records import log_message
from .sanitization import safe_text

_owner: ContextVar[object | None] = ContextVar("task_log_owner", default=None)


@asynccontextmanager
async def async_task_log_reporter(
    sink: Callable[[dict[str, Any]], Awaitable[None]], *, project_id: str, task_id: str, attempt: int
) -> AsyncIterator[Callable[[], Awaitable[None]]]:
    """Report ordinary logger messages in task order using the execution context.

    Args:
        sink: Async writer receiving log_id, project_id, task_id, attempt,
            occurred_at and a JSON payload of message, body, traceback and correlation.
        project_id: Authorized owning project.
        task_id: Durable task identity.
        attempt: Physical execution attempt, retained with every historical log.

    Yields:
        Async flush callback that waits for all currently queued log writes.
    """
    owner = object()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1000)

    class TaskHandler(logging.Handler):
        """Capture only records from this task's execution context."""

        def emit(self, record: logging.LogRecord) -> None:
            """Queue a prepared record only for its owning task.

            Args:
                record: Sanitized record with context attached by the shared filter.
            """
            if _owner.get() is not owner:
                return
            # Keep routine call telemetry out of progress without affecting the OTel handler.
            if record.getMessage() == "agent.call" and record.levelno < logging.WARNING:
                return
            payload = {
                key: record.__dict__.get(key)
                for key in (
                    "request_id",
                    "run_id",
                    "trace_id",
                    "span_id",
                    "tree_id",
                    "parent_id",
                    "node_id",
                    "group_id",
                )
            }
            payload.update(
                name=record.name,
                level=record.levelname,
                message=log_message(record),
                body=record.__dict__.get("event_body"),
                traceback=record.exc_text,
            )
            row = dict(
                log_id=uuid4().hex,
                project_id=project_id,
                task_id=task_id,
                attempt=attempt,
                occurred_at=datetime.fromtimestamp(record.created, UTC),
                payload=payload,
            )
            try:
                queue.put_nowait(row)
            except asyncio.QueueFull:
                sys.stderr.write(f"Task logging queue full | project_id={project_id} task_id={task_id}\n")

    async def write() -> None:
        """Serialize writes and report sink errors without recursive logging."""
        while (row := await queue.get()) is not None:
            try:
                await sink(row)
            except Exception:
                sys.stderr.write(
                    f"Task log persistence failed | project_id={project_id} task_id={task_id}\n"
                    f"{safe_text(traceback.format_exc())}\n"
                )
            finally:
                queue.task_done()

    handler = TaskHandler()
    handler.setLevel(logging.INFO)
    attach_capture_handler(handler)
    writer = asyncio.create_task(write())
    token = _owner.set(owner)
    try:
        with log_context(project_id=project_id, task_id=task_id, attempt=attempt):
            yield queue.join
    finally:
        _owner.reset(token)
        detach_capture_handler(handler)
        await queue.put(None)
        await writer
