"""Observe ASGI requests without consuming or buffering complete response streams."""

import asyncio
import json
from collections.abc import Callable
from time import perf_counter
from urllib.parse import parse_qs

from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from mindmemos_skill.common.ids import generate_id

from .config import get_logger
from .context import log_context
from .sanitization import safe_text
from .tracing import bind_tracer, get_tracer, log_failure

_BODY_LIMIT = 32768


def _body(data: bytearray, total: int, content_type: str) -> dict:
    """Render complete bounded JSON bodies and metadata for files or streams."""
    metadata = {"bytes": total, "content_type": content_type, "truncated": total > _BODY_LIMIT}
    if total <= _BODY_LIMIT and "json" in content_type and data:
        try:
            metadata["content"] = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            metadata["invalid_json"] = True
            metadata["content"] = safe_text(data.decode("utf-8", errors="replace"))
    return metadata


class RequestLoggingMiddleware:
    """Capture request outcomes, including authentication and validation failures."""

    def __init__(self, app: ASGIApp, *, tracer: Callable):
        """Borrow the current service tracer without taking resource ownership.

        Args:
            app: Wrapped ASGI application.
            tracer: Callback returning the initialized service's tracer, or None.
        """
        self.app = app
        self.tracer = tracer

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass ASGI messages through while retaining only bounded JSON samples.

        Args:
            scope: ASGI connection metadata.
            receive: Original request receiver.
            send: Original response sender; backpressure is preserved.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = generate_id("req_")
        scope.setdefault("state", {})["request_id"] = request_id
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        request_data, response_data = bytearray(), bytearray()
        pending_requests: list[Message] = []
        received = sent = 0
        status_code = 500
        response_type = ""
        completed = disconnected = False
        started = perf_counter()
        logger = get_logger(__name__)

        async def observe_receive() -> Message:
            nonlocal received, disconnected
            if pending_requests:
                return pending_requests.pop(0)
            message = await receive()
            if message["type"] == "http.request":
                data = message.get("body", b"")
                received += len(data)
                if "json" in headers.get("content-type", ""):
                    request_data.extend(data[: max(0, _BODY_LIMIT - len(request_data))])
            elif message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def observe_send(message: Message) -> None:
            nonlocal sent, status_code, response_type, completed
            if message["type"] == "http.response.start":
                status_code = message["status"]
                response_headers = dict(message.get("headers", []))
                response_type = response_headers.get(b"content-type", b"").decode("latin1")
            await send(message)
            if message["type"] == "http.response.body":
                data = message.get("body", b"")
                sent += len(data)
                if "json" in response_type:
                    response_data.extend(data[: max(0, _BODY_LIMIT - len(response_data))])
                completed = not message.get("more_body", False)

        provider = self.tracer() or get_tracer()
        with (
            log_context(request_id=request_id, project_id=None, service_name="mindmemos-skill"),
            bind_tracer(provider),
            provider.start_as_current_span(
                "http.request", context=TraceContextTextMapPropagator().extract(headers), record_exception=False
            ) as span,
        ):
            span.set_attribute("request_id", request_id)
            span.set_attribute("http.request.method", scope["method"])
            span.set_attribute("url.path", scope["path"])
            failure = False
            try:
                # Small JSON inputs are replayed so authentication rejection still has input evidence.
                length = headers.get("content-length", "")
                if "json" in headers.get("content-type", "") and length.isdecimal() and int(length) <= _BODY_LIMIT:
                    messages = []
                    while True:
                        message = await observe_receive()
                        messages.append(message)
                        if (
                            message["type"] != "http.request"
                            or not message.get("more_body", False)
                            or received >= _BODY_LIMIT
                        ):
                            break
                    pending_requests.extend(messages)
                await self.app(scope, observe_receive, observe_send)
            except asyncio.CancelledError:
                disconnected = True
                raise
            except Exception:
                failure = True
                with log_context(project_id=scope["state"].get("project_id")):
                    log_failure(logger, "http.request.failed", body={"path": scope["path"], "status_code": status_code})
                raise
            finally:
                body = {
                    "query": parse_qs(scope.get("query_string", b"").decode("utf-8", errors="replace")),
                    "method": scope["method"],
                    "path": scope["path"],
                    "route": scope["route"].path if "route" in scope else scope["path"],
                    "status_code": status_code,
                    "duration_ms": (perf_counter() - started) * 1000,
                    "input": _body(request_data, received, headers.get("content-type", "")),
                    "output": _body(response_data, sent, response_type),
                    "completed": completed,
                    "disconnected": disconnected,
                }
                span.set_attribute("http.response.status_code", status_code)
                span.set_attribute("duration_ms", body["duration_ms"])
                span.set_attribute("http.route", body["route"])
                span.set_status(Status(StatusCode.ERROR if failure or status_code >= 400 else StatusCode.OK))
                project_id = scope["state"].get("project_id")
                if project_id is not None:
                    span.set_attribute("project_id", project_id)
                with log_context(project_id=project_id):
                    logger.info("http.request.completed", extra={"event_body": body})
