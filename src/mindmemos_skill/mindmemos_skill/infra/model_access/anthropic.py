"""Keep Platform signing and streaming on the host side of a native Claude run."""

import asyncio
import socket
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from hmac import compare_digest
from secrets import token_urlsafe

import httpx
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from .platform import platform_connection

_MESSAGES_PATH = "/v1/messages"
_PROTOCOL_HEADERS = ("anthropic-version", "anthropic-beta")


class _LoopbackServer(uvicorn.Server):
    """Serve a subprocess without changing the parent worker's signal handlers."""

    @contextmanager
    def capture_signals(self) -> Iterator[None]:
        """Leave process-wide signal handling to the enclosing service."""
        yield


def _error(status: int, kind: str, message: str) -> JSONResponse:
    """Create a local failure understood by Claude Code."""
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


@asynccontextmanager
async def open_messages_relay() -> AsyncIterator[tuple[str, str]]:
    """Open a loopback relay for one CLI invocation and its authorized project.

    Yields:
        Loopback base URL and a per-invocation credential usable only on that listener.

    Raises:
        RuntimeError: No authorized Platform route is bound.
    """
    platform_connection()
    capability = token_urlsafe(32)
    async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(130, connect=10)) as client:

        async def relay(scope: Scope, receive: Receive, send: Send) -> None:
            """Authenticate the child and mint a fresh Platform token for this request."""
            request = Request(scope, receive)
            if request.method != "POST" or request.url.path != _MESSAGES_PATH:
                await _error(404, "not_found_error", "Unsupported Claude model endpoint")(scope, receive, send)
                return
            authorization = request.headers.get("authorization", "").encode()
            if not compare_digest(authorization, f"Bearer {capability}".encode()):
                await _error(401, "authentication_error", "Invalid Claude session credential")(scope, receive, send)
                return

            body = await request.body()
            base_url, token = platform_connection()
            headers = {name: request.headers[name] for name in _PROTOCOL_HEADERS if name in request.headers}
            headers.update(Authorization=f"Bearer {token}", **{"Content-Type": "application/json"})
            params = {"beta": "true"} if request.query_params.get("beta") == "true" else None
            response: Response
            try:
                upstream = await client.send(
                    client.build_request(
                        "POST", f"{base_url}{_MESSAGES_PATH}", content=body, headers=headers, params=params
                    ),
                    stream=True,
                )
            except httpx.RequestError:
                await _error(503, "api_error", "Platform model gateway is unavailable")(scope, receive, send)
                return
            try:
                if upstream.headers.get("content-type", "").startswith("text/event-stream"):
                    response = StreamingResponse(
                        upstream.aiter_bytes(), status_code=upstream.status_code, media_type="text/event-stream"
                    )
                else:
                    response = Response(
                        await upstream.aread(),
                        status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type"),
                    )
                await response(scope, receive, send)
            finally:
                await upstream.aclose()

        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = _LoopbackServer(
                uvicorn.Config(
                    relay,
                    lifespan="off",
                    interface="asgi3",
                    ws="none",
                    log_config=None,
                    access_log=False,
                    timeout_graceful_shutdown=1,
                )
            )
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("Claude Messages relay failed to start")
                    await asyncio.sleep(0.01)
                yield f"http://127.0.0.1:{port}", capability
            finally:
                server.should_exit = True
                await task
