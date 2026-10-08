"""Same-origin workbench backed by the configured MindMemOS SDK services."""

import secrets
from collections.abc import Callable
from pathlib import Path

import anyio
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.errors import ApiError, MindMemOSSDKError
from mindmemos_sdk.skills import ListRequest as SkillListRequest
from mindmemos_skill.common.skill_runtime import load_runtime_skill
from mindmemos_skill.errors import SkillConflictError
from mindmemos_skill.skill import Skill, SkillSnapshot
from starlette.middleware.trustedhost import TrustedHostMiddleware

from mindmemos_sdk import MindMemOSClient

from .config import SettingsUpdate, apply_sdk_settings
from .sdk_settings import sdk_settings_router
from .sdk_workbench import sdk_workbench_router
from .service_settings import service_settings_router
from .typing import MemorySearchRequest


async def create_app(
    config_manager: ConfigManager | None = None,
    *,
    skill_loader: Callable[[SkillSnapshot], Skill] = load_runtime_skill,
) -> FastAPI:
    """Build the local browser gateway around the configured SDK services.

    Args:
        config_manager: SDK settings owner and local export workspace selector.
        skill_loader: Runtime restoration used for task assembly previews.

    Returns:
        A loopback workbench that reads Skill data through the public SDK.
    """
    manager = config_manager or ConfigManager()
    app = FastAPI(title="MindMemOS UI", docs_url=None, redoc_url=None)
    app.include_router(service_settings_router(manager.config_dir))
    app.include_router(sdk_settings_router(manager.config_dir))
    app.include_router(sdk_workbench_router(manager, skill_loader=skill_loader))
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    token = secrets.token_urlsafe(32)
    lock = anyio.Lock()

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
                return JSONResponse({"detail": "Cross-origin request denied"}, status_code=403)
            if request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
                return JSONResponse({"detail": "Cross-origin request denied"}, status_code=403)
            if request.method not in {"GET", "HEAD"} and not secrets.compare_digest(
                request.headers.get("x-workbench-token", ""), token
            ):
                return JSONResponse({"detail": "Invalid workbench session"}, status_code=403)
            if request.method in {"GET", "HEAD"} or request.url.path in {
                "/api/configurations/choose-directory",
                "/api/skills/choose-directory",
            }:
                response = await call_next(request)
            else:
                async with lock:
                    response = await call_next(request)
            response.headers["Cache-Control"] = "no-store"
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"detail": "Invalid request fields"}, status_code=422)

    @app.exception_handler(MindMemOSSDKError)
    async def sdk_error(request, exc):
        if isinstance(exc, ApiError):
            status = exc.status_code if exc.status_code is not None and 400 <= exc.status_code <= 599 else 502
            body = exc.response_body
            detail = body.get("detail") if isinstance(body, dict) else None
            message = detail if isinstance(detail, str) and detail else str(exc)
            return JSONResponse({"detail": message}, status_code=status)
        return JSONResponse({"detail": "Service request failed"}, status_code=502)

    @app.exception_handler(SkillConflictError)
    async def conflict(request, exc):
        return JSONResponse({"detail": "Destination or Skill state changed"}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": "Invalid or unavailable resource"}, status_code=400)

    @app.exception_handler(OSError)
    async def file_error(request, exc):
        return JSONResponse({"detail": "Cannot access local files"}, status_code=400)

    @app.get("/api/session")
    def session():
        config = manager.load_or_default()
        _, skill_key = config.resolve_skill_connection()
        return {
            "token": token,
            "home": str(manager.config_dir),
            "graphs_connected": bool(skill_key),
        }

    @app.get("/api/config")
    def settings():
        config = manager.load_or_default()
        base_url, api_key = config.resolve_memory_connection()
        return {
            "base_url": base_url,
            "api_key_set": bool(api_key),
            "user_id": config.defaults.user_id,
            "timeout_seconds": config.network.timeout_seconds,
            "max_retries": config.network.max_retries,
            "path": str(manager.config_path),
        }

    @app.put("/api/config")
    def save_settings(body: SettingsUpdate):
        config = manager.load_or_default()
        apply_sdk_settings(config, body)
        manager.save(config)
        return settings()

    @app.post("/api/config/test")
    def test_connection():
        config = manager.load_or_default()
        with MindMemOSClient(config=config) as client:
            client.skills.list(SkillListRequest(limit=1))
        return {"ok": True}

    @app.get("/api/connections")
    def connections():
        """Check each service with a bounded, read-only SDK request."""
        config = manager.load_or_default()
        memory_url, memory_key = config.resolve_memory_connection()
        skill_url, skill_key = config.resolve_skill_connection()
        result = {
            "memory": {"url": memory_url, "status": "unconfigured"},
            "skill": {"url": skill_url, "status": "unconfigured"},
        }
        if not memory_key and not skill_key:
            return result

        probe_config = config.model_copy(deep=True)
        probe_config.network.timeout_seconds = min(config.network.timeout_seconds, 5)
        probe_config.network.max_retries = 0
        with MindMemOSClient(config=probe_config) as client:
            for name, key, probe in (
                ("memory", memory_key, lambda: client.memory.get(top_k=1)),
                ("skill", skill_key, lambda: client.skills.list(SkillListRequest(limit=1))),
            ):
                if not key:
                    continue
                try:
                    probe()
                except ApiError as exc:
                    result[name]["status"] = "unauthorized" if exc.status_code in {401, 403} else "unavailable"
                except (MindMemOSSDKError, ValueError):
                    result[name]["status"] = "unavailable"
                else:
                    result[name]["status"] = "connected"
        return result

    @app.get("/api/memories")
    def memories(page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100)):
        """List one page of memories through the configured Memory SDK client."""
        config = manager.load_or_default()
        with MindMemOSClient(config=config) as client:
            result = client.memory.list(page=page, page_size=page_size, user_id=config.defaults.user_id)
        return result.model_dump(mode="json")

    @app.post("/api/memories/search")
    def search_memories(body: MemorySearchRequest):
        """Return read-only Memory search results for one explicit query."""
        config = manager.load_or_default()
        with MindMemOSClient(config=config) as client:
            result = client.memory.search(
                body.query,
                top_k=body.top_k,
                user_id=body.user_id.strip() if body.user_id else config.defaults.user_id,
            )
        return result.model_dump(mode="json")

    static = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static / "assets", check_dir=False), name="assets")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
