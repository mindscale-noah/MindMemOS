"""Start the Skill HTTP service using Uvicorn."""

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from inspect import isawaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ...errors import SkillConflictError, SkillNotFoundError, UnsupportedPipelineError
from ...pipelines.typing import SkillPipelineRequest
from ..base import MindMemOSSkill
from ..factory import create_service
from .config import FastAPIConfig

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(
    service: MindMemOSSkill,
    *,
    principal_from_request: Callable[[Any], object | Awaitable[object]] | None = None,
    api_config: FastAPIConfig | None = None,
) -> "FastAPI":
    """Expose an existing service over HTTP without rebuilding its backends.

    Args:
        service: Assembled service shared by all HTTP routes.
        principal_from_request: Host account-system adapter returning an authenticated identity.
        api_config: HTTP pipeline allowlist; omission denies all pipeline execution.

    Returns:
        A FastAPI application ready for an ASGI server.
    """
    from fastapi import Depends, FastAPI, Request
    from fastapi.responses import JSONResponse

    from ..service import SkillService

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(title="MindMemOS Skill", lifespan=lifespan)
    app.state.service = service
    policy = api_config or FastAPIConfig()

    def allowed_pipeline(name: str | None, supported: set[str], kind: str) -> str:
        """Resolve only a pipeline explicitly allowed by this HTTP deployment."""
        selected = name or (next(iter(supported)) if len(supported) == 1 else None)
        if selected is None or selected not in supported:
            raise UnsupportedPipelineError(f"Unsupported {kind} pipeline {name!r}")
        return selected

    async def auto_evolve(request: SkillPipelineRequest | None = None):
        """Run an evolution pipeline only when allowed by the HTTP policy."""
        if request is None:
            return await service.auto_evolve_skill()
        selected = allowed_pipeline(request.pipeline, policy.evolution_pipelines, "evolution")
        return await service.auto_evolve_skill(request.model_copy(update={"pipeline": selected}))

    @app.exception_handler(UnsupportedPipelineError)
    async def unsupported(request: Request, exc: UnsupportedPipelineError) -> JSONResponse:
        """Report deployment capability limits with a stable error type."""
        return JSONResponse(status_code=422, content={"error": "unsupport", "detail": str(exc)})

    @app.exception_handler(NotImplementedError)
    async def not_implemented(request: Request, exc: NotImplementedError) -> JSONResponse:
        """Report pending business operations without claiming success."""
        return JSONResponse(status_code=501, content={"detail": str(exc)})

    @app.exception_handler(SkillNotFoundError)
    async def not_found(request: Request, exc: SkillNotFoundError) -> JSONResponse:
        """Report missing project-scoped objects."""
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(SkillConflictError)
    async def conflict(request: Request, exc: SkillConflictError) -> JSONResponse:
        """Report ambiguous references, stale commits, and retained dependencies."""
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc: ValueError) -> JSONResponse:
        """Report invalid content and operation settings."""
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PermissionError)
    async def denied(request: Request, exc: PermissionError) -> JSONResponse:
        """Reject missing or invalid project authorization."""
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(FileNotFoundError)
    async def missing_source(request: Request, exc: FileNotFoundError) -> JSONResponse:
        """Report a missing local import source."""
        return JSONResponse(status_code=404, content={"detail": "Local source path does not exist"})

    @app.get("/health")
    def health() -> dict[str, str]:
        """Report application availability, not backend readiness."""
        return {"status": "ok"}

    operations = {
        "/skills/register": service.register_skill,
        "/skills/delete": service.delete_skill,
        "/skills/list": service.list_skills,
        "/skills/show": service.show_skill,
        "/skills/export": service.export_skill,
        "/skills/commit": service.commit_skill,
        "/skills/history": service.skill_history,
        "/skills/diff": service.diff_skill,
        "/skills/switch": service.switch_skill,
        "/skills/set-meta": service.set_skill_meta,
        "/skills/auto-evolve": auto_evolve,
    }

    async def bind_principal(request: Request):
        """Bridge the host account identity into one service request."""
        if principal_from_request is None:
            yield
            return
        principal = principal_from_request(request)
        if isawaitable(principal):
            principal = await principal
        if not isinstance(service, SkillService):
            yield
            return
        with service.request_principal(principal):
            yield

    for path, operation in operations.items():
        app.add_api_route(
            path,
            operation,
            methods=["POST"],
            dependencies=[Depends(bind_principal)],
            responses=(
                {501: {"description": "Operation is not implemented yet"}, 422: {"description": "Unsupported pipeline"}}
                if path == "/skills/auto-evolve"
                else {404: {"description": "Not found"}, 409: {"description": "Conflict"}}
            ),
        )
    return app


async def _main() -> None:
    """Build a service, wrap it in HTTP routes, and start the ASGI server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Explicit service YAML file")
    parser.add_argument("--api-config", type=Path, help="HTTP pipeline allowlist YAML file")
    parser.add_argument("--home", type=Path, help="Local service home; omission uses the default home")
    parser.add_argument("--project-id", help="Optional local scope; empty uses the persisted generated scope")
    parser.add_argument("--host", default="127.0.0.1", help="Listening address")
    parser.add_argument("--port", type=int, default=8000, help="Listening port")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.config and (args.home or args.project_id):
        parser.error("--config cannot be combined with --home or --project-id")
    api_config = (
        FastAPIConfig.model_validate(yaml.safe_load(args.api_config.read_text(encoding="utf-8")))
        if args.api_config
        else FastAPIConfig()
    )
    service = await create_service(args.config, home=args.home, project_id=args.project_id or None)

    import uvicorn

    await uvicorn.Server(
        uvicorn.Config(create_app(service, api_config=api_config), host=args.host, port=args.port)
    ).serve()


def main() -> None:
    """Own the process event loop for the asynchronous entry point."""
    asyncio.run(_main())


if __name__ == "__main__":
    main()
