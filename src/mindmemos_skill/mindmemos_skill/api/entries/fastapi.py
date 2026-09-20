"""Start the Skill HTTP service using Uvicorn."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ...errors import SkillConflictError, SkillNotFoundError
from ..base import MindMemOSSkill
from ..factory import create_service

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(service: MindMemOSSkill) -> "FastAPI":
    """Expose an existing service over HTTP without rebuilding its backends.

    Args:
        service: Assembled service shared by all HTTP routes.

    Returns:
        A FastAPI application ready for an ASGI server.
    """
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            service.close()

    app = FastAPI(title="MindMemOS Skill", lifespan=lifespan)
    app.state.service = service

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
        "/skills/auto-evolve": service.auto_evolve_skill,
        "/skills/auto-train": service.auto_train_skill,
    }
    for path, operation in operations.items():
        app.add_api_route(
            path,
            operation,
            methods=["POST"],
            responses=(
                {501: {"description": "Operation is not implemented yet"}}
                if path in {"/skills/auto-evolve", "/skills/auto-train"}
                else {404: {"description": "Not found"}, 409: {"description": "Conflict"}}
            ),
        )
    return app


def main() -> None:
    """Build a service, wrap it in HTTP routes, and start the ASGI server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Service YAML file")
    parser.add_argument("--host", default="127.0.0.1", help="Listening address")
    parser.add_argument("--port", type=int, default=8000, help="Listening port")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    service = create_service(args.config)
    service.initialize_storage()

    import uvicorn

    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
