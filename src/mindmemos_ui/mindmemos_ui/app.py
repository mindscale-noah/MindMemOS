"""Same-origin workbench backed by the MindMemOSSkill service contract."""

import asyncio
import hashlib
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mindmemos_sdk.config import ConfigManager
from mindmemos_sdk.errors import ApiError, MindMemOSSDKError
from mindmemos_sdk.skills.cloud import SkillCloudClient
from mindmemos_sdk.transport import HttpTransport
from mindmemos_skill.api.factory import create_service
from mindmemos_skill.api.service import SkillService
from mindmemos_skill.envs.typing import Task
from mindmemos_skill.errors import SkillConflictError, SkillNotFoundError
from mindmemos_skill.pipelines import typing as requests
from mindmemos_skill.pipelines.content import compare_versions, snapshot_from_version
from mindmemos_skill.skill import Skill, SkillBundle, SkillResources, SkillSnapshot
from starlette.middleware.trustedhost import TrustedHostMiddleware

from mindmemos_skill import MindMemOSSkill

from .assembly import load_runtime_skill
from .connection_settings import connection_settings_router
from .config import SettingsUpdate, apply_sdk_settings
from .service_settings import service_settings_router
from .typing import (
    AssembleRequest,
    CommitRequest,
    ConfirmRequest,
    DiffRequest,
    ExportRequest,
    RegisterRequest,
    SwitchRequest,
    VersionRef,
)


def directory_hash(directory: Path) -> str:
    """Fingerprint a destination tree, rejecting symbolic links.

    Args:
        directory: Destination whose complete contents must remain unchanged until confirmation.
    """
    digest = hashlib.sha256()
    if directory.is_symlink():
        raise ValueError("Symbolic link destinations are not supported")
    if not directory.exists():
        return "absent"
    if not directory.is_dir():
        raise ValueError("Expected a directory")
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symbolic links are not supported")
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(b"D" if path.is_dir() else b"F")
        if path.is_file():
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


async def create_app(
    config_manager: ConfigManager | None = None,
    graph_service=None,
    *,
    service: MindMemOSSkill | None = None,
    project_id: str | None = None,
    skill_loader: Callable[[SkillSnapshot], Skill] = load_runtime_skill,
) -> FastAPI:
    """Build the workbench around one local service and project scope.

    Args:
        config_manager: Existing SDK settings owner and workspace home selector.
        graph_service: Optional borrowed query service for the same repository and project.
        service: Optional borrowed MindMemOSSkill implementation; never closed by this app.
        project_id: Required scope when a service is supplied.
        skill_loader: Explicit runtime restoration for installed Skill implementations.

    Returns:
        Local FastAPI application owning its service when none is supplied.
    """
    manager = config_manager or ConfigManager()
    if service is not None and not project_id:
        raise ValueError("project_id is required with an injected service")
    owned = service is None
    if owned:
        local = await create_service(home=manager.config_dir)
        repository = local
        scope = local.local_project_id
        graph_service = await local.graph_queries(scope)
    else:
        repository = service
        scope = project_id

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            if owned:
                await repository.close()

    app = FastAPI(title="MindMemOS UI", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.include_router(service_settings_router(manager.config_dir))
    app.include_router(connection_settings_router(manager.config_dir))
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    token = secrets.token_urlsafe(32)
    plans = OrderedDict()
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
            if request.method in {"GET", "HEAD"} or request.url.path == "/api/configurations/choose-directory":
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
        status = exc.status_code if isinstance(exc, ApiError) and exc.status_code in {401, 403, 404, 409, 429} else 502
        return JSONResponse(
            {"detail": "SDK operation failed; check configuration, connection or file state"}, status_code=status
        )

    @app.exception_handler(SkillNotFoundError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Skill or version not found"}, status_code=404)

    @app.exception_handler(SkillConflictError)
    async def conflict(request, exc):
        return JSONResponse({"detail": "Skill is referenced or its state has changed"}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": "Invalid or unavailable resource"}, status_code=400)

    @app.exception_handler(OSError)
    async def file_error(request, exc):
        return JSONResponse({"detail": "Cannot access local files"}, status_code=400)

    def graph():
        if graph_service is None:
            raise HTTPException(503, "Graph storage is not connected")
        return graph_service

    async def detail(ref: VersionRef):
        return await repository.show_skill(
            requests.ShowRequest(project_id=scope, skill_ref=ref.skill_id, version_id=ref.version_id)
        )

    def remember(kind, value):
        plan_id = secrets.token_urlsafe(24)
        plans[plan_id] = (time.monotonic(), kind, value)
        while len(plans) > 128:
            plans.popitem(last=False)
        return plan_id

    def consume(plan_id, kind):
        entry = plans.pop(plan_id, None)
        if entry is None or entry[1] != kind or time.monotonic() - entry[0] > 600:
            raise HTTPException(409, "Preview expired; preview again")
        return entry[2]

    @app.get("/api/session")
    def session():
        return {
            "token": token,
            "home": str(manager.config_dir),
            "project_id": scope,
            "graphs_connected": graph_service is not None,
        }

    @app.get("/api/config")
    def settings():
        c = manager.load_or_default()
        return {
            "base_url": c.base_url,
            "api_key_set": bool(c.auth.api_key),
            "user_id": c.defaults.user_id,
            "skill_cache_dir": c.storage.skill_cache_dir,
            "skill_backup_dir": c.storage.skill_backup_dir,
            "timeout_seconds": c.network.timeout_seconds,
            "max_retries": c.network.max_retries,
            "path": str(manager.config_path),
        }

    @app.put("/api/config")
    def save_settings(body: SettingsUpdate):
        c = manager.load_or_default()
        apply_sdk_settings(c, body)
        manager.save(c)
        plans.clear()
        return settings()

    @app.post("/api/config/test")
    def test_connection():
        config = manager.load_or_default()
        with HttpTransport(
            base_url=config.base_url,
            api_key=config.auth.api_key,
            timeout_seconds=config.network.timeout_seconds,
            max_retries=config.network.max_retries,
        ) as transport:
            SkillCloudClient(transport).list_skills()
        return {"ok": True}

    @app.get("/api/skills")
    async def skills():
        rows = []
        while True:
            page = await repository.list_skills(requests.ListRequest(project_id=scope, limit=100, offset=len(rows)))
            rows.extend(page)
            if len(page) < 100:
                result = []
                for row in rows:
                    current = (
                        await repository.show_skill(requests.ShowRequest(project_id=scope, skill_ref=row.skill_id))
                    ).version
                    history = await versions(row.skill_id)
                    latest = max(history, key=lambda version: version.created_at, default=current)
                    result.append(
                        {
                            **row.model_dump(),
                            "current_version_id": current.version_id,
                            "current_version_label": current.version_label,
                            "implementation": current.implementation,
                            "latest_version_id": latest.version_id,
                            "latest_version_label": latest.version_label,
                            "latest_implementation": latest.implementation,
                            "cloud_sync_status": await repository.skill_cloud_status(
                                requests.SkillRequest(project_id=scope, skill_ref=row.skill_id)
                            )
                            if isinstance(repository, SkillService)
                            else "unknown",
                        }
                    )
                return result

    @app.get("/api/skills/{skill_id}")
    async def show(skill_id: str):
        return await repository.show_skill(requests.ShowRequest(project_id=scope, skill_ref=skill_id))

    @app.get("/api/versions")
    async def versions(skill_id: str, source: str = "local"):
        VersionRef(source=source, skill_id=skill_id, version_id="current")
        rows = []
        while True:
            page = await repository.skill_history(
                requests.HistoryRequest(project_id=scope, skill_ref=skill_id, limit=100, offset=len(rows))
            )
            rows.extend(page)
            if len(page) < 100:
                return rows

    def displayed_content(version):
        """Read stored files without loading runtime behavior or synthesizing content."""
        files = {}
        binary_files = []
        resources = SkillResources.deserialize(version.resources).files
        for path, raw in resources.items():
            if b"\0" in raw:
                binary_files.append(path)
                continue
            try:
                files[path] = raw.decode("utf-8")
            except UnicodeDecodeError:
                binary_files.append(path)
        if version.bundle and version.bundle.strip() != "[]":
            files.update(SkillBundle.deserialize(version.bundle).files)
        return {"files": files, "resources": list(resources), "binary_files": binary_files}

    @app.get("/api/content")
    async def content(source: str, skill_id: str, version_id: str):
        stored = await detail(VersionRef(source=source, skill_id=skill_id, version_id=version_id))
        return displayed_content(stored.version)

    @app.get("/api/implementation")
    async def implementation(source: str, skill_id: str, version_id: str):
        stored = await detail(VersionRef(source=source, skill_id=skill_id, version_id=version_id))
        return {
            "implementation": stored.version.implementation,
            "dynamic": stored.version.implementation != "static",
        }

    @app.post("/api/assemble")
    async def assemble(body: AssembleRequest):
        if len({ref.skill_id for ref in body.versions}) != 1:
            raise HTTPException(400, "Compare versions of the same Skill")
        versions = [(await detail(ref)).version for ref in body.versions]
        task = Task(id=str(uuid4()), instruction=body.instruction)
        runtime = {}
        try:
            # Validate every source before any implementation can make external calls.
            for version in versions:
                if version.version_id not in runtime:
                    runtime[version.version_id] = skill_loader(snapshot_from_version(version))
        except NotImplementedError:
            raise HTTPException(501, "This Skill implementation does not support runtime assembly") from None
        except ValueError:
            raise HTTPException(400, "Skill source or runtime configuration is invalid") from None
        assembled = {}
        try:
            for version in versions:
                if version.version_id not in assembled:
                    snapshot = (
                        await asyncio.to_thread(runtime[version.version_id].assemble, task.model_copy(deep=True))
                    ).snapshot()
                    assembled[version.version_id] = version.model_copy(update=snapshot.model_dump())
        except Exception:
            # Provider exceptions may contain endpoints, credentials or task content.
            raise HTTPException(
                502, "Skill assembly failed; check the server model configuration and connection"
            ) from None
        outputs = [assembled[version.version_id] for version in versions]
        result = {"contents": [displayed_content(version) for version in outputs], "task": task.model_dump()}
        if len(outputs) == 2:
            result["comparison"] = compare_versions(*outputs).model_dump()
        return result

    @app.post("/api/diff")
    async def diff(body: DiffRequest):
        if body.before.skill_id != body.after.skill_id:
            raise HTTPException(400, "Compare versions of the same Skill")
        result = await repository.diff_skill(
            requests.DiffRequest(
                project_id=scope,
                skill_ref=body.after.skill_id,
                from_version_id=body.before.version_id,
                to_version_id=body.after.version_id,
            )
        )
        return {
            "diff": "\n".join(change.diff or f"{change.change}: {change.path}" for change in result.files),
            **result.model_dump(),
        }

    @app.post("/api/register/preview")
    def register_preview(body: RegisterRequest):
        snapshot = Skill.from_path(body.path).snapshot()
        return {
            "plan_id": remember("register", (body, snapshot)),
            "name": body.name,
            "path": body.path,
            "hash": snapshot.local_snapshot_hash,
            "files": ["SKILL.md", *Skill.from_snapshot(snapshot).resources.files],
        }

    @app.post("/api/register/confirm")
    async def register_confirm(body: ConfirmRequest):
        request, snapshot = consume(body.plan_id, "register")
        if Skill.from_path(request.path).snapshot() != snapshot:
            raise HTTPException(409, "Source changed; preview again")
        return await repository.register_skill(
            requests.RegisterRequest(
                project_id=scope,
                name=request.name,
                version_label=request.version_label,
                source=snapshot,
                alias=request.alias,
                message=request.message,
                registration_source=str(Path(request.path).expanduser().resolve()),
            )
        )

    @app.post("/api/commit")
    async def commit(body: CommitRequest):
        return await repository.commit_skill(
            requests.CommitRequest(
                project_id=scope,
                skill_ref=body.skill_id,
                source=body.path,
                base_version_id=body.base_version_id,
                message=body.message,
                version_label=body.version_label,
                version_bump=body.version_bump,
            )
        )

    @app.delete("/api/skills/{skill_id}")
    async def remove(skill_id: str):
        await repository.delete_skill(requests.SkillRequest(project_id=scope, skill_ref=skill_id))
        return {"ok": True}

    @app.post("/api/switch/preview")
    async def switch_preview(body: SwitchRequest):
        current = await repository.show_skill(requests.ShowRequest(project_id=scope, skill_ref=body.skill_id))
        result = await repository.diff_skill(
            requests.DiffRequest(project_id=scope, skill_ref=body.skill_id, to_version_id=body.version_id)
        )
        return {
            "plan_id": remember("switch", (body, current.skill.current_version_id)),
            "diff": "\n".join(change.diff or f"{change.change}: {change.path}" for change in result.files),
        }

    @app.post("/api/switch/confirm")
    async def switch_confirm(body: ConfirmRequest):
        request, expected = consume(body.plan_id, "switch")
        current = await repository.show_skill(requests.ShowRequest(project_id=scope, skill_ref=request.skill_id))
        if current.skill.current_version_id != expected:
            raise HTTPException(409, "Current version changed; preview again")
        return await repository.switch_skill(
            requests.SwitchRequest(project_id=scope, skill_ref=request.skill_id, version_id=request.version_id)
        )

    def export_directory(value):
        raw = Path(value).expanduser().absolute()
        if any(path.is_symlink() for path in (raw, *raw.parents)):
            raise ValueError("Symbolic link destinations are not supported")
        target = raw.resolve()
        home = manager.config_dir.resolve()
        if target == home or home.is_relative_to(target) or target.is_relative_to(home):
            raise ValueError("Export outside the workspace home")
        return target

    @app.post("/api/export/preview")
    async def export_preview(body: ExportRequest):
        target = export_directory(body.directory)
        stored = await detail(body.version)
        skill = Skill.from_snapshot(snapshot_from_version(stored.version))
        fingerprint = directory_hash(target)
        files = ["SKILL.md", *skill.resources.files]
        request = requests.ExportRequest(
            project_id=scope,
            skill_ref=body.version.skill_id,
            version_id=body.version.version_id,
            directory=str(target),
            overwrite=True,
        )
        return {
            "plan_id": remember("export", (request, fingerprint)),
            "directory": str(target),
            "files": files,
            "overwrite": target.exists() and any(target.iterdir()),
            "diff": "\n".join(files),
        }

    @app.post("/api/export/confirm")
    async def export_confirm(body: ConfirmRequest):
        request, expected = consume(body.plan_id, "export")
        target = export_directory(request.directory)
        if directory_hash(target) != expected:
            raise HTTPException(409, "Destination changed; preview again")
        return {"path": await repository.export_skill(request)}

    @app.get("/api/graphs")
    async def graphs():
        return await graph().list_graphs()

    @app.get("/api/graphs/{graph_id}/snapshots")
    async def graph_snapshots(graph_id: str):
        return await graph().snapshots(graph_id)

    @app.get("/api/graphs/{graph_id}/snapshots/{snapshot_id}")
    async def graph_snapshot(graph_id: str, snapshot_id: UUID):
        return await graph().snapshot(graph_id, snapshot_id)

    static = Path(__file__).parent / "static"
    app.mount("/assets", StaticFiles(directory=static / "assets", check_dir=False), name="assets")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    return app
