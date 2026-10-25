"""Expose Skill HTTP routes and the Uvicorn ASGI application."""

import base64
import binascii
import os
from contextlib import asynccontextmanager
from inspect import isawaitable
from ipaddress import ip_address
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from mindmemos_skill.common.ids import generate_id

from ...common.time_utils import utcnow
from ...config.config import FastAPIConfig
from ...config.loader import read_skill_config
from ...errors import (
    GraphConflictError,
    GraphNotFoundError,
    SkillAliasConflictError,
    SkillAmbiguousVersionError,
    SkillConflictError,
    SkillInUseError,
    SkillNotFoundError,
    SkillVersionNotFoundError,
)
from ...logging import log_context
from ...logging.http import RequestLoggingMiddleware
from ..auth import SkillHttpError, SkillHttpIdentity, build_identity_resolver
from ..base import MindMemOSSkill
from ..constants import SKILL_API_HOST_ENV, SKILL_CONFIG_PATH_ENV
from ..factory import create_service
from ..service import SkillService
from ..typing import (
    ActiveRequest,
    AddGraphMembersRequest,
    CommitRequest,
    DiffRequest,
    HistoryRequest,
    ListTrajectoriesRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    RemoveGraphMembersRequest,
    ReportTrajectoryRequest,
    SetMetaRequest,
    ShowRequest,
    ShowTrajectoryRequest,
    SkillHistoryCursor,
    SkillPageRequest,
    SkillRequest,
    TrajectoryStatisticsRequest,
    VersionResolveRequest,
)
from ..typing import (
    TrajectoryListCursor as ServiceTrajectoryCursor,
)
from .typing import (
    ActivateHttpRequest,
    AddGraphMembersHttpRequest,
    CommitHttpRequest,
    DeleteHttpRequest,
    DiffHttpRequest,
    EvolveHttpRequest,
    GraphResolveHttpRequest,
    GraphSnapshotHttpRequest,
    GraphSnapshotsHttpRequest,
    HistoryCursor,
    HistoryHttpRequest,
    IdentityResolver,
    ListCursor,
    ListHttpRequest,
    OrganizeHttpRequest,
    RecommendHttpRequest,
    RegisterHttpRequest,
    RemoveGraphMembersHttpRequest,
    ServiceFactory,
    ShowHttpRequest,
    SkillResolveHttpRequest,
    TaskShowHttpRequest,
    TrainingHttpRequest,
    TrajectoryListCursor,
    TrajectoryListHttpRequest,
    TrajectoryReportHttpRequest,
    TrajectoryShowHttpRequest,
    TrajectoryStatisticsHttpRequest,
    UpdateInfoHttpRequest,
    VersionResolveHttpRequest,
)

if TYPE_CHECKING:
    from fastapi import FastAPI


DEFAULT_CONFIG_PATH = "config/mindmemos_skill/dev.yaml"


def _family_data(family) -> dict:
    """Return portable family fields with public identity names."""
    return {
        "skill_id": family.skill_id,
        "name": family.name,
        "alias": family.alias,
        "metadata": family.metadata,
        "active_skill_version_id": family.active_version_id,
        "created_at": family.created_at,
        "updated_at": family.updated_at,
    }


def _version_data(version) -> dict:
    """Return complete source content without server-local annotations."""
    return {
        "skill_version_id": version.version_id,
        "version_label": version.version_label,
        "parent_skill_version_ids": version.parent_version_ids,
        "content_hash": version.content_hash,
        "skill_hash": version.skill_hash,
        "name": version.name,
        "description": version.description,
        "standard_format": version.standard_format,
        "metadata": version.metadata,
        "commit_message": version.commit_message,
        "implementation": version.implementation,
        "config_schema_version": version.config_schema_version,
        "config": version.config,
        "content": version.content,
        "resources": version.resources,
        "created_at": version.created_at,
    }


async def _selected_graph_version(
    service: MindMemOSSkill, graph_ref: str, version_ref: str | None, project_id: str
) -> tuple[str, str]:
    """Resolve a public graph and its pinned version for an existing operation."""
    graph = await service.resolve_graph(graph_ref, project_id)
    if version_ref is None:
        selected = graph["graph_version_id"]
    else:
        version = await service.resolve_graph(version_ref, project_id)
        if version["graph_id"] != graph["graph_id"]:
            raise ValueError("Graph version does not belong to the selected graph")
        selected = version["graph_version_id"]
    if selected is None:
        raise ValueError("Graph has no published version")
    return graph["graph_id"], selected


def _encode_cursor(cursor: ListCursor | TrajectoryListCursor | HistoryCursor) -> str:
    """Encode a validated continuation position for the next HTTP request."""
    return base64.urlsafe_b64encode(cursor.model_dump_json().encode("utf-8")).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> ListCursor:
    """Reject malformed or unsupported continuation tokens."""
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = base64.b64decode(padded, altchars=b"-_", validate=True)
        return ListCursor.model_validate_json(payload)
    except (binascii.Error, ValueError) as exc:
        raise SkillHttpError(422, "invalid_cursor", "Invalid Skill list cursor") from exc


def _decode_trajectory_cursor(value: str) -> TrajectoryListCursor:
    """Reject malformed or unsupported trajectory continuation tokens."""
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = base64.b64decode(padded, altchars=b"-_", validate=True)
        return TrajectoryListCursor.model_validate_json(payload)
    except (binascii.Error, ValueError) as exc:
        raise SkillHttpError(422, "invalid_cursor", "Invalid trajectory list cursor") from exc


def _decode_history_cursor(value: str) -> HistoryCursor:
    """Decode a validated immutable history continuation token."""
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = base64.b64decode(padded, altchars=b"-_", validate=True)
        return HistoryCursor.model_validate_json(payload)
    except (binascii.Error, ValueError) as exc:
        raise SkillHttpError(422, "invalid_cursor", "Invalid history cursor") from exc


def create_app(
    service: MindMemOSSkill | None = None,
    *,
    identity_from_bearer: IdentityResolver | None = None,
    api_config: FastAPIConfig | None = None,
    service_factory: ServiceFactory | None = None,
) -> "FastAPI":
    """Wrap one assembled service in the documented HTTP contract.

    Args:
        service: Shared service whose resources this application closes, if already assembled.
        identity_from_bearer: Trusted account adapter resolving MindMemOS Bearer
            credentials into an authorized project and Skill scopes.
        api_config: Deployment pipeline policy for future configured algorithms.
        service_factory: Async service constructor used by a process-managed deployment.

    Returns:
        FastAPI application using the existing service backends.
    """
    from fastapi import Depends, FastAPI, Request
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    if service is None and service_factory is None:
        raise ValueError("A service or service_factory is required")

    @asynccontextmanager
    async def lifespan(app):
        nonlocal service, policy, identity_from_bearer
        if service is None:
            assert service_factory is not None
            service, policy = await service_factory()
            identity_from_bearer = build_identity_resolver(policy.auth)
            app.state.service = service
            app.state.api_config = policy
        try:
            yield
        finally:
            if service is not None:
                await service.close()
            if service_factory is not None:
                service = None
                app.state.service = None

    policy = api_config or (service.api_config if isinstance(service, SkillService) else None)
    if service is not None and policy is None:
        raise ValueError("HTTP requires an explicit FastAPIConfig or a complete service YAML")
    if isinstance(service, SkillService) and policy is not None and identity_from_bearer is None:
        if policy.auth.mode == "none" and service.default_project_id != policy.auth.default_project_id:
            raise ValueError("HTTP default project must match the service deployment configuration")
        if policy.auth.mode != "none" and service.default_project_id is not None:
            raise ValueError("Authenticated HTTP requires a service without a default project")
    if service is not None and identity_from_bearer is None:
        identity_from_bearer = build_identity_resolver(policy.auth)

    app = FastAPI(title="MindMemOS Skill", lifespan=lifespan)
    app.state.service = service
    app.state.api_config = policy

    @app.middleware("http")
    async def assign_request_id(request: Request, call_next):
        """Assign a request ID even when body validation fails."""
        if not request.scope.get("state", {}).get("request_id"):
            request.state.request_id = generate_id("req_")
        if policy.auth.mode == "none":
            server = request.scope.get("server")
            if server is not None:
                try:
                    if not ip_address(server[0]).is_loopback:
                        return error(
                            request, 503, "authentication_unavailable", "Unauthenticated HTTP requires loopback"
                        )
                except ValueError:
                    pass
        return await call_next(request)

    app.add_middleware(
        RequestLoggingMiddleware,
        tracer=lambda: app.state.service.telemetry.tracer if isinstance(app.state.service, SkillService) else None,
    )

    def envelope(request: Request, data, *, status_code: int = 200) -> JSONResponse:
        """Serialize one successful business response."""
        return JSONResponse(
            status_code=status_code,
            content=jsonable_encoder(
                {
                    "code": "accepted" if status_code == 202 else "ok",
                    "request_id": request.state.request_id,
                    "data": data,
                }
            ),
        )

    def error(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
        """Serialize one stable error without exposing a traceback."""
        return JSONResponse(
            status_code=status_code,
            content={
                "code": "error",
                "request_id": request.state.request_id,
                "error": {"code": code, "message": message},
            },
        )

    @app.exception_handler(SkillHttpError)
    async def http_error(request: Request, exc: SkillHttpError) -> JSONResponse:
        """Return classified capability and identity errors."""
        return error(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return malformed public input as invalid_request."""
        return error(request, 422, "invalid_request", str(exc))

    @app.exception_handler(ValueError)
    async def invalid_value(request: Request, exc: ValueError) -> JSONResponse:
        """Return validation failures raised after body parsing."""
        return error(request, 422, "invalid_request", str(exc))

    @app.exception_handler(SkillNotFoundError)
    async def missing(request: Request, exc: SkillNotFoundError) -> JSONResponse:
        """Hide project ownership for missing or foreign records."""
        return error(request, 404, "skill_not_found", str(exc))

    @app.exception_handler(GraphNotFoundError)
    async def missing_graph(request: Request, exc: GraphNotFoundError) -> JSONResponse:
        """Hide graph ownership for missing or foreign identities."""
        return error(request, 404, "graph_not_found", str(exc))

    @app.exception_handler(GraphConflictError)
    async def ambiguous_graph(request: Request, exc: GraphConflictError) -> JSONResponse:
        """Reject ambiguous graph ID prefixes."""
        return error(request, 409, "graph_ambiguous", str(exc))

    @app.exception_handler(SkillVersionNotFoundError)
    async def missing_version(request: Request, exc: SkillVersionNotFoundError) -> JSONResponse:
        """Return a non-enumerating version lookup failure."""
        return error(request, 404, "skill_version_not_found", str(exc))

    @app.exception_handler(SkillInUseError)
    async def in_use(request: Request, exc: SkillInUseError) -> JSONResponse:
        """Prevent deletion while retained evidence references a Skill."""
        return error(request, 409, "skill_in_use", str(exc))

    @app.exception_handler(SkillConflictError)
    async def conflict(request: Request, exc: SkillConflictError) -> JSONResponse:
        """Expose a stable version or identity conflict category."""
        return error(request, 409, "version_conflict", str(exc))

    @app.exception_handler(SkillAliasConflictError)
    async def alias_conflict(request: Request, exc: SkillAliasConflictError) -> JSONResponse:
        """Separate alias collisions from version races."""
        return error(request, 409, "alias_conflict", str(exc))

    @app.exception_handler(SkillAmbiguousVersionError)
    async def ambiguous_version(request: Request, exc: SkillAmbiguousVersionError) -> JSONResponse:
        """Require an exact version ID when the same hash has multiple owners."""
        return error(request, 409, "ambiguous_skill_version", str(exc))

    @app.exception_handler(NotImplementedError)
    async def unavailable(request: Request, exc: NotImplementedError) -> JSONResponse:
        """Expose unavailable backend operations without claiming completion."""
        return error(request, 501, "capability_unavailable", str(exc))

    @app.exception_handler(PermissionError)
    async def forbidden(request: Request, exc: PermissionError) -> JSONResponse:
        """Reject a project authorization failure."""
        return error(request, 403, "permission_denied", str(exc))

    async def authenticated(request: Request):
        """Bind the credential's project and scopes for one service request."""
        if identity_from_bearer is None:
            if not isinstance(service, SkillService) or service.default_project_id is None:
                raise SkillHttpError(503, "authentication_unavailable", "Skill authentication is not configured")
            identity = SkillHttpIdentity(project_id=service.default_project_id, scopes={"skill:write"})
        else:
            authorization = request.headers.get("authorization")
            if not authorization:
                raise SkillHttpError(401, "auth.missing_authorization", "Missing Authorization header")
            scheme, _, credential = authorization.partition(" ")
            if scheme.lower() != "bearer" or not credential.strip():
                raise SkillHttpError(401, "auth.invalid_authorization_scheme", "Expected Bearer credential")
            resolved = identity_from_bearer(credential.strip())
            if isawaitable(resolved):
                resolved = await resolved
            identity = SkillHttpIdentity.model_validate(resolved)
            if identity.principal is None:
                identity.principal = resolved
        if identity.request_id is not None:
            request.state.request_id = identity.request_id
        request.state.project_id = identity.project_id
        with log_context(project_id=identity.project_id):
            if isinstance(service, SkillService):
                with service.request_principal(identity.principal if identity.principal is not None else identity):
                    yield identity
            else:
                yield identity

    def require(scope: str):
        """Check one Skill scope after credential resolution."""

        async def dependency(identity: SkillHttpIdentity = Depends(authenticated)) -> SkillHttpIdentity:
            """Return an identity authorized for this operation."""
            if scope not in identity.scopes and not (scope == "skill:read" and "skill:write" in identity.scopes):
                raise SkillHttpError(403, "auth.insufficient_scope", "Insufficient Skill scope")
            return identity

        return dependency

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Report process availability, not business readiness."""
        return {"status": "ok"}

    @app.post("/v1/skills/register", status_code=201)
    async def register(
        request: Request, body: RegisterHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Register one source document under the authenticated project."""
        result = await service.register_skill(
            RegisterRequest(project_id=identity.project_id, skill=body.skill, alias=body.alias)
        )
        return envelope(request, result, status_code=201)

    @app.post("/v1/skills/list")
    async def list_skills(
        request: Request, body: ListHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Filter families and return a bounded cursor page."""
        state = _decode_cursor(body.cursor) if body.cursor is not None else None
        if state is not None and (
            state.project_id != identity.project_id
            or ("filters" in body.model_fields_set and body.filters != state.filters)
        ):
            raise SkillHttpError(422, "invalid_cursor", "Cursor project or filters differ from this request")
        filters = state.filters if state is not None else body.filters
        snapshot_at = state.snapshot_at if state is not None else utcnow()
        page = await service.list_skill_page(
            SkillPageRequest(
                project_id=identity.project_id,
                name_pattern=filters.name,
                alias_pattern=filters.alias,
                user_id=filters.user_id,
                agent_id=filters.agent_id,
                limit=body.limit,
                snapshot_at=snapshot_at,
                after_created_at=state.last_created_at if state is not None else None,
                after_skill_id=state.last_skill_id if state is not None else None,
            )
        )
        selected = page.items
        next_cursor = (
            _encode_cursor(
                ListCursor(
                    project_id=identity.project_id,
                    filters=filters,
                    snapshot_at=snapshot_at,
                    last_created_at=selected[-1].created_at,
                    last_skill_id=selected[-1].skill_id,
                )
            )
            if page.has_more
            else None
        )
        return envelope(
            request,
            {
                "items": [_family_data(family) for family in selected],
                "limit": body.limit,
                "next_cursor": next_cursor,
                "has_more": next_cursor is not None,
            },
        )

    @app.post("/v1/skills/show")
    async def show(
        request: Request, body: ShowHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Return portable family metadata and the complete selected version."""
        result = await service.show_skill(
            ShowRequest(project_id=identity.project_id, skill_id=body.skill_id, version_id=body.skill_version_id)
        )
        return envelope(request, {"skill": _family_data(result.skill), "version": _version_data(result.version)})

    @app.post("/v1/skills/commit", status_code=201)
    async def commit(
        request: Request, body: CommitHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Create one immutable child and conditionally select it as active."""
        result = await service.commit_skill(CommitRequest(project_id=identity.project_id, **body.model_dump()))
        return envelope(
            request,
            {
                "skill_id": result.skill.skill_id,
                "skill_version_id": result.version.version_id,
                "version_label": result.version.version_label,
                "active": body.active,
                "content_hash": result.version.content_hash,
                "skill_hash": result.version.skill_hash,
                "active_skill_version_id": result.skill.active_version_id,
            },
            status_code=201,
        )

    @app.post("/v1/skills/history")
    async def history(
        request: Request, body: HistoryHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Return one bounded immutable version history page."""
        state = _decode_history_cursor(body.cursor) if body.cursor else None
        if state is not None and (
            state.project_id != identity.project_id or state.skill_id != body.skill_id or body.offset
        ):
            raise SkillHttpError(422, "invalid_cursor", "History cursor does not match this request")
        position = SkillHistoryCursor(
            offset=body.offset,
            snapshot_at=state.snapshot_at if state else (None if body.offset else utcnow()),
            after_created_at=state.after_created_at if state else None,
            after_version_id=state.after_version_id if state else None,
        )
        versions = await service.skill_history(
            HistoryRequest(
                project_id=identity.project_id,
                skill_id=body.skill_id,
                limit=None if body.limit is None else body.limit + 1,
                cursor=position,
            )
        )
        selected = versions[: body.limit]
        has_more = body.limit is not None and len(versions) > body.limit
        next_cursor = None
        if has_more and position.snapshot_at is not None:
            next_cursor = _encode_cursor(
                HistoryCursor(
                    project_id=identity.project_id,
                    skill_id=body.skill_id,
                    snapshot_at=position.snapshot_at,
                    after_created_at=selected[-1].created_at,
                    after_version_id=selected[-1].version_id,
                )
            )
        return envelope(
            request,
            {
                "items": [
                    {
                        "skill_version_id": version.version_id,
                        "version_label": version.version_label,
                        "commit_message": version.commit_message,
                        "parent_skill_version_ids": version.parent_version_ids,
                        "content_hash": version.content_hash,
                        "skill_hash": version.skill_hash,
                        "implementation": version.implementation,
                        "status": version.status,
                        "origin": version.origin,
                        "created_at": version.created_at,
                    }
                    for version in versions[: body.limit]
                ],
                "limit": body.limit,
                "offset": body.offset,
                "has_more": has_more,
                "next_cursor": next_cursor,
            },
        )

    @app.post("/v1/skills/diff")
    async def diff(
        request: Request, body: DiffHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Compare stored versions across two families in the same project."""
        source = await service.show_skill(
            ShowRequest(
                project_id=identity.project_id,
                skill_id=body.skill_id,
                version_id=body.from_skill_version_id,
            )
        )
        await service.show_skill(
            ShowRequest(
                project_id=identity.project_id,
                skill_id=body.to_skill_id or body.skill_id,
                version_id=body.to_skill_version_id,
            )
        )
        result = await service.diff_skill(
            DiffRequest(
                project_id=identity.project_id,
                from_version_id=source.version.version_id,
                to_version_id=body.to_skill_version_id,
            )
        )
        return envelope(
            request,
            {
                "from_skill_id": result.from_skill_id,
                "from_skill_version_id": result.from_version_id,
                "to_skill_id": result.to_skill_id,
                "to_skill_version_id": result.to_version_id,
                "files": result.files,
                "implementation_changed": result.implementation_changed,
            },
        )

    @app.post("/v1/skills/activate")
    async def activate(
        request: Request, body: ActivateHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Select an existing owned version as the family active version."""
        await service.show_skill(
            ShowRequest(
                project_id=identity.project_id,
                skill_id=body.skill_id,
                version_id=body.skill_version_id,
            )
        )
        family = await service.active_skill(
            ActiveRequest(
                project_id=identity.project_id,
                version_id=body.skill_version_id,
                expected_active_version_id=body.expected_active_version_id,
            )
        )
        return envelope(request, {"skill_id": family.skill_id, "active_skill_version_id": family.active_version_id})

    @app.post("/v1/skills/update-info")
    async def update_info(
        request: Request, body: UpdateInfoHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Replace requested family display fields without editing versions."""
        family = await service.set_skill_meta(
            SetMetaRequest(
                project_id=identity.project_id,
                skill_id=body.skill_id,
                name=body.name,
                alias=body.alias,
                clear_alias="alias" in body.model_fields_set and body.alias is None,
                metadata=body.metadata,
            )
        )
        return envelope(request, _family_data(family))

    @app.post("/v1/skills/delete")
    async def delete(
        request: Request, body: DeleteHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Soft-delete a family while retaining its stored history."""
        await service.delete_skill(SkillRequest(project_id=identity.project_id, skill_ref=body.skill_id))
        return envelope(request, {"deleted": True})

    @app.post("/v1/skills/version-resolve")
    async def resolve_version(
        request: Request, body: VersionResolveHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Resolve an exact source version or reject an ambiguous hash."""
        result = await service.resolve_skill_version(
            VersionResolveRequest(
                project_id=identity.project_id,
                content_hash=body.content_hash,
                skill_hash=body.skill_hash,
                skill_id=body.skill_id,
                skill_version_id=body.skill_version_id,
            )
        )
        return envelope(request, result)

    @app.post("/v1/skills/skill-resolve")
    async def skill_resolve(
        request: Request, body: SkillResolveHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Resolve one family alias or immutable version in the credential's project."""
        detail = await service.resolve_skill_reference(body.skill_ref, identity.project_id)
        return envelope(
            request,
            {
                "skill_id": detail.skill.skill_id,
                "skill_version_id": detail.version.version_id,
                "alias": detail.skill.alias,
                "is_active": detail.skill.active_version_id == detail.version.version_id,
                "version_label": detail.version.version_label,
            },
        )

    @app.post("/v1/skills/trajectories/statistics")
    async def trajectory_statistics(
        request: Request,
        body: TrajectoryStatisticsHttpRequest,
        identity: SkillHttpIdentity = Depends(require("skill:read")),
    ):
        """Return global and daily aggregate counts, never trajectory content."""
        result = await service.trajectory_statistics(
            TrajectoryStatisticsRequest(
                project_id=identity.project_id,
                skill_id=body.skill_id,
                version_id=body.skill_version_id,
            )
        )
        return envelope(request, result)

    @app.post("/v1/skills/trajectories/report", status_code=201)
    async def report_trajectory(
        request: Request,
        body: TrajectoryReportHttpRequest,
        identity: SkillHttpIdentity = Depends(require("skill:write")),
    ):
        """Store one execution against its exact loaded Skill version."""
        report = await service.report_trajectory(
            ReportTrajectoryRequest(
                project_id=identity.project_id,
                user_id=body.user_id,
                agent_id=body.agent_id,
                messages=body.messages,
                skill_version_id=body.skill_version_id,
                session_id=body.session_id,
                status=body.status,
                started_at=body.started_at,
                finished_at=body.finished_at,
            )
        )
        return envelope(
            request,
            {
                "trajectory_id": report.trajectory_id,
                "skill_version_id": report.skill_version_id,
                "stored_at": report.received_at,
            },
            status_code=201,
        )

    @app.post("/v1/skills/trajectories/show")
    async def show_trajectory(
        request: Request, body: TrajectoryShowHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Return one project-owned physical attempt and its version binding."""
        record = await service.show_trajectory(
            ShowTrajectoryRequest(project_id=identity.project_id, trajectory_id=body.trajectory_id)
        )
        return envelope(
            request,
            {
                "trajectory_id": record.trajectory_id,
                "task_id": record.task_id,
                "source": record.source,
                "skill_id": record.skill_id,
                "skill_version_id": record.skill_version_id,
                "skill_bindings": record.skill_bindings,
                "user_id": record.user_id,
                "agent_id": record.agent_id,
                "session_id": record.session_id,
                "task_instruction": record.task_instruction,
                "task_system_prompt": record.task_system_prompt,
                "environment": record.env_ref,
                "trajectory": record.trajectory,
                "status": record.status,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "received_at": record.received_at,
            },
        )

    @app.post("/v1/skills/trajectories/list")
    async def list_trajectories(
        request: Request, body: TrajectoryListHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Page one family's reported execution summaries by receipt order."""
        state = _decode_trajectory_cursor(body.cursor) if body.cursor is not None else None
        if state is not None and (
            state.project_id != identity.project_id
            or state.skill_id != body.skill_id
            or ("skill_version_id" in body.model_fields_set and state.skill_version_id != body.skill_version_id)
            or ("status" in body.model_fields_set and state.status != body.status)
            or ("received_at_from" in body.model_fields_set and state.received_at_from != body.received_at_from)
            or ("received_at_to" in body.model_fields_set and state.received_at_to != body.received_at_to)
        ):
            raise SkillHttpError(422, "invalid_cursor", "Cursor project or filters differ from this request")
        skill_version_id = state.skill_version_id if state is not None else body.skill_version_id
        status = state.status if state is not None else body.status
        received_at_from = state.received_at_from if state is not None else body.received_at_from
        received_at_to = state.received_at_to if state is not None else body.received_at_to
        snapshot_at = state.snapshot_at if state is not None else utcnow()
        rows = await service.list_trajectories(
            ListTrajectoriesRequest(
                project_id=identity.project_id,
                skill_id=body.skill_id,
                skill_version_id=skill_version_id,
                status=status,
                cursor=ServiceTrajectoryCursor(
                    received_at_from=received_at_from,
                    received_at_to=received_at_to,
                    snapshot_at=snapshot_at,
                    after_received_at=state.last_received_at if state is not None else None,
                    after_trajectory_id=state.last_trajectory_id if state is not None else None,
                    limit=None if body.limit is None else body.limit + 1,
                ),
            )
        )
        selected = rows[: body.limit]
        next_cursor = (
            _encode_cursor(
                TrajectoryListCursor(
                    project_id=identity.project_id,
                    skill_id=body.skill_id,
                    skill_version_id=skill_version_id,
                    status=status,
                    received_at_from=received_at_from,
                    received_at_to=received_at_to,
                    snapshot_at=snapshot_at,
                    last_received_at=selected[-1].received_at,
                    last_trajectory_id=selected[-1].trajectory_id,
                )
            )
            if body.limit is not None and len(rows) > body.limit
            else None
        )
        return envelope(
            request,
            {
                "items": [
                    {
                        "trajectory_id": row.trajectory_id,
                        "task_id": row.task_id,
                        "skill_id": row.skill_id,
                        "skill_version_id": row.skill_version_id,
                        "user_id": row.user_id,
                        "agent_id": row.agent_id,
                        "status": row.status,
                        "started_at": row.started_at,
                        "finished_at": row.finished_at,
                        "received_at": row.received_at,
                    }
                    for row in selected
                ],
                "next_cursor": next_cursor,
                "has_more": next_cursor is not None,
            },
        )

    @app.post("/v1/skills/algos/organize")
    async def organize(
        request: Request, body: OrganizeHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Publish an exact project-owned Skill collection through skill_manage."""
        try:
            result = await service.organize_skills(
                OrganizeRequest(
                    project_id=identity.project_id,
                    user_id=identity.actor_id,
                    **body.model_dump(),
                )
            )
        except NotImplementedError as exc:
            raise SkillHttpError(422, "pipeline_unavailable", str(exc)) from exc
        data = result.model_dump(mode="json")
        return envelope(request, data)

    @app.post("/v1/skills/graphs/members/add")
    async def add_graph_members(
        request: Request,
        body: AddGraphMembersHttpRequest,
        identity: SkillHttpIdentity = Depends(require("skill:write")),
    ):
        """Add a batch of new families using the authenticated project."""
        result = await service.add_graph_members(
            AddGraphMembersRequest(project_id=identity.project_id, user_id=identity.actor_id, **body.model_dump())
        )
        data = result.model_dump(mode="json")
        return envelope(request, data)

    @app.post("/v1/skills/graphs/members/remove")
    async def remove_graph_members(
        request: Request,
        body: RemoveGraphMembersHttpRequest,
        identity: SkillHttpIdentity = Depends(require("skill:write")),
    ):
        """Remove members while retaining their canonical Skill versions."""
        result = await service.remove_graph_members(
            RemoveGraphMembersRequest(project_id=identity.project_id, user_id=identity.actor_id, **body.model_dump())
        )
        data = result.model_dump(mode="json")
        return envelope(request, data)

    @app.post("/v1/skills/graphs/preview")
    async def preview_graph(
        request: Request,
        body: GraphResolveHttpRequest,
        identity: SkillHttpIdentity = Depends(require("skill:read")),
    ):
        """Read published topology without extracting Scenes or writing data."""
        item = await service.preview_graph(body.graph_ref, identity.project_id)
        return envelope(request, item)

    @app.post("/v1/skills/graph-resolve")
    async def graph_resolve(
        request: Request, body: GraphResolveHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Resolve one graph alias or immutable snapshot in the credential's project."""
        return envelope(request, await service.resolve_graph(body.graph_ref, identity.project_id))

    @app.post("/v1/skills/graphs/delete")
    async def delete_graph(
        request: Request, body: GraphResolveHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Soft-delete one graph in the credential's project."""
        await service.delete_graph(body.graph_ref, identity.project_id)
        return envelope(request, {"deleted": True})

    @app.post("/v1/skills/graphs/list")
    async def list_graphs(request: Request, identity: SkillHttpIdentity = Depends(require("skill:read"))):
        """List organization graphs in the credential's project."""
        items = await service.list_graph(identity.project_id)
        return envelope(request, {"items": items})

    @app.post("/v1/skills/graphs/snapshots")
    async def graph_snapshots(
        request: Request,
        body: GraphSnapshotsHttpRequest,
        identity: SkillHttpIdentity = Depends(require("skill:read")),
    ):
        """List immutable snapshots and revisions of one project-owned graph."""
        items = await service.list_graph_snapshots(body.graph_id, identity.project_id)
        return envelope(request, {"items": items})

    @app.post("/v1/skills/graphs/snapshot")
    async def graph_snapshot(
        request: Request,
        body: GraphSnapshotHttpRequest,
        identity: SkillHttpIdentity = Depends(require("skill:read")),
    ):
        """Read a graph snapshot without exposing another project's versions."""
        graph_id, version_id = await _selected_graph_version(
            service, body.graph_id, body.graph_version_id, identity.project_id
        )
        item = await service.get_graph_snapshot(graph_id, version_id, identity.project_id)
        return envelope(request, item)

    @app.post("/v1/skills/algos/recommend")
    async def recommend(
        request: Request, body: RecommendHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Rank exact versions in a pinned project-owned graph snapshot."""
        try:
            graph_id, version_id = await _selected_graph_version(
                service, body.graph_ref, body.graph_version_id, identity.project_id
            )
            result = await service.recommend_skills(
                RecommendRequest(
                    project_id=identity.project_id,
                    graph_id=graph_id,
                    graph_version_id=version_id,
                    task=body.query,
                    top_k=body.top_k or 5,
                    strategy=body.strategy,
                    include_scenes=body.include_scenes,
                )
            )
        except NotImplementedError as exc:
            raise SkillHttpError(422, "pipeline_unavailable", str(exc)) from exc
        data = result.model_dump(mode="json")
        return envelope(request, data)

    def evolution_pipeline(requested: str | None) -> str:
        """Resolve an explicitly permitted pipeline or the configured HTTP default."""
        selected = requested or policy.default_evolution_pipeline
        if selected is None:
            if len(policy.evolution_pipelines) != 1:
                raise SkillHttpError(422, "pipeline_unavailable", "An evolution pipeline must be selected")
            selected = next(iter(policy.evolution_pipelines))
        if selected not in policy.evolution_pipelines:
            raise SkillHttpError(422, "pipeline_unavailable", "The evolution pipeline is not permitted")
        return selected

    @app.post("/v1/skills/algos/evolve", status_code=202)
    async def evolve(
        request: Request, body: EvolveHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Accept evolution of one exact version or an authorized version collection."""
        pipeline = evolution_pipeline(body.pipeline)
        version_id = body.skill_version_ids if body.skill_version_ids is not None else body.skill_version_id
        options = {}
        if body.editable_skills is not None:
            options["editable_skills"] = body.editable_skills
        if not body.allow_create_skills:
            options["allow_create_skills"] = False
        try:
            snapshot = await service.submit_evolution(
                identity.project_id, version_id, pipeline=pipeline, auto_accept=body.auto_accept, **options
            )
        except (NotImplementedError, ValueError) as exc:
            raise SkillHttpError(422, "pipeline_unavailable", str(exc)) from exc
        return envelope(request, snapshot, status_code=202)

    @app.post("/v1/skills/tasks/show")
    async def show_task(
        request: Request, body: TaskShowHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:read"))
    ):
        """Read a task belonging to the authenticated project."""
        try:
            snapshot = await service.task_status(identity.project_id, body.task_id)
        except NotImplementedError as exc:
            raise SkillHttpError(501, "task_api_unavailable", "Skill task storage is not configured") from exc
        if snapshot is None:
            raise SkillHttpError(404, "task_not_found", "Skill task not found")
        return envelope(request, snapshot)

    @app.post("/v1/skills/algos/train", status_code=202)
    async def train(
        request: Request, body: TrainingHttpRequest, identity: SkillHttpIdentity = Depends(require("skill:write"))
    ):
        """Accept one authorized training request for background execution."""
        snapshot = await service.submit_training({**body.model_dump(mode="python"), "project_id": identity.project_id})
        return envelope(request, snapshot, status_code=202)

    @app.post("/v1/skills/tasks/progress")
    async def task_progress(
        request: Request,
        body: TaskShowHttpRequest,
        after: int = 0,
        identity: SkillHttpIdentity = Depends(require("skill:read")),
    ):
        """Read correlated task logs after the supplied sequence."""
        if after < 0:
            raise ValueError("after must be nonnegative")
        try:
            snapshot = await service.task_status(identity.project_id, body.task_id)
        except NotImplementedError as exc:
            raise SkillHttpError(501, "task_api_unavailable", "Skill task storage is not configured") from exc
        if snapshot is None:
            raise SkillHttpError(404, "task_not_found", "Skill task not found")
        return envelope(request, await service.task_progress(identity.project_id, body.task_id, after=after))

    return app


async def build_http_service() -> tuple[SkillService, FastAPIConfig]:
    """Build a service for the HTTP process from the combined configuration.

    Returns:
        Initialized service and its HTTP policy.
    """
    from ...logging import configure_logging

    configure_logging()
    load_dotenv(Path(".env"), override=False, interpolate=False)
    path = Path(os.getenv(SKILL_CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH))
    config = read_skill_config(path)
    policy = config.fastapi
    host = os.getenv(SKILL_API_HOST_ENV, "127.0.0.1")
    try:
        local_host = host == "localhost" or ip_address(host).is_loopback
    except ValueError:
        local_host = False
    if policy.auth.mode == "none" and not local_host:
        raise ValueError("Unauthenticated Skill HTTP requires a loopback address")
    service = await create_service(config)
    return service, policy


app = create_app(service_factory=build_http_service)
"""ASGI application started by the Skill Uvicorn command."""
