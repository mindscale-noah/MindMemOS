"""Gateway-only internal Memory Data Plane routes.

These endpoints are for the commercial Console BFF and must be exposed only on
private network paths. They authenticate Gateway-issued short-lived internal
tokens instead of standalone memory-system API keys.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel

from ..errors import (
    AuthenticationError,
    BadRequestError,
    MemoryNotFoundError,
    PermissionDeniedError,
    ResourceNotFoundError,
)
from ..infra.db import QdrantRecord, build_filter, match_text, match_value
from ..logging import get_logger
from ..pipelines.memory_db import (
    MemoryDbReader,
    MemoryGraph,
    MemoryInsights,
    MemoryInventoryItem,
    MemoryLifecycleProjection,
    VectorRepairRequest,
    VectorRepairResult,
    VectorRepairService,
    VectorRepairStatus,
    project_inventory_item,
    project_memory_lifecycle,
)
from ..provider_bindings import get_provider_binding_service
from .deps import ensure_scopes, get_internal_request_context
from .mappers import to_memory_request_context
from .schemas import ApiResponse, AuthContext

router = APIRouter(prefix="/internal/v1", tags=["internal-memory"])
logger = get_logger(__name__)


class InternalMemoryListData(BaseModel):
    """List payload for console memory visualization."""

    items: list[MemoryInventoryItem]
    next_cursor: str | None = None


InternalMemoryListResponse = ApiResponse[InternalMemoryListData]
InternalMemoryDetailResponse = ApiResponse[dict[str, Any]]
InternalMemoryInsightsResponse = ApiResponse[MemoryInsights]
InternalMemoryLifecycleResponse = ApiResponse[MemoryLifecycleProjection]
InternalMemoryGraphResponse = ApiResponse[MemoryGraph]
ProviderBindingResponse = ApiResponse[dict[str, Any]]
ProviderBindingListResponse = ApiResponse[dict[str, Any]]
VectorRepairResponse = ApiResponse[VectorRepairResult]
VectorRepairStatusResponse = ApiResponse[VectorRepairStatus]


class ProviderBindingCreateRequest(BaseModel):
    """Create a dynamic provider binding."""

    scope: dict[str, Any] = {}
    routers: dict[str, Any]


class ProviderBindingPatchRequest(BaseModel):
    """Patch a dynamic provider binding."""

    routers: dict[str, Any]


def get_vector_repair_service() -> VectorRepairService:
    """Build a repair service from the process database/client registry."""

    return VectorRepairService()


@router.get("/projects/{project_id}/memories", response_model=InternalMemoryListResponse)
async def list_project_memories(
    project_id: str = Path(min_length=1),
    q: str | None = Query(default=None),
    status: str | None = Query(default=None, min_length=1, max_length=50),
    algorithm: str | None = Query(default=None, min_length=1, max_length=50),
    mem_type: str | None = Query(default=None, min_length=1, max_length=50),
    limit: int = Query(default=50, ge=1, le=100),
    cursor: str | None = Query(default=None, min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryListResponse:
    """List project memories for the Console BFF."""

    _ensure_internal_read(ctx, project_id)
    query_filter = _inventory_filter(q=q, status=status, algorithm=algorithm, mem_type=mem_type)
    records, next_cursor = await MemoryDbReader().list_memory_records(
        ctx, filters=query_filter, limit=limit, cursor=cursor
    )
    items = [project_inventory_item(record.payload, point_id=str(record.point_id)) for record in records]
    data = InternalMemoryListData(items=items, next_cursor=str(next_cursor) if next_cursor is not None else None)
    return InternalMemoryListResponse(request_id=ctx.request_id, data=data)


@router.get("/projects/{project_id}/memories/{memory_id}", response_model=InternalMemoryDetailResponse)
async def get_project_memory(
    project_id: str = Path(min_length=1),
    memory_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryDetailResponse:
    """Return one project memory for the Console BFF."""

    _ensure_internal_read(ctx, project_id)
    record = await MemoryDbReader().get_memory_record(ctx, memory_id)
    if record is None:
        raise ResourceNotFoundError(str(MemoryNotFoundError(memory_id)), code="memory.not_found")
    return InternalMemoryDetailResponse(request_id=ctx.request_id, data=_record_to_item(record))


@router.get("/projects/{project_id}/memory-insights", response_model=InternalMemoryInsightsResponse)
async def get_project_memory_insights(
    project_id: str = Path(min_length=1),
    scan_limit: int = Query(default=5000, ge=1, le=10000),
    range_days: int = Query(default=30, ge=1, le=90),
    refresh: bool = Query(default=False),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryInsightsResponse:
    """Return bounded, evidence-backed lifecycle totals for one project."""

    _ensure_internal_read(ctx, project_id)
    result = await MemoryDbReader().get_memory_insights(
        ctx, scan_limit=scan_limit, range_days=range_days, refresh=refresh
    )
    return InternalMemoryInsightsResponse(request_id=ctx.request_id, data=result)


@router.get(
    "/projects/{project_id}/memories/{memory_id}/lifecycle",
    response_model=InternalMemoryLifecycleResponse,
)
async def get_project_memory_lifecycle(
    project_id: str = Path(min_length=1),
    memory_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryLifecycleResponse:
    """Return an evidence-only lifecycle projection for one memory."""

    _ensure_internal_read(ctx, project_id)
    reader = MemoryDbReader()
    record = await reader.get_memory_record(ctx, memory_id)
    if record is None:
        raise ResourceNotFoundError(str(MemoryNotFoundError(memory_id)), code="memory.not_found")
    lineage = (await reader.get_memory_lineage(ctx, [memory_id])).get(memory_id, [])
    provenance_unavailable = False
    try:
        add_record = await reader.get_memory_provenance(ctx, memory_id, record.payload)
    except (AuthenticationError, PermissionDeniedError):
        raise
    except Exception as exc:
        add_record = None
        provenance_unavailable = True
        logger.warning(
            "memory provenance read unavailable",
            request_id=ctx.request_id,
            project_id=project_id,
            memory_id=memory_id,
            error_type=type(exc).__name__,
        )
    result = project_memory_lifecycle(
        record.payload, add_record=add_record, lineage=lineage, provenance_unavailable=provenance_unavailable
    )
    return InternalMemoryLifecycleResponse(request_id=ctx.request_id, data=result)


@router.get(
    "/projects/{project_id}/memories/{memory_id}/graph",
    response_model=InternalMemoryGraphResponse,
)
async def get_project_memory_graph(
    project_id: str = Path(min_length=1),
    memory_id: str = Path(min_length=1),
    limit: int = Query(default=100, ge=1, le=200),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryGraphResponse:
    """Return a bounded semantic neighborhood with allowlisted graph types."""

    _ensure_internal_read(ctx, project_id)
    result = await MemoryDbReader().get_memory_graph(ctx, memory_id, limit=limit)
    return InternalMemoryGraphResponse(request_id=ctx.request_id, data=result)


@router.get(
    "/projects/{project_id}/graph/neighbors",
    response_model=InternalMemoryGraphResponse,
)
async def get_project_node_graph(
    project_id: str = Path(min_length=1),
    node_type: Literal["memory", "entity", "source"] = Query(),
    node_id: str = Query(min_length=1),
    limit: int = Query(default=100, ge=1, le=2000),
    cursor: str | None = Query(default=None, min_length=1, max_length=1024),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> InternalMemoryGraphResponse:
    """Return one read-only page of an allowlisted node's stored neighbors."""

    _ensure_internal_read(ctx, project_id)
    try:
        result = await MemoryDbReader().get_node_graph(
            ctx,
            node_id,
            node_type=node_type,
            limit=limit,
            cursor=cursor,
        )
    except ValueError as exc:
        raise BadRequestError(str(exc), code="memory.graph_cursor_invalid") from exc
    return InternalMemoryGraphResponse(request_id=ctx.request_id, data=result)


@router.post("/projects/{project_id}/vector-repairs", response_model=VectorRepairResponse)
async def repair_project_vectors(
    payload: VectorRepairRequest,
    project_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> VectorRepairResponse:
    """Repair one bounded project batch without exposing it publicly."""

    _ensure_internal_memory_write(ctx, project_id)
    result = await get_vector_repair_service().repair(to_memory_request_context(ctx), payload)
    return VectorRepairResponse(request_id=ctx.request_id, data=result)


@router.get("/projects/{project_id}/vector-repairs/status", response_model=VectorRepairStatusResponse)
async def get_project_vector_repair_status(
    project_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> VectorRepairStatusResponse:
    """Return aggregate vector-repair lifecycle counts for one project."""

    _ensure_internal_read(ctx, project_id)
    result = await get_vector_repair_service().status(to_memory_request_context(ctx))
    return VectorRepairStatusResponse(request_id=ctx.request_id, data=result)


@router.post("/projects/{project_id}/provider-bindings", response_model=ProviderBindingResponse)
async def create_project_provider_binding(
    payload: ProviderBindingCreateRequest,
    project_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> ProviderBindingResponse:
    """Create or replace one project provider binding for its scope."""

    _ensure_internal_write(ctx, project_id)
    result = await get_provider_binding_service().create_binding(
        project_id=project_id,
        scope=payload.scope,
        routers=payload.routers,
        request_id=ctx.request_id,
    )
    return ProviderBindingResponse(request_id=ctx.request_id, data=result)


@router.patch("/projects/{project_id}/provider-bindings/{binding_id}", response_model=ProviderBindingResponse)
async def patch_project_provider_binding(
    payload: ProviderBindingPatchRequest,
    project_id: str = Path(min_length=1),
    binding_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> ProviderBindingResponse:
    """Patch one project provider binding."""

    _ensure_internal_write(ctx, project_id)
    result = await get_provider_binding_service().patch_binding(
        project_id=project_id,
        binding_id=binding_id,
        routers=payload.routers,
        request_id=ctx.request_id,
    )
    return ProviderBindingResponse(request_id=ctx.request_id, data=result)


@router.get("/projects/{project_id}/provider-bindings", response_model=ProviderBindingListResponse)
async def list_project_provider_bindings(
    project_id: str = Path(min_length=1),
    ctx: AuthContext = Depends(get_internal_request_context),
) -> ProviderBindingListResponse:
    """List dynamic provider bindings for one project."""

    _ensure_internal_provider_read(ctx, project_id)
    items = await get_provider_binding_service().list_bindings(project_id=project_id)
    return ProviderBindingListResponse(request_id=ctx.request_id, data={"items": items})


def _ensure_internal_read(ctx: AuthContext, project_id: str) -> None:
    ensure_scopes(ctx, ("memory:read",))
    _ensure_project_scope(ctx, project_id)


def _ensure_internal_provider_read(ctx: AuthContext, project_id: str) -> None:
    ensure_scopes(ctx, ("provider:read",))
    _ensure_project_scope(ctx, project_id)


def _ensure_internal_write(ctx: AuthContext, project_id: str) -> None:
    ensure_scopes(ctx, ("provider:write",))
    _ensure_project_scope(ctx, project_id)


def _ensure_internal_memory_write(ctx: AuthContext, project_id: str) -> None:
    ensure_scopes(ctx, ("memory:write",))
    _ensure_project_scope(ctx, project_id)


def _ensure_project_scope(ctx: AuthContext, project_id: str) -> None:
    if ctx.project_id != project_id:
        raise PermissionDeniedError("project scope mismatch", code="auth.project_scope_mismatch")


def _record_to_item(record: QdrantRecord) -> dict[str, Any]:
    item = dict(record.payload)
    item.setdefault("id", str(record.point_id))
    item.setdefault("memory_id", str(item.get("memory_id") or record.point_id))
    return item


def _inventory_filter(*, q: str | None, status: str | None, algorithm: str | None, mem_type: str | None):
    conditions = []
    text = q.strip() if isinstance(q, str) else ""
    if text:
        conditions.append(match_text("content", text))
    if status:
        conditions.append(match_value("status", status))
    if algorithm:
        conditions.append(match_value("mem_extract_type", algorithm))
    if mem_type:
        conditions.append(match_value("mem_type", mem_type))
    return build_filter(must=conditions) if conditions else None
