"""Project-scoped cloud Skill v2 HTTP routes."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Path, Query
from mindmemos_skill.contracts import SkillTrajectory

from .deps import require_scopes
from .schemas import ApiResponse, AuthContext
from .services import SkillService, get_skill_service
from .skill_schemas import (
    SkillContentData,
    SkillEvolveData,
    SkillEvolveRequest,
    SkillListData,
    SkillRegisterData,
    SkillRegisterRequest,
    SkillRemoteSyncData,
    SkillRemoteSyncRequest,
    SkillSummaryData,
    SkillTrajectoryPageData,
    SkillTrajectoryReportData,
    SkillTrajectoryReportRequest,
    SkillVersionsData,
    SkillVersionStatusRequest,
)

router = APIRouter(prefix="/v1/skills", tags=["skills"])

SkillRegisterResponse = ApiResponse[SkillRegisterData]
SkillListResponse = ApiResponse[SkillListData]
SkillDetailResponse = ApiResponse[SkillSummaryData]
SkillVersionsResponse = ApiResponse[SkillVersionsData]
SkillContentResponse = ApiResponse[SkillContentData]
SkillEvolveResponse = ApiResponse[SkillEvolveData]
SkillRemoteSyncResponse = ApiResponse[SkillRemoteSyncData]
SkillTrajectoryReportResponse = ApiResponse[SkillTrajectoryReportData]
SkillTrajectoryPageResponse = ApiResponse[SkillTrajectoryPageData]
SkillTrajectoryResponse = ApiResponse[SkillTrajectory]

SCOPE_SKILL_READ = "skills:read"
SCOPE_SKILL_WRITE = "skills:write"
SCOPE_TRAJECTORY_READ = "skills:trajectory:read"
SCOPE_TRAJECTORY_WRITE = "skills:trajectory:write"
SCOPE_SKILL_EVOLVE = "skills:evolve"


@router.get("", response_model=SkillListResponse)
async def list_skills(
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillListResponse:
    return SkillListResponse(code="ok", request_id=auth.request_id, data=await service.list_skills(auth))


@router.post("/register", response_model=SkillRegisterResponse)
async def register_skill(
    payload: SkillRegisterRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_WRITE)),
    service: SkillService = Depends(get_skill_service),
) -> SkillRegisterResponse:
    return SkillRegisterResponse(code="ok", request_id=auth.request_id, data=await service.register(auth, payload))


@router.post("/sync", response_model=SkillRemoteSyncResponse)
async def sync_skills(
    payload: SkillRemoteSyncRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillRemoteSyncResponse:
    return SkillRemoteSyncResponse(code="ok", request_id=auth.request_id, data=await service.sync_remote(auth, payload))


@router.post("/trajectories", response_model=SkillTrajectoryReportResponse)
async def report_trajectories(
    payload: SkillTrajectoryReportRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_TRAJECTORY_WRITE)),
    service: SkillService = Depends(get_skill_service),
) -> SkillTrajectoryReportResponse:
    data = await service.report_trajectories(auth, payload)
    return SkillTrajectoryReportResponse(code="ok", request_id=auth.request_id, data=data)


@router.post("/evolve", response_model=SkillEvolveResponse)
async def evolve_skill(
    payload: SkillEvolveRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_EVOLVE)),
    service: SkillService = Depends(get_skill_service),
) -> SkillEvolveResponse:
    data = await service.evolve(auth, payload)
    return SkillEvolveResponse(code=data.status, request_id=auth.request_id, data=data)


@router.get("/trajectories", response_model=SkillTrajectoryPageResponse)
async def list_trajectories(
    cloud_skill_id: str = Query(min_length=1),
    version_id: str | None = Query(default=None),
    since: datetime | None = Query(default=None),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    status: str | None = Query(default=None),
    min_score: float | None = Query(default=None),
    include_events: bool = Query(default=True),
    auth: AuthContext = Depends(require_scopes(SCOPE_TRAJECTORY_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillTrajectoryPageResponse:
    data = await service.list_trajectories(
        auth,
        cloud_skill_id=cloud_skill_id,
        version_id=version_id,
        since=since,
        cursor=cursor,
        limit=limit,
        status=status,
        min_score=min_score,
    )
    if not include_events:
        data = data.model_copy(
            update={"items": [item.model_copy(update={"trajectory": []}) for item in data.items]}
        )
    return SkillTrajectoryPageResponse(code="ok", request_id=auth.request_id, data=data)


@router.get("/trajectories/{trajectory_id}", response_model=SkillTrajectoryResponse)
async def get_trajectory(
    trajectory_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_TRAJECTORY_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillTrajectoryResponse:
    return SkillTrajectoryResponse(
        code="ok",
        request_id=auth.request_id,
        data=await service.get_trajectory(auth, trajectory_id),
    )


@router.post("/versions/{version_id}/status", response_model=SkillRegisterResponse)
async def update_version_status(
    payload: SkillVersionStatusRequest,
    version_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_WRITE)),
    service: SkillService = Depends(get_skill_service),
) -> SkillRegisterResponse:
    return SkillRegisterResponse(
        code="ok",
        request_id=auth.request_id,
        data=await service.update_status(auth, version_id, payload),
    )


@router.get("/{cloud_skill_id}", response_model=SkillDetailResponse)
async def get_skill(
    cloud_skill_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillDetailResponse:
    return SkillDetailResponse(code="ok", request_id=auth.request_id, data=await service.get_skill(auth, cloud_skill_id))


@router.get("/{cloud_skill_id}/versions", response_model=SkillVersionsResponse)
async def list_versions(
    cloud_skill_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillVersionsResponse:
    return SkillVersionsResponse(
        code="ok",
        request_id=auth.request_id,
        data=await service.versions(auth, cloud_skill_id),
    )


@router.get("/{cloud_skill_id}/versions/{version_id}/content", response_model=SkillContentResponse)
async def get_version_content(
    cloud_skill_id: str = Path(min_length=1),
    version_id: str = Path(min_length=1),
    auth: AuthContext = Depends(require_scopes(SCOPE_SKILL_READ)),
    service: SkillService = Depends(get_skill_service),
) -> SkillContentResponse:
    return SkillContentResponse(
        code="ok",
        request_id=auth.request_id,
        data=await service.content(auth, cloud_skill_id, version_id),
    )


__all__ = ["router"]
