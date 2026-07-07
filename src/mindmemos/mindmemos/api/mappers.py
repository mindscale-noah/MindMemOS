"""Adapters between HTTP request models and pipeline contracts.

Actor identity (user_id / app_id / session_id / agent_id) arrives from request
bodies on add/search/feedback/dreaming. These helpers split each request model
into:

1. the pure business pipeline input (``mindmemos.typing.service``), and
2. a :class:`~mindmemos.typing.memory.MemoryRequestContext` assembled from the
   security-only :class:`~mindmemos.api.schemas.AuthContext` plus actor fields
   (``to_memory_request_context``).

Routes stay thin; the split happens in the service layer. Request models do not
inherit the pipeline input DTOs, so the field mapping is made explicit here.
"""

from __future__ import annotations

from ..config import get_config
from ..errors import BadRequestError
from ..typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    DeletePipelineInput,
    DeletePipelineResult,
    DreamingPipelineInput,
    DreamingPipelineResult,
    FeedbackPipelineInput,
    FeedbackPipelineResult,
    GetPipelineInput,
    GetPipelineResult,
    MemoryListPipelineInput,
    MemoryListPipelineResult,
    MemoryRequestContext,
    MemoryScrollPipelineInput,
    MemoryScrollPipelineResult,
    SearchPipelineInput,
    SearchPipelineResult,
    UpdatePipelineInput,
    UpdatePipelineResult,
)
from .schemas import (
    AddData,
    AddRequest,
    ApiResponse,
    AuthContext,
    DeleteRequest,
    DreamingRequest,
    FeedbackRequest,
    GetRequest,
    MemoryListData,
    MemoryPageData,
    MemoryPageRequest,
    MemoryScrollData,
    MemoryScrollRequest,
    SearchRequest,
    UpdateRequest,
)

# Actor identity fields carried by request models but owned by MemoryRequestContext.
_ACTOR_FIELDS = ("user_id", "app_id", "session_id", "agent_id")
_SKILL_FIELDS = ("skill_context", "score", "task_id")
MAX_MEMORY_PAGE_SIZE = 100
MAX_MEMORY_LIST_OFFSET = 10000
MAX_MEMORY_SCROLL_LIMIT = 500


def to_add_pipeline_input(req: AddRequest) -> AddPipelineInput:
    """Build pure add pipeline input from a public add request.

    Args:
        req: Public add request model.

    Returns:
        Add pipeline input with actor fields and trace annotations removed.
    """

    return AddPipelineInput.model_validate(
        req.model_dump(
            by_alias=True,
            exclude={*_ACTOR_FIELDS, *_SKILL_FIELDS},
            exclude_none=True,
        )
    )


def to_search_pipeline_input(
    req: SearchRequest,
    *,
    search_pipeline: str | None = None,
    search_pipline: str | None = None,
) -> SearchPipelineInput:
    """Build pure search pipeline input from a public search request."""

    _validate_request_top_k(req.top_k)
    data = req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS) | {"search_strategy"})
    data["search_pipeline"] = search_pipeline or search_pipline
    data["agentic"] = req.search_strategy == "agentic"
    return SearchPipelineInput.model_validate(data)


def _validate_request_top_k(top_k: int | None) -> None:
    if top_k is None:
        return
    top_k_max = get_config().algo_config.search.request_top_k_max
    if top_k > top_k_max:
        raise BadRequestError(
            f"top_k must be <= {top_k_max}; value={top_k}",
            code="search.top_k_too_large",
        )


def to_get_pipeline_input(req: GetRequest) -> GetPipelineInput:
    """Build get pipeline input from a public get request."""

    return GetPipelineInput.model_validate(req.model_dump(by_alias=True))


def to_memory_list_pipeline_input(req: MemoryPageRequest) -> MemoryListPipelineInput:
    """Build paged memory list pipeline input from a public list request."""

    _validate_memory_page(req.page, req.page_size)
    return MemoryListPipelineInput.model_validate(req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS)))


def to_memory_scroll_pipeline_input(req: MemoryScrollRequest) -> MemoryScrollPipelineInput:
    """Build cursor memory scroll pipeline input from a public scroll request."""

    if req.limit > MAX_MEMORY_SCROLL_LIMIT:
        raise BadRequestError(
            f"limit must be <= {MAX_MEMORY_SCROLL_LIMIT}; value={req.limit}",
            code="memory.scroll_limit_too_large",
        )
    return MemoryScrollPipelineInput.model_validate(req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS)))


def _validate_memory_page(page: int, page_size: int) -> None:
    if page_size > MAX_MEMORY_PAGE_SIZE:
        raise BadRequestError(
            f"page_size must be <= {MAX_MEMORY_PAGE_SIZE}; value={page_size}",
            code="memory.page_size_too_large",
        )
    offset = (page - 1) * page_size
    if offset > MAX_MEMORY_LIST_OFFSET:
        raise BadRequestError(
            f"(page - 1) * page_size must be <= {MAX_MEMORY_LIST_OFFSET}; value={offset}",
            code="memory.page_offset_too_large",
        )


def to_delete_pipeline_input(req: DeleteRequest) -> DeletePipelineInput:
    """Build delete pipeline input from a public delete request."""

    return DeletePipelineInput.model_validate(req.model_dump(by_alias=True))


def to_update_pipeline_input(req: UpdateRequest) -> UpdatePipelineInput:
    """Build update pipeline input from a public update request."""

    return UpdatePipelineInput.model_validate(req.model_dump(by_alias=True))


def to_feedback_pipeline_input(req: FeedbackRequest) -> FeedbackPipelineInput:
    """Build feedback pipeline input from a public feedback request."""

    return FeedbackPipelineInput.model_validate(
        req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS))
    )


def to_dreaming_pipeline_input(req: DreamingRequest) -> DreamingPipelineInput:
    """Build dreaming pipeline input from a public dreaming request."""

    return DreamingPipelineInput.model_validate(req.model_dump(by_alias=True, exclude=set(_ACTOR_FIELDS)))


def to_memory_request_context(
    auth: AuthContext,
    identity: AddRequest
    | SearchRequest
    | FeedbackRequest
    | DreamingRequest
    | MemoryPageRequest
    | MemoryScrollRequest
    | None = None,
    *,
    require_user_id: bool = False,
) -> MemoryRequestContext:
    """Build pipeline request context from auth and optional actor identity.

    Args:
        auth: Security context resolved by API dependencies.
        identity: Optional request model carrying actor identity fields.
        require_user_id: Whether the merged actor identity must include ``user_id``.

    Returns:
        The resolved memory request context.
    """

    resolved_actor = {field: getattr(identity, field, None) for field in _ACTOR_FIELDS} if identity is not None else {}
    if require_user_id and not resolved_actor.get("user_id"):
        raise BadRequestError("user_id is required in request body", code="missing_user_id")
    return MemoryRequestContext(
        request_id=auth.request_id,
        account_id=auth.account_id,
        project_id=auth.project_id,
        api_key_uuid=auth.api_key_uuid,
        memory_algorithm=auth.memory_algorithm,
        scopes=auth.scopes,
        **resolved_actor,
    )


# Flatten the pipeline's ``status`` / ``message`` into the ``ApiResponse``
# envelope (``code`` / ``message``) so ``data`` carries domain data only and the
# success signal is not duplicated across two layers.


def to_add_api_response(
    result: AddPipelineSyncResult | AddPipelineAsyncResult,
    request_id: str | None,
) -> ApiResponse[AddData]:
    """Convert an add pipeline result into an HTTP response envelope."""

    return ApiResponse[AddData](
        code=result.status,
        request_id=request_id,
        data=AddData(memories=result.memories),
    )


def to_memory_list_api_response(
    result: SearchPipelineResult | GetPipelineResult,
    request_id: str | None,
) -> ApiResponse[MemoryListData]:
    """Convert a search or get pipeline result into an HTTP response envelope."""

    return ApiResponse[MemoryListData](
        code=result.status,
        message=getattr(result, "message", None) or "",
        request_id=request_id,
        data=MemoryListData(memories=result.memories),
    )


def to_memory_page_api_response(
    result: MemoryListPipelineResult,
    request_id: str | None,
) -> ApiResponse[MemoryPageData]:
    """Convert a paged list pipeline result into an HTTP response envelope."""

    return ApiResponse[MemoryPageData](
        code=result.status,
        message=result.message or "",
        request_id=request_id,
        data=MemoryPageData(
            memories=result.memories,
            page=result.page,
            page_size=result.page_size,
            total=result.total,
            has_more=result.has_more,
        ),
    )


def to_memory_scroll_api_response(
    result: MemoryScrollPipelineResult,
    request_id: str | None,
) -> ApiResponse[MemoryScrollData]:
    """Convert a cursor scroll pipeline result into an HTTP response envelope."""

    return ApiResponse[MemoryScrollData](
        code=result.status,
        message=result.message or "",
        request_id=request_id,
        data=MemoryScrollData(memories=result.memories, next_cursor=result.next_cursor),
    )


def to_status_api_response(
    result: DeletePipelineResult | UpdatePipelineResult | FeedbackPipelineResult | DreamingPipelineResult,
    request_id: str | None,
) -> ApiResponse[None]:
    """Convert a status-only pipeline result into an HTTP response envelope."""

    return ApiResponse[None](
        code=result.status,
        message=getattr(result, "message", None) or "",
        request_id=request_id,
        data=None,
    )
