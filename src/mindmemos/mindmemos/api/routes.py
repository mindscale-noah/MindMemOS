"""Memory HTTP routes.

Each handler depends on :func:`get_request_context` (auth + context assembly)
and :func:`get_memory_service` (business logic). Handlers stay thin: validate
input via the typed request body, delegate to the service, wrap the result in
:class:`ApiResponse`.
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Query
from starlette.responses import Response, StreamingResponse

from .deps import get_image_access_context, require_scopes
from .mappers import (
    to_add_api_response,
    to_memory_list_api_response,
    to_memory_page_api_response,
    to_memory_scroll_api_response,
    to_status_api_response,
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
from .services import MemoryService, get_memory_service

router = APIRouter(
    prefix="/v1/memory",
    tags=["memory"],
)


AddResponse = ApiResponse[AddData]
GetResponse = ApiResponse[MemoryListData]
SearchResponse = ApiResponse[MemoryListData]
ListResponse = ApiResponse[MemoryPageData]
ScrollResponse = ApiResponse[MemoryScrollData]
DeleteResponse = ApiResponse[None]
UpdateResponse = ApiResponse[None]
FeedbackResponse = ApiResponse[None]
DreamingResponse = ApiResponse[None]
SCOPE_MEM_WRITE = "memory:write"
SCOPE_MEM_READ = "memory:read"


@router.post("/add", response_model=AddResponse)
async def add_memory(
    payload: AddRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> AddResponse:
    result = await service.add(auth, payload)
    return to_add_api_response(result, auth.request_id)


@router.post("/add/stream")
async def add_memory_stream(
    payload: AddRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> StreamingResponse:
    """Stream sync add progress as Server-Sent Events."""

    async def stream():
        async for item in service.add_stream(auth, payload, cancel_check=None):
            event_name = str(item.get("event") or "progress")
            data = {key: value for key, value in item.items() if key != "event"}
            yield f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/get", response_model=GetResponse)
async def get_memory(
    payload: GetRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: MemoryService = Depends(get_memory_service),
) -> GetResponse:
    result = await service.get(auth, payload)
    return to_memory_list_api_response(result, auth.request_id)


@router.post("/list", response_model=ListResponse)
async def list_memory(
    payload: MemoryPageRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: MemoryService = Depends(get_memory_service),
) -> ListResponse:
    result = await service.list(auth, payload)
    return to_memory_page_api_response(result, auth.request_id)


@router.post("/scroll", response_model=ScrollResponse)
async def scroll_memory(
    payload: MemoryScrollRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: MemoryService = Depends(get_memory_service),
) -> ScrollResponse:
    result = await service.scroll(auth, payload)
    return to_memory_scroll_api_response(result, auth.request_id)


@router.post("/delete", response_model=DeleteResponse)
async def delete_memory(
    payload: DeleteRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> DeleteResponse:
    result = await service.delete(auth, payload)
    return to_status_api_response(result, auth.request_id)


@router.post("/update", response_model=UpdateResponse)
async def update_memory(
    payload: UpdateRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> UpdateResponse:
    result = await service.update(auth, payload)
    return to_status_api_response(result, auth.request_id)


@router.post("/search", response_model=SearchResponse)
async def search_memory(
    payload: SearchRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_READ)),
    service: MemoryService = Depends(get_memory_service),
) -> SearchResponse:
    result = await service.search(auth, payload)
    return to_memory_list_api_response(result, auth.request_id)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_memory(
    payload: FeedbackRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> FeedbackResponse:
    result = await service.feedback(auth, payload)
    return to_status_api_response(result, auth.request_id)


@router.post("/dreaming", response_model=DreamingResponse)
async def dreaming_memory(
    payload: DreamingRequest,
    auth: AuthContext = Depends(require_scopes(SCOPE_MEM_WRITE)),
    service: MemoryService = Depends(get_memory_service),
) -> DreamingResponse:
    result = await service.dream(auth, payload)
    return to_status_api_response(result, auth.request_id)


@router.get("/images/{blob_id}", response_class=Response)
async def get_memory_image(
    blob_id: str,
    user_id: str = Query(min_length=1),
    project_id: str | None = Query(default=None, description="Signed-URL mode: project the signature was minted for."),
    expires: int | None = Query(default=None, description="Signed-URL mode: unix seconds until the link dies."),
    signature: str | None = Query(default=None, description="Signed-URL mode: HMAC over project/blob/owner/expiry."),
    auth: AuthContext = Depends(get_image_access_context),
    service: MemoryService = Depends(get_memory_service),
) -> Response:
    """Return the raw bytes of an image stored for this project (multimodal algorithms).

    Two authentication shapes: ``Authorization: Bearer`` key with ``memory:read`` (the SDK's
    ``get_image``), or — with no header — a valid unexpired ``project_id`` / ``expires`` /
    ``signature`` triple, i.e. exactly the short-lived link a search hit's ``metadata.image_url``
    carries in its signed-proxy shape (all three of ``metadata.image_url``'s shapes — public
    mirror, presigned, signed proxy — are directly fetchable without an API key; the signed
    proxy one is relative until ``blob_store.proxy_public_base_url`` is set). ``user_id`` must
    name the image's owner (``metadata.user_id`` of a search image result); it is verified
    against the project's image records, so another user's blob id is not readable (404
    image.not_found).
    """

    data, mime_type = await service.get_image(auth, blob_id, user_id)
    # a signed link must not outlive its signature in the receiving browser's cache
    max_age = 86400
    if signature is not None and expires is not None:
        max_age = max(0, min(86400, expires - int(time.time())))
    return Response(content=data, media_type=mime_type, headers={"Cache-Control": f"private, max-age={max_age}"})
