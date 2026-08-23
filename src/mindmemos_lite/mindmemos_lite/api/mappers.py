"""Map HTTP DTOs to transport-neutral service commands and results."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from ..service.schema import (
    AddMemoryRequest,
    AddMemoryResult,
    DeleteMemoryRequest,
    DialogueMessage,
    DreamingMemoryRequest,
    FeedbackMemoryRequest,
    FileMessage,
    GetMemoryRequest,
    MemoryItem,
    MemoryListResult,
    MemoryMutationResult,
    RequestContext,
    SearchMemoryRequest,
    SkillContext,
    TextMessage,
    UpdateMemoryRequest,
    UrlMessage,
)
from .schemas import (
    ActorIdentityRequest,
    AddData,
    AddRequest,
    ApiResponse,
    DeleteRequest,
    DialogueMessageInput,
    DreamingRequest,
    FeedbackRequest,
    FileMessageInput,
    GetRequest,
    MemoryAddEventResponse,
    MemoryItemResponse,
    MemoryLineageResponse,
    MemoryListData,
    SearchRequest,
    TaskEntityResponse,
    TaskSearchGroupData,
    TextMessageInput,
    UpdateRequest,
    UrlMessageInput,
)


def with_actor_identity(context: RequestContext, payload: ActorIdentityRequest | None) -> RequestContext:
    if payload is None:
        return context
    return replace(
        context,
        user_id=payload.user_id,
        app_id=payload.app_id,
        session_id=payload.session_id,
        agent_id=payload.agent_id,
    )


def to_add_command(payload: AddRequest) -> AddMemoryRequest:
    return AddMemoryRequest(
        messages=tuple(_to_message(message) for message in payload.messages),
        mode=payload.mode,
        infer=payload.infer,
        metadata=payload.metadata,
        skill_context=tuple(
            SkillContext(
                name=item.name,
                content_hash=item.content_hash,
                base_version_id=item.base_version_id,
                version_label=item.version_label,
                usage=item.usage,
            )
            for item in payload.skill_context
        ),
        score=payload.score,
        task_id=payload.task_id,
        task=payload.task,
    )


def to_search_command(payload: SearchRequest) -> SearchMemoryRequest:
    return SearchMemoryRequest(
        query=payload.query,
        memory_mode=payload.memory_mode,
        filters=payload.filters,
        top_k=payload.top_k,
        search_strategy=payload.search_strategy,
        rerank=payload.rerank,
        score_threshold=payload.score_threshold,
        max_rounds=payload.max_rounds,
        task_top_k=payload.task_top_k,
    )


def to_get_command(payload: GetRequest) -> GetMemoryRequest:
    return GetMemoryRequest(filters=payload.filters, top_k=payload.top_k)


def to_delete_command(payload: DeleteRequest) -> DeleteMemoryRequest:
    return DeleteMemoryRequest(memory_id=payload.memory_id)


def to_update_command(payload: UpdateRequest) -> UpdateMemoryRequest:
    return UpdateMemoryRequest(memory_id=payload.memory_id, content=payload.content)


def to_feedback_command(payload: FeedbackRequest) -> FeedbackMemoryRequest:
    return FeedbackMemoryRequest(
        feedback=payload.feedback,
        messages=tuple(_to_message(message) for message in payload.messages),
        recalled_memories=tuple(
            MemoryItem(
                memory_id=item.memory_id,
                content=item.content,
                memory_type=item.memory_type,
                updated_at=_parse_datetime(item.last_update_at),
                event_time=_parse_datetime(item.event_time),
                source_timestamp=_parse_datetime(item.source_timestamp),
            )
            for item in payload.recalled_memories
        ),
        mode=payload.mode,
    )


def to_dreaming_command(payload: DreamingRequest) -> DreamingMemoryRequest:
    return DreamingMemoryRequest(mode=payload.mode)


def to_add_response(result: AddMemoryResult, request_id: str) -> ApiResponse[AddData]:
    return ApiResponse[AddData](
        code=result.status,
        request_id=request_id,
        data=AddData(
            memories=[
                MemoryAddEventResponse(
                    operation=item.operation,
                    content=item.content,
                    memory_id=item.memory_id,
                    memory_type=item.memory_type,
                    confidence=item.confidence,
                    related_memory_ids=list(item.related_memory_ids),
                    graph_edge_count=item.graph_edge_count,
                )
                for item in result.memories
            ]
        ),
    )


def to_memory_list_response(result: MemoryListResult, request_id: str) -> ApiResponse[MemoryListData]:
    return ApiResponse[MemoryListData](
        code=result.status,
        message=result.message or "",
        request_id=request_id,
        data=MemoryListData(
            memories=[_to_memory_item(item) for item in result.memories],
            task=(
                TaskEntityResponse(entity_id=result.task_id, entity_name=result.task_name or "", entity_type="task")
                if result.task_id
                else None
            ),
            tasks=[
                TaskSearchGroupData(
                    task=TaskEntityResponse(
                        entity_id=group.task_id, entity_name=group.task_name or "", entity_type="task"
                    ),
                    memories=[_to_memory_item(item) for item in group.memories],
                )
                for group in result.tasks
            ],
        ),
    )


def to_status_response(result: MemoryMutationResult, request_id: str) -> ApiResponse[None]:
    return ApiResponse[None](
        code=result.status,
        message=result.message or "",
        request_id=request_id,
        data=None,
    )


def _to_message(message):
    if isinstance(message, DialogueMessageInput):
        return DialogueMessage(role=message.role, content=message.content, timestamp=message.timestamp)
    if isinstance(message, UrlMessageInput):
        return UrlMessage(url=message.url)
    if isinstance(message, FileMessageInput):
        return FileMessage(file_name=message.file_name, file_path=message.file_path, file_type=message.file_type)
    if isinstance(message, TextMessageInput):
        return TextMessage(text=message.text)
    raise TypeError(f"unsupported HTTP message: {type(message).__name__}")


def _to_memory_item(item: MemoryItem) -> MemoryItemResponse:
    lineage = item.lineage
    return MemoryItemResponse(
        id=item.memory_id,
        memory=item.content,
        memory_type=item.memory_type,
        last_update_at=_format_datetime(item.updated_at),
        event_time=_format_datetime(item.event_time),
        source_timestamp=_format_datetime(item.source_timestamp),
        lineage=(
            MemoryLineageResponse(
                role=lineage.role,
                derived_from_memory_ids=list(lineage.derived_from_memory_ids),
                derived_to_memory_ids=list(lineage.derived_to_memory_ids),
            )
            if lineage is not None
            else None
        ),
    )


def _format_datetime(value: datetime | None) -> str | None:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else None


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S") if value else None


__all__ = [
    "to_add_command",
    "to_add_response",
    "to_delete_command",
    "to_dreaming_command",
    "to_feedback_command",
    "to_get_command",
    "to_memory_list_response",
    "to_search_command",
    "to_status_response",
    "to_update_command",
    "with_actor_identity",
]
