"""Common memory-service orchestration independent of an algorithm pipeline.

Algorithm pipelines only execute synchronous add/search logic.  This base
service owns operation recording, async task submission, worker lifecycle
writeback, and conversion between the public service and pipeline contracts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from ..infra.tasking import TaskClient, TaskEnvelope
from ..logging import traced
from ..persistence import MemoryOperationRecorder
from ..persistence.memory import MemoryPersistence
from ..pipeline import AddPipeline, SearchPipeline
from ..typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    DialogueMessage,
    FileMessage,
    MemoryRequestContext,
    SearchPipelineInput,
    SearchPipelineResult,
    TextMessage,
    UrlMessage,
)
from .recording import suppress_recording_errors
from .schema import (
    AddMemoryRequest,
    AddMemoryResult,
    MemoryAddEvent,
    MemoryItem,
    MemoryLineage,
    MemoryListResult,
    MemoryTaskGroup,
    RequestContext,
    SearchMemoryRequest,
)

MEMORY_ADD_TASK = "memory.add"


class BaseMemoryService(ABC):
    """Default add/search service orchestration shared by all algorithms."""

    _default_add_pipeline_requires_task: bool = False
    """Whether the effective default add pipeline requires a non-empty ``task``.

    Subclasses resolve this from the configured pipeline's capabilities during
    construction; ``add()`` enforces it for inferred adds. Declared at class
    level so instances built without ``__init__`` (e.g. test doubles) default
    to not requiring a task.
    """

    def __init__(
        self,
        persistence: MemoryPersistence,
        *,
        add_pipeline: AddPipeline,
        search_pipeline: SearchPipeline,
        task_client: TaskClient | None = None,
        direct_add_pipeline: AddPipeline | None = None,
        direct_add_pipeline_factory: Callable[[], AddPipeline] | None = None,
        recorder: MemoryOperationRecorder | None = None,
    ) -> None:
        self._operation_recorder = recorder or MemoryOperationRecorder.from_service(persistence.service)
        self._algorithm_add_pipeline = add_pipeline
        self._algorithm_search_pipeline = search_pipeline
        self._direct_add_pipeline = direct_add_pipeline
        self._direct_add_pipeline_factory = direct_add_pipeline_factory
        self._memory_task_client = task_client
        if task_client is not None and MEMORY_ADD_TASK not in task_client.handlers.names():
            task_client.handlers.register(MEMORY_ADD_TASK, self._handle_add_task)

    @property
    @abstractmethod
    def search_pipeline_name(self) -> str:
        """Return the search strategy name persisted on search records."""

    @traced("memory.service.add")
    async def add(self, context: RequestContext, request: AddMemoryRequest) -> AddMemoryResult:
        """Run add with the default operation-record and async-task lifecycle."""

        if request.infer and self._default_add_pipeline_requires_task and not (request.task or "").strip():
            raise ValueError(
                "the configured default add pipeline requires a non-empty 'task' "
                "(set pipelines.default_add_pipeline, or pass task= on the request)"
            )

        pipeline_context = self.to_pipeline_context(context)
        messages = [_pipeline_message(message) for message in request.messages]
        if not request.infer:
            messages = [TextMessage(text="\n".join(_raw_message_text(message) for message in messages))]
        payload = AddPipelineInput(
            messages=messages,
            mode=request.mode,
            metadata=dict(request.metadata),
            task=request.task,
        )
        add_record_id = str(uuid4())
        request_submitted_at = _utcnow()
        try:
            await suppress_recording_errors(
                self._operation_recorder.record_add_input(
                    payload,
                    ctx=pipeline_context,
                    request_submitted_at=request_submitted_at,
                    add_record_id=add_record_id,
                    status="queued" if request.mode == "async" else "processing",
                ),
                operation="add",
            )
            if request.mode == "async":
                result: AddPipelineSyncResult | AddPipelineAsyncResult = await self._submit_add_task(
                    payload,
                    pipeline_context,
                    add_record_id=add_record_id,
                    infer=request.infer,
                )
            else:
                result = await self._run_add_sync(
                    payload,
                    pipeline_context,
                    add_record_id=add_record_id,
                    infer=request.infer,
                )
        except Exception as exc:
            await suppress_recording_errors(
                self._operation_recorder.mark_add_failed(pipeline_context, add_record_id, str(exc)),
                operation="add",
            )
            raise
        return _add_service_result(result)

    @traced("memory.service.search")
    async def search(self, context: RequestContext, request: SearchMemoryRequest) -> MemoryListResult:
        """Run search and persist the exact query/result snapshot."""

        pipeline_context = self.to_pipeline_context(context)
        memory_mode = request.memory_mode or pipeline_context.memory_algorithm or self.search_pipeline_name
        pipeline_context = pipeline_context.model_copy(update={"memory_algorithm": memory_mode})
        payload = SearchPipelineInput(
            query=request.query,
            filters=dict(request.filters) if request.filters is not None else None,
            top_k=request.top_k,
            search_pipeline=memory_mode,
            memory_mode=memory_mode,
            rerank=request.rerank,
            score_threshold=request.score_threshold,
            agentic=request.search_strategy == "agentic",
            max_rounds=request.max_rounds,
            task_top_k=request.task_top_k,
        )
        request_submitted_at = _utcnow()
        try:
            result = await self._algorithm_search_pipeline.search(payload, pipeline_context)
        except Exception as exc:
            await suppress_recording_errors(
                self._operation_recorder.record_search(
                    payload,
                    None,
                    ctx=pipeline_context,
                    request_submitted_at=request_submitted_at,
                    task_completed_at=_utcnow(),
                    error=str(exc),
                ),
                operation="search",
            )
            raise
        await suppress_recording_errors(
            self._operation_recorder.record_search(
                payload,
                result,
                ctx=pipeline_context,
                request_submitted_at=request_submitted_at,
                task_completed_at=_utcnow(),
            ),
            operation="search",
        )
        return _search_service_result(result)

    async def _run_add_sync(
        self,
        payload: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str,
        infer: bool,
    ) -> AddPipelineSyncResult:
        result = await self._get_add_pipeline(infer=infer).add_sync(payload, context)
        await suppress_recording_errors(
            self._operation_recorder.mark_add_completed(context, add_record_id, result),
            operation="add.sync",
        )
        return result

    async def _submit_add_task(
        self,
        payload: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str,
        infer: bool,
    ) -> AddPipelineAsyncResult:
        if self._memory_task_client is None:
            raise RuntimeError("async add requires a task backend configured on the memory service")
        await self._memory_task_client.submit(
            MEMORY_ADD_TASK,
            {
                "context": context.model_dump(),
                "input": payload.model_dump(by_alias=True),
                "add_record_id": add_record_id,
                "infer": infer,
            },
            dispatch_key=_memory_add_dispatch_key(context),
        )
        return AddPipelineAsyncResult(status="queued")

    async def _handle_add_task(self, task: TaskEnvelope) -> None:
        context = MemoryRequestContext.model_validate(task.payload["context"])
        payload = AddPipelineInput.model_validate(task.payload["input"])
        add_record_id = str(task.payload["add_record_id"])
        infer = bool(task.payload.get("infer", True))
        try:
            await suppress_recording_errors(
                self._operation_recorder.mark_add_processing(context, add_record_id),
                operation="add.worker",
            )
            await self._run_add_sync(
                payload,
                context,
                add_record_id=add_record_id,
                infer=infer,
            )
        except Exception as exc:
            await suppress_recording_errors(
                self._operation_recorder.mark_add_failed(context, add_record_id, str(exc)),
                operation="add.worker",
            )
            raise

    def _get_add_pipeline(self, *, infer: bool) -> AddPipeline:
        if infer:
            return self._algorithm_add_pipeline
        if self._direct_add_pipeline is None:
            if self._direct_add_pipeline_factory is None:
                raise RuntimeError("direct add is not configured for this memory service")
            self._direct_add_pipeline = self._direct_add_pipeline_factory()
        return self._direct_add_pipeline

    def to_pipeline_context(self, context: RequestContext) -> MemoryRequestContext:
        return MemoryRequestContext(
            request_id=context.request_id,
            account_id=context.account_id,
            project_id=context.project_id,
            api_key_uuid=context.api_key_uuid,
            memory_algorithm=context.memory_algorithm or self.search_pipeline_name,
            user_id=context.user_id,
            app_id=context.app_id,
            session_id=context.session_id,
            agent_id=context.agent_id,
            scopes=list(context.scopes),
        )


def parse_display_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _pipeline_message(message):
    data = asdict(message)
    name = type(message).__name__
    models = {
        "TextMessage": TextMessage,
        "UrlMessage": UrlMessage,
        "FileMessage": FileMessage,
        "DialogueMessage": DialogueMessage,
    }
    try:
        model = models[name]
    except KeyError as exc:
        raise TypeError(f"unsupported memory message: {name}") from exc
    return model.model_validate(data)


def _raw_message_text(message) -> str:
    if isinstance(message, DialogueMessage):
        return message.content
    if isinstance(message, TextMessage):
        return message.text
    if isinstance(message, UrlMessage):
        return message.url
    if isinstance(message, FileMessage):
        return message.file_path
    raise TypeError(f"unsupported pipeline message: {type(message).__name__}")


def _add_service_result(result: AddPipelineSyncResult | AddPipelineAsyncResult) -> AddMemoryResult:
    return AddMemoryResult(
        status=result.status,
        memories=tuple(
            MemoryAddEvent(
                operation=item.operation,
                content=item.content,
                memory_id=item.memory_id,
                memory_type=item.memory_type or item.mem_type,
                confidence=item.confidence,
                related_memory_ids=tuple(item.related_memory_ids),
                graph_edge_count=item.graph_edge_count,
            )
            for item in result.memories
        ),
    )


def _search_service_result(result: SearchPipelineResult) -> MemoryListResult:
    task = getattr(result, "task_entity", None)
    tasks = tuple(
        MemoryTaskGroup(
            task_id=group.task_entity.entity_id if group.task_entity else "",
            task_name=group.task_entity.entity_name if group.task_entity else "",
            memories=tuple(_to_memory_item(item) for item in group.memories),
        )
        for group in getattr(result, "tasks", ())
    )
    return MemoryListResult(
        status=result.status,
        memories=tuple(_to_memory_item(item) for item in result.memories),
        task_id=task.entity_id if task else None,
        task_name=task.entity_name if task else None,
        tasks=tasks,
    )


def _to_memory_item(item) -> MemoryItem:
    return MemoryItem(
        memory_id=item.id,
        content=item.memory,
        memory_type=item.memory_type,
        updated_at=parse_display_time(item.last_update_at),
        event_time=parse_display_time(item.event_time),
        source_timestamp=parse_display_time(item.source_timestamp),
        lineage=(
            MemoryLineage(
                role=item.lineage.role,
                derived_from_memory_ids=tuple(item.lineage.derived_from_memory_ids),
                derived_to_memory_ids=tuple(item.lineage.derived_to_memory_ids),
            )
            if item.lineage is not None
            else None
        ),
    )


def _memory_add_dispatch_key(context: MemoryRequestContext) -> str:
    return ":".join(
        value
        for value in (
            context.project_id,
            context.user_id,
            context.app_id,
            context.session_id,
            context.agent_id,
        )
        if value
    )


def _utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = ["BaseMemoryService", "MEMORY_ADD_TASK", "parse_display_time"]
