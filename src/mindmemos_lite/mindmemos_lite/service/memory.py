"""Transport-neutral memory service backed by the migrated vanilla algorithm."""

from __future__ import annotations

from dataclasses import asdict

from ..config import MemoryConfig, get_config
from ..infra.tasking import TaskClient, TaskEnvelope
from ..logging import traced
from ..mappers import parse_search_dsl
from ..persistence import MemoryOperationRecorder
from ..persistence.memory import MemoryPersistence
from ..pipeline import SearchPipeline, create_pipeline
from ..pipeline.dreaming.consolidation import MEMORY_DREAMING_TOPIC, MemoryConsolidationPipeline
from ..pipeline.feedback.default import MEMORY_FEEDBACK_TOPIC, DefaultFeedbackPipeline
from ..pipeline.mixed_memory import MixedAddPipeline
from ..pipeline.utils import (
    format_memory_event_time,
    format_source_timestamp,
)
from ..pipeline.vanilla_memory import VanillaAddPipeline, VanillaSearchPipeline
from ..typing import (
    DreamingPipelineInput,
    FeedbackPipelineInput,
    MemoryDbDeleteCommand,
    MemoryDbUpdateCommand,
    MemoryRequestContext,
    MemorySearchItem,
)
from .base import BaseMemoryService, parse_display_time
from .ports.memory import MemoryService
from .schema import (
    DeleteMemoryRequest,
    DreamingMemoryRequest,
    FeedbackAddAction,
    FeedbackDeleteAction,
    FeedbackMemoryRequest,
    FeedbackMemoryResult,
    FeedbackNoopAction,
    FeedbackUpdateAction,
    GetMemoryRequest,
    MemoryItem,
    MemoryListResult,
    MemoryMutationResult,
    RequestContext,
    UpdateMemoryRequest,
)


class VanillaMemoryService(BaseMemoryService):
    """Application service implementing ``MemoryService`` with vanilla add/search."""

    def __init__(
        self,
        persistence: MemoryPersistence,
        *,
        config: MemoryConfig | None = None,
        task_client: TaskClient | None = None,
        add_pipeline: VanillaAddPipeline | None = None,
        direct_add_pipeline: VanillaAddPipeline | None = None,
        search_pipeline: VanillaSearchPipeline | None = None,
        dreaming_pipeline: MemoryConsolidationPipeline | None = None,
        feedback_pipeline: DefaultFeedbackPipeline | None = None,
        recorder: MemoryOperationRecorder | None = None,
    ) -> None:
        self._persistence = persistence
        self._config = config or get_config()
        resolved_recorder = recorder or MemoryOperationRecorder.from_service(persistence.service)
        resolved_add_pipeline = add_pipeline or create_pipeline(
            type="add",
            name=self._config.pipelines.default_add_pipeline,
            config=self._config,
            persistence=persistence,
        )
        resolved_search_pipeline = search_pipeline or create_pipeline(
            type="search",
            name=self._config.pipelines.default_search_pipeline,
            config=self._config,
            persistence=persistence,
        )
        self._dreaming_pipeline = dreaming_pipeline
        self._dreaming_recorder = resolved_recorder
        self._feedback_pipeline = feedback_pipeline
        self._feedback_recorder = resolved_recorder
        # Feedback recall stays on generic vanilla search regardless of the
        # configured default search pipeline (task_experience_search would not
        # be a meaningful recall source).
        self._feedback_search_pipeline = search_pipeline or VanillaSearchPipeline.from_config(
            self._config,
            persistence=persistence,
        )
        super().__init__(
            persistence,
            add_pipeline=resolved_add_pipeline,
            search_pipeline=resolved_search_pipeline,
            task_client=task_client,
            direct_add_pipeline=direct_add_pipeline,
            direct_add_pipeline_factory=self._build_direct_add_pipeline,
            recorder=resolved_recorder,
        )
        # Resolve the default add pipeline's task requirement AFTER
        # BaseMemoryService.__init__ so the base default does not reset it.
        # Read the capability from the resolved pipeline itself, so injected
        # test doubles (e.g. plain Add classes) also resolve correctly.
        self._default_add_pipeline_requires_task = bool(getattr(resolved_add_pipeline, "requires_task", False))
        if task_client is not None and MEMORY_DREAMING_TOPIC not in task_client.handlers.names():
            task_client.handlers.register(MEMORY_DREAMING_TOPIC, self._handle_dreaming_task)
        if task_client is not None and MEMORY_FEEDBACK_TOPIC not in task_client.handlers.names():
            task_client.handlers.register(MEMORY_FEEDBACK_TOPIC, self._handle_feedback_task)

    @property
    def search_pipeline_name(self) -> str:
        return "vanilla"

    @classmethod
    def from_config(
        cls,
        persistence: MemoryPersistence,
        *,
        config: MemoryConfig | None = None,
        task_client: TaskClient | None = None,
    ) -> "VanillaMemoryService":
        return cls(
            persistence,
            config=config,
            task_client=task_client,
        )

    def _build_direct_add_pipeline(self) -> VanillaAddPipeline:
        return VanillaAddPipeline.from_config(
            self._config,
            persistence=self._persistence,
            llm_client=None,
        )

    @traced("memory.service.get")
    async def get(self, context: RequestContext, request: GetMemoryRequest) -> MemoryListResult:
        memories, _ = await self._persistence.list_memories(
            self.to_pipeline_context(context),
            filters=parse_search_dsl(dict(request.filters) if request.filters is not None else None),
            limit=request.top_k or 100,
        )
        return MemoryListResult(
            status="ok",
            memories=tuple(_service_memory_item(memory) for memory in memories),
        )

    @traced("memory.service.delete")
    async def delete(self, context: RequestContext, request: DeleteMemoryRequest) -> MemoryMutationResult:
        result = await self._persistence.delete_memory(
            self.to_pipeline_context(context),
            MemoryDbDeleteCommand(memory_id=request.memory_id, hard=False),
        )
        return MemoryMutationResult(
            status="ok",
            message=None if result.changed else "memory not found",
        )

    @traced("memory.service.update")
    async def update(self, context: RequestContext, request: UpdateMemoryRequest) -> MemoryMutationResult:
        result = await self._persistence.update_memory(
            self.to_pipeline_context(context),
            MemoryDbUpdateCommand(memory_id=request.memory_id, content=request.content),
        )
        return MemoryMutationResult(
            status="ok",
            message=None if result.changed else "memory not found",
        )

    @traced("memory.service.feedback")
    async def feedback(
        self,
        context: RequestContext,
        request: FeedbackMemoryRequest,
    ) -> FeedbackMemoryResult:
        pipeline_context = self.to_pipeline_context(context)
        payload = FeedbackPipelineInput(
            feedback=request.feedback,
            messages=[_feedback_pipeline_message(message) for message in request.messages],
            recalled_memories=[
                MemorySearchItem(
                    id=item.memory_id,
                    memory=item.content,
                    memory_type=item.memory_type,
                    last_update_at=_format_feedback_time(item.updated_at),
                    event_time=_format_feedback_time(item.event_time) or None,
                    source_timestamp=_format_feedback_time(item.source_timestamp) or None,
                )
                for item in request.recalled_memories
            ],
            mode=request.mode,
        )
        if request.mode == "async":
            if self._memory_task_client is None:
                raise RuntimeError("async feedback requires a task backend configured on the memory service")
            await self._memory_task_client.submit(
                MEMORY_FEEDBACK_TOPIC,
                {
                    "context": pipeline_context.model_dump(mode="json"),
                    "input": payload.model_dump(mode="json"),
                },
                dispatch_key=f"{pipeline_context.project_id}:{pipeline_context.user_id or ''}",
            )
            return FeedbackMemoryResult(status="queued", message="feedback queued")

        result = await self._get_feedback_pipeline().feedback_sync(payload, pipeline_context)
        return _feedback_service_result(result)

    @traced("memory.service.dream")
    async def dream(
        self,
        context: RequestContext,
        request: DreamingMemoryRequest,
    ) -> MemoryMutationResult:
        pipeline_context = self.to_pipeline_context(context)
        payload = DreamingPipelineInput(mode=request.mode)
        if request.mode == "async":
            if self._memory_task_client is None:
                raise RuntimeError("async dreaming requires a task backend configured on the memory service")
            await self._memory_task_client.submit(
                MEMORY_DREAMING_TOPIC,
                {
                    "context": pipeline_context.model_dump(mode="json"),
                    "input": payload.model_dump(mode="json"),
                },
                dispatch_key=f"{pipeline_context.project_id}:{pipeline_context.user_id or ''}",
            )
            return MemoryMutationResult(status="queued", message="consolidation queued")

        result = await self._get_dreaming_pipeline().dream_sync(payload, pipeline_context)
        return MemoryMutationResult(status=result.status, message=result.message)

    async def _handle_dreaming_task(self, task: TaskEnvelope) -> None:
        pipeline_context = self._pipeline_context_from_task(task)
        payload = DreamingPipelineInput.model_validate(task.payload["input"])
        await self._get_dreaming_pipeline().dream_sync(payload, pipeline_context)

    async def _handle_feedback_task(self, task: TaskEnvelope) -> None:
        pipeline_context = self._pipeline_context_from_task(task)
        payload = FeedbackPipelineInput.model_validate(task.payload["input"]).model_copy(update={"mode": "sync"})
        await self._get_feedback_pipeline().feedback_sync(payload, pipeline_context)

    def _get_dreaming_pipeline(self) -> MemoryConsolidationPipeline:
        if self._dreaming_pipeline is None:
            self._dreaming_pipeline = MemoryConsolidationPipeline.from_config(
                self._config,
                persistence=self._persistence,
                operation_recorder=self._dreaming_recorder,
            )
        return self._dreaming_pipeline

    def _get_feedback_pipeline(self) -> DefaultFeedbackPipeline:
        if self._feedback_pipeline is None:
            self._feedback_pipeline = DefaultFeedbackPipeline.from_config(
                self._config,
                persistence=self._persistence,
                operation_recorder=self._feedback_recorder,
                search_pipeline=self._feedback_search_pipeline,
            )
        return self._feedback_pipeline

    @staticmethod
    def _pipeline_context_from_task(task: TaskEnvelope) -> MemoryRequestContext:
        return MemoryRequestContext.model_validate(task.payload["context"])


class MixedMemoryService(VanillaMemoryService):
    """Memory service using config-driven add and search pipeline selection."""

    def __init__(
        self,
        persistence: MemoryPersistence,
        *,
        config: MemoryConfig | None = None,
        task_client: TaskClient | None = None,
        add_pipeline: MixedAddPipeline | None = None,
        search_pipeline: SearchPipeline | None = None,
        direct_add_pipeline: VanillaAddPipeline | None = None,
        dreaming_pipeline: MemoryConsolidationPipeline | None = None,
        feedback_pipeline: DefaultFeedbackPipeline | None = None,
        recorder: MemoryOperationRecorder | None = None,
    ) -> None:
        resolved_config = config or get_config()
        super().__init__(
            persistence,
            config=resolved_config,
            task_client=task_client,
            add_pipeline=add_pipeline
            or create_pipeline(
                type="add",
                name=resolved_config.pipelines.default_add_pipeline,
                config=resolved_config,
                persistence=persistence,
            ),
            search_pipeline=search_pipeline
            or create_pipeline(
                type="search",
                name=resolved_config.pipelines.default_search_pipeline,
                config=resolved_config,
                persistence=persistence,
            ),
            direct_add_pipeline=direct_add_pipeline,
            dreaming_pipeline=dreaming_pipeline,
            feedback_pipeline=feedback_pipeline,
            recorder=recorder,
        )

    @property
    def search_pipeline_name(self) -> str:
        """Return the configured mode used by backward-compatible searches."""

        return self._config.pipelines.default_search_mode

    @classmethod
    def from_config(
        cls,
        persistence: MemoryPersistence,
        *,
        config: MemoryConfig | None = None,
        task_client: TaskClient | None = None,
    ) -> "MixedMemoryService":
        return cls(
            persistence,
            config=config,
            task_client=task_client,
        )


def _service_memory_item(memory) -> MemoryItem:
    return MemoryItem(
        memory_id=memory.memory_id,
        content=memory.content,
        memory_type=memory.mem_type,
        updated_at=memory.update_at or memory.created_at,
        event_time=parse_display_time(format_memory_event_time(memory, fallback_to_source_timestamp=True)),
        source_timestamp=parse_display_time(format_source_timestamp(memory)),
        lineage=None,
    )


def _feedback_pipeline_message(message):
    from ..typing import DialogueMessage, FileMessage, TextMessage, UrlMessage

    model = {
        "DialogueMessage": DialogueMessage,
        "FileMessage": FileMessage,
        "TextMessage": TextMessage,
        "UrlMessage": UrlMessage,
    }.get(type(message).__name__)
    if model is None:
        raise TypeError(f"unsupported feedback message: {type(message).__name__}")
    return model.model_validate(asdict(message))


def _format_feedback_time(value) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S") if value is not None else ""


def _feedback_service_result(result) -> FeedbackMemoryResult:
    action_types = {
        "add": FeedbackAddAction,
        "update": FeedbackUpdateAction,
        "delete": FeedbackDeleteAction,
        "noop": FeedbackNoopAction,
    }
    return FeedbackMemoryResult(
        status=result.status,
        message=result.message,
        actions=tuple(
            action_types[action.action](
                **action.model_dump(exclude={"action"}),
            )
            for action in result.actions
        ),
    )


assert isinstance(VanillaMemoryService, type)
_port_check: type[MemoryService] = VanillaMemoryService
_mixed_port_check: type[MemoryService] = MixedMemoryService

__all__ = ["MixedMemoryService", "VanillaMemoryService"]
