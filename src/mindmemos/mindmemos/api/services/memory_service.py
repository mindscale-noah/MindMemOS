"""Memory HTTP business logic."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import nullcontext
from typing import Any, Literal
from uuid import uuid4

from ...config import bind_config_overrides, get_config, get_config_overrides
from ...errors import ConfigNotInitializedError, MemoryNotFoundError, ResourceNotFoundError
from ...logging import get_logger, traced
from ...pipelines import create_pipeline
from ...pipelines.add import AddPipeline
from ...pipelines.delete import DefaultDeletePipeline, DeletePipeline
from ...pipelines.dreaming import DreamingPipeline
from ...pipelines.feedback import FeedbackPipeline
from ...pipelines.get import DefaultGetPipeline, GetPipeline
from ...pipelines.memory_db import MemoryCatalog, MemoryOperationRecorder, suppress_recording_errors, utcnow
from ...pipelines.search import SearchPipeline
from ...pipelines.skill import SkillVersionStore, get_skill_version_store
from ...pipelines.update import DefaultUpdatePipeline, UpdatePipeline
from ...provider_bindings import ProviderBindingResolver, get_provider_binding_resolver
from ...typing import (
    AddPipelineAsyncResult,
    AddPipelineSyncResult,
    AddStreamCancelled,
    DeletePipelineResult,
    DreamingPipelineResult,
    FeedbackPipelineResult,
    GetPipelineResult,
    MemoryListPipelineResult,
    MemoryScrollPipelineResult,
    SearchPipelineResult,
    SkillBinding,
    SkillContext,
    UpdatePipelineResult,
)
from ..algorithm import binding_for_memory_algorithm
from ..deps import annotate_request_trace
from ..mappers import (
    to_add_pipeline_input,
    to_delete_pipeline_input,
    to_dreaming_pipeline_input,
    to_feedback_pipeline_input,
    to_get_pipeline_input,
    to_memory_list_pipeline_input,
    to_memory_request_context,
    to_memory_scroll_pipeline_input,
    to_search_pipeline_input,
    to_update_pipeline_input,
)
from ..schemas import (
    AddRequest,
    AuthContext,
    DeleteRequest,
    DreamingRequest,
    FeedbackRequest,
    GetRequest,
    MemoryPageRequest,
    MemoryScrollRequest,
    SearchRequest,
    UpdateRequest,
)

logger = get_logger(__name__)

PipelineKind = Literal["add", "search", "get", "delete", "update", "feedback", "dreaming"]
SEARCH_PIPELINE_NAME = "search_pipeline"


class MemoryService:
    """Stateless facade routing memory endpoints to their pipelines."""

    def __init__(
        self,
        *,
        get_pipeline: GetPipeline | None = None,
        catalog: MemoryCatalog | None = None,
        add_pipeline: AddPipeline | None = None,
        search_pipeline: SearchPipeline | None = None,
        delete_pipeline: DeletePipeline | None = None,
        update_pipeline: UpdatePipeline | None = None,
        feedback_pipeline: FeedbackPipeline | None = None,
        dreaming_pipeline: DreamingPipeline | None = None,
        add_pipeline_name: str | None = None,
        search_pipeline_name: str | None = None,
        get_pipeline_name: str | None = None,
        delete_pipeline_name: str | None = None,
        update_pipeline_name: str | None = None,
        feedback_pipeline_name: str | None = None,
        dreaming_pipeline_name: str | None = None,
        operation_recorder: MemoryOperationRecorder | None = None,
        skill_store: SkillVersionStore | None = None,
        provider_binding_resolver: ProviderBindingResolver | None = None,
    ) -> None:
        self._add = add_pipeline
        if search_pipeline is None and search_pipeline_name is None:
            search_pipeline_name = SEARCH_PIPELINE_NAME
        self._search = search_pipeline
        self._get = get_pipeline if get_pipeline is not None else (None if get_pipeline_name else DefaultGetPipeline())
        self._catalog = catalog or MemoryCatalog()
        self._delete = (
            delete_pipeline
            if delete_pipeline is not None
            else (None if delete_pipeline_name else DefaultDeletePipeline())
        )
        self._update = (
            update_pipeline
            if update_pipeline is not None
            else (None if update_pipeline_name else DefaultUpdatePipeline())
        )
        self._feedback = feedback_pipeline
        self._dreaming = dreaming_pipeline
        self._pipeline_names: dict[str, tuple[PipelineKind, str | None]] = {
            "_add": ("add", add_pipeline_name),
            "_search": ("search", search_pipeline_name),
            "_get": ("get", get_pipeline_name),
            "_delete": ("delete", delete_pipeline_name),
            "_update": ("update", update_pipeline_name),
            "_feedback": ("feedback", feedback_pipeline_name),
            "_dreaming": ("dreaming", dreaming_pipeline_name),
        }
        self._recorder = operation_recorder or MemoryOperationRecorder()
        self._skill_store = skill_store if skill_store is not None else get_skill_version_store()
        self._algorithm_add_pipelines: dict[str, AddPipeline] = {}
        self._provider_binding_resolver = provider_binding_resolver

    def _pipeline(self, attr: str):
        pipeline = getattr(self, attr)
        if pipeline is not None:
            return pipeline
        pipeline_type, pipeline_name = self._pipeline_names[attr]
        if pipeline_name is None:
            return None
        pipeline = create_pipeline(type=pipeline_type, name=pipeline_name)
        setattr(self, attr, pipeline)
        return pipeline

    def _add_pipeline_for_algorithm(self, memory_algorithm: str) -> tuple[AddPipeline | None, str | None]:
        pipeline_name = binding_for_memory_algorithm(memory_algorithm).add_pipeline
        pipeline = self._algorithm_add_pipelines.get(pipeline_name)
        if pipeline is None:
            pipeline = create_pipeline(type="add", name=pipeline_name)
            self._algorithm_add_pipelines[pipeline_name] = pipeline
        return pipeline, pipeline_name

    def _add_pipeline_for_auth(self, auth: AuthContext) -> tuple[AddPipeline | None, str | None]:
        if self._add is not None:
            return self._add, self._pipeline_names["_add"][1]
        return self._add_pipeline_for_algorithm(auth.memory_algorithm)

    async def _provider_config_context(self, ctx):
        """Return a temporary config binding that includes dynamic provider overrides."""

        try:
            resolver = self._provider_binding_resolver or get_provider_binding_resolver()
            dynamic_project_config = await resolver.resolve(ctx)
        except ConfigNotInitializedError:
            return nullcontext()
        if not dynamic_project_config:
            return nullcontext()
        overrides = get_config_overrides()
        tenant_config = overrides.tenant_config if overrides is not None else None
        project_config = _deep_merge_dicts(
            overrides.project_config if overrides is not None else None,
            dynamic_project_config,
        )
        return bind_config_overrides(tenant_config=tenant_config, project_config=project_config)

    @traced("memory_service.add")
    async def add(
        self,
        auth: AuthContext,
        request: AddRequest,
    ) -> AddPipelineSyncResult | AddPipelineAsyncResult:
        """Run the add pipeline according to the requested mode."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline, _add_pipeline_name = self._add_pipeline_for_auth(auth)
        if pipeline is None:
            raise NotImplementedError("add pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth, request, require_user_id=True)
        payload = to_add_pipeline_input(request)
        add_record_id = str(uuid4())
        request_submitted_at = utcnow()
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            skill_bindings = await self._bind_skill_context(ctx.project_id, add_record_id, request.skill_context)
            return await self._add_with_context(
                pipeline,
                payload,
                ctx,
                request,
                add_record_id,
                request_submitted_at,
                skill_bindings,
            )

    @traced("memory_service.add_stream")
    async def add_stream(
        self,
        auth: AuthContext,
        request: AddRequest,
        *,
        cancel_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run sync add while streaming progress events."""

        annotate_request_trace(auth)
        pipeline, _add_pipeline_name = self._add_pipeline_for_auth(auth)
        if pipeline is None:
            raise NotImplementedError("add pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth, request, require_user_id=True)
        payload = to_add_pipeline_input(request)
        if payload.mode != "sync":
            yield {
                "event": "error",
                "stage": "accepted",
                "message": "streaming add only supports sync mode",
            }
            return

        add_record_id = str(uuid4())
        request_submitted_at = utcnow()
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            skill_bindings = await self._bind_skill_context(ctx.project_id, add_record_id, request.skill_context)
            yield {
                "event": "progress",
                "stage": "accepted",
                "message": "Add request accepted.",
                "percent": 1,
            }
            await suppress_recording_errors(
                self._recorder.record_add_input(
                    payload,
                    ctx=ctx,
                    request_submitted_at=request_submitted_at,
                    add_record_id=add_record_id,
                    status="processing",
                    skill_bindings=skill_bindings,
                    score=request.score,
                    task_id=request.task_id,
                ),
                operation="add",
            )

            if hasattr(pipeline, "add_sync_stream"):
                async for event in self._stream_pipeline_add(
                    pipeline,
                    payload,
                    ctx,
                    add_record_id=add_record_id,
                    cancel_check=cancel_check,
                ):
                    yield event
                return

            try:
                result = await pipeline.add_sync(payload, ctx, add_record_id=add_record_id)
            except Exception as exc:
                await suppress_recording_errors(
                    self._recorder.mark_add_failed(ctx, add_record_id, str(exc)),
                    operation="add",
                )
                yield {"event": "error", "stage": "error", "message": str(exc)}
                return
            yield {
                "event": "completed",
                "stage": "completed",
                "message": "Add completed.",
                "data": {"memories": [memory.model_dump(mode="json") for memory in result.memories]},
            }

    async def _stream_pipeline_add(
        self,
        pipeline: AddPipeline,
        payload,
        ctx,
        *,
        add_record_id: str,
        cancel_check: Callable[[], Awaitable[bool]] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def progress(
            stage: str,
            message: str,
            percent: int | None = None,
            data: dict[str, Any] | None = None,
        ) -> None:
            event: dict[str, Any] = {
                "event": "progress",
                "stage": stage,
                "message": message,
            }
            if percent is not None:
                event["percent"] = percent
            if data:
                message_i18n = data.get("message_i18n")
                if isinstance(message_i18n, dict):
                    event["message_i18n"] = message_i18n
                remaining_data = {key: value for key, value in data.items() if key != "message_i18n"}
                if remaining_data:
                    event["data"] = remaining_data
            await queue.put(event)

        async def run_pipeline() -> None:
            try:
                result_or_stream = pipeline.add_sync_stream(
                    payload,
                    ctx,
                    add_record_id=add_record_id,
                    progress=progress,
                    cancel_check=cancel_check,
                )
                if hasattr(result_or_stream, "__aiter__"):
                    async for event in result_or_stream:
                        await queue.put(event)
                else:
                    result = await result_or_stream
                    await queue.put(
                        {
                            "event": "completed",
                            "stage": "completed",
                            "message": "Add completed.",
                            "data": {
                                "memories": [
                                    memory.model_dump(mode="json") for memory in result.memories
                                ]
                            },
                        }
                    )
            except AddStreamCancelled as exc:
                await suppress_recording_errors(
                    self._recorder.mark_add_cancelled(ctx, add_record_id, exc.message),
                    operation="add",
                )
                await queue.put(
                    {
                        "event": "cancelled",
                        "stage": exc.stage,
                        "message": exc.message,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                await suppress_recording_errors(
                    self._recorder.mark_add_failed(ctx, add_record_id, str(exc)),
                    operation="add",
                )
                await queue.put({"event": "error", "stage": "error", "message": str(exc)})
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_pipeline())
        heartbeat_seconds = getattr(self, "stream_heartbeat_seconds", 10.0)
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=heartbeat_seconds)
                except asyncio.TimeoutError:
                    yield {
                        "event": "heartbeat",
                        "stage": "waiting",
                        "message": "Memory extraction is still running.",
                        "message_i18n": {
                            "zh": "记忆提取仍在进行，请稍候",
                            "en": "Memory extraction is still running.",
                        },
                    }
                    continue
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                def _consume_task_result(done_task: asyncio.Task) -> None:
                    try:
                        done_task.exception()
                    except asyncio.CancelledError:
                        pass

                task.add_done_callback(_consume_task_result)

    async def _add_with_context(
        self,
        pipeline: AddPipeline,
        payload,
        ctx,
        request: AddRequest,
        add_record_id: str,
        request_submitted_at,
        skill_bindings,
    ):
        try:
            if payload.mode == "async":
                record_metadata = {
                    "request_submitted_at": request_submitted_at.isoformat(),
                    "skill_bindings": [binding.model_dump(mode="json") for binding in skill_bindings or []],
                    "score": request.score,
                    "task_id": request.task_id,
                }
                await suppress_recording_errors(
                    self._recorder.record_add_input(
                        payload,
                        ctx=ctx,
                        request_submitted_at=request_submitted_at,
                        add_record_id=add_record_id,
                        status="queued",
                        skill_bindings=skill_bindings,
                        score=request.score,
                        task_id=request.task_id,
                    ),
                    operation="add",
                )
                return await pipeline.add_async(
                    payload,
                    ctx,
                    add_record_id=add_record_id,
                    record_metadata=record_metadata,
                )
            await suppress_recording_errors(
                self._recorder.record_add_input(
                    payload,
                    ctx=ctx,
                    request_submitted_at=request_submitted_at,
                    add_record_id=add_record_id,
                    status="processing",
                    skill_bindings=skill_bindings,
                    score=request.score,
                    task_id=request.task_id,
                ),
                operation="add",
            )
            return await pipeline.add_sync(payload, ctx, add_record_id=add_record_id)
        except Exception as exc:
            await suppress_recording_errors(
                self._recorder.mark_add_failed(ctx, add_record_id, str(exc)),
                operation="add",
            )
            raise

    async def _bind_skill_context(
        self,
        project_id: str,
        add_record_id: str,
        skill_context: list[SkillContext] | None,
    ) -> list[SkillBinding] | None:
        """Resolve ``skill_context`` to trace bindings, swallowing all failures."""

        if self._skill_store is None or not skill_context:
            return None
        try:
            return await self._skill_store.bind_skill_context(
                project_id=project_id,
                add_record_id=add_record_id,
                skill_context=skill_context,
            )
        except Exception as exc:  # noqa: BLE001 - binding must never block add
            logger.warning("skill_context binding failed", error=str(exc), add_record_id=add_record_id)
            return None

    @traced("memory_service.search")
    async def search(
        self,
        auth: AuthContext,
        request: SearchRequest,
    ) -> SearchPipelineResult:
        """Run the search pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        ctx = to_memory_request_context(auth, request, require_user_id=True)
        binding = binding_for_memory_algorithm(auth.memory_algorithm)
        payload = to_search_pipeline_input(request, search_pipeline=binding.search_pipeline)
        pipeline = self._pipeline("_search")
        if pipeline is None:
            raise NotImplementedError("search pipeline implementation is not wired yet")
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            return await self._search_with_context(pipeline, payload, ctx)

    async def _search_with_context(self, pipeline: SearchPipeline, payload, ctx) -> SearchPipelineResult:
        request_submitted_at = utcnow()
        try:
            result = await pipeline.search(payload, ctx)
        except Exception:
            task_completed_at = utcnow()
            await suppress_recording_errors(
                self._recorder.record_search(
                    payload,
                    None,
                    ctx=ctx,
                    request_submitted_at=request_submitted_at,
                    task_completed_at=task_completed_at,
                ),
                operation="search",
            )
            raise
        task_completed_at = utcnow()
        await suppress_recording_errors(
            self._recorder.record_search(
                payload,
                result,
                ctx=ctx,
                request_submitted_at=request_submitted_at,
                task_completed_at=task_completed_at,
            ),
            operation="search",
        )
        return result

    @traced("memory_service.get")
    async def get(self, auth: AuthContext, request: GetRequest) -> GetPipelineResult:
        """Run the get pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_get")
        if pipeline is None:
            raise NotImplementedError("get pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth)
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            return await pipeline.get(to_get_pipeline_input(request), ctx)

    @traced("memory_service.list")
    async def list(self, auth: AuthContext, request: MemoryPageRequest) -> MemoryListPipelineResult:
        """Run the paged memory list pipeline."""

        annotate_request_trace(auth)
        ctx = to_memory_request_context(auth, request)
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            return await self._catalog.list(to_memory_list_pipeline_input(request), ctx)

    @traced("memory_service.scroll")
    async def scroll(self, auth: AuthContext, request: MemoryScrollRequest) -> MemoryScrollPipelineResult:
        """Run the cursor memory scroll pipeline."""

        annotate_request_trace(auth)
        ctx = to_memory_request_context(auth, request)
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            return await self._catalog.scroll(to_memory_scroll_pipeline_input(request), ctx)

    @traced("memory_service.delete")
    async def delete(self, auth: AuthContext, request: DeleteRequest) -> DeletePipelineResult:
        """Run the delete pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_delete")
        if pipeline is None:
            raise NotImplementedError("delete pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth)
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            try:
                return await pipeline.delete(to_delete_pipeline_input(request), ctx)
            except MemoryNotFoundError as exc:
                raise ResourceNotFoundError(str(exc), code="memory.not_found") from exc

    @traced("memory_service.update")
    async def update(self, auth: AuthContext, request: UpdateRequest) -> UpdatePipelineResult:
        """Run the update pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_update")
        if pipeline is None:
            raise NotImplementedError("update pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth, request)
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            try:
                return await pipeline.update(to_update_pipeline_input(request), ctx)
            except MemoryNotFoundError as exc:
                raise ResourceNotFoundError(str(exc), code="memory.not_found") from exc

    @traced("memory_service.feedback")
    async def feedback(self, auth: AuthContext, request: FeedbackRequest) -> FeedbackPipelineResult:
        """Run the feedback pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_feedback")
        if pipeline is None:
            raise NotImplementedError("feedback pipeline implementation is not wired yet")
        ctx = to_memory_request_context(auth, request, require_user_id=True)
        payload = to_feedback_pipeline_input(request)
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            if payload.mode == "async":
                return await pipeline.feedback_async(payload, ctx)
            return await pipeline.feedback_sync(payload, ctx)

    @traced("memory_service.dream")
    async def dream(self, auth: AuthContext, request: DreamingRequest) -> DreamingPipelineResult:
        """Run the dreaming pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_dreaming")
        if pipeline is None:
            raise NotImplementedError("dreaming pipeline implementation is not wired yet")
        payload = to_dreaming_pipeline_input(request)
        ctx = to_memory_request_context(auth, request)
        config_ctx = await self._provider_config_context(ctx)
        with config_ctx:
            if payload.mode == "sync":
                return await pipeline.dream_sync(payload, ctx)
            return await pipeline.dream(payload, ctx)


_service: MemoryService | None = None
_service_key: tuple[str, ...] | None = None


def get_memory_service() -> MemoryService:
    """Process-global service singleton, used as a FastAPI dependency."""

    global _service, _service_key
    cfg = get_config().pipelines
    get_pipeline = cfg["get"]
    delete_pipeline = cfg["delete"]
    update_pipeline = cfg["update"]
    feedback_pipeline = cfg["feedback"]
    dreaming_pipeline = cfg["dreaming"]
    service_key = (
        get_pipeline,
        delete_pipeline,
        update_pipeline,
        feedback_pipeline,
        dreaming_pipeline,
    )
    if _service is None or _service_key != service_key:
        _service = MemoryService(
            get_pipeline_name=get_pipeline,
            delete_pipeline_name=delete_pipeline,
            update_pipeline_name=update_pipeline,
            feedback_pipeline_name=feedback_pipeline,
            dreaming_pipeline_name=dreaming_pipeline,
        )
        _service_key = service_key
    return _service


def _deep_merge_dicts(base: dict[str, Any] | None, override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base or {})
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(current, value)
        else:
            result[key] = value
    return result
