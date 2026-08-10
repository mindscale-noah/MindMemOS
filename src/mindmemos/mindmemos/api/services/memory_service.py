"""Memory HTTP business logic."""

from uuid import NAMESPACE_URL, uuid4, uuid5

from mindmemos_skill.contracts import (
    SkillTrajectoryReportRequest,
    SkillTrajectoryUpload,
    SkillTrajectoryUploadItem,
    TrajectorySource,
    compute_trajectory_hash,
)

from ...config import get_config
from ...errors import BadRequestError
from ...logging import get_logger, traced
from ...pipelines import PipelineType, create_pipeline
from ...pipelines.add import AddPipeline
from ...pipelines.delete import DefaultDeletePipeline, DeletePipeline
from ...pipelines.dreaming import DreamingPipeline
from ...pipelines.feedback import FeedbackPipeline
from ...pipelines.get import DefaultGetPipeline, GetPipeline
from ...pipelines.memory_db import MemoryOperationRecorder, suppress_recording_errors, utcnow
from ...pipelines.search import SearchPipeline
from ...pipelines.update import DefaultUpdatePipeline, UpdatePipeline
from ...typing import (
    AddPipelineAsyncResult,
    AddPipelineSyncResult,
    DeletePipelineResult,
    DreamingPipelineResult,
    FeedbackPipelineResult,
    GetPipelineResult,
    SearchPipelineResult,
    SkillBinding,
    SkillContext,
    UpdatePipelineResult,
)
from ..algorithm import binding_for_memory_algorithm
from ..deps import annotate_request_trace, ensure_scopes
from ..mappers import (
    to_add_pipeline_input,
    to_delete_pipeline_input,
    to_dreaming_pipeline_input,
    to_feedback_pipeline_input,
    to_get_pipeline_input,
    to_memory_request_context,
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
    SearchRequest,
    UpdateRequest,
)
from .skill_service import SkillService, get_skill_service

logger = get_logger(__name__)

SEARCH_PIPELINE_NAME = "search_pipeline"


class MemoryService:
    """Stateless facade routing memory endpoints to their pipelines."""

    def __init__(
        self,
        *,
        get_pipeline: GetPipeline | None = None,
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
        skill_store: object | None = None,
        skill_service: SkillService | None = None,
    ) -> None:
        self._add = add_pipeline
        if search_pipeline is None and search_pipeline_name is None:
            search_pipeline_name = SEARCH_PIPELINE_NAME
        self._search = search_pipeline
        self._get = get_pipeline if get_pipeline is not None else (None if get_pipeline_name else DefaultGetPipeline())
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
        self._pipeline_names: dict[str, tuple[PipelineType, str | None]] = {
            "_add": (PipelineType.ADD, add_pipeline_name),
            "_search": (PipelineType.SEARCH, search_pipeline_name),
            "_get": (PipelineType.GET, get_pipeline_name),
            "_delete": (PipelineType.DELETE, delete_pipeline_name),
            "_update": (PipelineType.UPDATE, update_pipeline_name),
            "_feedback": (PipelineType.FEEDBACK, feedback_pipeline_name),
            "_dreaming": (PipelineType.DREAMING, dreaming_pipeline_name),
        }
        self._recorder = operation_recorder or MemoryOperationRecorder()
        del skill_store  # Legacy injection point retained for constructor compatibility; the old store is never written.
        self._skill_service = skill_service
        self._algorithm_add_pipelines: dict[str, AddPipeline] = {}

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
            pipeline = create_pipeline(type=PipelineType.ADD, name=pipeline_name)
            self._algorithm_add_pipelines[pipeline_name] = pipeline
        return pipeline, pipeline_name

    def _add_pipeline_for_auth(self, auth: AuthContext) -> tuple[AddPipeline | None, str | None]:
        if self._add is not None:
            return self._add, self._pipeline_names["_add"][1]
        return self._add_pipeline_for_algorithm(auth.memory_algorithm)

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
        trajectory_result = await self._ingest_skill_trajectory(auth, add_record_id, request)
        trajectory_ref = None
        if trajectory_result is not None:
            trajectory_ref = {
                "trajectory_id": trajectory_result["trajectory_id"],
                "trajectory_hash": trajectory_result["trajectory_hash"],
                "delivery": request.skill_trajectory_delivery,
            }
        skill_bindings = await self._bind_skill_context(ctx.project_id, add_record_id, request.skill_context)
        try:
            if payload.mode == "async":
                record_metadata = {
                    "request_submitted_at": request_submitted_at.isoformat(),
                    "skill_bindings": [binding.model_dump(mode="json") for binding in skill_bindings or []],
                    "score": request.score,
                    "task_id": request.task_id,
                    "skill_trajectory_ref": trajectory_ref,
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
                        extra_payload={"skill_trajectory_ref": trajectory_ref} if trajectory_ref else None,
                    ),
                    operation="add",
                )
                result = await pipeline.add_async(
                    payload,
                    ctx,
                    add_record_id=add_record_id,
                    record_metadata=record_metadata,
                )
                return result.model_copy(update={"skill_trajectory": trajectory_result})
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
                    extra_payload={"skill_trajectory_ref": trajectory_ref} if trajectory_ref else None,
                ),
                operation="add",
            )
            result = await pipeline.add_sync(payload, ctx, add_record_id=add_record_id)
            return result.model_copy(update={"skill_trajectory": trajectory_result})
        except Exception as exc:
            await suppress_recording_errors(
                self._recorder.mark_add_failed(ctx, add_record_id, str(exc)),
                operation="add",
            )
            raise

    async def _ingest_skill_trajectory(
        self,
        auth: AuthContext,
        add_record_id: str,
        request: AddRequest,
    ) -> dict[str, str] | None:
        upload = request.skill_trajectory
        if upload is None:
            return None
        ensure_scopes(auth, ("skills:trajectory:write",))
        if upload.source is not TrajectorySource.MEMORY_ADD:
            raise BadRequestError(
                "Memory Add skill_trajectory requires source=memory_add",
                code="skill.trajectory_invalid",
                status_code=422,
            )
        if upload.source_add_record_id not in {None, add_record_id}:
            raise BadRequestError(
                "Memory Add skill_trajectory source_add_record_id is server-assigned",
                code="skill.trajectory_invalid",
                status_code=422,
            )
        source = upload.model_dump(mode="json")
        source["source_add_record_id"] = add_record_id
        source["trajectory_hash"] = compute_trajectory_hash(source)
        upload = SkillTrajectoryUpload.model_validate(source)
        operation_id = str(
            uuid5(
                NAMESPACE_URL,
                f"mindmemos:memory-add-trajectory:{upload.trajectory_id}:{upload.trajectory_hash}",
            )
        )
        skill_service = self._skill_service or get_skill_service()
        report = await skill_service.report_trajectories(
            auth,
            SkillTrajectoryReportRequest(
                operation_id=operation_id,
                mode="sync" if request.skill_trajectory_delivery == "required" else "async",
                items=[SkillTrajectoryUploadItem(trajectory=upload)],
            ),
        )
        item = report.items[0]
        return {
            "trajectory_id": item.trajectory_id,
            "trajectory_hash": upload.trajectory_hash.removeprefix("sha256:"),
            "status": item.status,
        }

    async def _bind_skill_context(
        self,
        project_id: str,
        add_record_id: str,
        skill_context: list[SkillContext] | None,
    ) -> list[SkillBinding] | None:
        """Project legacy ``skill_context`` into add-audit metadata without touching old Skill collections."""

        del project_id, add_record_id
        if not skill_context:
            return None
        return [
            SkillBinding(
                name=context.name,
                content_hash=context.content_hash,
                base_version_id=context.base_version_id,
                version_id=context.base_version_id or None,
                version_label=context.version_label,
                usage=context.usage,
            )
            for context in skill_context
        ]

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
        return await pipeline.get(to_get_pipeline_input(request), to_memory_request_context(auth))


    @traced("memory_service.delete")
    async def delete(self, auth: AuthContext, request: DeleteRequest) -> DeletePipelineResult:
        """Run the delete pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_delete")
        if pipeline is None:
            raise NotImplementedError("delete pipeline implementation is not wired yet")
        return await pipeline.delete(to_delete_pipeline_input(request), to_memory_request_context(auth))

    @traced("memory_service.update")
    async def update(self, auth: AuthContext, request: UpdateRequest) -> UpdatePipelineResult:
        """Run the update pipeline."""

        # Stamp request identity onto the handler-root span so downstream LLM spans
        # (which live in this trace, not the auth dependency's trace) are attributable.
        annotate_request_trace(auth)
        pipeline = self._pipeline("_update")
        if pipeline is None:
            raise NotImplementedError("update pipeline implementation is not wired yet")
        return await pipeline.update(to_update_pipeline_input(request), to_memory_request_context(auth))

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
