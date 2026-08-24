"""Schema add-memory pipeline implementation."""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ....components.chunker import EpisodeBoundary, EpisodesChunker
from ....components.extractor import _records as add_record_ops
from ....components.extractor.schema import (
    SchemaAddExtractor,
    SchemaAddPlanner,
    build_episode_entity,
)
from ....components.memory_modeling.schema import EntityManager, get_entity_manager
from ....components.text import SparseVectorEncoder, detect_prompt_language, get_text_preprocessor
from ....config import get_config
from ....infra.kafka import get_producer
from ....llm import EmbedClient, LLMClient, get_embed_client, get_llm_client, require_model_endpoint
from ....logging import get_logger, traced, traced_awaitable
from ....prompts import AddPromptSet, get_add_prompts
from ....typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    AddStreamCancelled,
    EntityVectorWrite,
    EntityWrite,
    MemoryAddEventItem,
    MemoryDbEntityUpdateCommand,
    MemoryDbMutationPlan,
    MemoryDbWritePlan,
    MemoryRequestContext,
)
from ...base import MemoryDbPipelineMixin
from ...memory_db import (
    AddRecordBuffer,
    BufferedAddRecord,
    MemoryOperationRecorder,
    buffer_key,
    context_from_record,
    suppress_recording_errors,
    utcnow,
)
from ...registry import register
from ..base import AddPipeline

logger = get_logger(__name__)

SCHEMA_ADD_DRAIN_TOPIC = "memory.add.drain"
SCHEMA_ADD_EPISODE_TOPIC = "memory.add.episode"
ProgressReporter = Callable[[str, str, int | None, dict[str, Any] | None], Awaitable[None]]
CancelCheck = Callable[[], Awaitable[bool]]

_STAGE_MESSAGES: dict[str, dict[str, str]] = {
    "buffering": {"zh": "正在接收输入", "en": "Buffering source messages"},
    "chunking": {"zh": "正在分析内容边界", "en": "Detecting memory episode boundaries"},
    "llm_extracting": {"zh": "正在提取结构化记忆", "en": "Extracting structured memory"},
    "search_fielding": {"zh": "正在生成检索线索", "en": "Generating search hints"},
    "memory_planning": {"zh": "正在整理记忆结构", "en": "Planning memory structure"},
    "embedding": {"zh": "正在生成记忆向量", "en": "Generating memory embeddings"},
    "relationship_building": {"zh": "正在建立记忆关系", "en": "Building memory relationships"},
    "ready_to_persist": {"zh": "准备写入记忆", "en": "Preparing to persist memory"},
    "persisting": {"zh": "正在写入记忆", "en": "Persisting memory"},
    "completed": {"zh": "记忆提取完成", "en": "Memory extraction completed"},
}


async def _report_progress(
    progress: ProgressReporter | None,
    stage: str,
    message: str,
    percent: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    if progress is not None:
        stage_message = _STAGE_MESSAGES.get(stage)
        if stage_message is not None:
            message = stage_message["en"]
            data = {**(data or {}), "message_i18n": stage_message}
        await progress(stage, message, percent, data)


async def _raise_if_cancelled(cancel_check: CancelCheck | None, stage: str) -> None:
    if cancel_check is not None and await cancel_check():
        raise AddStreamCancelled(stage, "Add stream cancelled before persistence.")


@dataclass(slots=True)
class _EpisodeTask:
    """A chunked episode ready for memory generation."""

    episode_id: str
    records: list[BufferedAddRecord]
    chunk_index: int = 0
    chunk_count: int = 1
    start_idx: int = 0
    end_idx: int = 0
    title: str = ""


@dataclass(slots=True)
class _SchemaAddRuntime:
    """Per-drain-loop resolved schema-add deps.

    Built from the request-scoped config (ContextVar) once per ``_process_loop`` call
    and never cached on the singleton pipeline instance, so one project's config can
    never leak into another. Mirrors the entity_manager per-request resolution pattern.
    """

    schema_cfg: Any
    project_em: Any
    chunker: EpisodesChunker
    extractor: SchemaAddExtractor
    planner: SchemaAddPlanner
    search_fields_max: int


def _override(explicit: Any, default: Any) -> Any:
    """Return the explicit override when provided, else the (request-scoped) default."""

    return explicit if explicit is not None else default


@register(type="add", name="schema_add")
class SchemaAddPipeline(MemoryDbPipelineMixin, AddPipeline):
    """Schema-driven add pipeline migrated from the original algorithm."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        embed_client: EmbedClient | None = None,
        entity_manager: EntityManager | None = None,
        add_buffer: AddRecordBuffer | None = None,
        chunker: EpisodesChunker | None = None,
        recorder: MemoryOperationRecorder | None = None,
        enable_schema_selection: bool | None = None,
        entity_recall_top_k: int | None = None,
        search_fields_max: int | None = None,
        episode_edge_top_k: int | None = None,
        prompt_language: str | None = None,
        prompt_set: AddPromptSet | None = None,
        extractor: SchemaAddExtractor | None = None,
        planner: SchemaAddPlanner | None = None,
        consistency: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # This pipeline is held by a process-wide singleton (MemoryService.
        # _algorithm_add_pipelines), so it MUST stay project-agnostic: all
        # project-scoped deps (LLM/embed clients, entity manager, prompts, chunker,
        # extractor, planner, text preprocessor, sparse encoder, and every algo
        # parameter) are resolved per drain loop from the
        # request-scoped ContextVar config (see get_config() and _resolve_add_runtime).
        # The explicit injections/overrides below are for tests only; production
        # leaves them None so each request reads its own project's config.
        self.add_buffer = add_buffer or AddRecordBuffer()
        self._recorder = recorder or MemoryOperationRecorder()
        self.recorder = self._recorder
        self._processing_by_key: dict[str, bool] = defaultdict(bool)
        self._process_lock_by_key: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._explicit_consistency = consistency
        self._explicit_llm = llm_client
        self._explicit_embed = embed_client
        self._explicit_entity_manager = entity_manager
        self._explicit_chunker = chunker
        self._explicit_extractor = extractor
        self._explicit_planner = planner
        self._explicit_prompts = prompt_set
        self._explicit_prompt_language = prompt_language
        # Algo overrides (None -> use the request-scoped config value at drain time).
        self._explicit_enable_schema_selection = enable_schema_selection
        self._explicit_entity_recall_top_k = entity_recall_top_k
        self._explicit_search_fields_max = search_fields_max
        self._explicit_episode_edge_top_k = episode_edge_top_k

    def _get_consistency(self) -> str:
        if self._explicit_consistency is not None:
            return self._explicit_consistency
        return _default_consistency()

    def _get_schema_add_config(self):
        return get_config().algo_config.add.schema

    def _resolve_add_runtime(self, context: MemoryRequestContext) -> _SchemaAddRuntime:
        """Resolve all project-scoped deps from the request-scoped config (ContextVar).

        Built once per drain loop (per ``_process_loop`` call) and never cached on this
        singleton pipeline instance, so the first project's config can never leak into
        another project. Mirrors the entity_manager per-request resolution pattern.
        """
        schema_cfg = self._get_schema_add_config()
        if self._explicit_llm is None:
            require_model_endpoint("chat")
        if self._explicit_embed is None:
            require_model_endpoint("embedding")
        llm_client = self._explicit_llm or get_llm_client()
        embed_client = self._explicit_embed or get_embed_client()
        project_em = self._explicit_entity_manager or get_entity_manager(project_id=context.project_id)
        text_preprocessor = get_text_preprocessor()
        sparse_encoder = SparseVectorEncoder(get_config().algo_config.text_processing)
        prompt_language = self._explicit_prompt_language or get_config().algo_config.common.prompt_language
        prompts = self._explicit_prompts or get_add_prompts(prompt_language)

        chunker = self._explicit_chunker or EpisodesChunker(
            mode=schema_cfg.chunker.split_mode,
            llm_client=llm_client,
            max_messages=schema_cfg.chunker.max_episode_length,
            max_minutes_from_first=schema_cfg.chunker.max_minutes_from_first,
            split_on_user_speaker=schema_cfg.chunker.split_on_user_speaker,
            boundary_prompt=prompts.conv_boundary_detection,
            resplit_prompt=prompts.conv_forced_resplit,
            streaming_window_size=schema_cfg.chunker.streaming_window_size,
        )

        enable_schema_selection = _override(
            self._explicit_enable_schema_selection, schema_cfg.extraction.enable_schema_selection
        )
        extractor = self._explicit_extractor or SchemaAddExtractor(
            llm_client=llm_client,
            prompt_set=prompts,
            entity_manager=project_em,
            enable_schema_selection=enable_schema_selection,
        )

        entity_recall_top_k = _override(self._explicit_entity_recall_top_k, schema_cfg.merge.entity_recall_top_k)
        episode_edge_top_k = _override(self._explicit_episode_edge_top_k, schema_cfg.episode_edge.top_k)
        search_fields_max = _override(self._explicit_search_fields_max, schema_cfg.extraction.search_fields_max)

        planner = self._explicit_planner or SchemaAddPlanner(
            llm_client=llm_client,
            embed_client=embed_client,
            db_reader=self.db_reader,
            db_writer=self.db_writer,
            entity_manager=project_em,
            prompt_set=prompts,
            entity_recall_top_k=entity_recall_top_k,
            episode_edge_top_k=episode_edge_top_k,
            max_entities_per_conversation=schema_cfg.extraction.max_entities_per_conversation,
            max_properties_per_entity=schema_cfg.extraction.max_properties_per_entity,
            text_preprocessor=text_preprocessor,
            sparse_encoder=sparse_encoder,
        )

        return _SchemaAddRuntime(
            schema_cfg=schema_cfg,
            project_em=project_em,
            chunker=chunker,
            extractor=extractor,
            planner=planner,
            search_fields_max=search_fields_max,
        )

    @traced("add_pipeline.sync", record_args=False)
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        """Append messages to the add buffer and drain them synchronously.

        Args:
            inp: Add request payload, including messages and force-generation options.
            context: Tenant and project context used for storage isolation.
            add_record_id: Optional add record id to write the output back onto.

        Returns:
            The generated memory events for this synchronous add request.
        """

        await self.add_buffer.append(
            context,
            inp,
            force_generation=inp.force_generation,
            source_add_record_id=add_record_id,
        )
        events = await self._ensure_drain_and_wait(
            context,
            consistency=self._get_consistency(),
            force=True,
        )
        result = AddPipelineSyncResult(status="ok", memories=events)
        # Sync drains inline and produces the full output in one shot, so overwrite
        # the request-level record directly. The inline path does not thread the
        # trigger id into episodes, so there is no double write.
        await suppress_recording_errors(
            self.recorder.mark_add_completed(context, add_record_id, result),
            operation="add.schema_add.sync",
        )
        return result

    async def add_sync_stream(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> AddPipelineSyncResult:
        """Append messages, drain synchronously, and report progress for SSE callers."""

        await _report_progress(progress, "buffering", "Buffering source messages.", 8)
        record_ids = await self.add_buffer.append(
            context,
            inp,
            force_generation=inp.force_generation,
            source_add_record_id=add_record_id,
        )
        try:
            await _raise_if_cancelled(cancel_check, "buffering")
            events = await self._ensure_drain_and_wait(
                context,
                consistency=self._get_consistency(),
                force=True,
                progress=progress,
                cancel_check=cancel_check,
            )
        except AddStreamCancelled:
            records = await self.add_buffer.get_by_ids(context, record_ids)
            await self.add_buffer.delete_processed(context, records)
            raise
        result = AddPipelineSyncResult(status="ok", memories=events)
        await _report_progress(progress, "completed", "Memory extraction completed.", 100)
        await suppress_recording_errors(
            self.recorder.mark_add_completed(context, add_record_id, result),
            operation="add.schema_add.sync",
        )
        return result

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> AddPipelineAsyncResult:
        """Append messages to the add buffer and queue background draining.

        Args:
            inp: Add request payload, including messages and force-generation options.
            context: Tenant and project context used for storage isolation.
            add_record_id: Optional triggering add record id. Every episode produced
                by the drain this request kicks off is accumulated onto this record
                (trigger binding, not message provenance).

        Returns:
            A queued status result.

        Raises:
            RuntimeError: If Kafka is disabled for asynchronous add processing.
        """
        if not get_config().kafka.enabled:
            raise RuntimeError(
                "schema_add add_async requires Kafka to be enabled (kafka.enabled=true). "
                "Use mode='sync' or enable Kafka in config."
            )
        await self.add_buffer.append(
            context,
            inp,
            force_generation=inp.force_generation,
            source_add_record_id=add_record_id,
        )
        await self._ensure_drain_started(
            context,
            inp,
            force=inp.force_generation,
            trigger_record_id=add_record_id,
            record_metadata=record_metadata,
        )
        return AddPipelineAsyncResult(status="queued")

    async def has_pending(self, context: MemoryRequestContext) -> bool:
        """Check whether the project buffer still has unprocessed add records.

        Args:
            context: Tenant and project context used to select the buffer.

        Returns:
            True when buffered records are still pending.
        """
        return await self.add_buffer.has_pending(context)

    async def drain_buffer(
        self,
        context: MemoryRequestContext,
        *,
        consistency: str | None = None,
        force: bool = False,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        """Drain buffered add records from an external worker entry point."""
        contexts = [context]
        if not await self.add_buffer.list_buffered(context, limit=1):
            contexts = await self._contexts_for_project(context.project_id, limit=100)

        events: list[MemoryAddEventItem] = []
        dispatched = 0
        for drain_context in contexts:
            if not await self._try_start_loop(drain_context):
                continue
            loop_events, loop_dispatched = await self._process_loop(
                drain_context,
                consistency=consistency or self._get_consistency(),
                force=force,
                trigger_record_id=trigger_record_id,
            )
            events.extend(loop_events)
            dispatched += loop_dispatched
        # Trigger produced no episode this drain: finalize it as ok/empty so the
        # request-level record does not linger at "queued". When episodes were
        # dispatched, the episode workers accumulate output onto the trigger.
        if trigger_record_id and dispatched == 0:
            await suppress_recording_errors(
                self.recorder.mark_add_completed(
                    context, trigger_record_id, AddPipelineSyncResult(status="ok", memories=[])
                ),
                operation="add.schema_add.drain_buffer",
            )
        return events

    async def generate_episode(
        self,
        context: MemoryRequestContext,
        add_record_ids: list[str],
        *,
        episode_id: str,
        consistency: str | None = None,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        """Generate one episode from an external worker entry point."""
        records = await self.add_buffer.get_by_ids(context, add_record_ids)
        if not records:
            logger.warning(
                "episode generation skipped: records not found in buffer",
                episode_id=episode_id,
                expected_count=len(add_record_ids),
            )
            return []
        await self.add_buffer.mark_processing(context, records)
        return await self._execute_episode_task(
            _EpisodeTask(episode_id=episode_id, records=records),
            context=context,
            consistency=consistency or self._get_consistency(),
            trigger_record_id=trigger_record_id,
            rt=self._resolve_add_runtime(context),
        )

    # Internal: drain orchestration

    async def _ensure_drain_started(
        self,
        context: MemoryRequestContext,
        inp: AddPipelineInput,
        *,
        force: bool,
        trigger_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish the drain trigger to Kafka; async mode is Kafka-only."""

        if not get_config().kafka.enabled:
            raise RuntimeError(
                "schema_add async drain requires Kafka to be enabled (kafka.enabled=true). "
                "Use mode='sync' or enable Kafka in config."
            )
        try:
            await self._publish_drain_task(
                context,
                inp,
                force=force,
                trigger_record_id=trigger_record_id,
                record_metadata=record_metadata,
            )
        except Exception:
            logger.error("schema add drain task publish failed", exc_info=True)
            raise

    async def _ensure_drain_and_wait(
        self,
        context: MemoryRequestContext,
        *,
        consistency: str,
        force: bool,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> list[MemoryAddEventItem]:
        """Start the drain loop and wait for generated events."""
        key = buffer_key(context)
        while True:
            async with self._process_lock_by_key[key]:
                if not self._processing_by_key[key]:
                    self._processing_by_key[key] = True
                    break
            await asyncio.sleep(0.05)
        events, _dispatched = await self._process_loop(
            context,
            consistency=consistency,
            force=force,
            inline=True,
            progress=progress,
            cancel_check=cancel_check,
        )
        return events

    async def _publish_drain_task(
        self,
        context: MemoryRequestContext,
        inp: AddPipelineInput,
        *,
        force: bool,
        trigger_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> None:
        key = buffer_key(context)
        await get_producer().send(
            SCHEMA_ADD_DRAIN_TOPIC,
            value={
                "context": context.model_dump(mode="json"),
                "input": inp.model_dump(mode="json", by_alias=True),
                "force": force,
                "consistency": self._get_consistency(),
                "trigger_record_id": trigger_record_id,
                "record_metadata": record_metadata,
            },
            dispatch_key=key,
        )

    async def _try_start_loop(self, context: MemoryRequestContext) -> bool:
        """Acquire processing ownership for the drain loop."""
        key = buffer_key(context)
        async with self._process_lock_by_key[key]:
            if self._processing_by_key[key]:
                return False
            self._processing_by_key[key] = True
            return True

    async def _finish_processing(self, context: MemoryRequestContext) -> None:
        key = buffer_key(context)
        async with self._process_lock_by_key[key]:
            self._processing_by_key[key] = False

    async def _context_for_buffer_key(self, project_id: str, key: str) -> MemoryRequestContext | None:
        records = await self.add_buffer.list_buffered_for_key(project_id, key, limit=1)
        if not records:
            return None
        return context_from_record(records[0])

    async def _contexts_for_project(self, project_id: str, *, limit: int) -> list[MemoryRequestContext]:
        contexts: list[MemoryRequestContext] = []
        for pending_key in await self.add_buffer.list_buffer_keys_with_new_records(limit=limit):
            if pending_key.project_id != project_id:
                continue
            context = await self._context_for_buffer_key(pending_key.project_id, pending_key.buffer_key)
            if context is not None:
                contexts.append(context)
        return contexts

    # Core: two-phase process loop (chunking → dispatch)

    async def _process_loop(
        self,
        context: MemoryRequestContext,
        *,
        consistency: str,
        force: bool,
        inline: bool = False,
        trigger_record_id: str | None = None,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> tuple[list[MemoryAddEventItem], int]:
        """Run the two-phase drain loop until no processable episodes remain.

        Returns the generated events plus the number of episodes dispatched in
        this drain, so the async caller can finalize a trigger record that
        produced nothing (``ok`` with empty output) instead of leaving it queued.
        """
        events: list[MemoryAddEventItem] = []
        dispatched = 0
        try:
            rt = self._resolve_add_runtime(context)
            while True:
                # Phase 1: Chunking
                await _raise_if_cancelled(cancel_check, "chunking")
                await _report_progress(progress, "chunking", "Detecting memory episode boundaries.", 18)
                episode_tasks = await self._chunk_episodes(context, force=force, rt=rt)
                if not episode_tasks:
                    break

                # Phase 2: Dispatch
                dispatched += len(episode_tasks)
                await _raise_if_cancelled(cancel_check, "llm_extracting")
                round_events = await (
                    self._dispatch_episodes_inline(
                        episode_tasks,
                        context=context,
                        consistency=consistency,
                        progress=progress,
                        cancel_check=cancel_check,
                        rt=rt,
                    )
                    if inline
                    else self._dispatch_episodes_kafka(
                        episode_tasks,
                        context=context,
                        consistency=consistency,
                        trigger_record_id=trigger_record_id,
                    )
                )
                events.extend(round_events)

                if not inline or not round_events:
                    break
        finally:
            await self._finish_processing(context)
        return events, dispatched

    async def _chunk_episodes(
        self, context: MemoryRequestContext, *, force: bool, rt: _SchemaAddRuntime
    ) -> list[_EpisodeTask]:
        """Split buffered records into episode generation tasks.

        Uses a streaming window approach: entries are processed in windows of
        ``streaming_window_size``.  For each non-final window only completed
        episodes (boundaries that do not touch the window tail) are kept.  The
        remaining messages carry over into the next window.  The final window
        (when *force* is True) keeps all boundaries so every message is consumed.
        """
        records = await self.add_buffer.list_buffered(context, limit=rt.schema_cfg.chunker.max_buffer_size)
        entries = add_record_ops.to_chunker_entries(records)
        if len(entries) < rt.schema_cfg.chunker.min_episode_length:
            if records:
                await self.add_buffer.mark_split_attempted(context, records)
            return []

        sample_text = " ".join(str(e.get("content", "")) for e in entries[:20])
        request_prompts = get_add_prompts(_prompt_language_for_records(records, sample_text))

        detect_force = force or add_record_ops.force_generation(records)
        window_size = rt.chunker.streaming_window_size

        tasks: list[_EpisodeTask] = []
        queued_record_ids: set[str] = set()
        global_offset = 0

        while global_offset < len(entries):
            window_entries = entries[global_offset : global_offset + window_size]
            is_final_window = global_offset + len(window_entries) >= len(entries)
            window_force = detect_force and is_final_window

            boundaries = await traced_awaitable(
                "schema_add.chunk_episodes.detect_boundaries",
                rt.chunker.detect_boundaries(
                    window_entries,
                    force=window_force,
                    boundary_prompt=request_prompts.conv_boundary_detection,
                    resplit_prompt=request_prompts.conv_forced_resplit,
                ),
                attributes={
                    "project_id": context.project_id,
                    "entry_count": len(window_entries),
                    "force": window_force,
                    "chunker.mode": rt.chunker.mode,
                    "chunker.max_messages": rt.chunker.max_messages,
                    "window_offset": global_offset,
                    "total_entries": len(entries),
                },
                record_result=True,
                tracer_name=__name__,
            )

            if not boundaries and window_force and len(window_entries) >= rt.schema_cfg.chunker.min_episode_length:
                boundaries = [EpisodeBoundary(start_idx=0, end_idx=len(window_entries) - 1)]

            if not boundaries:
                if len(window_entries) >= window_size:
                    boundaries = [EpisodeBoundary(start_idx=0, end_idx=len(window_entries) - 1)]
                else:
                    break

            for boundary in boundaries:
                global_start = boundary.start_idx + global_offset
                global_end = boundary.end_idx + global_offset
                episode_records = records[global_start : global_end + 1]
                if not episode_records:
                    continue
                episode_id = str(uuid4())
                await self.add_buffer.mark_episode_queued(context, episode_records, episode_id=episode_id)
                queued_record_ids.update(record.add_record_id for record in episode_records)
                tasks.append(
                    _EpisodeTask(
                        episode_id=episode_id,
                        records=episode_records,
                        chunk_index=len(tasks),
                        chunk_count=0,
                        start_idx=global_start,
                        end_idx=global_end,
                        title=boundary.title,
                    )
                )

            global_offset += boundaries[-1].end_idx + 1

        for task in tasks:
            task.chunk_count = len(tasks)

        remaining_records = [record for record in records if record.add_record_id not in queued_record_ids]
        if remaining_records:
            await self.add_buffer.mark_split_attempted(context, remaining_records)
        return tasks

    async def _dispatch_episodes_kafka(
        self,
        tasks: list[_EpisodeTask],
        *,
        context: MemoryRequestContext,
        consistency: str,
        trigger_record_id: str | None = None,
    ) -> list[MemoryAddEventItem]:
        """Publish episode generation tasks to Kafka."""
        producer = get_producer()
        key = buffer_key(context)
        for task in tasks:
            try:
                await producer.send(
                    SCHEMA_ADD_EPISODE_TOPIC,
                    value={
                        "context": context.model_dump(mode="json"),
                        "add_record_ids": [r.add_record_id for r in task.records],
                        "episode_id": task.episode_id,
                        "consistency": consistency,
                        "trigger_record_id": trigger_record_id,
                    },
                    dispatch_key=key,
                )
            except Exception:
                logger.error(
                    "failed to publish episode task to kafka; restoring records to buffered",
                    episode_id=task.episode_id,
                    exc_info=True,
                )
                await self.add_buffer.restore_buffered(context, task.records, error="kafka episode publish failed")
        return []

    async def _dispatch_episodes_inline(
        self,
        tasks: list[_EpisodeTask],
        *,
        context: MemoryRequestContext,
        consistency: str,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
        rt: _SchemaAddRuntime,
    ) -> list[MemoryAddEventItem]:
        """Execute episode generation tasks in the current process."""
        events: list[MemoryAddEventItem] = []
        for task in tasks:
            task_events = await self._execute_episode_task(
                task,
                context=context,
                consistency=consistency,
                progress=progress,
                cancel_check=cancel_check,
                rt=rt,
            )
            events.extend(task_events)
        return events

    # Episode execution with retry, failure recording

    async def _execute_episode_task(
        self,
        task: _EpisodeTask,
        *,
        context: MemoryRequestContext,
        consistency: str,
        trigger_record_id: str | None = None,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
        rt: _SchemaAddRuntime | None = None,
    ) -> list[MemoryAddEventItem]:
        """Trace and execute one episode generation task."""
        if rt is None:
            rt = self._resolve_add_runtime(context)
        return await traced_awaitable(
            "schema_add.episode_chunk",
            self._execute_episode_task_inner(
                task,
                context=context,
                consistency=consistency,
                trigger_record_id=trigger_record_id,
                rt=rt,
                progress=progress,
                cancel_check=cancel_check,
            ),
            attributes={
                "project_id": context.project_id,
                "episode_id": task.episode_id,
                "chunk.index": task.chunk_index,
                "chunk.count": task.chunk_count,
                "chunk.start_idx": task.start_idx,
                "chunk.end_idx": task.end_idx,
                "chunk.record_count": len(task.records),
                "chunk.title": task.title,
                "consistency": consistency,
            },
            record_result=False,
            tracer_name=__name__,
        )

    async def _execute_episode_task_inner(
        self,
        task: _EpisodeTask,
        *,
        context: MemoryRequestContext,
        consistency: str,
        trigger_record_id: str | None = None,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
        rt: _SchemaAddRuntime | None = None,
    ) -> list[MemoryAddEventItem]:
        if rt is None:
            rt = self._resolve_add_runtime(context)
        for attempt in range(rt.schema_cfg.drain.episode_generation_max_retries):
            try:
                episode_events = await self._generate_episode_memory(
                    task.records,
                    context=context,
                    consistency=consistency,
                    episode_title=task.title,
                    rt=rt,
                    progress=progress,
                    cancel_check=cancel_check,
                )
                await self.add_buffer.mark_processed(
                    context,
                    task.records,
                    episode_id=task.episode_id,
                    events=_events_to_payload(episode_events),
                )
                if rt.schema_cfg.drain.cleanup_processed_buffer:
                    try:
                        await self.add_buffer.delete_processed(context, task.records)
                    except Exception:
                        logger.warning(
                            "failed to cleanup processed buffer records",
                            episode_id=task.episode_id,
                            exc_info=True,
                        )
                # Trigger binding: accumulate this episode's output onto the request
                # that kicked off the drain (async/Kafka path only; inline sync writes
                # the full output via add_sync, so trigger_record_id is None there).
                await suppress_recording_errors(
                    self.recorder.append_add_output(context, trigger_record_id, episode_events),
                    operation="add.schema_add.episode_chunk",
                )
                return episode_events
            except AddStreamCancelled:
                raise
            except Exception as exc:
                if attempt < rt.schema_cfg.drain.episode_generation_max_retries - 1:
                    delay = min(
                        rt.schema_cfg.drain.episode_retry_backoff_base * (2**attempt),
                        rt.schema_cfg.drain.episode_retry_backoff_max,
                    )
                    jitter = delay * random.random()
                    logger.warning(
                        "episode memory generation failed; retrying",
                        attempt=attempt + 1,
                        episode_id=task.episode_id,
                        delay=round(jitter, 2),
                        exc_info=True,
                    )
                    await asyncio.sleep(jitter)
                else:
                    error_msg = str(exc)
                    logger.error(
                        "episode memory generation failed permanently",
                        episode_id=task.episode_id,
                        exc_info=True,
                    )
                    try:
                        await self.add_buffer.mark_failed(context, task.records, error=error_msg)
                    except Exception:
                        logger.error("failed to mark episode records as failed", exc_info=True)
                    if trigger_record_id:
                        await suppress_recording_errors(
                            self._recorder.mark_add_failed(context, trigger_record_id, error_msg),
                            operation="add",
                        )
                    else:
                        await self._record_episode_failure(task.records, context=context)
        return []

    async def _record_episode_failure(self, records: list[BufferedAddRecord], *, context: MemoryRequestContext) -> None:
        """Record a failed episode generation attempt for audit history."""
        records_time = add_record_ops.records_added_datetime(records)
        reconstructed_input = _reconstruct_input_from_records(records)
        await suppress_recording_errors(
            self._recorder.record_add(
                reconstructed_input,
                None,
                ctx=context,
                request_submitted_at=records_time,
                task_completed_at=utcnow(),
            ),
            operation="add",
        )

    async def _generate_episode_memory(
        self,
        records: list[BufferedAddRecord],
        *,
        context: MemoryRequestContext,
        consistency: str,
        episode_title: str = "",
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
        rt: _SchemaAddRuntime | None = None,
    ) -> list[MemoryAddEventItem]:
        """Generate schema entities, vectors, and write events for one episode.

        Orchestrates two parallel siblings per episode:
          - 二.1 episode memory: objectify + episode entity + episode edges.
          - 二.2 schema entity memory: schema selection -> reference recall -> entity generation.
        """
        if rt is None:
            rt = self._resolve_add_runtime(context)
        conversation_text = add_record_ops.to_conversation_text(records)
        if not conversation_text.strip():
            return []

        request_prompts = get_add_prompts(_prompt_language_for_records(records, conversation_text))

        episode_context = add_record_ops.context(records, context)
        event_at = add_record_ops.records_datetime(records)
        added_at = add_record_ops.records_added_datetime(records)
        dialogue_timestamp = add_record_ops.dialogue_timestamp(event_at)
        dialogue_date = dialogue_timestamp.split(" ", 1)[0]

        project_em = rt.project_em
        episode_name_hint = _episode_name_hint(episode_title, conversation_text)

        await _report_progress(progress, "llm_extracting", "Extracting structured memory with LLM.", 35)
        await _raise_if_cancelled(cancel_check, "llm_extracting")
        # 二.1 and 二.2 are parallel siblings, all derived from the raw conversation
        # text. Guard them with one TaskGroup so a failure in the serial 二.2 chain
        # (schema selection -> reference recall -> entity generation) cancels the
        # still-running 二.1 tasks instead of leaving them orphaned across retries.
        try:
            async with asyncio.TaskGroup() as tg:
                # 二.1: episode memory generation (parallel).
                objectify_task = tg.create_task(
                    rt.extractor.objectify_conversation(
                        conversation_text, dialogue_timestamp, prompt_set=request_prompts
                    )
                )
                episode_entity_task = tg.create_task(
                    rt.extractor.generate_episode_entity(
                        conversation_text, dialogue_timestamp, rt.search_fields_max, prompt_set=request_prompts
                    )
                )
                episode_edge_task = tg.create_task(
                    rt.planner.plan_episode_edges(
                        episode_name=episode_name_hint,
                        episode_description=conversation_text,
                        context=episode_context,
                        prompt_set=request_prompts,
                    )
                )
                # 二.2.a schema selection and 二.2.b reference recall (embed, not chat).
                schema_selection_task = tg.create_task(
                    rt.extractor.select_schema(
                        conversation_text,
                        rt.extractor.schema_for_generation(entity_manager=project_em),
                        prompt_set=request_prompts,
                    )
                )
                recall_task = tg.create_task(
                    rt.planner.recall_reference_entities(conversation_text=conversation_text, context=episode_context)
                )

                selected_schema = await schema_selection_task
                reference_entities = await recall_task
                # 二.2.c single entity-generation call.
                raw_memory = await rt.extractor.generate_memory(
                    entity_schema=selected_schema,
                    reference_entities=reference_entities,
                    dialogue_timestamp=dialogue_timestamp,
                    conversation_text=conversation_text,
                    prompt_set=request_prompts,
                    entity_manager=project_em,
                )
                raw_memory = rt.extractor.prepare_raw_memory(raw_memory, dialogue_timestamp)

                objectified_content = await objectify_task
                episode_entity_info = await episode_entity_task
                episode_edges = await episode_edge_task
        except BaseExceptionGroup as group_exc:
            # TaskGroup wraps task failures into an ExceptionGroup. Unwrap to
            # the first real error so the outer retry loop keeps its original
            # exception/message semantics. A pure-cancellation group (only
            # CancelledError) is re-raised unchanged so cooperative shutdown
            # is not mistaken for a retryable failure.
            _cancelled, rest = group_exc.split(asyncio.CancelledError)
            if rest is not None and rest.exceptions:
                raise rest.exceptions[0] from None
            raise
        await _raise_if_cancelled(cancel_check, "memory_planning")

        episode_entity = build_episode_entity(
            objectified_content=objectified_content,
            title=episode_entity_info["title"] or episode_name_hint,
            content=episode_entity_info["content"],
            dialogue_date=dialogue_date,
            search_fields=episode_entity_info["search_fields"],
        )

        await _report_progress(progress, "memory_planning", "Planning memory structure.", 60)
        plan, events = await rt.planner.build_write_plan(
            raw_entities=raw_memory.get("entities", []),
            raw_edges=raw_memory.get("edges", []),
            episode_entity=episode_entity,
            reference_entities=reference_entities,
            episode_edges=episode_edges,
            context=episode_context,
            request_metadata=add_record_ops.metadata(records),
            created_at=added_at,
            progress=progress,
        )

        entity_updates = _split_entity_updates(plan)
        mutation_plan = MemoryDbMutationPlan.from_write_plan(plan)
        mutation_plan.entity_updates.extend(_to_entity_update_commands(entity_updates, consistency=consistency))
        await _report_progress(progress, "ready_to_persist", "Memory is ready to persist.", 82)
        await _raise_if_cancelled(cancel_check, "ready_to_persist")
        await _report_progress(progress, "persisting", "Persisting memory to storage.", 94)
        await self.db_writer.apply_mutation_plan(
            episode_context,
            mutation_plan,
            consistency=consistency,
        )
        return events


def _events_to_payload(events: list[MemoryAddEventItem]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="python") for event in events]


def _episode_name_hint(episode_title: str, conversation_text: str) -> str:
    """Return a chunk-title fallback name for the episode edge prompt (方案1)."""
    title = (episode_title or "").strip()
    if title:
        return title
    first_line = conversation_text.splitlines()[0] if conversation_text else ""
    return first_line[:80] or "Episode"


def _split_entity_updates(plan: MemoryDbWritePlan) -> list[tuple[EntityWrite, list[EntityVectorWrite]]]:
    update_ids = {
        entity.entity_id
        for entity in plan.entities
        if isinstance(entity.metadata, dict) and entity.metadata.get("merge_action") == "update"
    }
    if not update_ids:
        return []

    vectors_by_entity: dict[str, list[EntityVectorWrite]] = defaultdict(list)
    remaining_vectors: list[EntityVectorWrite] = []
    for vector in plan.entity_vectors:
        owner_id = vector.entity_id.split("#sf", 1)[0]
        if owner_id in update_ids:
            vectors_by_entity[owner_id].append(vector)
        else:
            remaining_vectors.append(vector)

    updates: list[tuple[EntityWrite, list[EntityVectorWrite]]] = []
    remaining_entities: list[EntityWrite] = []
    for entity in plan.entities:
        if entity.entity_id in update_ids:
            updates.append((entity, vectors_by_entity.get(entity.entity_id, [])))
        else:
            remaining_entities.append(entity)

    plan.entities = remaining_entities
    plan.entity_vectors = remaining_vectors
    return updates


def _to_entity_update_commands(
    updates: list[tuple[EntityWrite, list[EntityVectorWrite]]],
    *,
    consistency: str,
) -> list[MemoryDbEntityUpdateCommand]:
    commands: list[MemoryDbEntityUpdateCommand] = []
    for entity, vectors in updates:
        commands.append(
            MemoryDbEntityUpdateCommand(
                entity_id=entity.entity_id,
                entity=entity,
                core_vector=next((vector for vector in vectors if vector.entity_id == entity.entity_id), None),
                search_field_vectors=[vector for vector in vectors if vector.entity_id != entity.entity_id],
                consistency=consistency,
            )
        )
    return commands


def _reconstruct_input_from_records(records: list[BufferedAddRecord]) -> AddPipelineInput:
    """Rebuild a minimal add pipeline input from buffer records."""
    messages = []
    metadata: dict[str, Any] = {}
    prompt_language: str | None = None
    for record in records:
        payload = record.payload
        record_messages = payload.get("messages", [])
        messages.extend(record_messages)
        if not metadata and payload.get("metadata"):
            metadata = payload["metadata"]
        if prompt_language is None and payload.get("prompt_language") in {"EN", "ZH"}:
            prompt_language = payload["prompt_language"]
    try:
        return AddPipelineInput(messages=messages, metadata=metadata, prompt_language=prompt_language)
    except Exception:
        return AddPipelineInput(metadata=metadata, prompt_language=prompt_language)


def _prompt_language_for_records(records: list[BufferedAddRecord], text: str) -> str:
    """Return request-level prompt language or fall back to auto detection."""

    for record in records:
        value = record.payload.get("prompt_language")
        if value in {"EN", "ZH"}:
            return value
    return detect_prompt_language(
        text,
        fallback=get_config().algo_config.common.prompt_language,
    )


def _default_consistency() -> str:
    value = get_config().database.default_consistency
    return value if value in {"fast", "strong"} else "fast"
