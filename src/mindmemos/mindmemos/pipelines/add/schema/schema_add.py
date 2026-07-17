"""Schema add-memory pipeline implementation."""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from ....components.chunker import EpisodeBoundary, EpisodesChunker
from ....components.extractor import _records as add_record_ops
from ....components.extractor.schema import (
    SchemaAddExtractor,
    SchemaAddPlanner,
    SchemaSearchFieldExtractor,
    build_episode_entity,
)
from ....components.memory_modeling.schema import EntityManager, get_entity_manager
from ....components.text import SparseVectorEncoder, detect_prompt_language, get_text_preprocessor
from ....config import get_config
from ....infra.kafka import get_producer
from ....llm import EmbedClient, LLMClient, get_embed_client, get_llm_client
from ....logging import get_logger, traced, traced_awaitable
from ....prompts import AddPromptSet, get_add_prompts
from ....typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
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
    search_field_extractor: SchemaSearchFieldExtractor
    use_search_fields: bool
    search_fields_max: int
    episode_search_fields_augment: bool
    episode_augment_count: int


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
        enable_entity_merge_decision: bool | None = None,
        entity_recall_top_k: int | None = None,
        max_merge_retries: int | None = None,
        use_property_merge: bool | None = None,
        secondary_search_limit: int | None = None,
        secondary_search_retries: int | None = None,
        use_search_fields: bool | None = None,
        search_fields_max: int | None = None,
        episode_search_fields_augment: bool | None = None,
        episode_augment_count: int | None = None,
        higher_order_enabled: bool | None = None,
        higher_order_top_k: int | None = None,
        higher_order_min_evidence_count: int | None = None,
        episode_edge_top_k: int | None = None,
        prompt_language: str | None = None,
        prompt_set: AddPromptSet | None = None,
        search_field_extractor: SchemaSearchFieldExtractor | None = None,
        extractor: SchemaAddExtractor | None = None,
        planner: SchemaAddPlanner | None = None,
        consistency: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # This pipeline is held by a process-wide singleton (MemoryService.
        # _algorithm_add_pipelines), so it MUST stay project-agnostic: all
        # project-scoped deps (LLM/embed clients, entity manager, prompts, chunker,
        # extractor, planner, search-field extractor, text preprocessor, sparse
        # encoder, and every algo parameter) are resolved per drain loop from the
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
        self._explicit_search_field_extractor = search_field_extractor
        self._explicit_prompts = prompt_set
        self._explicit_prompt_language = prompt_language
        # Algo overrides (None -> use the request-scoped config value at drain time).
        self._explicit_enable_schema_selection = enable_schema_selection
        self._explicit_enable_entity_merge_decision = enable_entity_merge_decision
        self._explicit_entity_recall_top_k = entity_recall_top_k
        self._explicit_max_merge_retries = max_merge_retries
        self._explicit_use_property_merge = use_property_merge
        self._explicit_secondary_search_limit = secondary_search_limit
        self._explicit_secondary_search_retries = secondary_search_retries
        self._explicit_use_search_fields = use_search_fields
        self._explicit_search_fields_max = search_fields_max
        self._explicit_episode_search_fields_augment = episode_search_fields_augment
        self._explicit_episode_augment_count = episode_augment_count
        self._explicit_higher_order_enabled = higher_order_enabled
        self._explicit_higher_order_top_k = higher_order_top_k
        self._explicit_higher_order_min_evidence_count = higher_order_min_evidence_count
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

        enable_entity_merge_decision = _override(
            self._explicit_enable_entity_merge_decision, schema_cfg.merge.enable_entity_merge_decision
        )
        entity_recall_top_k = _override(self._explicit_entity_recall_top_k, schema_cfg.merge.entity_recall_top_k)
        max_merge_retries = _override(self._explicit_max_merge_retries, schema_cfg.merge.max_merge_retries)
        use_property_merge = _override(self._explicit_use_property_merge, schema_cfg.merge.use_property_merge)
        secondary_search_limit = _override(
            self._explicit_secondary_search_limit, schema_cfg.merge.secondary_search_limit
        )
        secondary_search_retries = _override(
            self._explicit_secondary_search_retries, schema_cfg.merge.secondary_search_retries
        )
        higher_order_enabled = _override(self._explicit_higher_order_enabled, schema_cfg.higher_order.enabled)
        higher_order_top_k = _override(self._explicit_higher_order_top_k, schema_cfg.higher_order.top_k)
        higher_order_min_evidence_count = _override(
            self._explicit_higher_order_min_evidence_count, schema_cfg.higher_order.min_evidence_count
        )
        episode_edge_top_k = _override(self._explicit_episode_edge_top_k, schema_cfg.episode_edge.top_k)

        planner = self._explicit_planner or SchemaAddPlanner(
            llm_client=llm_client,
            embed_client=embed_client,
            db_reader=self.db_reader,
            db_writer=self.db_writer,
            entity_manager=project_em,
            prompt_set=prompts,
            enable_entity_merge_decision=enable_entity_merge_decision,
            entity_recall_top_k=entity_recall_top_k,
            max_merge_retries=max_merge_retries,
            use_property_merge=use_property_merge,
            secondary_search_limit=secondary_search_limit,
            secondary_search_retries=secondary_search_retries,
            higher_order_enabled=higher_order_enabled,
            higher_order_top_k=higher_order_top_k,
            higher_order_min_evidence_count=higher_order_min_evidence_count,
            episode_edge_top_k=episode_edge_top_k,
            max_entity_resolve_concurrency=schema_cfg.extraction.max_entity_resolve_concurrency,
            max_entities_per_conversation=schema_cfg.extraction.max_entities_per_conversation,
            max_properties_per_entity=schema_cfg.extraction.max_properties_per_entity,
            secondary_search_retry_backoff_base=schema_cfg.merge.secondary_search_retry_backoff_base,
            secondary_search_retry_backoff_max=schema_cfg.merge.secondary_search_retry_backoff_max,
            text_preprocessor=text_preprocessor,
            sparse_encoder=sparse_encoder,
        )

        search_field_extractor = self._explicit_search_field_extractor or SchemaSearchFieldExtractor(
            llm_client=llm_client,
            prompt_set=prompts,
        )

        use_search_fields = _override(self._explicit_use_search_fields, schema_cfg.extraction.use_search_fields)
        search_fields_max = _override(self._explicit_search_fields_max, schema_cfg.extraction.search_fields_max)
        episode_search_fields_augment = _override(
            self._explicit_episode_search_fields_augment, schema_cfg.extraction.episode_search_fields_augment
        )
        episode_augment_count = _override(
            self._explicit_episode_augment_count, schema_cfg.extraction.episode_augment_count
        )

        return _SchemaAddRuntime(
            schema_cfg=schema_cfg,
            project_em=project_em,
            chunker=chunker,
            extractor=extractor,
            planner=planner,
            search_field_extractor=search_field_extractor,
            use_search_fields=use_search_fields,
            search_fields_max=search_fields_max,
            episode_search_fields_augment=episode_search_fields_augment,
            episode_augment_count=episode_augment_count,
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
    ) -> list[MemoryAddEventItem]:
        """Start the drain loop and wait for generated events."""
        key = buffer_key(context)
        while True:
            async with self._process_lock_by_key[key]:
                if not self._processing_by_key[key]:
                    self._processing_by_key[key] = True
                    break
            await asyncio.sleep(0.05)
        events, _dispatched = await self._process_loop(context, consistency=consistency, force=force, inline=True)
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
    ) -> tuple[list[MemoryAddEventItem], int]:
        """Run the two-phase drain loop until no processable episodes remain.

        Returns the generated events plus the number of episodes dispatched in
        this drain, so the async caller can finalize a trigger record that
        produced nothing (``ok`` with empty output) instead of leaving it queued.
        """
        events: list[MemoryAddEventItem] = []
        dispatched = 0
        rt = self._resolve_add_runtime(context)
        try:
            while True:
                # Phase 1: Chunking
                episode_tasks = await self._chunk_episodes(context, force=force, rt=rt)
                if not episode_tasks:
                    break

                # Phase 2: Dispatch
                dispatched += len(episode_tasks)
                round_events = await (
                    self._dispatch_episodes_inline(episode_tasks, context=context, consistency=consistency, rt=rt)
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
        detected_lang = detect_prompt_language(
            sample_text,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = get_add_prompts(detected_lang)

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
        rt: _SchemaAddRuntime,
    ) -> list[MemoryAddEventItem]:
        """Execute episode generation tasks in the current process."""
        events: list[MemoryAddEventItem] = []
        for task in tasks:
            task_events = await self._execute_episode_task(
                task, context=context, consistency=consistency, rt=rt
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
                    rt=rt,
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
        rt: _SchemaAddRuntime | None = None,
    ) -> list[MemoryAddEventItem]:
        """Generate schema entities, vectors, and write events for one episode."""
        if rt is None:
            rt = self._resolve_add_runtime(context)
        conversation_text = add_record_ops.to_conversation_text(records)
        if not conversation_text.strip():
            return []

        detected_lang = detect_prompt_language(
            conversation_text,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = get_add_prompts(detected_lang)

        episode_context = add_record_ops.context(records, context)
        event_at = add_record_ops.records_datetime(records)
        added_at = add_record_ops.records_added_datetime(records)
        dialogue_timestamp = add_record_ops.dialogue_timestamp(event_at)

        project_em = rt.project_em

        # Kick off the three independent LLM calls together, but guard them
        # with a TaskGroup. Schema selection is awaited first, so if it (or
        # the synchronous extract/prepare steps that follow) raises, the
        # still-running objectify/description tasks are cancelled instead of
        # being left as orphans. Without this, the outer episode-retry loop
        # would spawn a fresh trio on top of the stranded ones, multiplying
        # LLM calls and piling up concurrency during backend outages.
        try:
            async with asyncio.TaskGroup() as tg:
                objectify_task = tg.create_task(
                    rt.extractor.objectify_conversation(
                        conversation_text, dialogue_timestamp, prompt_set=request_prompts
                    )
                )
                description_task = tg.create_task(
                    rt.extractor.generate_episode_description(
                        conversation_text, dialogue_timestamp, prompt_set=request_prompts
                    )
                )
                schema_selection_task = tg.create_task(
                    rt.extractor.select_schema(
                        conversation_text,
                        rt.extractor.schema_for_generation(entity_manager=project_em),
                        prompt_set=request_prompts,
                    )
                )

                selected_schema = await schema_selection_task
                raw_memory = await rt.extractor.extract_memory(
                    entity_schema=selected_schema,
                    dialogue_timestamp=dialogue_timestamp,
                    conversation_text=conversation_text,
                    prompt_set=request_prompts,
                    entity_manager=project_em,
                )

                _raw_before_prepare = raw_memory.get("entities", [])
                logger.info(
                    "schema_add drain: BEFORE prepare_raw_memory: %d entities, types=%s",
                    len(_raw_before_prepare),
                    [e.get("entity_type") for e in _raw_before_prepare],
                )

                raw_memory = rt.extractor.prepare_raw_memory(raw_memory, dialogue_timestamp)

                _raw_entities = raw_memory.get("entities", [])
                _entity_types = [e.get("entity_type") for e in _raw_entities]
                logger.info(
                    "schema_add drain: AFTER prepare_raw_memory: %d entities, types=%s, selected_schema_types=%s",
                    len(_raw_entities),
                    _entity_types,
                    [s.get("entity_type") for s in selected_schema],
                )

                objectified_content = await objectify_task
                episode_description = await description_task
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
        episode_search_fields = (
            await rt.search_field_extractor.extract_search_fields(
                entities=raw_memory.get("entities", []),
                context_text=conversation_text,
                max_fields=rt.search_fields_max,
                augment=rt.episode_search_fields_augment,
                augment_count=rt.episode_augment_count,
                prompt_set=request_prompts,
            )
            if rt.use_search_fields
            else []
        )
        episode_entity = build_episode_entity(
            objectified_content=objectified_content,
            episode_description=episode_description,
            dialogue_date=dialogue_timestamp.split(" ", 1)[0],
            search_fields=episode_search_fields,
        )

        plan, events, pending_archives, pending_updates = await rt.planner.build_write_plan(
            raw_entities=raw_memory.get("entities", []),
            raw_edges=raw_memory.get("edges", []),
            episode_entity=episode_entity,
            context=episode_context,
            request_metadata=add_record_ops.metadata(records),
            created_at=added_at,
            episode_time=dialogue_timestamp,
            prompt_set=request_prompts,
        )

        entity_updates = _split_entity_updates(plan)
        memory_update_commands = await rt.planner.build_memory_update_commands(
            episode_context,
            pending_updates,
            consistency=consistency,
        )
        memory_delete_commands = rt.planner.build_archive_memory_commands(pending_archives, consistency=consistency)
        mutation_plan = MemoryDbMutationPlan.from_write_plan(plan)
        mutation_plan.entity_updates.extend(_to_entity_update_commands(entity_updates, consistency=consistency))
        mutation_plan.memory_updates.extend(memory_update_commands)
        mutation_plan.memory_deletes.extend(memory_delete_commands)
        write_result = await self.db_writer.apply_mutation_plan(
            episode_context,
            mutation_plan,
            consistency=consistency,
        )
        update_results = write_result.mutations[: len(memory_update_commands)]
        update_events = rt.planner.memory_update_events(pending_updates, update_results)
        return events + update_events


def _events_to_payload(events: list[MemoryAddEventItem]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="python") for event in events]


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
    for record in records:
        payload = record.payload
        record_messages = payload.get("messages", [])
        messages.extend(record_messages)
        if not metadata and payload.get("metadata"):
            metadata = payload["metadata"]
    try:
        return AddPipelineInput(messages=messages, metadata=metadata)
    except Exception:
        return AddPipelineInput(metadata=metadata)


def _default_consistency() -> str:
    value = get_config().database.default_consistency
    return value if value in {"fast", "strong"} else "fast"
