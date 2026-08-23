"""Orchestration of trajectory task+experience building into a write plan."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

from ....config import MessageChunkerConfig, TrajectoryAddConfig
from ....logging import get_logger
from ....typing import (
    AddPipelineInput,
    Entity,
    EntityVectorWrite,
    EntityWrite,
    MemoryAddEventItem,
    MemoryDbUpdateCommand,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryWrite,
    NormalizedMessage,
    PreprocessedText,
    SourceRef,
    VectorWrite,
)
from ...chunker import MessageChunker, SourceAwareSegment
from ...id import generate_entity_id, generate_trajectory_source_id
from ...memory_modeling.vanilla import build_extracted_from_edge, build_task_experience_edge
from ...text import TextPreprocessor, MemoryVectorizer
from ..vanilla._entity import build_entity_write
from ..vanilla._update_commands import build_update_command
from .dedup import ExperienceDeduplicator
from .extractor import TrajectoryExperienceExtractor
from .schema import ExtractedExperienceCandidate, ExperienceResolution

logger = get_logger(__name__)

# Garbled/zero-width bytes that slip into trajectory files and would otherwise
# fragment identical tasks into separate entity names (U+FFFD replacement char,
# BOM, zero-width joiners, directional marks, ASCII control).
_GARBLED_TEXT_RE = re.compile(r"[\ufeff\u0000-\u001f\u007f\u200b-\u200d\u200e\u200f\u2060\ufffd]")

def _clean_task_text(text: str) -> str:
    return _GARBLED_TEXT_RE.sub("", text or "").strip()


def _normalized_to_turn(message: NormalizedMessage) -> dict[str, Any]:
    """Convert a chunker-normalized message into the turn dict the prompt expects.

    ``message_index`` keeps the original trajectory position so extracted
    ``source_message_indices`` stay stable across chunks and imports.
    """
    return {
        "message_index": message.message_index,
        "role": str(message.role),
        "text": message.text,
        "timestamp": message.timestamp,
    }


def _dominant_lang(preprocessed: list[PreprocessedText]) -> str:
    zh = sum(1 for pp in preprocessed if pp.lang == "zh")
    return "zh" if zh and zh >= max(1, len(preprocessed) // 2) else "en"


def _task_ref(task_entity_id: str, task_text: str) -> dict[str, str]:
    return {"task_entity_id": task_entity_id, "task_text": task_text}


def _existing_task_refs(metadata: dict[str, Any] | None) -> list[dict[str, str]]:
    metadata = dict(metadata or {})
    refs = [ref for ref in (metadata.get("task_refs") or []) if isinstance(ref, dict)]
    if refs:
        return refs
    # Migration fallback for nodes written before task_refs existed: treat the
    # recorded scalar creator pair as the first (and only) task reference.
    entity_id = metadata.get("task_entity_id")
    if entity_id:
        return [_task_ref(str(entity_id), str(metadata.get("task_text") or ""))]
    return []


def _merge_task_refs(
    existing_metadata: dict[str, Any] | None,
    task_entity_id: str,
    task_text: str,
) -> list[dict[str, str]]:
    """Append the current task to a shared experience's task_refs, de-duplicated."""
    refs = _existing_task_refs(existing_metadata)
    seen = {ref.get("task_entity_id") for ref in refs}
    if task_entity_id not in seen:
        refs.append(_task_ref(task_entity_id, task_text))
    return refs


def _source_segment(turn: dict[str, Any], source_ref: SourceRef) -> SourceAwareSegment:
    text = turn.get("text") or ""
    return SourceAwareSegment(
        segment_id=f"traj_msg{turn['message_index']}",
        text=text,
        source_ref=source_ref,
        message_index=int(turn["message_index"]),
        role=turn.get("role"),
        timestamp=turn.get("timestamp"),
        start_offset=0,
        end_offset=len(text),
        metadata={"message_type": "trajectory", "raw_role": turn.get("role")},
    )


class TrajectoryExperienceBuilder:
    """Build one task entity plus reusable experience memories for a trajectory.

    Components are injected so callers can reuse the config-driven text,
    LLM, and embedding resources already owned by the runtime.
    """

    def __init__(
        self,
        *,
        text_preprocessor: TextPreprocessor,
        extractor: TrajectoryExperienceExtractor,
        deduplicator: ExperienceDeduplicator,
        vectorizer: MemoryVectorizer,
        chunker_config: MessageChunkerConfig | None = None,
        llm_client=None,
    ) -> None:
        self._text_preprocessor = text_preprocessor
        self._extractor = extractor
        self._deduplicator = deduplicator
        self._vectorizer = vectorizer
        self._chunker_config = chunker_config
        self._llm_client = llm_client

    async def _preprocess(self, text: str) -> PreprocessedText:
        return await asyncio.to_thread(self._text_preprocessor.preprocess_text, text, include_entities=False)

    async def build(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        consistency: str = "fast",
        config: TrajectoryAddConfig | None = None,
    ) -> tuple[MemoryDbWritePlan, list[MemoryAddEventItem], list[MemoryDbUpdateCommand]]:
        cfg = config or TrajectoryAddConfig()
        task_text = _clean_task_text(inp.task if inp.task else "")
        now = datetime.now(UTC)

        memories: list[MemoryWrite] = []
        entities_by_id: dict[str, EntityWrite] = {}
        relationships: list = []
        vectors: list[VectorWrite] = []
        entity_vectors: list[EntityVectorWrite] = []
        pending_memory_vectors: list[tuple[str, PreprocessedText, str]] = []
        update_commands: list[MemoryDbUpdateCommand] = []
        events: list[MemoryAddEventItem] = []

        if not task_text:
            return (
                MemoryDbWritePlan(entities=list(entities_by_id.values())),
                events,
                update_commands,
            )

        # 1. Task entity: one deterministic entity per task text.
        task_entity = Entity(
            name=task_text,
            canonical_name=task_text,
            entity_type="task",
            description=f"trajectory task: {task_text}",
            extractor="trajectory",
        )
        task_entity_id = generate_entity_id(context.project_id, task_entity)
        entities_by_id[task_entity_id] = build_entity_write(task_entity, task_entity_id, context, now)

        # 2. Reuse vanilla's MessageChunker so long trajectories stay within LLM
        #    budget. Extractable messages keep their original message_index for
        #    stable per-message source refs and cross-chunk provenance.
        chunking_result = await MessageChunker(self._chunker_config, llm_client=self._llm_client).split(
            inp.messages
        )
        turn_by_index: dict[int, dict[str, Any]] = {}
        preprocessed_by_index: dict[int, PreprocessedText] = {}
        source_by_index: dict[int, SourceRef] = {}
        for prepared in chunking_result.chunks:
            for message in prepared.extractable_messages:
                index = message.message_index
                if index in turn_by_index:
                    continue
                turn = _normalized_to_turn(message)
                turn_by_index[index] = turn
                pp = await self._preprocess(turn["text"])
                preprocessed_by_index[index] = pp
                source_by_index[index] = SourceRef(
                    source_type="message",
                    source_id=generate_trajectory_source_id(context.project_id, index, pp.content_hash),
                    is_parsed=True,
                    metadata={"message_index": index},
                )
        lang = _dominant_lang(list(preprocessed_by_index.values()))

        # 3. Extract experiences per chunk (task text stays fixed for the whole trace).
        candidates: list[ExtractedExperienceCandidate] = []
        for prepared in chunking_result.chunks:
            chunk_turns = [_normalized_to_turn(message) for message in prepared.extractable_messages]
            if not chunk_turns:
                continue
            chunk_candidates = await self._extractor.extract(task_text, chunk_turns, lang, context)
            for index, candidate in enumerate(chunk_candidates):
                candidates.append(candidate.model_copy(update={"ref_id": f"c{prepared.chunk.chunk_index}_e{index}"}))

        # 4. Resolve each candidate; reuse experiences already created by this
        #    import so repeated lessons across chunks do not duplicate nodes.
        batch_target_by_hash: dict[str, str] = {}
        import_experiences: list[tuple[str, str]] = []
        for candidate in candidates:
            content = candidate.content.strip()
            if len(content) < cfg.min_content_chars:
                continue
            pp = await self._preprocess(content)

            # Guard against byte-identical duplicates within this very import:
            # writes are batched at the end, so a repeated candidate cannot recall
            # the node it would otherwise create - link it to the first target.
            batch_duplicate_target = batch_target_by_hash.get(pp.content_hash)
            if batch_duplicate_target is not None:
                relationships.append(
                    build_task_experience_edge(task_entity_id, batch_duplicate_target, task_text, context)
                )
                edge_count = 1
                for message_index in candidate.source_message_indices:
                    source_ref = source_by_index.get(message_index)
                    if source_ref is None:
                        continue
                    segment = _source_segment(turn_by_index[message_index], source_ref)
                    relationships.append(
                        build_extracted_from_edge(batch_duplicate_target, source_ref, context, segment)
                    )
                    edge_count += 1
                events.append(
                    MemoryAddEventItem(
                        operation="add",
                        content=content,
                        memory_id=batch_duplicate_target,
                        mem_type="experience",
                        confidence=candidate.confidence,
                        graph_edge_count=edge_count,
                    )
                )
                continue

            resolution: ExperienceResolution = await self._deduplicator.resolve(
                context,
                content,
                pp,
                lang=lang,
                import_experiences=import_experiences,
            )
            target_id = resolution.target_memory_id
            edge_count = 0
            batch_target_by_hash[pp.content_hash] = target_id

            if resolution.action == "create":
                memory = self._new_experience_memory(
                    candidate,
                    pp,
                    task_entity_id,
                    task_text,
                    context,
                    now,
                    lang=lang,
                    memory_id=target_id,
                )
                memories.append(memory)
                pending_memory_vectors.append((memory.memory_id, pp, memory.content))
                event_content = memory.content
            elif resolution.action == "merge":
                merged_content = resolution.merged_content or content
                existing_metadata = resolution.existing_memory.metadata if resolution.existing_memory is not None else None
                update_commands.append(
                    build_update_command(
                        target_id,
                        merged_content,
                        context,
                        now,
                        metadata_refresh={
                            "task_refs": _merge_task_refs(existing_metadata, task_entity_id, task_text)
                        },
                    )
                )
                event_content = merged_content
            else:  # reuse
                event_content = content

            import_experiences.append((target_id, event_content))

            relationships.append(build_task_experience_edge(task_entity_id, target_id, task_text, context))
            edge_count += 1
            for message_index in candidate.source_message_indices:
                source_ref = source_by_index.get(message_index)
                if source_ref is None:
                    continue
                segment = _source_segment(turn_by_index[message_index], source_ref)
                relationships.append(build_extracted_from_edge(target_id, source_ref, context, segment))
                edge_count += 1

            events.append(
                MemoryAddEventItem(
                    operation="update" if resolution.action == "merge" else "add",
                    content=event_content,
                    memory_id=target_id,
                    mem_type="experience",
                    confidence=candidate.confidence,
                    graph_edge_count=edge_count,
                )
            )

        # 5. Vectorize new experiences (and optionally the task entity).
        if pending_memory_vectors:
            vectors, _ = await self._vectorizer.vectorize_many(
                pending_memory_vectors,
                consistency,
                batch_size=cfg.embedding_batch_size,
            )
        if cfg.enable_task_entity_embedding and entities_by_id:
            entity_vectors, _ = await self._vectorizer.vectorize_entities(
                list(entities_by_id.values()),
                memories_by_entity={},
                consistency=consistency,
                batch_size=cfg.embedding_batch_size,
            )

        plan = MemoryDbWritePlan(
            memories=memories,
            entities=list(entities_by_id.values()),
            vectors=vectors,
            entity_vectors=entity_vectors,
            relationships=relationships,
        )
        return plan, events, update_commands

    def _new_experience_memory(
        self,
        candidate: ExtractedExperienceCandidate,
        preprocessed: PreprocessedText,
        task_entity_id: str,
        task_text: str,
        context: MemoryRequestContext,
        now: datetime,
        *,
        lang: str,
        memory_id: str,
    ) -> MemoryWrite:
        source_indices = [int(index) for index in candidate.source_message_indices]
        return MemoryWrite(
            memory_id=memory_id,
            account_id=context.account_id,
            project_id=context.project_id,
            api_key_uuid=context.api_key_uuid,
            user_id=context.user_id,
            app_id=context.app_id,
            session_id=context.session_id,
            agent_id=context.agent_id,
            request_id=context.request_id,
            content=candidate.content,
            mem_type="experience",
            mem_extract_type="trajectory",
            mem_extract_version="trajectory_experience_v1",
            metadata={
                "task_entity_id": task_entity_id,
                "task_text": task_text,
                "task_refs": [_task_ref(task_entity_id, task_text)],
                "source_message_indices": source_indices,
                "content_hash": preprocessed.content_hash,
                "bm25_text": preprocessed.bm25_text,
                "tokens": list(preprocessed.tokens),
                "lang": preprocessed.lang or lang,
                "confidence": candidate.confidence,
                "importance": candidate.importance,
                "extractor": "trajectory_experience",
                "reason": candidate.reason if candidate.reason else None,
                "task_experience": True,
            },
            created_at=now,
            root_id=[memory_id],
        )