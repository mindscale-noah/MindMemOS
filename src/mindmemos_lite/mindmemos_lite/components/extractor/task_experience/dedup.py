"""Recall and LLM-judged dedup for trajectory experience candidates."""

from __future__ import annotations

import json
from typing import Any

from ....logging import get_logger
from ....persistence import MemoryPersistence
from ....typing import (
    FieldCondition,
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemoryView,
    PreprocessedText,
    SearchFilter,
)
from ...id import generate_experience_id
from .schema import ExperienceDedupVerdict, ExperienceResolution, parse_experience_json

logger = get_logger(__name__)

_EXPERIENCE_TYPE = "experience"


def _dedup_prompt_messages(candidate_text: str, existing: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    from ....prompts import get_trajectory_dedup_prompt

    return [
        {"role": "system", "content": get_trajectory_dedup_prompt(lang)},
        {"role": "user", "content": json.dumps({"new_candidate": candidate_text, "existing": existing}, ensure_ascii=False)},
    ]


def _parse_verdict(raw: Any, hit_count: int) -> ExperienceDedupVerdict:
    if not isinstance(raw, dict):
        return ExperienceDedupVerdict(verdict="different")
    verdict = str(raw.get("verdict") or "different").lower().strip()
    if verdict not in {"same_no_delta", "same_with_delta", "different"}:
        verdict = "different"
    match_index = raw.get("match_index")
    if isinstance(match_index, str) and match_index.isdigit():
        match_index = int(match_index)
    if not isinstance(match_index, int) or isinstance(match_index, bool) or not (0 <= match_index < hit_count):
        match_index = None
    merged_content = raw.get("merged_content") if isinstance(raw.get("merged_content"), str) else None
    return ExperienceDedupVerdict(verdict=verdict, match_index=match_index, merged_content=merged_content)


class ExperienceDeduplicator:
    """Resolve each experience candidate against existing experiences.

    Recall surfaces dense-semantic candidates scoped to ``mem_type=experience``;
    the LLM then decides create / merge-with-delta / reuse. When either client is
    unavailable the resolver conservatively creates a new node.
    """

    def __init__(
        self,
        *,
        persistence: MemoryPersistence,
        embed_client=None,
        llm_client=None,
        top_k: int = 5,
    ) -> None:
        self._persistence = persistence
        self._embed_client = embed_client
        self._llm_client = llm_client
        self._top_k = max(1, top_k)

    async def recall(
        self,
        ctx: MemoryRequestContext,
        candidate_text: str,
        preprocessed: PreprocessedText,
    ) -> list[MemoryDbSearchHit]:
        if self._embed_client is None:
            return []
        response = await self._embed_client.embed(task="memory.trajectory.experience.recall", text=candidate_text)
        if not response or not response.embeddings or not response.embeddings[0]:
            return []
        result = await self._persistence.search_dense(
            ctx,
            MemoryDbSearchQuery(
                query=candidate_text,
                top_k=self._top_k,
                mode="semantic",
                ranking="score",
                filters=SearchFilter(
                    must=[
                        FieldCondition(field="mem_type", op="match", value=_EXPERIENCE_TYPE),
                        FieldCondition(field="status", op="match", value="active"),
                    ]
                ),
            ),
            dense_vector=list(response.embeddings[0]),
        )
        # Keep only hits whose memory content resolves; the LLM judges against text.
        return [hit for hit in result.hits if hit.memory is not None]

    async def judge(
        self,
        ctx: MemoryRequestContext,
        candidate_text: str,
        hits: list[MemoryDbSearchHit],
        lang: str,
    ) -> ExperienceDedupVerdict:
        if not hits:
            return ExperienceDedupVerdict(verdict="different")
        if self._llm_client is None:
            logger.warning(
                "trajectory_experience_dedup_llm_unavailable",
                request_id=ctx.request_id,
                hit_count=len(hits),
            )
            return ExperienceDedupVerdict(verdict="different")
        existing = [{"memory_id": hit.memory_id, "content": hit.memory.content} for hit in hits]
        try:
            response = await self._llm_client.chat(
                task="memory.add.trajectory_experience_dedup",
                messages=_dedup_prompt_messages(candidate_text, existing, lang),
                format_parser=parse_experience_json,
            )
            return _parse_verdict(response.parsed if response is not None else None, len(hits))
        except Exception:
            logger.warning(
                "trajectory_experience_dedup_failed",
                request_id=ctx.request_id,
                exc_info=True,
            )
            return ExperienceDedupVerdict(verdict="different")

    async def resolve(
        self,
        ctx: MemoryRequestContext,
        candidate_text: str,
        preprocessed: PreprocessedText,
        *,
        lang: str,
        import_experiences: list[tuple[str, str]] | None = None,
    ) -> ExperienceResolution:
        hits = await self.recall(ctx, candidate_text, preprocessed)

        # Include experiences already resolved by THIS import: writes are batched
        # at the end of the trajectory pipeline, so a second chunk that repeats
        # the lesson cannot recall them via the database yet. Surfacing them to
        # the LLM lets long traces dedup cross-chunk instead of duplicating nodes.
        existing_ids: set[str] = set()
        for hit in hits:
            existing_ids.add(hit.memory_id)
        for memory_id, content in import_experiences or ():
            if not content or not content.strip() or memory_id in existing_ids:
                continue
            hits.append(
                MemoryDbSearchHit(
                    memory_id=memory_id,
                    score=1.0,
                    memory=MemoryView(
                        memory_id=memory_id,
                        project_id=ctx.project_id,
                        content=content,
                        mem_type="experience",
                        status="active",
                    ),
                    source="import",
                )
            )
            existing_ids.add(memory_id)

        verdict = await self.judge(ctx, candidate_text, hits, lang)

        def _create() -> ExperienceResolution:
            memory_id = generate_experience_id(ctx.project_id, preprocessed.normalized_text)
            return ExperienceResolution(action="create", target_memory_id=memory_id, preprocessed=preprocessed)

        if verdict.verdict == "different":
            return _create()

        match_index = verdict.match_index
        if match_index is None or not (0 <= match_index < len(hits)):
            # LLM claimed a match but pointed outside the recalled list; stay safe.
            return _create()

        target = hits[match_index].memory_id
        matched_memory = hits[match_index].memory
        if verdict.verdict == "same_with_delta":
            return ExperienceResolution(
                action="merge",
                target_memory_id=target,
                merged_content=verdict.merged_content or candidate_text,
                existing_memory=matched_memory,
            )
        return ExperienceResolution(action="reuse", target_memory_id=target, existing_memory=matched_memory)