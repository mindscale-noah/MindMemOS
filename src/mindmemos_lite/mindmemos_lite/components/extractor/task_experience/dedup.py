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
    PreprocessedText,
    SearchFilter,
)
from ...id import generate_experience_id
from .schema import ExperienceDedupVerdict, ExperienceResolution, parse_experience_json

logger = get_logger(__name__)

_EXPERIENCE_TYPE = "experience"

# How many judge rounds we run when the model drops one or more candidates. Each
# round only re-sends the still-missing candidates, so already-returned verdicts
# are never thrown away.
_DEDUP_MAX_ATTEMPTS = 3


def _parsed_candidate_ids(raw: Any, wanted: set[int]) -> set[int]:
    """candidate_ids for which the model actually returned a verdict entry."""

    entries = raw.get("verdicts") if isinstance(raw, dict) else None
    covered: set[int] = set()
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        candidate_id = _coerce_int(entry.get("candidate_id"))
        if candidate_id is not None and candidate_id in wanted:
            covered.add(candidate_id)
    return covered


def _dedup_prompt_messages(
    candidates: list[tuple[int, str]],
    existing: list[dict[str, Any]],
    lang: str,
) -> list[dict[str, Any]]:
    from ....prompts import get_trajectory_dedup_prompt

    return [
        {"role": "system", "content": get_trajectory_dedup_prompt(lang)},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "candidates": [{"candidate_id": candidate_id, "content": text} for candidate_id, text in candidates],
                    "existing": existing,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _coerce_int(value: Any) -> int | None:
    """Parse a JSON integer, tolerating a numeric string. Returns None otherwise.

    ``str.isdecimal`` rather than ``str.isdigit``: ``isdigit`` accepts characters
    like ``"²"`` that ``int()`` then rejects, which would raise mid-parse.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip())
    return None


def _parse_verdict(raw: Any, hit_count: int) -> ExperienceDedupVerdict:
    if not isinstance(raw, dict):
        return ExperienceDedupVerdict(verdict="different")
    verdict = str(raw.get("verdict") or "different").lower().strip()
    if verdict not in {"same_no_delta", "same_with_delta", "different"}:
        verdict = "different"
    match_index = _coerce_int(raw.get("match_index"))
    if match_index is None or not (0 <= match_index < hit_count):
        match_index = None
    merged_content = raw.get("merged_content") if isinstance(raw.get("merged_content"), str) else None
    return ExperienceDedupVerdict(verdict=verdict, match_index=match_index, merged_content=merged_content)


def _parse_verdicts(
    raw: Any,
    candidates: list[tuple[int, str]],
    hit_count: int,
) -> dict[int, ExperienceDedupVerdict]:
    """Map ``candidate_id`` to its verdict, defaulting skipped candidates to new.

    A candidate the model omitted, mislabelled, or duplicated is treated as
    ``different`` rather than dropped: losing an experience is worse than
    storing a near-duplicate.
    """

    wanted = {candidate_id for candidate_id, _ in candidates}
    verdicts: dict[int, ExperienceDedupVerdict] = {}
    entries = raw.get("verdicts") if isinstance(raw, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        candidate_id = _coerce_int(entry.get("candidate_id"))
        if candidate_id is None or candidate_id not in wanted or candidate_id in verdicts:
            continue
        verdicts[candidate_id] = _parse_verdict(entry, hit_count)
    for candidate_id in wanted:
        verdicts.setdefault(candidate_id, ExperienceDedupVerdict(verdict="different"))
    return verdicts


class ExperienceDeduplicator:
    """Resolve a batch of experience candidates against existing experiences.

    Recall surfaces dense-semantic candidates scoped to ``mem_type=experience``;
    one LLM call then decides create / merge-with-delta / reuse for every
    candidate at once. When either client is unavailable the resolver
    conservatively creates new nodes.
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

    async def _search(
        self,
        ctx: MemoryRequestContext,
        query_text: str,
        dense_vector: list[float],
    ) -> list[MemoryDbSearchHit]:
        result = await self._persistence.search_dense(
            ctx,
            MemoryDbSearchQuery(
                query=query_text,
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
            dense_vector=dense_vector,
        )
        # Keep only hits whose memory content resolves; the LLM judges against text.
        return [hit for hit in result.hits if hit.memory is not None]

    async def recall_many(
        self,
        ctx: MemoryRequestContext,
        candidates: list[tuple[int, str]],
    ) -> dict[int, list[MemoryDbSearchHit]]:
        """Recall existing experiences for every candidate in one embedding pass."""

        if self._embed_client is None or not candidates:
            return {candidate_id: [] for candidate_id, _ in candidates}
        response = await self._embed_client.embed(
            task="memory.trajectory.experience.recall",
            text=[text for _, text in candidates],
        )
        vectors = list(response.embeddings) if response and response.embeddings else []
        recalled: dict[int, list[MemoryDbSearchHit]] = {}
        for position, (candidate_id, text) in enumerate(candidates):
            vector = vectors[position] if position < len(vectors) else None
            recalled[candidate_id] = await self._search(ctx, text, list(vector)) if vector else []
        return recalled

    async def judge_many(
        self,
        ctx: MemoryRequestContext,
        candidates: list[tuple[int, str]],
        existing: list[MemoryDbSearchHit],
        lang: str,
    ) -> dict[int, ExperienceDedupVerdict]:
        """Judge every candidate, blind-re-sending the whole batch when verdicts drop.

        The full batch is sent; if the model's JSON omits some candidates, the same
        whole batch is re-sent up to ``_DEDUP_MAX_ATTEMPTS`` rounds, mirroring
        chat()'s own parse-failure retry. Candidates already answered keep their
        first verdict, so only genuinely missing candidates are picked up on later
        rounds; anything still missing when the budget is spent defaults to new
        rather than being dropped.
        """

        def _all_new() -> dict[int, ExperienceDedupVerdict]:
            return {candidate_id: ExperienceDedupVerdict(verdict="different") for candidate_id, _ in candidates}

        if not existing or self._llm_client is None:
            if existing and self._llm_client is None:
                logger.warning(
                    "trajectory_experience_dedup_llm_unavailable",
                    request_id=ctx.request_id,
                    hit_count=len(existing),
                    candidate_count=len(candidates),
                )
            return _all_new()

        payload = [
            {"memory_index": index, "content": hit.memory.content} for index, hit in enumerate(existing)
        ]
        hit_count = len(existing)
        wanted = {candidate_id for candidate_id, _ in candidates}

        verdicts: dict[int, ExperienceDedupVerdict] = {}
        missing = set(wanted)
        for _attempt in range(_DEDUP_MAX_ATTEMPTS):
            if not missing:
                break
            # Blind re-send of the whole batch, exactly as chat() retries its own
            # parse failures. Candidates already answered keep their first verdict;
            # only still-missing candidates are accepted from later rounds.
            try:
                response = await self._llm_client.chat(
                    task="memory.add.trajectory_experience_dedup",
                    messages=_dedup_prompt_messages(candidates, payload, lang),
                    format_parser=parse_experience_json,
                )
            except Exception:
                # JSON stayed unparsable after the client's own retries; keep what we have.
                logger.warning(
                    "trajectory_experience_dedup_failed",
                    request_id=ctx.request_id,
                    exc_info=True,
                )
                break
            covered = _parsed_candidate_ids(response.parsed, missing)
            if not covered:
                break
            parsed = _parse_verdicts(response.parsed, candidates, hit_count)
            for candidate_id in covered:
                verdicts[candidate_id] = parsed[candidate_id]
            missing -= covered

        exhausted = wanted - set(verdicts)
        if exhausted:
            logger.warning(
                "trajectory_experience_dedup_incomplete",
                request_id=ctx.request_id,
                exhausted=sorted(exhausted),
            )
            for candidate_id in exhausted:
                verdicts[candidate_id] = ExperienceDedupVerdict(verdict="different")
        return verdicts

    async def resolve_many(
        self,
        ctx: MemoryRequestContext,
        candidates: list[tuple[int, str, PreprocessedText]],
        *,
        lang: str,
    ) -> dict[int, ExperienceResolution]:
        """Resolve a whole extraction batch with one recall pass and one judge call.

        Every candidate is judged against the same ``existing`` array — the union
        of all candidates' recalled hits — so ``match_index`` stays unambiguous
        across the batch. To keep a match honest, a candidate may only merge/reuse
        a memory from its OWN recall: zero-recall candidates are stored as new
        without an LLM round trip, and a model match pointing at a memory that
        candidate never recalled is treated as no match.
        """

        recall_input = [(candidate_id, text) for candidate_id, text, _ in candidates]
        hits_by_candidate = await self.recall_many(ctx, recall_input)

        allowed_by_candidate: dict[int, set[str]] = {}
        existing: list[MemoryDbSearchHit] = []
        seen: set[str] = set()
        for candidate_id, _, _ in candidates:
            hits = hits_by_candidate.get(candidate_id, [])
            allowed_by_candidate[candidate_id] = {hit.memory_id for hit in hits}
            for hit in hits:
                if hit.memory_id in seen:
                    continue
                seen.add(hit.memory_id)
                existing.append(hit)

        # Zero-recall candidates have no possible match; do not spend an LLM call
        # on them (the old per-candidate resolve() created them directly).
        judge_input = [
            (candidate_id, text)
            for candidate_id, text, _ in candidates
            if allowed_by_candidate[candidate_id]
        ]
        verdicts = await self.judge_many(ctx, judge_input, existing, lang)
        return {
            candidate_id: self._to_resolution(
                ctx,
                verdicts.get(candidate_id, ExperienceDedupVerdict(verdict="different")),
                existing,
                text,
                preprocessed,
                allowed_by_candidate[candidate_id],
            )
            for candidate_id, text, preprocessed in candidates
        }

    @staticmethod
    def _to_resolution(
        ctx: MemoryRequestContext,
        verdict: ExperienceDedupVerdict,
        existing: list[MemoryDbSearchHit],
        candidate_text: str,
        preprocessed: PreprocessedText,
        allowed: set[str],
    ) -> ExperienceResolution:
        def _create() -> ExperienceResolution:
            memory_id = generate_experience_id(ctx.project_id, preprocessed.normalized_text)
            return ExperienceResolution(action="create", target_memory_id=memory_id, preprocessed=preprocessed)

        if verdict.verdict == "different":
            return _create()
        match_index = verdict.match_index
        if match_index is None or not (0 <= match_index < len(existing)):
            # LLM claimed a match but pointed outside the recalled list; stay safe.
            return _create()

        target = existing[match_index]
        if target.memory_id not in allowed:
            # The model matched a memory this candidate never recalled; trust the
            # semantic search over the guess and store as new rather than polluting
            # an unrelated existing node.
            return _create()
        if verdict.verdict == "same_with_delta":
            return ExperienceResolution(
                action="merge",
                target_memory_id=target.memory_id,
                merged_content=verdict.merged_content or candidate_text,
                existing_memory=target.memory,
            )
        return ExperienceResolution(action="reuse", target_memory_id=target.memory_id, existing_memory=target.memory)
