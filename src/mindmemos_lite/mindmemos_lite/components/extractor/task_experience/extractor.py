"""LLM extraction of reusable experiences from a trajectory."""

from __future__ import annotations

import json
from typing import Any

from ....logging import get_logger
from ....typing import MemoryRequestContext
from .schema import (
    ExtractedExperienceCandidate,
    parse_experience_json,
)

logger = get_logger(__name__)


def _experience_prompt_messages(task_text: str, turns: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    from ....prompts import get_trajectory_experience_prompt

    return [
        {"role": "system", "content": get_trajectory_experience_prompt(lang)},
        {"role": "user", "content": json.dumps({"task": task_text, "turns": turns}, ensure_ascii=False)},
    ]


def _payload_experiences(payload: Any) -> list[ExtractedExperienceCandidate]:
    if not isinstance(payload, dict):
        return []
    candidates: list[ExtractedExperienceCandidate] = []
    for index, raw in enumerate(payload.get("experiences") or []):
        if not isinstance(raw, dict):
            continue
        content = (raw.get("content") or "").strip()
        if not content:
            continue
        candidates.append(
            ExtractedExperienceCandidate(
                ref_id=f"e{index}",
                content=content,
                confidence=_optional_float(raw.get("confidence")),
                importance=_optional_float(raw.get("importance")),
                source_message_indices=[
                    int(item)
                    for item in (raw.get("source_message_indices") or [])
                    if isinstance(item, int) and not isinstance(item, bool)
                ],
                reason=raw.get("reason") if isinstance(raw.get("reason"), str) else None,
                metadata=dict(raw.get("metadata") or {}),
            )
        )
    return candidates


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class TrajectoryExperienceExtractor:
    """Extract reusable experience candidates from a trajectory's task and turns.

    Falls back to an empty candidate list when the LLM is unavailable; the
    trajectory pipeline then still persists the task entity.
    """

    def __init__(self, *, llm_client=None) -> None:
        self._llm_client = llm_client

    async def extract(
        self,
        task_text: str,
        turns: list[dict[str, Any]],
        lang: str,
        context: MemoryRequestContext,
    ) -> list[ExtractedExperienceCandidate]:
        if self._llm_client is None:
            logger.debug("trajectory_experience_llm_unavailable", request_id=context.request_id)
            return []
        try:
            response = await self._llm_client.chat(
                task="memory.add.trajectory_experience",
                messages=_experience_prompt_messages(task_text, turns, lang),
                format_parser=parse_experience_json,
            )
            return _payload_experiences(response.parsed if response is not None else None)
        except Exception:
            logger.warning(
                "trajectory_experience_extraction_failed",
                request_id=context.request_id,
                exc_info=True,
            )
            return []