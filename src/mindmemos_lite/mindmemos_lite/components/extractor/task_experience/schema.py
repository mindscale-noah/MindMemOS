"""Data shapes for trajectory task+experience extraction and dedup."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from ....typing import MemoryView, MemoryWrite, PreprocessedText


class ExtractedExperienceCandidate(BaseModel):
    """One reusable experience distilled from a trajectory."""

    ref_id: str
    content: str
    confidence: float | None = None
    importance: float | None = None
    source_message_indices: list[int] = Field(default_factory=list)
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperienceDedupVerdict(BaseModel):
    """LLM verdict on whether a candidate experience duplicates an existing one."""

    verdict: Literal["same_no_delta", "same_with_delta", "different"] = "different"
    match_index: int | None = None
    merged_content: str | None = None


class ExperienceResolution(BaseModel):
    """Resolved write action for one experience candidate."""

    action: Literal["create", "merge", "reuse"]
    target_memory_id: str
    new_memory: MemoryWrite | None = None
    preprocessed: PreprocessedText | None = None
    merged_content: str | None = None
    existing_memory: MemoryView | None = None
    """For merge/reuse: the existing node selected by the LLM judge, so the caller
    can read its metadata (e.g. accumulated task_refs) before rewriting it."""


def parse_experience_json(content: str) -> dict[str, Any]:
    """Parse trajectory JSON output, tolerating simple markdown code fences."""

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").strip()
        text = text.removesuffix("```").strip()
    return json.loads(text)