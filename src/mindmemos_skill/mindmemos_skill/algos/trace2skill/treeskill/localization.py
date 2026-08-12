"""Atomic evidence localization against one immutable initial Skill tree."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .analysis import ChatModel
from .json_utils import extract_json_object, strict_json_schema_response_format
from .models import LocalizationFailure, LocatedEvidence, TrajectoryAnalysisRecord
from .prompts import LOCALIZATION_SYSTEM_PROMPT, localization_user_prompt
from .tree import MarkdownSkillTree


class _RawEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    reusable_lesson: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


_LOCALIZATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["instance_id", "evidence"],
    "properties": {
        "instance_id": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "reusable_lesson", "target_node_id", "rationale"],
                "properties": {
                    "evidence_id": {"type": "string"},
                    "reusable_lesson": {"type": "string"},
                    "target_node_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


class TreeSkillEvidenceLocator:
    """Locate every analysis record against the same initial tree snapshot."""

    def __init__(
        self,
        *,
        chat_model: ChatModel,
        task: str,
        concurrency: int,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._chat_model = chat_model
        self._task = task
        self._concurrency = concurrency
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def locate(
        self,
        tree: MarkdownSkillTree,
        records: list[TrajectoryAnalysisRecord],
    ) -> tuple[list[LocatedEvidence], list[LocalizationFailure]]:
        semaphore = asyncio.Semaphore(self._concurrency)
        known_ids = set(tree.node_by_id)

        async def run(
            record: TrajectoryAnalysisRecord,
        ) -> tuple[list[LocatedEvidence], LocalizationFailure | None]:
            if not record.items:
                return [], None
            messages = [
                {"role": "system", "content": LOCALIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": localization_user_prompt(tree, record)},
            ]

            last_error: Exception | None = None
            async with semaphore:
                for _attempt, token_budget in ((1, self._max_tokens), (2, min(self._max_tokens * 2, 8192))):
                    try:
                        response = await self._chat_model.chat(
                            task=self._task,
                            messages=list(messages),
                            temperature=self._temperature,
                            max_tokens=token_budget,
                            response_format=strict_json_schema_response_format(
                                "tree_fusion_locator",
                                _LOCALIZATION_SCHEMA,
                            ),
                        )
                        located, rejected = _parse_localization_items(
                            response.content or "",
                            record=record,
                            known_ids=known_ids,
                        )
                        rejection = (
                            LocalizationFailure(
                                instance_id=record.instance_id,
                                error="; ".join(rejected),
                            )
                            if rejected
                            else None
                        )
                        return located, rejection
                    except Exception as exc:
                        last_error = exc
            assert last_error is not None
            return [], LocalizationFailure(
                instance_id=record.instance_id,
                error=f"{type(last_error).__name__}: {last_error}",
            )

        results = await asyncio.gather(*(run(record) for record in records))
        evidence = [item for items, _ in results for item in items]
        failures = [failure for _, failure in results if failure is not None]
        return evidence, failures


def _parse_localization_items(
    text: str,
    *,
    record: TrajectoryAnalysisRecord,
    known_ids: set[str],
) -> tuple[list[LocatedEvidence], list[str]]:
    payload = extract_json_object(text)
    if payload.get("instance_id") != record.instance_id:
        raise ValueError("localization instance_id does not match the analysis record")
    raw_items = payload.get("evidence")
    if not isinstance(raw_items, list):
        raise ValueError("localization evidence must be a list")

    located: list[LocatedEvidence] = []
    rejected: list[str] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_items, start=1):
        try:
            item = _RawEvidence.model_validate(raw)
        except ValidationError as exc:
            rejected.append(f"evidence[{index}] is invalid: {exc.errors()[0]['msg']}")
            continue
        if item.evidence_id in seen_ids:
            rejected.append(f"evidence[{index}] duplicates evidence_id {item.evidence_id!r}")
            continue
        seen_ids.add(item.evidence_id)
        if item.target_node_id not in known_ids:
            rejected.append(f"{item.evidence_id}: unknown target_node_id {item.target_node_id!r}")
            continue
        located.append(
            LocatedEvidence(
                instance_id=record.instance_id,
                evidence_id=item.evidence_id,
                record_source=record.record_source,
                reusable_lesson=item.reusable_lesson,
                target_node_id=item.target_node_id,
                rationale=item.rationale,
            )
        )
    return located, rejected


__all__ = ["TreeSkillEvidenceLocator"]
