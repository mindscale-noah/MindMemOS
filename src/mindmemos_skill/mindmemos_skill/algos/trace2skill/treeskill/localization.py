"""Atomic evidence localization against one immutable initial Skill tree."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .analysis import ChatModel
from .json_utils import parse_model
from .models import LocalizationFailure, LocatedEvidence, TrajectoryAnalysisRecord
from .prompts import LOCALIZATION_SYSTEM_PROMPT, localization_user_prompt
from .tree import MarkdownSkillTree


class _RawEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    reusable_lesson: str = Field(min_length=1)
    target_node_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class _RawLocalization(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str = Field(min_length=1)
    evidence: tuple[_RawEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_unique_evidence_ids(self) -> _RawLocalization:
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("localization evidence_id values must be unique within a record")
        return self


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

            def parse(text: str) -> _RawLocalization:
                payload = parse_model(text, _RawLocalization)
                if payload.instance_id != record.instance_id:
                    raise ValueError("localization instance_id does not match the analysis record")
                unknown = [item.target_node_id for item in payload.evidence if item.target_node_id not in known_ids]
                if unknown:
                    raise ValueError(f"localization returned unknown target node ids: {sorted(set(unknown))}")
                return payload

            try:
                async with semaphore:
                    response = await self._chat_model.chat(
                        task=self._task,
                        messages=messages,
                        format_parser=parse,
                        feedback_on_parse_error=True,
                        temperature=self._temperature,
                        max_tokens=self._max_tokens,
                    )
                payload = getattr(response, "parsed", None) or parse(response.content or "")
                located = [
                    LocatedEvidence(
                        instance_id=record.instance_id,
                        evidence_id=item.evidence_id,
                        record_source=record.record_source,
                        reusable_lesson=item.reusable_lesson,
                        target_node_id=item.target_node_id,
                        rationale=item.rationale,
                    )
                    for item in payload.evidence
                ]
                return located, None
            except Exception as exc:
                return [], LocalizationFailure(instance_id=record.instance_id, error=f"{type(exc).__name__}: {exc}")

        results = await asyncio.gather(*(run(record) for record in records))
        evidence = [item for items, _ in results for item in items]
        failures = [failure for _, failure in results if failure is not None]
        return evidence, failures


__all__ = ["TreeSkillEvidenceLocator"]
