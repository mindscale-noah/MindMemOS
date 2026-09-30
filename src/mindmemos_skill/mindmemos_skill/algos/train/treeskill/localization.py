"""Atomic evidence localization against one immutable initial Skill tree."""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import ValidationError

from ....common.json import extract_json_object, strict_json_schema_response_format
from .prompts import LOCALIZATION_SYSTEM_PROMPT, localization_user_prompt
from .typing import (
    ChatModel,
    LocalizationFailure,
    LocatedEvidence,
    MarkdownSkillTree,
    TrajectoryAnalysisRecord,
    _RawEvidence,
)
from .validation import analysis_artifact_reason

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
        """init  .

        Args:
            chat_model: Model client implementing the recorded chat request protocol.
            task: Stage audit label or public task supplied by the caller.
            concurrency: Maximum simultaneous requests in this stage.
            temperature: Sampling temperature passed unchanged to the model.
            max_tokens: Completion-token budget passed to the model.
        """
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
        """Locate.

        Args:
            tree: Parsed Markdown tree with stable heading identities.
            records: Validated trajectory analyses to localize against the same initial tree.
        """
        semaphore = asyncio.Semaphore(self._concurrency)
        known_ids = set(tree.node_by_id)

        async def run(
            record: TrajectoryAnalysisRecord,
        ) -> tuple[list[LocatedEvidence], LocalizationFailure | None]:
            """Run.

            Args:
                record: Validated analysis record to render for localization.
            """
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
                    except Exception as exc:
                        last_error = exc
                        continue
                    try:
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

        tasks = [asyncio.create_task(run(record)) for record in records]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
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
        artifact_reason = analysis_artifact_reason(item.reusable_lesson)
        if artifact_reason:
            rejected.append(f"{item.evidence_id}: {artifact_reason}")
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
