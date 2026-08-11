"""Generate a source-prioritized patch from accepted experience sets only."""

from __future__ import annotations

from typing import Any

from ....typing import Skill
from ..skill_grpo_with_replay_buffer.contracts import EditSupport, PatchProposalRecord, SkillTextEdit
from ..skill_grpo_with_replay_buffer.fileedit import EditError, load_json_object, validate_edits
from ..skill_grpo_with_replay_buffer.models import ChatModel, chat_content
from ..skill_grpo_with_replay_buffer.prompts import patch_repair_message
from .contracts import ExperienceSource, ExtractedExperienceSet
from .prompts import experience_patch_messages


class PatchProposer:
    def __init__(self, chat_model: ChatModel, *, max_edits: int, max_attempts: int) -> None:
        self._chat_model = chat_model
        self._max_edits = max_edits
        self._max_attempts = max_attempts

    async def propose(self, skill: Skill, experiences: list[ExtractedExperienceSet]) -> PatchProposalRecord:
        if not experiences:
            raise ValueError("patch proposal requires at least one accepted experience")
        messages = experience_patch_messages(skill=skill, experiences=experiences, max_edits=self._max_edits)
        raw = ""
        response: dict[str, Any] | None = None
        parsed: list[SkillTextEdit] | None = None
        last_error = ""
        attempts = 0
        for attempt in range(1, self._max_attempts + 1):
            attempts = attempt
            raw = await chat_content(
                self._chat_model,
                task=(
                    "skill_grpo_experience_validation.patch"
                    if attempt == 1
                    else "skill_grpo_experience_validation.patch_repair"
                ),
                messages=messages,
            )
            try:
                response = load_json_object(raw)
                parsed = self._parse_edits(response)
                break
            except (TypeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self._max_attempts:
                    messages = [*messages, {"role": "assistant", "content": raw}, patch_repair_message(last_error)]
        if response is None or parsed is None:
            raise EditError(f"patch proposal failed after {attempts} attempt(s): {last_error}")

        valid, validation_errors = validate_edits(parsed, skill.content)
        valid = self._deduplicate(valid)
        support = self._support(response, valid, len(experiences))
        source_rank = {ExperienceSource.CONTRAST: 0, ExperienceSource.FAILURE: 1, ExperienceSource.SUCCESS: 2}

        def priority(item: EditSupport) -> tuple[int, int]:
            ranks = [
                source_rank[experiences[index - 1].source]
                for index in item.supporting_experience_sets
                if 1 <= index <= len(experiences)
            ]
            return min(ranks, default=len(source_rank)), -len(item.supporting_experience_sets)

        support.sort(key=priority)
        return PatchProposalRecord(
            raw_content=raw,
            proposed_edit_count=len(parsed),
            validation_errors=validation_errors,
            edit_support=support[: self._max_edits],
            attempts=attempts,
        )

    @staticmethod
    def _parse_edits(response: dict[str, Any]) -> list[SkillTextEdit]:
        raw: Any = response.get("edits", response)
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise EditError("expected an edit object or a list of edits")
        edits: list[SkillTextEdit] = []
        for item in raw:
            if not isinstance(item, dict):
                raise EditError(f"edit must be an object, got {type(item).__name__}")
            if "find" not in item and "replace" not in item:
                raise EditError("edit missing both 'find' and 'replace'")
            edits.append(SkillTextEdit(find=str(item.get("find", "")), replace=str(item.get("replace", ""))))
        return edits

    @staticmethod
    def _deduplicate(edits: list[SkillTextEdit]) -> list[SkillTextEdit]:
        output: list[SkillTextEdit] = []
        seen: set[tuple[str, str]] = set()
        for edit in edits:
            key = edit.find, edit.replace
            if key not in seen:
                seen.add(key)
                output.append(edit)
        return output

    @staticmethod
    def _support(response: dict[str, Any], edits: list[SkillTextEdit], experience_count: int) -> list[EditSupport]:
        by_key = {(edit.find, edit.replace): set() for edit in edits}
        for item in response.get("edits", []):
            if not isinstance(item, dict):
                continue
            key = str(item.get("find", "")), str(item.get("replace", ""))
            if key not in by_key:
                continue
            indices = item.get("supporting_experience_sets", [])
            if isinstance(indices, list):
                by_key[key].update(
                    index
                    for index in indices
                    if isinstance(index, int) and not isinstance(index, bool) and 1 <= index <= experience_count
                )
        return [
            EditSupport(edit=edit, supporting_experience_sets=sorted(by_key[(edit.find, edit.replace)]))
            for edit in edits
        ]


__all__ = ["PatchProposer"]
