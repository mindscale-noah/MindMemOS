"""Summarize rejected experiences for the immediately following batch only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from ..skill_grpo_with_replay_buffer.models import ChatModel, chat_content
from .contracts import (
    ExperienceExtractionPayload,
    ExperienceValidationDecision,
    ExperienceValidationRecord,
    ExtractedExperienceSet,
)

_MAX_SUMMARY_ATTEMPTS = 2
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

INVALID_EXPERIENCE_FEEDBACK_SYSTEM = """# Role
You summarize experience guidance rejected by a paired behavioral gate.

# Objective
Aggregate the supplied rejected experience sets into a concise warning for the
next batch's experience extractor. Every supplied set is behaviorally invalid,
including sets whose injected performance stayed flat instead of improving.
Describe what guidance must not be rediscovered and why the gate rejected it.
Do not convert invalid guidance into positive advice, and do not invent evidence.

# Output contract
Return exactly one valid JSON object and no other text:
{
  "summary": "concise synthesis of invalid guidance and common failure patterns",
  "invalid_experiences": [
    {
      "experience_index": 1,
      "invalid_guidance": "concise description of guidance not to reuse",
      "failure_mechanism": "why the supplied validation shows it is invalid"
    }
  ]
}
Return exactly one invalid_experiences item for every supplied experience
index, using each index exactly once.
"""


class _StrictSummaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _InvalidExperienceSummaryPayload(_StrictSummaryModel):
    experience_index: int = Field(ge=1)
    invalid_guidance: NonEmptyText
    failure_mechanism: NonEmptyText


class _InvalidExperienceSummaryPayloads(_StrictSummaryModel):
    summary: NonEmptyText
    invalid_experiences: list[_InvalidExperienceSummaryPayload] = Field(min_length=1)


@dataclass(frozen=True)
class RejectedExperienceEvidence:
    experience_index: int
    experience: ExtractedExperienceSet
    targeted_case_count: int
    baseline_success_count: int
    injected_success_count: int
    baseline_first_success_attempt: int | None
    injected_first_success_attempt: int | None
    gate_reason: str

    @property
    def degraded_success_count(self) -> int:
        return self.baseline_success_count - self.injected_success_count


@dataclass(frozen=True)
class InvalidExperienceFeedbackItem:
    evidence: RejectedExperienceEvidence
    invalid_guidance: str
    failure_mechanism: str


@dataclass(frozen=True)
class PreviousBatchInvalidExperienceFeedback:
    source_batch_index: int
    summary: str
    invalid_experiences: tuple[InvalidExperienceFeedbackItem, ...]

    @property
    def targeted_case_count(self) -> int:
        return sum(item.evidence.targeted_case_count for item in self.invalid_experiences)

    @property
    def baseline_success_count(self) -> int:
        return sum(item.evidence.baseline_success_count for item in self.invalid_experiences)

    @property
    def injected_success_count(self) -> int:
        return sum(item.evidence.injected_success_count for item in self.invalid_experiences)

    @property
    def degraded_success_count(self) -> int:
        return self.baseline_success_count - self.injected_success_count

    def render_for_extraction(self) -> str:
        lines = [
            "# Previous Batch Invalid Experience Feedback",
            "",
            f"This feedback comes only from the immediately preceding batch (batch {self.source_batch_index + 1}).",
            "Every experience listed below was rejected by behavioral validation and is invalid. Do not repeat, "
            "rephrase, combine, or infer new guidance from it during extraction.",
            "",
            f"Aggregate warning: {self.summary}",
        ]
        for item in self.invalid_experiences:
            evidence = item.evidence
            attempt_line = ""
            if evidence.baseline_first_success_attempt is not None:
                attempt_line = (
                    f" First success attempt: {evidence.baseline_first_success_attempt} -> "
                    f"{evidence.injected_first_success_attempt}."
                )
            lines.extend(
                [
                    "",
                    (
                        f"## Invalid experience set {evidence.experience_index + 1} "
                        f"[source={evidence.experience.source.value}]"
                    ),
                    "",
                    (
                        f"Original correct: {evidence.baseline_success_count}/{evidence.targeted_case_count}; "
                        f"with this experience: {evidence.injected_success_count}/{evidence.targeted_case_count}; "
                        f"degraded by: {evidence.degraded_success_count}.{attempt_line}"
                    ),
                    f"Gate rejection: {evidence.gate_reason}",
                    f"Invalid guidance: {item.invalid_guidance}",
                    f"Observed failure mechanism: {item.failure_mechanism}",
                ]
            )
        return "\n".join(lines)

    def event_payload(self) -> dict[str, object]:
        return {
            "source_batch_index": self.source_batch_index,
            "rejected_experience_count": len(self.invalid_experiences),
            "targeted_case_count": self.targeted_case_count,
            "baseline_success_count": self.baseline_success_count,
            "injected_success_count": self.injected_success_count,
            "degraded_success_count": self.degraded_success_count,
            "summary": self.summary,
            "invalid_experiences": [
                {
                    "experience_index": item.evidence.experience_index,
                    "source": item.evidence.experience.source.value,
                    "task_ids": item.evidence.experience.task_ids,
                    "targeted_case_count": item.evidence.targeted_case_count,
                    "baseline_success_count": item.evidence.baseline_success_count,
                    "injected_success_count": item.evidence.injected_success_count,
                    "degraded_success_count": item.evidence.degraded_success_count,
                    "baseline_first_success_attempt": item.evidence.baseline_first_success_attempt,
                    "injected_first_success_attempt": item.evidence.injected_first_success_attempt,
                    "gate_reason": item.evidence.gate_reason,
                    "invalid_guidance": item.invalid_guidance,
                    "failure_mechanism": item.failure_mechanism,
                }
                for item in self.invalid_experiences
            ],
        }


class InvalidExperienceFeedbackSummarizer:
    def __init__(self, chat_model: ChatModel) -> None:
        self._chat_model = chat_model

    async def summarize(
        self,
        *,
        batch_index: int,
        experiences: list[ExtractedExperienceSet],
        validations: list[ExperienceValidationRecord],
    ) -> PreviousBatchInvalidExperienceFeedback | None:
        rejected = self._rejected_experiences(experiences, validations)
        if not rejected:
            return None

        current_messages = self._messages(rejected)
        for attempt in range(1, _MAX_SUMMARY_ATTEMPTS + 1):
            task = f"skill_grpo_experience_validation.invalid_experience_feedback.batch_{batch_index}"
            raw = await chat_content(
                self._chat_model,
                task=task if attempt == 1 else f"{task}.repair",
                messages=current_messages,
            )
            try:
                payload = self._validated_payload(raw, rejected)
            except ValueError as exc:
                if attempt == _MAX_SUMMARY_ATTEMPTS:
                    return self._fallback(batch_index, rejected)
                current_messages = [
                    *current_messages,
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "The previous response violated the invalid-experience feedback schema: "
                            f"{type(exc).__name__}: {exc}. Return one corrected JSON object and no other text."
                        ),
                    },
                ]
                continue
            return self._feedback(batch_index, rejected, payload)
        raise AssertionError("unreachable")

    @staticmethod
    def _rejected_experiences(
        experiences: list[ExtractedExperienceSet], validations: list[ExperienceValidationRecord]
    ) -> list[RejectedExperienceEvidence]:
        rejected: list[RejectedExperienceEvidence] = []
        for validation in validations:
            if validation.decision is not ExperienceValidationDecision.REJECTED or not validation.rollouts:
                continue
            targeted_case_count = len(validation.task_ids)
            rejected.append(
                RejectedExperienceEvidence(
                    experience_index=validation.experience_index,
                    experience=experiences[validation.experience_index],
                    targeted_case_count=targeted_case_count,
                    baseline_success_count=round(validation.baseline_success_rate * targeted_case_count),
                    injected_success_count=round(validation.injected_success_rate * targeted_case_count),
                    baseline_first_success_attempt=validation.baseline_first_success_attempt,
                    injected_first_success_attempt=validation.injected_first_success_attempt,
                    gate_reason=validation.reason,
                )
            )
        return rejected

    @staticmethod
    def _messages(rejected: list[RejectedExperienceEvidence]) -> list[dict[str, str]]:
        blocks = []
        for evidence in rejected:
            attempt_line = ""
            if evidence.baseline_first_success_attempt is not None:
                attempt_line = (
                    f"\nFirst success attempt: {evidence.baseline_first_success_attempt} -> "
                    f"{evidence.injected_first_success_attempt}"
                )
            blocks.append(
                (
                    f"## Experience set {evidence.experience_index + 1} "
                    f"[source={evidence.experience.source.value}]\n\n"
                    f"Original correct: {evidence.baseline_success_count}/{evidence.targeted_case_count}\n"
                    f"With experience correct: {evidence.injected_success_count}/{evidence.targeted_case_count}\n"
                    f"Degraded by: {evidence.degraded_success_count}{attempt_line}\n"
                    f"Gate rejection: {evidence.gate_reason}\n\n"
                    f"Extracted experience JSON:\n{evidence.experience.content}"
                )
            )
        return [
            {"role": "system", "content": INVALID_EXPERIENCE_FEEDBACK_SYSTEM},
            {"role": "user", "content": "# Rejected Experience Sets\n\n" + "\n\n".join(blocks)},
        ]

    @staticmethod
    def _validated_payload(
        raw: str, rejected: list[RejectedExperienceEvidence]
    ) -> _InvalidExperienceSummaryPayloads:
        payload = _InvalidExperienceSummaryPayloads.model_validate_json(raw)
        expected = {item.experience_index + 1 for item in rejected}
        actual = [item.experience_index for item in payload.invalid_experiences]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"expected invalid experience indices {sorted(expected)}, got {actual}")
        return payload

    @staticmethod
    def _feedback(
        batch_index: int,
        rejected: list[RejectedExperienceEvidence],
        payload: _InvalidExperienceSummaryPayloads,
    ) -> PreviousBatchInvalidExperienceFeedback:
        summary_by_index = {item.experience_index - 1: item for item in payload.invalid_experiences}
        return PreviousBatchInvalidExperienceFeedback(
            source_batch_index=batch_index,
            summary=payload.summary,
            invalid_experiences=tuple(
                InvalidExperienceFeedbackItem(
                    evidence=evidence,
                    invalid_guidance=summary_by_index[evidence.experience_index].invalid_guidance,
                    failure_mechanism=summary_by_index[evidence.experience_index].failure_mechanism,
                )
                for evidence in rejected
            ),
        )

    @staticmethod
    def _fallback(
        batch_index: int, rejected: list[RejectedExperienceEvidence]
    ) -> PreviousBatchInvalidExperienceFeedback:
        items = []
        for evidence in rejected:
            payload = ExperienceExtractionPayload.model_validate_json(evidence.experience.content)
            items.append(
                InvalidExperienceFeedbackItem(
                    evidence=evidence,
                    invalid_guidance="; ".join(item.lesson for item in payload.experiences),
                    failure_mechanism=evidence.gate_reason,
                )
            )
        return PreviousBatchInvalidExperienceFeedback(
            source_batch_index=batch_index,
            summary="The listed guidance failed its paired behavioral gate and must not be reused.",
            invalid_experiences=tuple(items),
        )


__all__ = [
    "InvalidExperienceFeedbackItem",
    "InvalidExperienceFeedbackSummarizer",
    "PreviousBatchInvalidExperienceFeedback",
    "RejectedExperienceEvidence",
]
