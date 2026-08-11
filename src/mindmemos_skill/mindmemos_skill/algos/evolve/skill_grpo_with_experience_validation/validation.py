"""Build transient experience-injected Skills and assess their behavioral re-runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from ....persistence.enums import SkillVersionOrigin, SkillVersionStatus
from ....typing import Skill, compute_skill_content_hash
from ..skill_grpo_with_replay_buffer.contracts import RolloutOutcome
from .contracts import (
    ExperienceSource,
    ExperienceValidationDecision,
    ExperienceValidationRecord,
    ExtractedExperienceSet,
)


def render_experience_guidance(content: str) -> str:
    """Render only reusable lesson/reason fields for agent injection."""

    try:
        payload: Any = json.loads(content)
    except (TypeError, ValueError):
        return content.strip()
    if not isinstance(payload, dict):
        return content.strip()
    items = payload.get("experiences")
    if not isinstance(items, list):
        return content.strip()
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lesson = item.get("lesson")
        reason = item.get("reason")
        if not isinstance(lesson, str) or not lesson.strip():
            continue
        line = f"- {lesson.strip()}"
        if isinstance(reason, str) and reason.strip():
            line += f" Rationale: {reason.strip()}"
        lines.append(line)
    return "\n".join(lines)


def inject_experience(
    base: Skill,
    experience: ExtractedExperienceSet,
    *,
    run_id: str,
    batch_index: int,
    experience_index: int,
) -> Skill | None:
    guidance = render_experience_guidance(experience.content)
    if not guidance:
        return None
    content = (
        f"{base.content.rstrip()}\n\n"
        "## Candidate guidance for this experimental re-run\n\n"
        f"{guidance}\n"
    )
    blob = {"SKILL.md": content}
    now = datetime.now(UTC)
    return base.model_copy(
        update={
            "version_id": f"experience-validation:{run_id}:{batch_index}:{experience_index}",
            "parent_version_ids": [base.version_id],
            "content_hash": compute_skill_content_hash(blob),
            "status": SkillVersionStatus.DRAFT,
            "origin": SkillVersionOrigin.EVOLUTION,
            "blob": blob,
            "commit_message": "transient experience validation candidate",
            "created_at": now,
            "updated_at": now,
            "metadata": {
                **base.metadata,
                "experience_validation": {
                    "run_id": run_id,
                    "batch_index": batch_index,
                    "experience_index": experience_index,
                    "source": experience.source.value,
                    "transient": True,
                },
            },
        },
        deep=True,
    )


def assess_experience(
    experience: ExtractedExperienceSet,
    *,
    experience_index: int,
    injected_outcomes: list[RolloutOutcome],
    baseline_first_success_attempt: int | None,
    success_reward: float,
) -> ExperienceValidationRecord:
    injected_successes = [
        outcome
        for outcome in injected_outcomes
        if outcome.trajectory is not None
        and outcome.trajectory.reward.score is not None
        and outcome.trajectory.reward.score >= success_reward
    ]
    task_count = len(experience.task_ids)
    successful_task_ids = {outcome.spec.task.task_id for outcome in injected_successes}
    injected_rate = len(successful_task_ids) / task_count

    if experience.source is ExperienceSource.CONTRAST:
        baseline_rate = 1.0
        injected_attempt = next(
            (outcome.spec.sample_index + 1 for outcome in injected_outcomes if outcome in injected_successes),
            None,
        )
        accepted = (
            baseline_first_success_attempt is not None
            and injected_attempt is not None
            and injected_attempt < baseline_first_success_attempt
        )
        reason = (
            f"first success improved from attempt {baseline_first_success_attempt} to {injected_attempt}"
            if accepted
            else f"first success did not improve from attempt {baseline_first_success_attempt} to {injected_attempt}"
        )
    elif experience.source is ExperienceSource.SUCCESS:
        baseline_rate = 1.0
        injected_attempt = None
        accepted = injected_rate >= baseline_rate
        reason = (
            f"one-shot success rate preserved at {injected_rate:.3f}"
            if accepted
            else f"one-shot success rate dropped from {baseline_rate:.3f} to {injected_rate:.3f}"
        )
    else:
        baseline_rate = 0.0
        injected_attempt = None
        accepted = injected_rate > baseline_rate
        reason = (
            f"one-shot success rate improved from {baseline_rate:.3f} to {injected_rate:.3f}"
            if accepted
            else f"one-shot success rate did not improve from {baseline_rate:.3f}"
        )

    return ExperienceValidationRecord(
        experience_index=experience_index,
        source=experience.source,
        task_ids=experience.task_ids,
        baseline_success_rate=baseline_rate,
        injected_success_rate=injected_rate,
        baseline_first_success_attempt=baseline_first_success_attempt,
        injected_first_success_attempt=injected_attempt,
        decision=(ExperienceValidationDecision.ACCEPTED if accepted else ExperienceValidationDecision.REJECTED),
        reason=reason,
        rollouts=injected_outcomes,
    )


def rejected_empty_experience(
    experience: ExtractedExperienceSet,
    *,
    experience_index: int,
    baseline_first_success_attempt: int | None,
) -> ExperienceValidationRecord:
    baseline_rate = 0.0 if experience.source is ExperienceSource.FAILURE else 1.0
    return ExperienceValidationRecord(
        experience_index=experience_index,
        source=experience.source,
        task_ids=experience.task_ids,
        baseline_success_rate=baseline_rate,
        injected_success_rate=0.0,
        baseline_first_success_attempt=baseline_first_success_attempt,
        decision=ExperienceValidationDecision.REJECTED,
        reason="extraction produced no injectable guidance",
        rollouts=[],
    )


__all__ = [
    "assess_experience",
    "inject_experience",
    "rejected_empty_experience",
    "render_experience_guidance",
]
