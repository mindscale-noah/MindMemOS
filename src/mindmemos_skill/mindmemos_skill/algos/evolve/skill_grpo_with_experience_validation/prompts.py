"""Prompts for three-stream experience extraction and accepted-experience patching."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ....typing import Skill, Trajectory
from ..skill_grpo_with_replay_buffer.prompts import _render_events
from .contracts import ExtractedExperienceSet

if TYPE_CHECKING:
    from .invalid_experience_feedback import PreviousBatchInvalidExperienceFeedback

COMMON_OUTPUT = """
# Output contract
Return exactly one valid JSON object and no other text:
{
  "experiences": [
    {
      "topic": "short task-independent label",
      "reason": "causal explanation grounded in the supplied evidence",
      "lesson": "concise reusable guidance",
      "evidence": [
        {"task_id": "id", "rollout": 1, "observation": "observable behavior"}
      ]
    }
  ]
}
Return at most {max_experiences} experiences. Return {"experiences": []} when
the evidence does not justify a Skill change. Lessons must not contain
task-specific names, values, paths, or benchmark details.
"""

CROSS_TASK_SUCCESS_SYSTEM = (
    """# Role
You are a cross-task success-pattern analyst. You receive the last successful
trajectory from each task in one mini-batch and the current Skill.

# Objective
Identify high-impact behaviors that recur across multiple tasks and should be
preserved. Retain only patterns supported by at least two distinct tasks and
missing or unclear in the current Skill. Do not promote isolated tricks or
infer behavior from reward alone. Ground each lesson in observed actions or
verification.
"""
    + COMMON_OUTPUT
)

CROSS_TASK_FAILURE_SYSTEM = (
    """# Role
You are a cross-task failure-pattern analyst. You receive the first failed
trajectory from each task in one mini-batch and the current Skill.

# Objective
Identify prevalent systematic failure mechanisms across multiple tasks. Trace
observed failure -> producing action, omission, or unsupported assumption ->
corrective guidance. Ignore infrastructure failures before meaningful
execution and avoid guidance already covered by the Skill.
"""
    + COMMON_OUTPUT
)

FAILURE_TO_SUCCESS_SYSTEM = (
    """# Role
You are a failure-to-success contrast analyst. You receive chronological
rollouts of one task: failures followed by its first success, plus the Skill.

# Objective
Identify the smallest decisive reusable change that plausibly produced the
earlier success. Trace how reflection changed strategy, sequencing, tool use,
checks, or recovery. Separate correlation from causal evidence. Emit no
experience when no reusable improvement is supported.
"""
    + COMMON_OUTPUT
)

EXPERIENCE_PATCH_SYSTEM = """# Role
You update a SKILL.md using only experience sets that passed behavioral re-run
validation. Sets are ordered by priority: CONTRAST, FAILURE, SUCCESS.

# Objective
Produce at most {max_edits} coherent edits. Prefer validated CONTRAST guidance,
then validated FAILURE corrections, then non-duplicative validated SUCCESS
guidance. Preserve successful workflows and keep the Skill concise.

# Rules
- Every edit must cite supplied experience-set indices; invent no advice.
- Prefer local integration into an existing section.
- Every edit targets the original Current Skill; edits must not overlap.
- `find` is an exact unique substring and `replace` substitutes it.
- Empty `find` appends content only when no local anchor exists.
- The resulting Skill must not mention experiments, tasks, rollouts, rewards,
  experience sources, or validation.

# Output contract
Return exactly one JSON object and no other text:
{
  "edits": [
    {
      "find": "exact text copied from Current Skill",
      "replace": "replacement text",
      "supporting_experience_sets": [1, 3],
      "support_count": 2
    }
  ]
}
Return {"edits": []} when no edit is warranted.
"""


def cross_task_messages(
    *,
    skill: Skill,
    items: list[tuple[str, Trajectory]],
    source: str,
    max_experiences: int,
    max_tool_observation_chars: int | None = None,
    previous_batch_invalid_experience_feedback: PreviousBatchInvalidExperienceFeedback | None = None,
) -> list[dict[str, str]]:
    system = CROSS_TASK_SUCCESS_SYSTEM if source == "success" else CROSS_TASK_FAILURE_SYSTEM
    blocks = [
        _trajectory_block(
            index,
            task_id,
            trajectory,
            skill,
            max_tool_observation_chars=max_tool_observation_chars,
        )
        for index, (task_id, trajectory) in enumerate(items, start=1)
    ]
    feedback = _invalid_experience_feedback_suffix(previous_batch_invalid_experience_feedback)
    user = (
        f"# Current Skill\n\n{skill.content}{feedback}\n\n"
        f"# {source.title()} trajectories from {len(items)} tasks\n\n" + "\n\n".join(blocks)
    )
    return [
        {"role": "system", "content": system.replace("{max_experiences}", str(max_experiences))},
        {"role": "user", "content": user},
    ]


def failure_to_success_messages(
    *,
    skill: Skill,
    task_id: str,
    trajectories: list[Trajectory],
    max_experiences: int,
    max_tool_observation_chars: int | None = None,
    previous_batch_invalid_experience_feedback: PreviousBatchInvalidExperienceFeedback | None = None,
) -> list[dict[str, str]]:
    blocks = [
        _trajectory_block(
            index,
            task_id,
            trajectory,
            skill,
            max_tool_observation_chars=max_tool_observation_chars,
        )
        for index, trajectory in enumerate(trajectories, start=1)
    ]
    feedback = _invalid_experience_feedback_suffix(previous_batch_invalid_experience_feedback)
    user = (
        f"# Task\n\n{trajectories[0].task.instruction}\n\n"
        f"# Current Skill\n\n{skill.content}{feedback}\n\n"
        f"# Chronological rollouts ({len(trajectories)} total)\n\n" + "\n\n".join(blocks)
    )
    return [
        {
            "role": "system",
            "content": FAILURE_TO_SUCCESS_SYSTEM.replace("{max_experiences}", str(max_experiences)),
        },
        {"role": "user", "content": user},
    ]


def experience_patch_messages(
    *, skill: Skill, experiences: list[ExtractedExperienceSet], max_edits: int
) -> list[dict[str, str]]:
    blocks = [
        f"## Validated Experience Set {index} [source={experience.source.value}]\n\n{experience.content}"
        for index, experience in enumerate(experiences, start=1)
    ]
    user = f"# Current Skill\n\n{skill.content}\n\n# Validated Experience Sets\n\n" + "\n\n".join(blocks)
    return [
        {"role": "system", "content": EXPERIENCE_PATCH_SYSTEM.replace("{max_edits}", str(max_edits))},
        {"role": "user", "content": user},
    ]


def _trajectory_block(
    index: int,
    task_id: str,
    trajectory: Trajectory,
    skill: Skill,
    *,
    max_tool_observation_chars: int | None,
) -> str:
    error = trajectory.execution.error_info or trajectory.metadata.get("error")
    error_block = f"\n\n### Error\n\n{error}" if error else ""
    return (
        f"## Task trajectory {index}\n\n"
        f"### Task ID\n\n{task_id}\n\n"
        f"### Task\n\n{trajectory.task.instruction}\n\n"
        f"### Verified score\n\n{trajectory.reward.score}"
        f"{error_block}\n\n"
        "### Agent trajectory\n\n"
        f"{_render_extraction_events(trajectory.events, skill.content, max_tool_observation_chars)}"
    )


def _render_extraction_events(
    events: list[dict[str, Any]],
    skill_content: str,
    max_tool_observation_chars: int | None,
) -> str:
    if max_tool_observation_chars is None:
        return _render_events(events, skill_content)

    bounded_events: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("role") or "").lower() != "tool":
            bounded_events.append(event)
            continue
        content = event.get("content")
        if content in (None, ""):
            bounded_events.append(event)
            continue
        rendered_content = (
            content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2, default=str)
        )
        truncated_content = _truncate_middle(rendered_content, max_tool_observation_chars)
        bounded_events.append(
            event if truncated_content == rendered_content else {**event, "content": truncated_content}
        )
    return _render_events(bounded_events, skill_content)


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = f"\n...[tool observation middle truncated; original_chars={len(text)}]...\n"
    available = max_chars - len(marker)
    if available <= 1:
        return marker[:max_chars]
    head_chars = (available + 1) // 2
    tail_chars = available - head_chars
    return text[:head_chars] + marker + text[-tail_chars:]


def _invalid_experience_feedback_suffix(feedback: PreviousBatchInvalidExperienceFeedback | None) -> str:
    return f"\n\n{feedback.render_for_extraction()}" if feedback is not None else ""


__all__ = ["cross_task_messages", "experience_patch_messages", "failure_to_success_messages"]
