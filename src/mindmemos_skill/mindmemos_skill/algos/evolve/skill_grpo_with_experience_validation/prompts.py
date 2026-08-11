"""Prompts for three-stream experience extraction and accepted-experience patching."""

from __future__ import annotations

from ....typing import Skill, Trajectory
from ..skill_grpo_with_replay_buffer.prompts import _render_events
from .contracts import ExtractedExperienceSet

COMMON_OUTPUT = """
# Output contract
Return exactly one valid JSON object and no other text:
{
  "experiences": [
    {
      "topic": "short task-independent label",
      "lesson": "concise reusable guidance",
      "reason": "causal explanation grounded in the supplied evidence",
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
) -> list[dict[str, str]]:
    system = CROSS_TASK_SUCCESS_SYSTEM if source == "success" else CROSS_TASK_FAILURE_SYSTEM
    blocks = [
        _trajectory_block(index, task_id, trajectory, skill)
        for index, (task_id, trajectory) in enumerate(items, start=1)
    ]
    user = (
        f"# Current Skill\n\n{skill.content}\n\n"
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
) -> list[dict[str, str]]:
    blocks = [
        _trajectory_block(index, task_id, trajectory, skill) for index, trajectory in enumerate(trajectories, start=1)
    ]
    user = (
        f"# Task\n\n{trajectories[0].task.instruction}\n\n"
        f"# Current Skill\n\n{skill.content}\n\n"
        f"# Chronological rollouts ({len(trajectories)} total)\n\n" + "\n\n".join(blocks)
    )
    return [
        {"role": "system", "content": FAILURE_TO_SUCCESS_SYSTEM.replace("{max_experiences}", str(max_experiences))},
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


def _trajectory_block(index: int, task_id: str, trajectory: Trajectory, skill: Skill) -> str:
    error = trajectory.execution.error_info or trajectory.metadata.get("error")
    error_block = f"\n\n### Error\n\n{error}" if error else ""
    return (
        f"## Task trajectory {index}\n\n"
        f"### Task ID\n\n{task_id}\n\n"
        f"### Task\n\n{trajectory.task.instruction}\n\n"
        f"### Verified score\n\n{trajectory.reward.score}"
        f"{error_block}\n\n"
        f"### Agent trajectory\n\n{_render_events(trajectory.events, skill.content)}"
    )


__all__ = ["cross_task_messages", "experience_patch_messages", "failure_to_success_messages"]
