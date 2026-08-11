"""Replay-free extraction and source-prioritized patch prompts.

The cross-task prompts adapt the common-pattern constraints used by SkillOpt
and the evidence-traceability/preservation rules used by Trace2Skill.
"""

from __future__ import annotations

from ....typing import Skill, Trajectory
from ..skill_grpo_with_replay_buffer.prompts import _render_events
from .contracts import ReplayFreeExtractedExperience

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
the evidence does not justify a Skill change. Do not include task-specific
names, values, paths, or benchmark details in a lesson.
"""

CROSS_TASK_SUCCESS_SYSTEM = (
    """# Role
You are a cross-task success-pattern analyst. You receive the last successful
trajectory from each of several distinct tasks and the current Skill.

# Objective
Identify high-impact behaviors that recur across MULTIPLE tasks and are worth
preserving in the Skill. Read every trajectory, retain only patterns supported
by at least two distinct tasks, and patch only guidance missing or unclear in
the current Skill. Prefer compact reinforcement of an existing section. Do not
promote an isolated trick, incidental detail, or success inferred from reward
alone. Every lesson must trace to concrete observed actions or verification.
"""
    + COMMON_OUTPUT
)

CROSS_TASK_FAILURE_SYSTEM = (
    """# Role
You are a cross-task failure-pattern analyst. You receive the first failed
trajectory from each of several distinct tasks and the current Skill.

# Objective
Identify the most prevalent systematic failure mechanisms across the group.
Read every trajectory and prioritize patterns repeated across MULTIPLE tasks,
not individual edge cases. For each lesson verify the causal chain: observed
failure -> producing action, omission, or unsupported assumption -> corrective
guidance. Do not learn from infrastructure failures that occurred before
meaningful execution, and do not duplicate guidance already covered by the
current Skill. Every lesson must trace to concrete evidence.
"""
    + COMMON_OUTPUT
)

FAILURE_TO_SUCCESS_SYSTEM = (
    """# Role
You are a failure-to-success contrast analyst. You receive the chronological
rollouts of ONE task: one or more failed attempts followed by the first
successful attempt, plus the current Skill.

# Objective
Compare the attempts directly and identify the smallest decisive change that
caused progress from failure to success. Trace how reflections changed the
strategy, actions, sequencing, tool use, checks, or recovery behavior. Separate
mere correlation from a supported causal watershed: state what failed, what
changed, and why that change plausibly enabled success. Prefer the improvement
factor shared by several failed attempts over superficial differences. If the
successful attempt does not provide evidence for a reusable improvement, emit
no experience. Do not restate the task solution or infer behavior from reward
alone.
"""
    + COMMON_OUTPUT
)

EXPERIENCE_PATCH_SYSTEM = """# Role
You update a SKILL.md from replay-free rollout experiences. The experience sets
are already sorted by source priority:
1. CONTRAST: same-task failure-to-success causal comparisons (highest priority)
2. FAILURE: cross-task first-failure patterns
3. SUCCESS: cross-task last-success patterns (lowest priority)

# Objective
Produce a small coherent patch. You may use at most {max_edits} edits and should
use fewer when possible. First adopt well-supported CONTRAST guidance; then use
FAILURE guidance to cover remaining recurring gaps; only then add SUCCESS
guidance that is not already addressed. A lower-priority source must not replace,
dilute, or duplicate a compatible higher-priority correction. Preserve proven
successful workflows when resolving conflicts.

# Fusion and traceability rules
- Treat each numbered Experience Set as one evidence source.
- Deduplicate equivalent lessons and discard task-specific singleton details.
- Every edit must trace to supplied experience-set indices; never invent advice.
- Prefer local integration into an existing section and keep the Skill concise.
- Every pair of edits must target independent, non-overlapping regions.

# Patch rules
- Every edit addresses the original Current Skill, not another edit's output.
- `find` must be an exact unique verbatim substring; `replace` substitutes it.
- An empty `find` appends content and should be used only when no local anchor exists.
- The resulting Skill must not mention tasks, rollouts, rewards, sources, or evidence.

# Output contract
Return exactly one valid JSON object and no other text:
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
When no edit is warranted, return {"edits": []}.
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
        f"# {source.title()} trajectories from {len(items)} distinct tasks\n\n" + "\n\n".join(blocks)
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
        {
            "role": "system",
            "content": FAILURE_TO_SUCCESS_SYSTEM.replace("{max_experiences}", str(max_experiences)),
        },
        {"role": "user", "content": user},
    ]


def experience_patch_messages(
    *, skill: Skill, experiences: list[ReplayFreeExtractedExperience], max_edits: int
) -> list[dict[str, str]]:
    blocks = [
        f"## Experience Set {index} [source={experience.source.value}; "
        f"tasks={','.join(experience.task_ids)}]\n\n{experience.content}"
        for index, experience in enumerate(experiences, start=1)
    ]
    user = f"# Current Skill\n\n{skill.content}\n\n# Ordered Experience Sets\n\n" + "\n\n".join(blocks)
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
