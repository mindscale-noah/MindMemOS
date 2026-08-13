"""Prompts for trajectory summaries and source-grounded subtask decomposition."""

from __future__ import annotations

import json

from ....typing import Skill
from .models import TrajectoryKeyPoints

SUMMARY_SYSTEM = """You are a concise expert in AI trajectory analysis. Analyze exactly one agent trajectory and return its key points.

Use only facts supported by the trajectory. Cover the user-facing goal, the main action sequence, important strategy changes or failures, how the injected Skill guidance was used, and the final outcome. `task_family` must describe the concrete independently completable subtask demonstrated by this trajectory, not a library, knowledge topic, document heading, or generic workflow phase. Do not invent missing facts.

Return exactly one JSON object with no Markdown fence:
{
  "task_goal": "user-facing goal",
  "task_family": "independently completable subtask type",
  "key_actions": ["concrete action"],
  "turning_points": ["trigger, changed strategy, and effect"],
  "skill_usage": ["specific Skill guidance followed, useful, missing, misleading, or ignored"],
  "outcome": "result, unresolved risk, and evidence-based confidence"
}
"""

DECOMPOSE_SYSTEM = """You split one existing agent Skill into a flat set of virtual Skills organized by independently completable subtask goals.

Trajectory summaries determine which subtask boundaries are useful. The original SKILL.md is the ONLY allowed source of virtual-Skill instructions. Trajectories may never contribute new instructions.

Hard rules:
1. Generate a virtual Skill only when both conditions hold: sampled trajectories demonstrate that subtask, and the original SKILL.md contains explicit actionable guidance for it.
2. Each virtual Skill must independently complete one small function and have an observable result. Do not split by library, topic, heading, setup phase, inspection phase, or verification phase unless that phase itself is the complete demonstrated task.
3. Virtual Skills have no dependencies. If one needs another to finish, merge them.
4. `source_excerpts` must be exact, verbatim, contiguous excerpts copied from the provided SKILL.md. Include all original guidance needed for the subtask. Do not paraphrase, expand, repair, or invent instructions.
5. `supporting_trajectory_ids` may reference only the supplied sampled summaries and must genuinely demonstrate the subtask.
6. If trajectories demonstrate a subtask that the source Skill does not teach, do not generate it. If the source teaches something absent from sampled trajectories, do not generate it.
7. Descriptions are routing metadata only: state the triggering task goal and produced result without adding operational advice.
8. Prefer fewer well-supported Skills. Return an empty list when no subtask satisfies both evidence gates.

Return exactly one JSON object with no Markdown fence:
{
  "skill_name": "...",
  "description": "...",
  "virtual_skills": [
    {
      "skill_id": "independent-subtask",
      "name": "Independent subtask",
      "description": "Select for ...; produces ...",
      "supporting_trajectory_ids": ["trajectory-id"],
      "source_excerpts": ["exact excerpt copied from SKILL.md"]
    }
  ]
}
"""


def summarize_trajectory_user(
    *, skill_name: str, trajectory_id: str, task_instruction: str, transcript: str
) -> str:
    return (
        f"# Injected Skill\n{skill_name}\n\n"
        f"# Trajectory ID\n{trajectory_id}\n\n"
        f"# User-facing task\n{task_instruction}\n\n"
        f"# Complete agent trajectory\n{transcript}"
    )


def decomposition_user(
    *,
    skill: Skill,
    summaries: list[TrajectoryKeyPoints],
    max_virtual_skills: int,
) -> str:
    blocks = []
    for item in summaries:
        blocks.append(
            f"## Trajectory {item.trajectory_id}\n"
            f"task_id: {item.task_id}\n"
            f"score: {'unknown' if item.score is None else item.score}\n"
            f"summary: {json.dumps(item.model_dump(mode='json'), ensure_ascii=False)}"
        )
    joined = "\n\n".join(blocks) if blocks else "(no sampled summaries)"
    return (
        f"# Maximum virtual Skills\n{max_virtual_skills}\n\n"
        f"# Original Skill name\n{skill.name}\n\n"
        f"# Original SKILL.md — sole instruction source\n{skill.content}\n\n"
        f"# Sampled trajectory summaries — boundary evidence only\n{joined}\n\n"
        "Produce only virtual Skills supported by both sections."
    )


__all__ = [
    "DECOMPOSE_SYSTEM",
    "SUMMARY_SYSTEM",
    "decomposition_user",
    "summarize_trajectory_user",
]
