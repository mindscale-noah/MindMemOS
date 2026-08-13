"""Concise prompts for reflection and tree-local Skill improvement."""

from __future__ import annotations

import json
from dataclasses import asdict

from ....typing import Task, Trajectory
from .models import SkillTree, TreeEditProposal

_MAX_MESSAGE_CHARS = 2_000
_TRUNCATION_MARKER = "\n...[message middle truncated]...\n"

# Design sources:
# - Reflexion: diagnose the attempted strategy and produce a concise new plan grounded in specific actions.
#   https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/generate_reflections.py
# - Self-Refine: feedback should identify specific problems and give concrete, actionable corrections.
#   https://proceedings.neurips.cc/paper_files/paper/2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf
# - CRITIC: verify claims against available external feedback before correcting the answer.
#   https://proceedings.iclr.cc/paper_files/paper/2024/file/fef126561bbf9d4467dbb8d27334b8fe-Paper-Conference.pdf
REFLECTION_SYSTEM_PROMPT = """# Role
You are a conservative failure analyst. Convert one unsuccessful agent trajectory into specific verbal feedback that
will improve the immediately following attempt.

# Analysis rules
1. Reconstruct the attempted strategy from the task, actions, and observations. Do not merely summarize the trajectory.
2. Identify the earliest pivotal error, omission, or unsupported assumption that is most likely to explain the failure.
   Ground it in quoted or precisely referenced trajectory evidence. Treat the binary outcome as a result signal, not a
   diagnosis, and state uncertainty when the evidence does not isolate a cause.
3. Separate agent-controllable mistakes from tool, environment, evaluator, and infrastructure failures. Do not invent a
   strategic lesson or a hidden expected answer from external failure alone.
4. Preserve steps that were already correct. Propose one materially different, concise, ordered plan that directly
   addresses the diagnosis, includes a verification checkpoint, and says how to recover if the key assumption is
   contradicted by the next observation.
5. Preserve the environment's required reasoning, action, tool-use, and final-answer protocol.

# Output contract
Return only these sections in at most 250 words:
Diagnosis: <one likely root cause, with calibrated uncertainty>
Evidence: <the decisive action/observation and its consequence>
Preserve: <correct behavior worth retaining, or "none identified">
New plan: <numbered, executable steps for the next attempt>
Verification: <what evidence will confirm the correction, plus a fallback if it does not>
"""

REFLECTION_USER_PROMPT = """# Original task
{task}

# Outcome
{outcome}

# Failed trajectory
{trajectory}

Generate the reflection. Use only information available above; if it is insufficient, say so and make the first step
an information-gathering or state-recovery action."""

RETRY_SYSTEM_CONTEXT = """# Previous unsuccessful attempt
Previous answer:
{previous_answer}

Reflection:
{reflection}

Treat the reflection as a fallible hypothesis: preserve correct prior work, execute the corrected plan, and check it
against new tool results or environment observations before committing. If evidence contradicts it, use the stated
fallback instead of repeating the failed strategy. Continue to follow the environment's required reasoning, action,
tool-use, and final-answer protocol."""

TREE_EDIT_SYSTEM_PROMPT = """# Role
Improve one Markdown Skill using one task's chronological rollout attempts. The first attempt failed and later attempts
were made with reflection. Produce at most one high-impact, reusable tree edit.

# Evidence and quality rules
- Analyze every attempt independently. Ground conclusions in observed decisions, actions, omissions, errors, recovery,
  or verification behavior; never infer an improvement from reward alone or transfer details between attempts.
- For failure-to-success, identify the smallest decisive behavioral difference. For repeated failure, verify the causal
  chain: observed failure -> producing action, omission, or assumption -> corrective guidance. Reject unsupported links.
- Determine whether the current Skill is missing guidance, incorrect, unclear, misleading, or already adequate but not
  followed. Do not duplicate or rewrite adequate guidance merely because the agent failed to follow it.
- The guidance must remain useful when task-specific names, identifiers, values, paths, answers, locations, and bounds
  are replaced. Never mention the task, benchmark, attempts, rewards, evidence, or reflection in Skill content.
- Treat the current target node as canonical history: preserve distinct correct guidance, integrate the new lesson,
  deduplicate equivalent points, resolve contradictions only when evidence supports the correction, and match the
  existing format and style. Keep the result concise, actionable, and internally consistent.

# Operations
- edit_node: edit the target node's direct content with an exact literal find/replace. `find` must be copied verbatim
  from the current direct content and occur exactly once. `replace` is the text that replaces it and may be empty.
- append_node: append new guidance to the end of the target node's direct content. Append only genuinely new content;
  do not repeat or paraphrase guidance already present.
- create_child: add one non-duplicate child node or recursive subtree when the lesson warrants a separate section and no
  existing node is a suitable location.
- reject: make no change when evidence is weak, conflicting, task-specific, infrastructure-only, already covered, or
  does not justify a general Skill improvement.

# Output contract
Return exactly one valid JSON object and no other text. Use exactly one of these schemas:
{"operation":"edit_node","target_node_id":"node-a1b2c3d4e5f6","find":"exact original text occurring once","replace":"replacement text","reason":"brief evidence-based reason"}
{"operation":"append_node","target_node_id":"node-a1b2c3d4e5f6","content":"new text appended after current direct content","reason":"brief evidence-based reason"}
{"operation":"create_child","target_node_id":"node-a1b2c3d4e5f6","child":{"heading":"New section","content":"guidance","children":[]},"reason":"brief evidence-based reason"}
{"operation":"reject","reason":"brief reason"}
"""

TREE_FUSION_SYSTEM_PROMPT = """# Role
Fuse independent task-local proposals that were all produced against the same frozen Skill tree and target region.
Do not reinterpret or combine the tasks' raw trajectories; treat each proposal as one already-grounded candidate.

# Fusion rules
- The original target node direct content is provided exactly once and is canonical history. The proposals contain only
  edit information produced against that same original content.
- Apply compatible edit_node and append_node proposals, preserve distinct correct guidance and existing style, deduplicate
  equivalent wording, and keep the clearest actionable form.
- Resolve contradictions only when a proposal clearly corrects current guidance. Otherwise preserve the current content.
- Never invent guidance absent from the proposals. Keep all resulting Skill content general and concise.
- Merge semantically equivalent create_child proposals into one child or subtree; keep genuinely distinct children separate.
- Do not return a child whose heading duplicates an existing direct child.
- Omit weak, conflicting, redundant, or already-covered proposals.
- Return the target node's complete final direct content, not another patch. If no direct-content change is warranted,
  copy the original direct content unchanged. Return only genuinely new child nodes; existing children remain unchanged.

# Output contract
Return exactly one valid JSON object and no other text:
{"content":"complete merged direct content","children":[{"heading":"New section","content":"guidance","children":[]}]}
Use an empty children list when no child should be added.
"""


def reflection_messages(trajectory: Trajectory) -> list[dict[str, str]]:
    """Build one evidence-grounded reflection request from the immediately preceding failure."""

    return [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REFLECTION_USER_PROMPT.format(
                task=trajectory.task.instruction,
                outcome="success" if (trajectory.reward.score or 0.0) > 0.0 else "fail",
                trajectory=_render_events(trajectory),
            ),
        },
    ]


def task_with_reflection(task: Task, *, previous_answer: str, reflection: str) -> Task:
    """Put only the immediately preceding reflection in the retry system prompt."""

    context = RETRY_SYSTEM_CONTEXT.format(
        previous_answer=previous_answer[-6_000:].strip(),
        reflection=reflection[-4_000:].strip(),
    )
    system_prompt = f"{task.system_prompt.rstrip()}\n\n{context}" if task.system_prompt else context
    return task.model_copy(update={"system_prompt": system_prompt}, deep=True)


def tree_edit_messages(tree: SkillTree, trajectories: list[Trajectory]) -> list[dict[str, str]]:
    """Build one attribution request for the attempts of exactly one task."""

    attempts: list[str] = []
    for attempt_index, trajectory in enumerate(trajectories, start=1):
        outcome = "success" if (trajectory.reward.score or 0.0) > 0.0 else "failure"
        attempts.append(
            f"## Attempt {attempt_index}: {outcome}\n"
            f"Reward: {trajectory.reward.score}\n"
            f"Reflection system context: {trajectory.task.system_prompt or '(none)'}\n"
            f"Trajectory:\n{_render_events(trajectory)}"
        )
    user = (
        "# Current Skill tree\n\n"
        f"{json.dumps(asdict(tree), ensure_ascii=False, indent=2)}\n\n"
        f"# Task\n\ntask_id={trajectories[0].task.task_id}\n{trajectories[0].task.instruction}\n\n"
        "# Chronological attempts\n\n" + "\n\n".join(attempts)
    )
    return [{"role": "system", "content": TREE_EDIT_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def tree_fusion_messages(
    tree: SkillTree,
    *,
    target_node_id: str,
    proposals: list[TreeEditProposal],
) -> list[dict[str, str]]:
    """Build one node-local fusion request from task-local proposals."""

    target = tree.node(target_node_id)
    original = {
        "node_id": target.node_id,
        "heading": target.heading,
        "content": target.content,
        "existing_child_headings": [child.heading for child in target.children],
    }
    edits = [proposal.edit.model_dump(mode="json", exclude_none=True) for proposal in proposals]
    user = (
        "# Original target node\n\n"
        f"{json.dumps(original, ensure_ascii=False, indent=2)}\n\n"
        "# Independent edit information\n\n"
        f"{json.dumps(edits, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": TREE_FUSION_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _render_events(trajectory: Trajectory) -> str:
    messages: list[str] = []
    for index, event in enumerate(
        (event for event in trajectory.events if event.get("role") != "system"),
        start=1,
    ):
        rendered = json.dumps(event, ensure_ascii=False, indent=2, default=str)
        if len(rendered) > _MAX_MESSAGE_CHARS:
            available = _MAX_MESSAGE_CHARS - len(_TRUNCATION_MARKER)
            head_chars = (available + 1) // 2
            tail_chars = available // 2
            rendered = rendered[:head_chars] + _TRUNCATION_MARKER + rendered[-tail_chars:]
        messages.append(f"## Message {index}\n{rendered}")
    return "\n\n".join(messages)


__all__ = [
    "REFLECTION_SYSTEM_PROMPT",
    "REFLECTION_USER_PROMPT",
    "RETRY_SYSTEM_CONTEXT",
    "TREE_EDIT_SYSTEM_PROMPT",
    "TREE_FUSION_SYSTEM_PROMPT",
    "reflection_messages",
    "task_with_reflection",
    "tree_edit_messages",
    "tree_fusion_messages",
]
