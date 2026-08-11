"""Every LLM prompt used by SkillGrpoWithReplayBuffer.

No algorithm component should define prompt prose outside this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ....typing import Skill, Task, Trajectory
from .contracts import ExtractedExperience

EXPERIENCE_EXTRACTION_SYSTEM = """# Role
You are an expert at extracting reusable task-solving experience. You will be
given multiple agent trajectories produced for the same task together with the
current Skill document. Analyze the trajectories and extract lessons that can
generalize across task instances and may later be incorporated into the Skill.
The task may involve coding, research, question answering, tool use, data
processing, or interaction with an environment. Do not assume a particular
domain or benchmark.

# Objective
Identify the highest-impact patterns that explain success or failure. Extract
only lessons supported by the supplied trajectories. Do not infer a lesson from
the final reward alone; ground it in observable decisions, actions, omissions,
errors, or recovery behavior in the trajectories.

You may extract three types of experience:

## 1. Success pattern
- The pattern is not already covered by the current Skill.
- The pattern appears in at least two successful trajectories.
- The lesson is concise and generalizes beyond this specific task instance.

## 2. Failure pattern
A failure-derived lesson may identify any of the following:
- Guidance missing from the Skill that is useful for this class of tasks.
- Existing Skill guidance that is incorrect, unclear, or misleading.
- Existing Skill guidance that the agent failed to follow.
- A problem the agent encountered but did not resolve effectively.
- A premature action taken without sufficient exploration or verification.

## 3. Success-failure contrast
- The lesson captures a decisive behavioral or decision-making difference
  between successful and failed trajectories.
- The useful behavior is present in at least one successful trajectory and
  absent from at least one failed trajectory.
- The lesson generalizes beyond this specific task instance.

# Selection rules
- Extract at most {max_experiences} experiences. Return fewer when the evidence
  does not support that many.
- Prioritize high-impact, non-duplicate, actionable lessons that are not already
  adequately covered by the current Skill.
- The topic and lesson must remain useful if task-specific names, identifiers,
  exact values, locations, and boundaries are replaced. Keep such details in
  evidence only when needed to show what happened.
- Analyze each rollout independently. Every evidence observation must describe
  something actually present in that rollout and directly support the lesson;
  never transfer a detail between rollouts or group different failure causes.
- For a failure-derived lesson, verify one causal chain: observed failure ->
  producing action, omission, or assumption -> corrective guidance that would
  address that failure. If any link is unsupported, omit the experience.
- API errors, timeouts, and other infrastructure failures that occur before
  meaningful task execution are not valid evidence for task-solving lessons.

# Output contract
Return exactly one valid JSON object matching the schema below. Do not wrap it
in Markdown fences. Do not include comments, prose, or additional keys.

{
  "experiences": [
    {
      "type": "success_pattern | failure_pattern | success_failure_contrast",
      "topic": "short, stable, task-independent topic label without incidental instance details",
      "skill_assessment": {
        "status": "missing | incorrect | unclear | misleading | not_followed",
        "reason": "how the current Skill relates to this lesson and why a change may be needed"
      },
      "lesson": "specific, concise, reusable guidance stated at the causal mechanism level",
      "reason": "concise causal explanation of why this lesson follows from the observed behavior",
      "evidence": [
        {
          "rollout": 3,
          "outcome": "success | failure",
          "observation": "specific action, omission, decision, or error directly observed in this rollout"
        }
      ]
    }
  ]
}

Use rollout indices exactly as supplied in the input. Keep each observation
brief and factual: describe what is visible in that rollout instead of merely
restating the lesson. A success pattern must include evidence from at least two
successful rollouts. A failure pattern must include evidence from at least one
failed rollout that reached meaningful task execution. A success-failure
contrast must include at least one successful and one failed rollout, and the
observations must identify the concrete behavioral difference between them. If
no experience satisfies the requirements, return {"experiences": []}.
"""


EXPERIENCE_PATCH_SYSTEM = """
# Role
You update a SKILL.md from reusable experiences extracted across multiple
tasks. Treat the experiences as evidence to synthesize, not as instructions to
copy verbatim.

# Objective
Propose a small, coherent patch that adds only high-impact guidance missing
from the current Skill or corrects guidance shown to be wrong. Treat each
numbered Task Experience Set as one source, regardless of how many rollout
observations it contains. If the Skill already covers the supported lessons,
return no edits.

# Fusion rules
- Deduplicate equivalent lessons and keep the clearest, most actionable wording.
- Resolve contradictory guidance about the same point by choosing the
  better-supported version or synthesizing a version justified by both.
- Preserve distinct, non-duplicate corrective insights.
- Prefer patterns supported by multiple Task Experience Sets because they are
  more likely to be systematic. A singleton may be kept when it is a strong,
  general corrective insight, but discard singleton task-specific details.
- Every pair of final edits must target independent, non-overlapping regions.
- For each edit, list the distinct supporting Task Experience Set indices and
  set `support_count` to the length of that list.

# Patch rules
- Return at most {max_edits} non-overlapping edits.
- Every edit addresses the original Current Skill, not the result of another
  edit in the same response.
- `find` is an exact, unique, verbatim substring of the Current Skill.
- `replace` is the text that replaces `find`; an empty string deletes it.
- Prefer a local replacement that integrates with an existing section. Use an
  empty `find` only to append content when no suitable location exists.
- Keep the resulting Skill general, concise, internally consistent, and free
  of benchmark, task, rollout, evidence, or experience references.

# Output contract
Return exactly one valid JSON object and no other text:

{
  "edits": [
    {
      "find": "exact text copied from the Current Skill",
      "replace": "replacement text",
      "supporting_experience_sets": [1, 3],
      "support_count": 2
    }
  ]
}

`support_count` counts Task Experience Sets, not rollout observations or
repeated statements within one set. When no patch is warranted, return
{"edits": []}.
"""


PATCH_REPAIR_USER = (
    "The previous response violated the patch output contract: {error}\n"
    "Return one complete corrected JSON object. Every item in "
    "`edits` must contain both string fields `find` and `replace`."
)


FUSION_SYSTEM = (
    "You maintain the single CANONICAL version of one recurring SKILL.md edit as "
    "new corroborating variants arrive over time.\n\n"
    "You receive:\n"
    "- HISTORY: the current consolidated replacement text. It already summarizes "
    "N_HISTORY earlier corroborating edits (empty on first sight).\n"
    "- NEW: one or more freshly proposed replacement texts for the SAME span.\n\n"
    "Produce ONE consolidated replacement that:\n"
    "1. INTEGRATES every distinct lesson from HISTORY and NEW.\n"
    "2. DEDUPLICATES points repeated across inputs.\n"
    "3. WEIGHTS HISTORY by its size: it already stands for N_HISTORY prior edits, "
    "so preserve its established content and let NEW refine or extend it rather "
    "than overwrite it — unless NEW clearly corrects it.\n"
    "4. PRESERVES FORMAT — match the inputs' style (bullets, short sentences, "
    "headings).\n"
    "5. NEVER invents content ungrounded in an input.\n\n"
    "Also emit CENTROID_TEXT: a <=160 char description naming the edit's location "
    "and intent, fusing the prior description with the new keys.\n\n"
    "Output ONLY JSON, no prose:\n"
    '{"merged_replace": "<consolidated replacement>", "centroid_text": "<description>"}'
)

SKILL_CONTENT_OMITTED = "[Skill content omitted: identical to the Current Skill shown above.]"

LIVEMATH_SYSTEM = """You are an expert mathematical reasoning agent solving multiple-choice questions.

{skill_section}## Task Format
You will receive one mathematics multiple-choice question and its answer choices.
Reason carefully about quantifiers, hypotheses, extremal wording, and exact equality conditions.

## Answer Format
Think step by step, then provide your final answer inside <answer>...</answer> tags.
Inside the tags, output only the single choice label, such as A or C.

Example:
<answer>B</answer>
"""


SPREADSHEET_SYSTEM = (
    "You are an expert spreadsheet assistant. Your working directory contains a "
    "source Excel file named 'input.xlsx'. Do NOT modify 'input.xlsx'. Instead, "
    "produce a new file 'output.xlsx' in the same directory that fully satisfies "
    "the user's request (start from a copy of 'input.xlsx' and apply your changes).\n"
    "Work by writing and running Python (openpyxl is available) through the shell "
    "tool — do not answer from memory. Inspect the sheets first, apply the changes, "
    "save to 'output.xlsx', and verify.\n"
    "IMPORTANT: 'output.xlsx' is graded by reading cached cell VALUES, with no "
    "formula recalculation. Write the final computed values into the target cells "
    "(not bare formulas), since an unevaluated formula reads back as empty.\n"
    "When you are done, stop without calling any tool."
)


def experience_extraction_messages(
    *,
    task: Task,
    skill: Skill,
    trajectories: list[Trajectory],
    max_experiences: int,
) -> list[dict[str, str]]:
    blocks = []
    for index, trajectory in enumerate(trajectories, start=1):
        error = trajectory.execution.error_info or trajectory.metadata.get("error")
        error_block = f"\n\n### Error\n\n{error}" if error else ""
        blocks.append(
            f"## Rollout {index}\n\n"
            f"### Verified Score\n\n"
            f"{trajectory.reward.score}"
            f"{error_block}\n\n"
            f"### Agent Trajectory\n\n"
            f"{_render_events(trajectory.events, skill.content)}"
        )
    user = (
        f"# Task\n\n{task.instruction}\n\n"
        f"# Current Skill\n\n{skill.content}\n\n"
        f"# Rollouts ({len(trajectories)} total)\n\n" + "\n\n".join(blocks)
    )
    return [
        {
            "role": "system",
            "content": EXPERIENCE_EXTRACTION_SYSTEM.replace("{max_experiences}", str(max_experiences)),
        },
        {"role": "user", "content": user},
    ]


def experience_patch_messages(
    *,
    skill: Skill,
    experiences: list[ExtractedExperience],
    max_edits: int,
) -> list[dict[str, str]]:
    blocks = [
        f"## Task Experience Set {index}\n\n{experience.content}"
        for index, experience in enumerate(experiences, start=1)
    ]
    user = f"# Current Skill\n\n{skill.content}\n\n# Task Experience Sets ({len(experiences)} total)\n\n" + "\n\n".join(
        blocks
    )
    return [
        {"role": "system", "content": EXPERIENCE_PATCH_SYSTEM.replace("{max_edits}", str(max_edits))},
        {"role": "user", "content": user},
    ]


def patch_repair_message(error: str) -> dict[str, str]:
    return {"role": "user", "content": PATCH_REPAIR_USER.format(error=error)}


def fusion_messages(
    *,
    history_replace: str | None,
    history_count: int,
    new_replaces: list[str],
    history_centroid_text: str,
    new_key_texts: list[str],
) -> list[dict[str, str]]:
    history = history_replace or "(none — this is the first observation of this edit)"
    proposals = "\n\n".join(
        f"## New proposal {index}\n{replace if replace.strip() else '(empty — deletes the span)'}"
        for index, replace in enumerate(new_replaces)
    )
    keys = "\n".join(f"- {key}" for key in new_key_texts) or "(none)"
    user = (
        f"# HISTORY (N_HISTORY = {history_count} prior edit(s))\n"
        f"Prior description: {history_centroid_text or '(none)'}\n"
        f"Prior consolidated replacement:\n{history}\n\n"
        f"# NEW ({len(new_replaces)} proposal(s) this batch)\n{proposals}\n\n"
        f"# New edit keys (for the description)\n{keys}\n\n"
        "Merge HISTORY and NEW into one consolidated replacement (weighting "
        f"HISTORY as {history_count} edits) and emit the fused CENTROID_TEXT."
    )
    return [{"role": "system", "content": FUSION_SYSTEM}, {"role": "user", "content": user}]


def spreadsheet_messages(*, task: Task, skill_names: list[str]) -> list[dict[str, str]]:
    system = SPREADSHEET_SYSTEM
    if skill_names:
        names = ", ".join(skill_names)
        system += (
            f"\n\nA `skill` tool is available with expert skills: {names}. "
            f'Call it first (e.g. skill(name="{skill_names[0]}")) to load detailed '
            "guidance and the absolute path to reusable reference scripts before you start."
        )
    user = (
        "The source file 'input.xlsx' is in your working directory.\n\n"
        f"Task:\n{task.instruction}\n\n"
        "Complete the task and save the result as 'output.xlsx' "
        "(do not modify 'input.xlsx')."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def livemath_system_prompt(skill_content: str) -> str:
    skill_section = f"## Skill\n{skill_content.strip()}\n\n" if skill_content.strip() else ""
    return LIVEMATH_SYSTEM.format(skill_section=skill_section)


def livemath_user_prompt(item: Mapping[str, Any], use_theorem: bool, use_sketch: bool) -> str:
    choices = "\n".join(f"{choice['label']}. {choice['text']}" for choice in item["choices"])
    parts = [f"## Question\n{item['question']}", f"## Choices\n{choices}"]
    if use_theorem and item.get("theorem"):
        parts.append(f"## Theorem\n{item['theorem']}")
    if use_sketch and item.get("sketch"):
        parts.append(f"## Proof Sketch\n{item['sketch']}")
    return "\n\n".join(parts)


def livemath_refinement_prompt(previous_response: str) -> str:
    return (
        f"Your previous answer was:\n{previous_response}\n\n"
        "Re-evaluate the exact option wording. If needed, correct it. "
        "Output only the final choice label inside <answer>...</answer>."
    )


def _render_events(events: list[dict[str, Any]], skill_content: str) -> str:
    rendered: list[str] = []
    for index, event in enumerate(events, start=1):
        if "action" in event and "env_feedback" in event:
            rendered.append(_render_step_event(event))
            continue
        role = str(event.get("role") or "?").upper()
        parts = [f"#### [{index}] {role}"]

        content = _message_content(event)
        if content:
            content = _omit_duplicate_skill_content(content, skill_content).strip()
            if role == "TOOL":
                parts.append(f"Observation:\n{_fenced_block(content, 'text')}")
            else:
                parts.append(content)

        for call in event.get("tool_calls") or []:
            parts.append(f"Action:\n{_format_tool_action(call)}")

        rendered.append("\n\n".join(parts))
    return "\n\n---\n\n".join(rendered)


def _render_step_event(event: dict[str, Any]) -> str:
    """Match SkillOpt's analyst-facing ALFWorld trajectory format."""

    step = event.get("step", "?")
    reasoning = str(event.get("reasoning") or "")[:300]
    action = str(event.get("action") or "")[:200]
    feedback = str(event.get("env_feedback") or "")[:500]
    lines: list[str] = []
    if reasoning:
        lines.append(f"[step {step} think] {reasoning}")
    lines.append(f"[step {step} action] {action}")
    lines.append(f"[step {step} obs]    {feedback}")
    return "\n".join(lines)


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content in (None, ""):
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2, default=str)


def _format_tool_action(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    action = {"name": function.get("name", "?"), "arguments": arguments}
    return json.dumps(action, ensure_ascii=False, indent=2, default=str)


def _fenced_block(content: str, language: str) -> str:
    fence = "```"
    while fence in content:
        fence += "`"
    return f"{fence}{language}\n{content}\n{fence}"


def _omit_duplicate_skill_content(text: str, skill_content: str) -> str:
    normalized_skill = _normalize_newlines(skill_content).strip()
    if "\n" not in normalized_skill:
        return text
    normalized_text = _normalize_newlines(text)
    if normalized_skill not in normalized_text:
        return text
    return normalized_text.replace(normalized_skill, SKILL_CONTENT_OMITTED)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


__all__ = [
    "EXPERIENCE_EXTRACTION_SYSTEM",
    "EXPERIENCE_PATCH_SYSTEM",
    "FUSION_SYSTEM",
    "LIVEMATH_SYSTEM",
    "PATCH_REPAIR_USER",
    "SPREADSHEET_SYSTEM",
    "experience_extraction_messages",
    "experience_patch_messages",
    "fusion_messages",
    "livemath_refinement_prompt",
    "livemath_system_prompt",
    "livemath_user_prompt",
    "patch_repair_message",
    "spreadsheet_messages",
]
