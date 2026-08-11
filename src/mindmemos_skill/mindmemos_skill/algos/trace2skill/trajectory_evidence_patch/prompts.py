"""Prompts for summarizing offline trajectory evidence and patching a Skill."""

from __future__ import annotations

import json

from .models import TrajectorySummary

SUMMARY_SYSTEM = """You are a concise expert in AI trajectory analysis. Given an agent trajectory that used a Skill, produce an analytical summary in 8-15 sentences.

Cover the user-facing goal, the main sequence of actions, important strategy changes, tool failures, verification behavior, the Skill guidance that helped or hurt, and the final outcome. Preserve causal relationships and concrete evidence. Do not propose a Skill patch and do not invent facts not supported by the trajectory. Output only a compact plain-text paragraph."""

PATCH_QUALITY_PRINCIPLES = """Follow these principles for every Skill edit:
1. Failure mechanism encoding: explain why the failure occurs instead of saying only to be careful.
2. Actionable specificity: provide an executable step, not a vague attitude.
3. High-risk action blacklist: explicitly name dangerous actions and the failures they cause."""

PROPOSE_PATCH_SYSTEM = f"""You maintain a reusable SKILL.md that guides an autonomous agent on a class of tasks. You are given the current Skill and analytical summaries from several different real sessions. The sessions have no outcome labels.

Treat them as field observations. Infer recurring helpful behavior, repeated mistakes or dead ends, and missing guidance that generalizes to future tasks. Never overfit to one task's values, filenames, contents, or exact answer. Propose a minimal general change plan with the exact guidance to add, revise, or remove and a short evidence-based rationale. If the current Skill already covers the transferable lessons, explicitly propose no edits.

{PATCH_QUALITY_PRINCIPLES}"""

PROPOSE_PATCH_ANNOTATED_SYSTEM = f"""You maintain a reusable SKILL.md that guides an autonomous agent on a class of tasks. You are given the current Skill and analytical summaries from several different real sessions. Some or all observations carry outcome scores or evaluator feedback.

Use available annotations as the primary outcome signal. Reinforce behavior recurring in higher-score sessions and discourage mistakes or risky actions recurring in lower-score sessions. An observation marked unknown is unlabeled, not a failure. If behavior appears in both good and bad outcomes, do not claim that it discriminates outcome. Favor patterns that generalize across tasks and never hard-code a task value, filename, exact answer, or score into the Skill.

Propose a minimal general change plan with exact guidance and an evidence-based rationale. If the evidence does not justify a change, explicitly propose no edits.

{PATCH_QUALITY_PRINCIPLES}"""

APPLY_PATCH_SYSTEM = """Apply an approved change plan to SKILL.md using line-addressed JSON edit operations. The current document is shown with a 1-based `N|` gutter; the gutter is not part of the document.

Return one JSON object with an `edits` array. Supported operations are:
{"op":"replace","start":1,"end":1,"new":"replacement","old_string_prefix":"old line prefix"}
{"op":"delete","start":1,"end":1,"old_string_prefix":"old line prefix"}
{"op":"insert","after":1,"new":"inserted text"}

Ranges are inclusive and refer to the original line numbers. Use `after: 0` to prepend. Replace/delete ranges may not overlap. Include `old_string_prefix` on replace/delete to guard against a wrong line number. Apply only the approved change plan. If no edits are needed, return {"edits":[]}. Output JSON only."""

REWRITE_SKILL_SYSTEM = """You are a Markdown format-repair editor. Repair only presentation damage introduced by automated edits, such as fused bullets, headings, or missing line breaks. Preserve every instruction, the complete frontmatter, code blocks, wording, and order. Do not add, delete, merge, deduplicate, summarize, or reinterpret content. Output only the complete repaired SKILL.md."""


def summarize_trajectory_user(skill_name: str, transcript: str) -> str:
    return f"# Injected Skill\n{skill_name}\n\n# Complete agent trajectory\n{transcript}"


def propose_patch_user(skill_name: str, skill_md: str, summaries: list[TrajectorySummary]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(summaries, start=1):
        score = "unknown" if item.score is None else f"{item.score:g}"
        detail = item.annotation_detail or "unknown"
        metadata = json.dumps(item.annotation_metadata, ensure_ascii=False, sort_keys=True) or "{}"
        blocks.append(
            f"## Observation {index}\n"
            f"trajectory_id: {item.trajectory_id}\n"
            f"task_id: {item.task_id}\n"
            f"score: {score}\n"
            f"evaluator_feedback: {detail}\n"
            f"annotation_metadata: {metadata}\n"
            f"summary: {item.summary}"
        )
    joined = "\n\n".join(blocks) if blocks else "(no observations)"
    return (
        f"# Skill name\n{skill_name}\n\n"
        f"# Current SKILL.md\n{skill_md}\n\n"
        f"# Offline trajectory evidence\n{joined}\n\n"
        "Propose the smallest general change plan justified by the evidence."
    )


def apply_patch_user(skill_md: str, patch_plan: str, numbered_skill_md: str) -> str:
    return (
        "# Current SKILL.md with line-number gutter\n"
        f"{numbered_skill_md}\n\n"
        f"# Approved change plan\n{patch_plan}\n\n"
        "Return the JSON edits object only."
    )


def rewrite_skill_user(skill_md: str) -> str:
    return f"# SKILL.md to format-repair\n{skill_md}\n\nReturn only the repaired complete document."


__all__ = [
    "APPLY_PATCH_SYSTEM",
    "PROPOSE_PATCH_ANNOTATED_SYSTEM",
    "PROPOSE_PATCH_SYSTEM",
    "REWRITE_SKILL_SYSTEM",
    "SUMMARY_SYSTEM",
    "apply_patch_user",
    "propose_patch_user",
    "rewrite_skill_user",
    "summarize_trajectory_user",
]
