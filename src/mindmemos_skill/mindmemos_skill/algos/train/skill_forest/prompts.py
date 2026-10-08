"""Prompt rendering for trajectory reflection and forest synthesis."""

import difflib
import json

from pydantic import JsonValue

_PROMPTS: dict[str, str] = {
    "analyst_error": """\
You are an expert failure-analysis agent for AI agent tasks.

You will be given MULTIPLE failed agent trajectories from a single minibatch
and the current skill document.
Your job is to identify the most important COMMON failure patterns across
the batch and propose a concise set of skill edits.

## Analysis Process
1. Read ALL trajectories in the minibatch.
2. Identify the most prevalent, systematic failure patterns across them.
3. For each pattern, classify its failure type.
4. Propose skill edits that address the COMMON patterns — not individual edge cases.
5. Edits must be generalizable; do not hardcode task-specific values.
6. Only patch gaps in the skill — do not duplicate existing content.

You will be told the maximum number of edits (the budget L). Produce AT MOST L edits,
focusing on the highest-impact patterns. You may produce fewer if warranted.

Respond ONLY with a valid JSON object (no markdown fences, no extra text):
{
  "batch_size": <number of trajectories analysed>,
  "failure_summary": [
    {"failure_type": "<type>", "count": <int>, "description": "<one-line>"}
  ],
  "patch": {
    "reasoning": "<why these edits address the batch's common failures>",
    "edits": [
      {"op": "append",       "content": "<markdown to add at end of skill>"},
      {"op": "insert_after", "target": "<exact heading/text to insert after>", "content": "<markdown>"},
      {"op": "replace",      "target": "<exact text to replace>",              "content": "<replacement>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}
Only include edits that are needed. "edits" can be an empty list if no patch is warranted.
""",
    "analyst_success": """\
You are an expert success-pattern analyst for AI agents.

You will be given MULTIPLE successful agent trajectories from a single minibatch
and the current skill document. Your job is to identify generalizable behavior
patterns that are COMMON across the batch and worth encoding in the skill.

## Rules
- Only propose patches for patterns NOT already covered in the skill.
- Focus on patterns that appear across MULTIPLE trajectories in the batch.
- Be concise. Patterns must generalize beyond specific tasks.
- Prefer reinforcing existing sections over adding new top-level sections.

You will be told the maximum number of edits (the budget L). Produce AT MOST L edits,
focusing on the most broadly applicable patterns. You may produce fewer if warranted.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number of trajectories analysed>,
  "success_patterns": ["<pattern 1>", "<pattern 2>"],
  "patch": {
    "reasoning": "<why these patterns are worth encoding>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}
"edits" may be empty if the skill already covers all observed patterns.
""",
    "merge_failure": """\
You are a skill-edit coordinator. You receive multiple independently-proposed patches
from FAILURE analysis of agent trajectories. Merge them into ONE coherent, non-redundant patch.

Merge guidelines:
1. **Deduplicate**: keep the best-worded version of similar edits.
2. **Resolve conflicts**: if patches contradict on the same point,
   choose the one with stronger justification or synthesize both.
3. **Preserve unique insights**: include all non-redundant corrective edits.
4. **Prevalent-pattern bias**: edits appearing consistently across multiple patches
   address systematic failures — preserve them with HIGH priority.
   Edits from only one patch may be discarded if task-specific.
5. **Independence**: no two edits in the merged patch may target the same text region.
6. **Support count**: for each merged edit, estimate how many source patches support it.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary of key consolidation decisions>",
    "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if insert_after or replace or delete>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "failure"
    }
  ]
}
""",
    "merge_success": """\
You are a skill-edit coordinator. You receive multiple independently-proposed patches
from SUCCESS analysis of agent trajectories. Merge them into ONE coherent patch
that reinforces effective patterns.

Merge guidelines:
1. **Deduplicate**: keep only the most generalizable version of similar patterns.
2. **Be conservative**: success-driven patches reinforce existing behavior.
   Only include edits for patterns NOT already in the skill.
3. **Prevalent-pattern bias**: patterns seen across many successful trajectories
   are most worth encoding.
4. **Support count**: estimate how many source patches support each merged edit.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "success"
    }
  ]
}
""",
    "merge_final": """\
You are a skill-edit coordinator performing the FINAL merge. You receive two
pre-merged patch groups, in priority order:
1. **Failure-driven patches** (corrective)
2. **Success-driven patches** (reinforcement, lowest priority)

Merge guidelines:
1. **FAILURE TAKES PRIORITY OVER SUCCESS**: failure-driven edits should be
   preserved unless they directly conflict with a well-supported success pattern.
2. **Deduplicate**: if a lower-priority edit covers the same point as a higher-
   priority one, keep the higher-priority version.
3. **Preserve insights**: include failure and success edits that cover patterns
   NOT addressed by higher-priority edits.
4. **Higher-level merges represent broader consensus**: edits that survived
   previous merge rounds (higher level) should be given priority.
5. **Carry forward support_count and source_type for each edit.**

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary of priority decisions>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "failure|success"
    }
  ]
}
""",
    "hierarchical_merge_skills": """\
You are the hierarchical merger for Skill Forest. Your task is not to summarize
two texts, but to merge two branches into one complete, directly executable
global Skill without losing useful experience.

**I. Merge objective**

You will receive two Skills independently explored on different tasks, together
with their scores on the same validation set. Merge them into a complete Skill
with a clear structure and no redundant repetition.

Follow these principles:

- Combine complementary rules or experience from both Skills.
- For conflicting rules or experience, use those from the Skill with the higher validation score.
- Combine similar or partially consistent rules into a complete rule without
losing details. For example, if rule 1 applies to apples and oranges and rule 2
applies to bananas, do not generalize them to "fruit"; explicitly retain
apples, oranges, and bananas.
- Fully preserve both Skills' task scopes, output formats, tool constraints,
safety constraints, verification requirements, stopping conditions, and other
binding contracts. Except when resolving conflicts, do not omit, delete, or
narrow any contract.

**II. Input format**

You will receive three components:

- `Skill Candidate 1`: the first complete Skill and its validation score;
- `Skill Candidate 2`: the second complete Skill and its validation score;
- `Diff`: a unified diff between the complete Skills to help locate textual differences.

The diff is only an auxiliary view and cannot replace the two complete Skills:

- A line starting with `-` appears only in Skill Candidate 1; this does not mean it should be deleted.
- A line starting with `+` appears only in Skill Candidate 2; this does not mean it should be preferred.
- A line starting with a space provides shared context near a difference.
- The left and right sides do not indicate quality, authority, or ancestry.
- When an entire rule is rewritten, the diff may show large deletions and
additions. Return to the complete Skills and assess equivalence through their
trigger conditions, specific actions, scope, and verification procedures.

Use validation scores only to resolve genuine conflicts. Prefer the higher-scoring
side only when both sides provide valid numerical scores from the same validation
set. If either side is marked "unverified", or the scores are not directly
comparable, do not delete either side's experience on the basis of scores;
preserve both sets of rules where possible and distinguish them with explicit
applicability conditions.

**III. Merge criteria**

**1. Inventory each item before writing.**
First inspect each Skill's task scope, output format, tool constraints, safety
constraints, procedures, verification requirements, stopping conditions, and
failure-recovery strategies. For each specific piece of experience, determine:

- which tasks or conditions trigger it;
- which objects, locations, tools, actions, or states it concerns;
- which actions are required and in what order;
- which restrictions, exceptions, attempt limits, or fallbacks it includes;
- how it determines success, failure, or task completion.

Do not judge whether two rules are identical solely from their titles or topics.

**2. Combine complementary experience.**
Compatible experience present in only one Skill must enter the final Skill.
Do not remove a rule because it appears in only one candidate, has a narrow scope,
is highly specific, or would make the final Skill longer.

**3. Combine similar or partially consistent experience.**
Deduplicate only when two rules require the same actions under the same
conditions and produce the same effects.

- If their trigger objects differ, list every specific object; do not replace them with broad categories.
- If their applicability conditions differ, retain each condition.
- If one rule adds actions, ordering, restrictions, exceptions, or verification steps, retain these details.
- If one rule states a principle and the other provides executable steps, retain both; do not keep only the abstract principle.
- Do not compress multiple specific rules into a weaker summary that cannot be executed directly.

**4. Resolve conflicting experience.**
First determine whether the rules genuinely conflict. If they apply to different
tasks, objects, or environment states, retain both with explicit applicability
conditions rather than choosing one.
Only rules requiring incompatible actions under the same conditions genuinely conflict:

- If both sides have valid scores from the same validation set, adopt the higher-scoring candidate's rule.
- If scores are tied, missing, unverified, or incomparable, prefer the more precise, executable rule with explicit conditions.
- If a safe choice remains impossible, retain both alternatives with explicit guards; do not invent a compromise absent from the inputs.
- Do not prefer a candidate merely because it appears first or occupies a particular side of the diff.

**5. Prevent weakening or loss of detail.**
During merging, do not:

- replace precise object, container, location, tool, action, or state names with generic terms that change trigger conditions;
- delete constraint words such as "must", "must not", "only if", "exactly", "before", or "after";
- delete task types, preconditions, guards, exceptions, search priors, fallbacks, anti-loop rules, or recovery from acting on the wrong object;
- change the order of procedural steps;
- delete quantity requirements, retry limits, success verification, completion criteria, or stopping conditions;
- replace specific actions with non-executable advice such as "act carefully", "process efficiently", or "check the result";
- reduce applicability or operational information merely to shorten the text;
- replace an original rule with one that appears more general but actually covers fewer cases.

**6. Check backward after completion.**
After writing the final Skill, check backward from each input Skill:

- Has every piece of complementary experience entered the final Skill?
- Have all objects, conditions, actions, ordering, exceptions, and verification details of similar rules been preserved?
- Has every genuine conflict been handled according to comparable validation scores?
- Are both Skills' task scopes, output formats, tool constraints, safety constraints, verification requirements, and stopping conditions complete?
- Has every substantive difference in the diff been retained, merged, or resolved according to the conflict rules?

If any input experience lacks an equivalent rule with no narrower applicability
in the final Skill, revise the final Skill before returning it.

**IV. Output requirements**

- Output only one complete Markdown Skill file.
- The first non-whitespace character must be `#`.
- Do not output JSON, code fences, merge reports, audit tables, explanations, prefixes, or suffixes.
- Do not mention Skill Candidate 1, Skill Candidate 2, the diff, validation scores, or the merging process in the final Skill.
- The final Skill must be usable independently of the current input.
- You may reorganize sections and remove genuine repetition, but must not sacrifice the operational semantics of any non-conflicting experience.
- Consider concision only after fully preserving trigger conditions, actions, ordering, restrictions, exceptions, recovery, and verification steps.
""",
}


def load_prompt(name: str) -> str:
    """Return a generic algorithm prompt.

    Args:
        name: Prompt stem.

    Returns:
        The bundled prompt text, preserving its source whitespace.
    """
    return _PROMPTS[name]


def format_trajectory(conversation: list[dict[str, JsonValue]]) -> str:
    """Render public trajectory records for evidence review.

    Args:
        conversation: Public agent messages, tool calls or action/feedback steps.

    Returns:
        The untruncated analyst-readable conversation.
    """

    return json.dumps(conversation, ensure_ascii=False, indent=2)


def reflection_user(skill: str, trajectories: str, *, count: int, success: bool, edit_budget: int) -> str:
    """Build a reference-format analyst request.

    Args:
        skill: Parent Skill text.
        trajectories: Rendered training evidence.
        count: Number of training examples.
        success: Shared outcome of this minibatch.
        edit_budget: Maximum requested edits.

    Returns:
        Complete Markdown optimizer request.
    """
    outcome = "Successful" if success else "Failed"
    return (
        f"## Current Skill\n{skill}\n\n"
        f"## Edits Budget\nProduce at most L={edit_budget} edits.\n\n"
        f"## {outcome} Trajectories ({count} total)\n{trajectories}"
    )


def patch_merge_user(skill: str, patches: list[dict[str, JsonValue]], *, level: int) -> str:
    """Serialize same-outcome patches for reduction.

    Args:
        skill: Unchanged parent Skill text.
        patches: Nonempty patch evidence.
        level: One-based reduction level.

    Returns:
        A JSON request for a single reduced patch.
    """
    return json.dumps({"skill": skill, "merge_level": level, "patches": patches}, ensure_ascii=False)


def final_patch_user(skill: str, groups: list[tuple[str, dict[str, JsonValue]]]) -> str:
    """Serialize outcome groups for conflict resolution.

    Args:
        skill: Unchanged parent Skill text.
        groups: Outcome names paired with their reduced patches.

    Returns:
        A JSON request preserving each group's evidence origin.
    """
    return json.dumps(
        {"skill": skill, "groups": [{"outcome": name, "patch": patch} for name, patch in groups]},
        ensure_ascii=False,
    )


def skill_merge_user(left: str, right: str, left_score: float, right_score: float) -> str:
    """Render two complete candidates with comparable scores and a text diff.

    Args:
        left: Complete left Skill.
        right: Complete right Skill.
        left_score: Left full-training Hard score.
        right_score: Right score on the same complete training pool.

    Returns:
        Two-candidate synthesis request.
    """
    lines = list(
        difflib.unified_diff(
            left.strip().splitlines(),
            right.strip().splitlines(),
            fromfile="skill_candidate_1",
            tofile="skill_candidate_2",
            n=3,
            lineterm="",
        )
    )
    diff = "\n".join(lines) + "\n" if lines else "The Skill texts have no line differences.\n"
    return (
        f"## Skill Candidate 1\n\nValidation Score: {left_score:.6g}\n\n{left.strip()}\n\n"
        f"## Skill Candidate 2\n\nValidation Score: {right_score:.6g}\n\n{right.strip()}\n\n"
        f"## Diff\n\n```diff\n{diff}```\n"
    )
