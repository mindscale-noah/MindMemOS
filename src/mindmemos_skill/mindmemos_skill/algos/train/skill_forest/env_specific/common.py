"""Shared paper-protocol headers, conversations, and optimizer requests."""

import json

from ..env_adapter import SkillForestEnvAdapter


def format_conversation(events: list[dict]) -> str:
    """Render recorded events without truncating evidence.

    Args:
        events: Ordered messages, action steps, or paired tool calls.

    Returns:
        Analyst-readable conversation in the reference project's format.
    """
    lines = []
    for event in events:
        if event.get("type") == "tool_call":
            lines.extend([f"[action] {event.get('cmd') or ''}", f"[obs]    {event.get('obs') or ''}"])
        elif "action" in event and "env_feedback" in event:
            step = event.get("step", "?")
            if event.get("reasoning"):
                lines.append(f"[step {step} think] {event['reasoning']}")
            lines.extend([f"[step {step} action] {event['action']}", f"[step {step} obs]    {event['env_feedback']}"])
        else:
            role = "verification" if event.get("role") == "system" else event.get("role", "agent")
            lines.append(f"[{role}] {event.get('content') or ''}")
    return "\n".join(lines)


def render_record(index, task_id, instruction, task_type, failure, steps, events, *, system="", user="") -> str:
    """Render one reference-format trajectory with recorded target context.

    Args:
        index: One-based position in the minibatch.
        task_id: Dataset identity.
        instruction: Task description used in the reference header.
        task_type: Released task category, or empty when absent.
        failure: Evaluator or execution failure reason.
        steps: Environment-specific reference turn count.
        events: Ordered analyst conversation.
        system: Actual initial system prompt, when recorded.
        user: Actual initial user prompt, when recorded.

    Returns:
        Complete trajectory section.
    """
    header = f"### Trajectory {index} (id={task_id})\nTask: {instruction}\nTask type: {task_type}\n"
    if failure:
        header += f"Failure reason: {failure}\n"
    header += f"Steps: {steps}\n"
    if system:
        header += f"\n#### Target System Prompt\n{system}\n"
    if user:
        header += f"\n#### Target User Prompt\n{user}\n"
    return header + "\n" + format_conversation(events)


def recorded_context(result) -> tuple[str, str]:
    """Read actual initial inputs rather than reconstructing current prompts.

    Args:
        result: Completed rollout containing recorded agent inputs.

    Returns:
        Initial system and user messages.

    Raises:
        ValueError: Required input context or assistant output is absent.
    """
    events = result.trajectory
    if (
        len(events) < 3
        or events[0].get("role") != "system"
        or events[1].get("role") != "user"
        or not any(event.get("role") == "assistant" for event in events)
    ):
        raise ValueError("Reflection requires recorded system, user and assistant messages")
    return events[0]["content"], events[1]["content"]


def tool_conversation(events: list[dict], *, assistant_role: str = "assistant") -> list[dict]:
    """Pair native calls with observations while preserving call order and arguments.

    Args:
        events: Recorded conversation after the initial system prompt.
        assistant_role: Reference role label for model messages.

    Returns:
        Conversation with native tool exchanges represented as action/observation pairs.

    Raises:
        ValueError: A native tool observation cannot be matched to its call.
    """
    conversation, pending = [], {}
    for event in events:
        if event.get("role") == "assistant":
            conversation.append({"role": assistant_role, "content": event.get("content", "")})
            for call in event.get("tool_calls", []):
                function = call["function"]
                item = {"type": "tool_call", "cmd": f"{function['name']}({function['arguments']})", "obs": ""}
                if call["id"] in pending:
                    raise ValueError("Duplicate native tool call identity")
                pending[call["id"]] = item
                conversation.append(item)
        elif event.get("role") == "tool":
            identity = event.get("tool_call_id")
            if identity not in pending:
                raise ValueError("Unmatched native tool observation")
            pending.pop(identity)["obs"] = event.get("content", "")
        else:
            conversation.append(event)
    if pending:
        raise ValueError("Missing native tool observation")
    return conversation


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


def patch_merge_user(skill: str, patches: list[dict], *, level: int) -> str:
    """Render the historical same-outcome patch aggregation request.

    Args:
        skill: Parent Skill text.
        patches: Ordered patch payloads from reflection.
        level: One-based aggregation depth.

    Returns:
        Complete optimizer user message.
    """
    return (
        f"## Current Skill\n{skill}\n\n"
        f"## Patches to merge ({len(patches)} total, merge level {level})\n"
        f"{json.dumps(patches, ensure_ascii=False, indent=2)}"
    )


def final_patch_user(skill: str, groups: list[tuple[str, dict]]) -> str:
    """Render ordered failure/success patch groups for final aggregation.

    Args:
        skill: Parent Skill text.
        groups: Nonempty outcome groups in priority order.

    Returns:
        Complete optimizer user message.
    """
    descriptions = "\n".join(
        f"Group {index} ({name}, priority {index}): {len(patch['edits'])} edits"
        for index, (name, patch) in enumerate(groups, 1)
    )
    return (
        f"## Current Skill\n{skill}\n\n"
        f"## Pre-merged patch groups to combine (higher priority first)\n"
        f"Group order and priority:\n{descriptions}\n\n"
        f"{json.dumps([patch for name, patch in groups], ensure_ascii=False, indent=2)}"
    )


class SkillForestReferenceEnvAdapter(SkillForestEnvAdapter):
    """Share the released optimizer protocol across benchmark adapters."""

    MERGE_FAILURE_PROMPT = """\
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
7. **PROTECTED SECTION**: The skill may contain a section between
   <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
   Do NOT merge or produce any edits that target content within these markers.

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
"""
    """Default merge failure instructions for this adapter."""

    MERGE_SUCCESS_PROMPT = """\
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
5. **PROTECTED SECTION**: The skill may contain a section between
   <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
   Do NOT merge or produce any edits that target content within these markers.

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
"""
    """Default merge success instructions for this adapter."""

    MERGE_FINAL_PROMPT = """\
You are a skill-edit coordinator performing the FINAL merge. You receive three
pre-merged patch groups, in priority order:
1. **Contrast-driven patches** (failure→success evidence; the most reliable signal)
2. **Failure-driven patches** (corrective)
3. **Success-driven patches** (reinforcement, lowest priority)

Merge guidelines:
1. **CONTRAST PATCHES TAKE PRIORITY**: patches derived from a same-task
   failure→success contrast isolate the decisive reusable change, so preserve
   them unless they directly conflict with a stronger, overlapping edit.
2. **FAILURE TAKES PRIORITY OVER SUCCESS**: failure-driven edits should be
   preserved unless they directly conflict with a well-supported contrast or
   success pattern.
3. **Deduplicate**: if a lower-priority edit covers the same point as a higher-
   priority one, keep the higher-priority version.
4. **Preserve insights**: include failure and success edits that cover patterns
   NOT addressed by higher-priority edits.
5. **Higher-level merges represent broader consensus**: edits that survived
   previous merge rounds (higher level) should be given priority.
6. **Carry forward support_count and source_type for each edit.**
7. **PROTECTED SECTION**: The skill may contain a section between
   <!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
   Do NOT merge or produce any edits that target content within these markers.

Respond ONLY with a valid JSON object:
{
  "reasoning": "<summary of priority decisions>",
  "edits": [
    {
      "op": "append|insert_after|replace|delete",
      "target": "<if needed>",
      "content": "<markdown>",
      "support_count": <integer>,
      "source_type": "contrast|failure|success"
    }
  ]
}
"""
    """Default merge final instructions for this adapter."""

    def _load_prompt(self, name: str) -> str:
        """Return prompt constants for this adapter and inherit other stages."""
        if name == "merge_failure":
            return self.MERGE_FAILURE_PROMPT
        if name == "merge_success":
            return self.MERGE_SUCCESS_PROMPT
        if name == "merge_final":
            return self.MERGE_FINAL_PROMPT
        return super()._load_prompt(name)

    def build_reflection_messages(self, context) -> list[dict[str, str]]:
        """Build a released-format reflection request.

        Args:
            context: Skill, ordered training trajectories and edit budget.

        Returns:
            Complete reference optimizer messages.
        """
        return [
            {
                "role": "system",
                "content": self.prompts.analyst_success if context.success else self.prompts.analyst_error,
            },
            {
                "role": "user",
                "content": reflection_user(
                    context.skill,
                    "\n\n---\n\n".join(self.render_trajectory(item) for item in context.trajectories),
                    count=len(context.trajectories),
                    success=context.success,
                    edit_budget=context.edit_budget,
                ),
            },
        ]

    def build_patch_merge_messages(self, context) -> list[dict[str, str]]:
        """Build a released same-outcome reduction request.

        Args:
            context: Parent Skill, ordered patches and reduction level.

        Returns:
            Complete reference optimizer messages.
        """
        return [
            {
                "role": "system",
                "content": self.prompts.merge_success if context.success else self.prompts.merge_failure,
            },
            {
                "role": "user",
                "content": patch_merge_user(
                    context.skill,
                    [patch.model_dump(exclude_unset=True) for patch in context.patches],
                    level=context.level,
                ),
            },
        ]

    def build_final_patch_messages(self, context) -> list[dict[str, str]]:
        """Build a released failure-first reconciliation request.

        Args:
            context: Reduced patches and parent Skill.

        Returns:
            Complete reference optimizer messages.
        """
        return [
            {"role": "system", "content": self.prompts.merge_final},
            {
                "role": "user",
                "content": final_patch_user(
                    context.skill,
                    [
                        ("failure", context.failure.model_dump(exclude_unset=True)),
                        ("success", context.success.model_dump(exclude_unset=True)),
                    ],
                ),
            },
        ]
