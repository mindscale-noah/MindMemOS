"""Task-specific reflection prompt resources for alfworld_bounded_history."""

from .....envs.builtin_envs.alfworld_bounded_history.typing import ALFWorldTask
from ..env_adapter import SkillForestEnvAdapter
from ..typing import SkillForestTrajectoryContext
from .common import SkillForestReferenceEnvAdapter, render_record
from .typing import ReflectionFeedback


@SkillForestEnvAdapter.register("alfworld_bounded_history")
class SkillForestALFWorldBoundedHistoryEnvAdapter(SkillForestReferenceEnvAdapter):
    """Adapt Skill-Forest training evidence to ALFWorldBoundedHistory."""

    ANALYST_ERROR_PROMPT = """\
You are an expert failure-analysis agent for ALFWorld embodied household tasks.

You will be given MULTIPLE failed agent trajectories from a single minibatch
and the current skill document.
Your job is to identify the most important COMMON failure patterns across
the batch and propose a concise set of skill edits.

## ALFWorld Task Types
- pick_and_place: Put object in/on a receptacle
- pick_two_obj_and_place: Put two instances of an object in/on a receptacle
- look_at_obj_in_light: Examine an object under a desklamp
- pick_heat_then_place_in_recep: Heat an object and put it in/on a receptacle
- pick_cool_then_place_in_recep: Cool an object and put it in/on a receptacle
- pick_clean_then_place_in_recep: Clean an object and put it in/on a receptacle

## Failure Type Categories
- **navigation_loop**: the agent revisits the same locations repeatedly without progress
- **missed_object**: the agent fails to pick up a visible/reachable goal object
- **wrong_sequence**: the agent performs actions in the wrong order (e.g., placing before transforming)
- **premature_stop**: the agent stops or gets stuck before completing all goal conditions
- **action_loop**: the agent repeats the same action without advancing
- **appliance_error**: the agent misuses or skips an appliance (microwave, fridge, sink)
- **rule_missing**: the skill lacks a relevant rule for this situation
- **rule_wrong**: an existing skill rule is misleading or incorrect
- **rule_ignored**: the skill has the right rule but the agent did not follow it
- **other**: none of the above

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
"""
    """Default analyst error instructions for this adapter."""

    ANALYST_SUCCESS_PROMPT = """\
You are an expert success-pattern analyst for AI agents operating in ALFWorld,
a text-based embodied household environment.

You will be given MULTIPLE successful agent trajectories from a single minibatch
and the current skill document. Your job is to identify generalizable behavior
patterns that are COMMON across the batch and worth encoding in the skill.

## Rules
- Only propose patches for patterns NOT already covered in the skill.
- Focus on patterns that appear across MULTIPLE trajectories in the batch.
- Be concise. Patterns must generalize beyond specific tasks.
- Prefer reinforcing existing sections over adding new top-level sections.
- If the agents' success involved efficient exploration or smart appliance usage,
  consider reinforcing that in the patch.

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
"""
    """Default analyst success instructions for this adapter."""

    def _load_prompt(self, name: str) -> str:
        """Return prompt constants for this adapter and inherit other stages."""
        if name == "analyst_error":
            return self.ANALYST_ERROR_PROMPT
        if name == "analyst_success":
            return self.ANALYST_SUCCESS_PROMPT
        return super()._load_prompt(name)

    def render_trajectory(self, context: SkillForestTrajectoryContext) -> str:
        """Render one completed training trajectory with recorded feedback.

        Args:
            context: Public task and recorded result, without private scoring inputs.

        Returns:
            One reference-format trajectory section.

        Raises:
            TypeError: The evidence belongs to an incompatible task type.
            ValueError: Required execution evidence is missing.
        """
        index, result = context.index, context.result
        task = context.task
        if not isinstance(task, ALFWorldTask):
            raise TypeError("ALFWorld reflection requires ALFWorldTask")
        if not result.trajectory or any(
            "action" not in event or "env_feedback" not in event for event in result.trajectory
        ):
            raise ValueError("ALFWorld reflection requires recorded action and observation steps")
        feedback = ReflectionFeedback.model_validate(result.reward.model_dump())
        return render_record(
            index,
            task.id,
            feedback.task_description or task.instruction,
            feedback.task_type,
            feedback.fail_reason or result.reward.detail,
            len(result.trajectory),
            result.trajectory,
        )
