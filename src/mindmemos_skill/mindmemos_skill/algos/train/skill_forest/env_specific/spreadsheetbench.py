"""Task-specific reflection prompt resources for spreadsheetbench."""

from .....envs.builtin_envs.spreadsheetbench.typing import SpreadsheetBenchTask
from ..env_adapter import SkillForestEnvAdapter
from ..typing import SkillForestTrajectoryContext
from .common import SkillForestReferenceEnvAdapter, recorded_context, render_record, tool_conversation
from .typing import ReflectionFeedback


@SkillForestEnvAdapter.register("spreadsheetbench")
class SkillForestSpreadsheetBenchEnvAdapter(SkillForestReferenceEnvAdapter):
    """Adapt Skill-Forest training evidence to SpreadsheetBench."""

    ANALYST_ERROR_PROMPT = """\
You are an expert failure-analysis agent for spreadsheet manipulation tasks.

You will be given MULTIPLE failed agent trajectories from a single minibatch
and the current skill document.
Your job is to identify the most important COMMON failure patterns across
the batch and propose a concise set of skill edits.

## Failure Type Categories
- **rule_missing**: the skill lacks a relevant rule for this type of task
- **rule_wrong**: an existing skill rule is misleading or incorrect
- **rule_ignored**: the skill has the right rule but the agent did not follow it
- **data_exploration**: the agent did not read enough data from the spreadsheet
- **code_error**: the agent's code has a bug unrelated to the skill
- **other**: none of the above

## Analysis Process
1. Read ALL failed trajectories in the minibatch.
2. Identify the most prevalent, systematic failure patterns across them.
3. For each pattern, classify its failure type.
4. Propose skill edits that address the COMMON patterns — not individual edge cases.
5. Edits must be generalizable; do not hardcode task-specific values
   (file paths, cell addresses, expected values).
6. Only patch gaps in the skill — do not duplicate existing content.
7. If the failure is because the agent did not read enough spreadsheet rows/columns
   to understand the data, propose a patch encouraging broader data exploration.

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
You are an expert success-pattern analyst for AI spreadsheet agents.

You will be given MULTIPLE successful agent trajectories from a single minibatch
and the current skill document. Your job is to identify generalizable behavior
patterns that are COMMON across the batch and worth encoding in the skill.

## Rules
- Only propose patches for patterns NOT already covered in the skill.
- Focus on patterns that appear across MULTIPLE trajectories in the batch.
- Be concise. Patterns must generalize beyond specific tasks.
- Prefer reinforcing existing sections over adding new top-level sections.
- If the agents' success involved reading enough data rows or using a smart
  exploration strategy, consider reinforcing that in the patch.

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
        if not isinstance(task, SpreadsheetBenchTask):
            raise TypeError("SpreadsheetBench reflection requires SpreadsheetBenchTask")
        system, user = recorded_context(result)
        feedback = ReflectionFeedback.model_validate(result.reward.model_dump())
        events = [
            event
            for event in result.trajectory
            if not (event.get("role") == "system" and event.get("content") == result.reward.detail)
        ]
        conversation = tool_conversation(events)
        conversation.append({"role": "system", "content": result.reward.detail})
        steps = feedback.n_turns
        if steps is None:
            steps = sum(event.get("role") == "assistant" for event in events)
        return render_record(
            index,
            task.id,
            task.instruction,
            task.instruction_type,
            feedback.fail_reason,
            steps,
            conversation,
            system=system,
            user=user,
        )
