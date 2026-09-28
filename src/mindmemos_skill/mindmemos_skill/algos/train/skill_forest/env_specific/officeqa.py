"""Task-specific reflection prompt resources for officeqa."""

from .....envs.builtin_envs.officeqa.typing import OfficeQATask
from ..env_adapter import SkillForestEnvAdapter
from ..typing import SkillForestTrajectoryContext
from .common import SkillForestReferenceEnvAdapter, recorded_context, render_record, tool_conversation
from .typing import ReflectionFeedback


@SkillForestEnvAdapter.register("officeqa")
class SkillForestOfficeQAEnvAdapter(SkillForestReferenceEnvAdapter):
    """Adapt Skill-Forest training evidence to OfficeQA."""

    ANALYST_ERROR_PROMPT = """\
You are an expert failure-analysis agent for OfficeQA document-retrieval question answering tasks.

You will be given MULTIPLE failed OfficeQA trajectories from a single minibatch and the current skill document. The trajectories may include local document tool calls such as file search, grep, and partial reads.

Your job is to identify COMMON failure patterns across the batch and propose concise skill edits.

## Failure Type Categories
- retrieval_miss: the agent searched the wrong file or failed to narrow to the right file
- evidence_miss: the agent read documents but missed the decisive evidence span
- operand_error: the agent extracted the wrong value or the wrong operands
- calculation_error: the agent identified the right evidence but computed the result incorrectly
- answer_format: the agent reached the right result but formatted it wrong
- other: none of the above

## Rules
- Focus on patterns common across multiple trajectories.
- Prefer general retrieval and evidence-grounding rules over task-specific hacks.
- Only patch gaps in the skill; do not duplicate rules already present.
- Do not hardcode file names, years, or question-specific constants unless the pattern truly requires a reusable retrieval heuristic.

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
You are an expert success-pattern analyst for OfficeQA document-retrieval question answering tasks.

You will be given MULTIPLE successful OfficeQA trajectories from a single minibatch and the current skill document. Your job is to identify common retrieval, evidence-selection, and numeric-grounding behaviors worth encoding in the skill.

## Rules
- Focus on patterns shared across multiple successful trajectories.
- Prefer reusable retrieval and extraction discipline over question-specific tips.
- Reinforce compact, high-value behaviors such as narrowing files early, reading only the relevant span, building a clean operand ledger, and copying the final answer from checked evidence.
- Only propose patches for patterns not already captured in the current skill.

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
        if not isinstance(context.task, OfficeQATask):
            raise TypeError("OfficeQA reflection requires OfficeQATask")
        system, user = recorded_context(result)
        feedback = ReflectionFeedback.model_validate(result.reward.model_dump())
        events = tool_conversation(result.trajectory[1:], assistant_role="agent")
        return render_record(
            index,
            context.task.id,
            context.task.instruction,
            context.task.task_type,
            feedback.fail_reason or result.reward.detail,
            len(events),
            events,
            system=system,
            user=user,
        )
