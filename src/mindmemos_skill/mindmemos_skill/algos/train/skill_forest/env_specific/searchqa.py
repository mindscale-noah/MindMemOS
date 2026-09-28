"""Task-specific reflection prompt resources for searchqa."""

from .....envs.builtin_envs.searchqa.typing import SearchQAReward, SearchQATask
from ..env_adapter import SkillForestEnvAdapter
from ..typing import SkillForestTrajectoryContext
from .common import SkillForestReferenceEnvAdapter, recorded_context, render_record


@SkillForestEnvAdapter.register("searchqa")
class SkillForestSearchQAEnvAdapter(SkillForestReferenceEnvAdapter):
    """Adapt Skill-Forest training evidence to SearchQA."""

    ANALYST_ERROR_PROMPT = """\
You are an expert failure-analysis agent for question answering tasks.

You will be given MULTIPLE failed QA agent responses from a single minibatch
and the current skill document. Each trajectory includes the agent's response
and an evaluation result showing the predicted answer vs. the gold answer(s).

Your job is to identify the most important COMMON failure patterns across
the batch and propose a concise set of skill edits.

## Failure Type Categories
- **rule_missing**: the skill lacks a relevant rule for this type of question
- **rule_wrong**: an existing skill rule is misleading or incorrect
- **rule_ignored**: the skill has the right rule but the agent did not follow it
- **answer_format**: the agent found the right information but formatted it incorrectly
- **other**: none of the above

## Analysis Process
1. Read ALL failed trajectories in the minibatch.
2. Carefully compare each predicted answer against the gold answer(s) —
   understand exactly WHY the Exact Match failed.
3. Identify the most prevalent, systematic failure patterns across them.
4. For each pattern, classify its failure type.
5. Propose skill edits that address the COMMON patterns — not individual edge cases.
6. Edits must be generalizable; do not hardcode question-specific values.
7. Only patch gaps in the skill — do not duplicate existing content.

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
You are an expert success-pattern analyst for AI question answering agents.

You will be given MULTIPLE successful QA agent responses from a single minibatch
and the current skill document. Your job is to identify generalizable behavior
patterns that are COMMON across the batch and worth encoding in the skill.

## Rules
- Only propose patches for patterns NOT already covered in the skill.
- Focus on patterns that appear across MULTIPLE trajectories in the batch.
- Be concise. Patterns must generalize beyond specific questions.
- Prefer reinforcing existing sections over adding new top-level sections.
- If the agents' success involved a smart reading strategy or disambiguation
  approach, consider reinforcing that in the patch.

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
        if not isinstance(context.task, SearchQATask):
            raise TypeError("SearchQA reflection requires SearchQATask")
        system, user = recorded_context(result)
        reward = SearchQAReward.model_validate(result.reward.model_dump())
        answers = [{"content": event["content"]} for event in result.trajectory if event.get("role") == "assistant"]
        failure = (
            ""
            if reward.success
            else (f"EM=0: predicted '{reward.predicted_answer}' but expected {reward.gold_answers}")
        )
        return render_record(
            index,
            context.task.id,
            "",
            "",
            failure,
            len(answers),
            [*answers, {"role": "system", "content": reward.detail}],
            system=system,
            user=user,
        )
