"""LiveMath reflection rendering from the shared environment result contract."""

from .....envs.builtin_envs.livemath.typing import LiveMathReward, LiveMathTask
from ..env_adapter import SkillForestEnvAdapter
from ..typing import SkillForestTrajectoryContext
from .common import SkillForestReferenceEnvAdapter


@SkillForestEnvAdapter.register("livemath")
class SkillForestLiveMathEnvAdapter(SkillForestReferenceEnvAdapter):
    """Adapt Skill-Forest training evidence to LiveMath."""

    ANALYST_ERROR_PROMPT = """\
You are an expert failure-analysis agent for theorem-grounded mathematical multiple-choice questions.

You will be given MULTIPLE failed trajectories from a single minibatch and the current skill document.
Each trajectory includes the target's response and an evaluation result showing the predicted option
versus the correct option.

Your job is to identify COMMON reasoning failures across the batch and propose concise skill edits.

## Failure Type Categories
- **quantifier_miss**: the agent missed exact quantifiers, scope, or existence/uniqueness conditions
- **strength_mismatch**: the agent preferred a weaker or stronger statement than what was proved
- **condition_miss**: the agent ignored hypotheses, equality cases, or domain restrictions
- **option_confusion**: the agent confused similar answer choices or failed to compare them exactly
- **other**: none of the above

## Rules
1. Focus on patterns that recur across the minibatch.
2. Prefer edits that improve exact choice discrimination, not theorem-specific memorization.
3. Do not hardcode paper-specific content.
4. Only patch gaps not already covered by the skill.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number>,
  "failure_summary": [
    {"failure_type": "<type>", "count": <int>, "description": "<one-line>"}
  ],
  "patch": {
    "reasoning": "<why these edits address the common failures>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}
"""
    """Default analyst error instructions for this adapter."""

    ANALYST_SUCCESS_PROMPT = """\
You are an expert success-pattern analyst for theorem-grounded mathematical multiple-choice questions.

You will be given MULTIPLE successful trajectories from a minibatch and the current skill document.
Identify generalizable behavior patterns that are genuinely helping the agent choose the exact correct option.

## Rules
- Focus on broadly useful reasoning behaviors.
- Prefer patterns about exact comparison of options, quantifiers, and equality conditions.
- Do not add theorem-specific facts.
- "edits" may be empty if the skill already captures the useful patterns.

Respond ONLY with a valid JSON object:
{
  "batch_size": <number>,
  "success_patterns": ["<pattern 1>", "<pattern 2>"],
  "patch": {
    "reasoning": "<why these patterns matter>",
    "edits": [
      {"op": "append",       "content": "<markdown>"},
      {"op": "insert_after", "target": "<heading/text>", "content": "<markdown>"},
      {"op": "replace",      "target": "<old text>",     "content": "<new text>"},
      {"op": "delete",       "target": "<exact text to remove>"}
    ]
  }
}
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
        if not isinstance(task, LiveMathTask) or not isinstance(result.output, str):
            raise TypeError("LiveMath reflection requires LiveMathTask and text output")
        reward = LiveMathReward.model_validate(result.reward.model_dump(mode="json"))
        systems = [event["content"] for event in result.trajectory if event.get("role") == "system"]
        users = [event["content"] for event in result.trajectory if event.get("role") == "user"]
        answers = [event["content"] for event in result.trajectory if event.get("role") == "assistant"]
        if not systems or not users or not answers:
            raise ValueError("LiveMath reflection requires recorded system, user and assistant messages")
        header = f"### Trajectory {index} (id={task.id})\nTask: \nTask type: {task.task_type}\n"
        if not reward.success:
            header += (
                f"Failure reason: MCQ=0: predicted '{reward.predicted_label}' but expected '{reward.correct_label}'\n"
            )
        header += f"Steps: {len(answers)}\n"
        header += f"\n#### Target System Prompt\n{systems[0]}\n"
        header += f"\n#### Target User Prompt\n{users[0]}\n"
        verification = (
            f"[EVALUATION RESULT]\nQuestion: {task.instruction}\n"
            f"Predicted label: {reward.predicted_label!r}\n"
            f"Predicted text: {reward.predicted_text.strip()!r}\n"
            f"Correct label: {reward.correct_label!r}\n"
            f"Correct text: {reward.correct_text.strip()!r}\n"
            f"Exact Match: {float(reward.success)}"
        )
        conversation = "\n".join([*(f"[agent] {answer}" for answer in answers), f"[verification] {verification}"])
        return header + "\n" + conversation
