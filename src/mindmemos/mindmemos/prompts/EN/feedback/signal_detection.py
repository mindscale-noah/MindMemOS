SIGNAL_DETECTION_PROMPT = """Filter and classify implicit feedback signals in session conversation rounds.

Input contains compact rounds with only the original user query and the final assistant reply.

Return JSON only, shaped as:
{
  "signals": [
    {
      "round_index": 0,
      "category": "task_temporary" | "scenario_specific" | "long_term",
      "reason": "..."
    }
  ]
}

Detect every round where the user expresses actionable feedback, including negative feedback, correction,
dissatisfaction, requested revision, changed preference, future behavior instruction, or durable working rule.
Do not omit feedback simply because the assistant successfully complied in the same round.
Do not include purely positive feedback with no requested change, preference, correction, or future instruction.
Use the zero-based round_index from the input.

One round may contain multiple independent feedback signals. Return one signal object for each independent signal,
even when multiple signal objects share the same round_index. Do not merge multiple feedback points into one
conservative category.

For every selected signal, first infer the concrete reason behind that specific feedback point from the surrounding
conversation:
- What did the user object to or correct?
- Why did the user want the change?
- Is that reason caused by a property of only the current task/artifact, by a reusable class of scenarios with the same feature, or by an unconditional preference/fact/rule?

Classify each signal based on that inferred reason:
- task_temporary: The reason can be judged from context as having no reusable value — the feedback is tied to a one-off state, artifact, or execution problem that cannot be summarized or generalized. Use this category only when the current scenario cannot be abstracted into a reusable rule or pattern. Examples: "this command failed" because this specific run had a transient error; "undo the previous change" because the user wants to revert a single mistaken edit with no broader implication.
- scenario_specific: The reason depends on a reusable scenario feature shared by a class of future tasks. Use this category only when the conversation explicitly shows that the feedback is scoped to a constrained scenario — the user's words must clearly limit applicability to a specific task type, project, or condition. If no such explicit scope is stated, classify as long_term instead. When used, the memory must carry an explicit scenario precondition. Examples: "for papers whose method implementation is simple, keep the method summary short" (the user explicitly limits to simple-method papers); "when reviewing PRs, be stricter about tests" (the user explicitly limits to PR review context).
- long_term: The reason has no task/scenario precondition and is a stable user preference, objective rule, fixed fact, correction to user knowledge, or generally applicable behavior. The memory should be written as a general statement without scenario preconditions. Examples: "I use uv, not conda"; "my team uses Beijing time"; "backend API request parameters should never use `any` placeholders"; "write future code comments in Chinese"; "I prefer detailed answers".

The category must follow the reason, not just the surface form of the user's wording. A correction can be long_term if the reason is a general rule; a request can be scenario_specific if the reason is a reusable scenario feature; a strong complaint can still be task_temporary if it only concerns the current implementation.
In the reason field, explicitly state the inferred reason and why that reason maps to the chosen category.
The reason may cite or paraphrase the relevant signal briefly for explainability.
The reason must describe only this signal. If the same round contains another independent feedback point with a
different category or reason, return it as a separate signal with the same round_index.

Important boundary cases:
- If the user asks a follow-up for more detail because the previous answer was unclear, incomplete, too shallow, or missed a specific concept, include it as a negative implicit feedback signal. If the follow-up reveals a gap only relevant to the current artifact, classify it as task_temporary. If the gap reveals a reusable knowledge or style need, classify it as scenario_specific.
- If the user edits or rejects the assistant's output because of a reusable property scoped to a constrained scenario, classify it as scenario_specific, not task_temporary. The named artifact is evidence, but the reusable scenario should come from the property/reason and be explicitly bounded. Example: "this paper's method is simple, no need to expand" means the scenario is papers with simple method implementation, not only that named paper.
- If the user states a general preference without limiting it to a specific artifact/task/scenario, classify it as long_term. Example: "I always prefer concise method sections".
- If the user says future outputs should follow a style without limiting it to a task/scenario, classify it as long_term. Example: "use Chinese comments in code from now on" is a long-term preference, not just the current code block.
- If the user corrects a general coding rule without limiting it to the current implementation, classify it as long_term. Example: "backend interface parameter types must not use `any`; define strict types" is a general coding rule.

If uncertain between task_temporary and scenario_specific, prefer scenario_specific — as long as the current scenario can be summarized or generalized into a reusable condition, it belongs in scenario_specific. Reserve task_temporary only when the context makes it clear that the feedback has no reusable value and cannot be abstracted.
If uncertain between scenario_specific and long_term, prefer long_term unless the conversation explicitly shows the feedback is scoped to a constrained scenario, task type, or project. Do not infer a scenario scope that the user did not explicitly state. When no explicit scope is stated, the feedback is long_term.
Do not output a feedback field.
"""
