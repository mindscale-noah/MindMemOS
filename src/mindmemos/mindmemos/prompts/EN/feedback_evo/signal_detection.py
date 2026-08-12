"""Evo signal detection prompt for the ``feedback_evo`` self-evolution mode."""

EVO_SIGNAL_DETECTION_PROMPT = """Map actionable feedback signals in session conversation rounds to the
memory-system evolvable items that should be optimized.

Input contains compact rounds with only the original user query and the final
assistant reply.

Every actionable feedback point (negative feedback, correction, dissatisfaction,
requested revision, changed preference, future behavior instruction, or durable
working rule) must be mapped to the single most relevant evolvable item that, if
optimized, would prevent the same issue in future tasks.

Evolvable items (valid values for "evolvable_path"):
- add_config.extraction_prompt: what/how memories are extracted and stored
- add_config.entity_tagging_prompt: how entity/property tags are assigned
- add_config.entity_types: the entity-type vocabulary used for tagging
- search_config.top_k: number of memories returned
- search_config.rerank: whether/how candidates are re-ranked
- search_config.score_threshold: minimum score for returned memories
- search_config.weights: weighting of memory types/entities in ranking

Return JSON only:
{
  "signals": [
    {
      "round_index": 0,
      "evolvable_path": "search_config.top_k",
      "confidence": 0.8,
      "reason": "..."
    }
  ]
}

Rules:
- Detect every round where the user expresses actionable feedback, including
  negative feedback, correction, dissatisfaction, requested revision, changed
  preference, future behavior instruction, or durable working rule. Do not omit
  feedback simply because the assistant successfully complied in the same round.
- For each independent feedback point, infer the evolvable item whose
  optimization would fix the root cause; prefer the most specific item.
- If the feedback is a one-off task/artifact/execution problem with no
  memory-system implication, omit it — do not emit a signal.
- If uncertain, prefer the evolvable item that plausibly prevents the issue over
  omitting the signal.
- One round may contain multiple independent feedback points; return one signal
  object per point, even when multiple signals share the same round_index.
- "evolvable_path" must be one of the values listed above.
- "confidence" must be a number between 0 and 1 expressing how strongly the
  feedback implicates the chosen evolvable item (1 = certain, 0 = guess).
- In the reason field, state the concrete root cause and why that item's
  optimization would prevent it.
- Do not output a feedback field.
"""
