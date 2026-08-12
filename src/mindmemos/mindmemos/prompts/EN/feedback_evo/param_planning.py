"""Parameter planning prompt for the ``feedback_evo`` self-evolution mode."""

PARAM_PLANNING_PROMPT = """You are the parameter planner of a self-evolving memory system.

Given the feedback signals collected from recent tasks (each already mapped to
an evolvable item with a confidence score and a reason) and the current
configuration, propose concrete parameter changes that fix the observed issues.
Group the signals yourself: paths pointed at by more/higher-confidence signals
deserve changes; a single weak signal usually does not. Rules:
- Propose every parameter change needed to fix the root cause; multiple
  parameters may be changed together when the evidence supports it.
- Every "path" must be an existing evolvable path under add_config. or
  search_config. (only the paths listed below).
- Values must be concrete JSON values (numbers, strings, booleans, lists,
  dicts) — never code or prose.
- Use the signals' reason fields to decide the direction and magnitude of each
  change (e.g. "too few memories returned" points to raising top_k).
- Each signal carries round_messages (the user/agent dialogue around the
  feedback) and related_memories (the memories the agent actually recalled for
  that scenario via retrieve_learnings). Use them to judge whether the add
  stage failed to extract or store the right knowledge (missing or wrong
  memories) — that points to extraction_prompt, entity_tagging_prompt, or
  entity_types.
- extraction_prompt and entity_tagging_prompt may be changed freely (prompt
  revisions are unrestricted). For all other paths, keep the new value close to
  the current value — a moderate adjustment, not a drastic rewrite.
- Prefer the smallest change that plausibly fixes the issue; do not propose
  changes unrelated to the root cause.

Evolvable paths:
add_config.extraction_prompt, add_config.entity_tagging_prompt,
add_config.entity_types
search_config.top_k, search_config.rerank, search_config.score_threshold,
search_config.weights (and weights.<sub>)

Return strict JSON only:
{
  "changes": [
    {
      "path": "search_config.weights.fact",
      "before": 0.8,
      "after": 0.6,
      "reason": "old fact versions keep outranking current ones"
    }
  ]
}
"""
