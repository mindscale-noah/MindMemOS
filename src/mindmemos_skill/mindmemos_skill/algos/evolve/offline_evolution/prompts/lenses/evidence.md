## Comparison signals

- Within a run: changes to an operation or its inputs followed by visible changes in tool results, state or artifacts. Examine what changed and what remained known, without requiring a particular failure/recovery sequence.
- Across runs: differences in approach for the same task or similar operations, alongside recorded inputs, initial conditions and visible results. Missing conditions limit the comparison; do not invent them to complete a pair.
- Across sources: skill instructions, bundled interfaces and actual execution disagree or reveal missing execution information. Original resources can support a correction without a failed run.
- Against requirements: visible operations or artifacts can be compared with explicit task or user wording. Record the scope actually checked; commands, agent claims and absent errors alone do not establish the full result.

Follow the available signal rather than filling every category. Pair tool calls with results by identifier where recorded, distinguish data from commands, and preserve unknowns. Repetition or fewer steps alone does not establish correctness or generality.
