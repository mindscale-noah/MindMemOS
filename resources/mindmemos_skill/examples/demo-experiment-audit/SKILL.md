---
name: demo-experiment-audit
description: Audit experiment results for missing runs, invalid attempts, split leakage, and mismatched metric aggregation before writing a comparison.
---

# Experiment Result Audit

Identify the expected run IDs, seeds, dataset split, and selection protocol from the supplied experiment description. Keep train, validation, held-out test, and oracle diagnostics separate.

Compute each reported rate from explicit numerator and denominator. When averaging across tasks, state whether the result is a macro-average of task rates or a micro-average over all examples. Do not silently replace one with the other.

Distinguish completed scored attempts, unfinished runs, and infrastructure failures. A provider failure is not evidence of model failure; report its count separately rather than inventing a score. Report any resulting coverage limitation.

Use [the sample results](references/results.csv) to practice the audit. Adapt [the audit template](assets/audit-template.md) to the actual evidence. Do not infer causality from a comparison where several settings changed together.
