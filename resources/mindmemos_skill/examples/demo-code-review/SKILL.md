---
name: demo-code-review
description: Review a supplied code change for concrete correctness bugs, with reproducible evidence and focused regression checks.
---

# Focused Code Review

Read the supplied change and the directly affected callers. Report only actionable issues introduced by the change. Separate confirmed failures from hypotheses; state which checks were actually run.

For each finding, identify the triggering input, the observed or predicted incorrect result, and the affected file and line. Prioritize data loss, incorrect behavior, and broken contracts over style preferences.

Use [the report template](assets/review-template.md) when delivering multiple findings. Read [the sample change](references/sample-change.diff) for a small practice review.

Before proposing a correction, check empty input, boundary values, and the calling contract. Suggest one focused regression check per confirmed defect. If no defect is supported, report that result and name the remaining unverified boundary.

Review by default; modify files only when the user asks for a fix. Match the user's language in the report.
