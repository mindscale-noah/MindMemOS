---
name: demo-code-review
description: Review a supplied code change for concrete correctness bugs and explain the evidence for each finding.
---

# Focused Code Review

Read the supplied change and its affected callers. Report actionable issues introduced by the change.

For each finding, identify the triggering input, incorrect result, and affected file and line. Separate confirmed failures from hypotheses and state which checks were run.

Read [the sample change](references/sample-change.diff) for practice. Review by default; modify files only when asked. Match the user's language.
