---
name: livemath
description: Solving multiple-choice mathematical (MCQ) problems. Use when a task presents a math question with a set of candidate answers and the result should be the single best option — the skill guides choosing among options and producing only the option label as the answer.
---

# Live Mathematical MCQ Heuristics

## Option Comparison
- Compare all options before committing. The correct choice is often the strongest statement justified by the question, while nearby distractors are weaker, overstrong, or miss an equality case.
- Track exact quantifiers such as "there exists", "for every", "if and only if", and "exactly when".

## Theorem-Level Precision
- Check whether an option weakens the conclusion by dropping a characterization, equality clause, or full equivalence.
- Check whether an option overstates the theorem by upgrading regularity, removing scale restrictions, or changing an existential statement into a universal one.

## Hypotheses
- Verify the hypotheses and domain carefully. Distractors often keep the theorem shape but alter the required assumptions.
- Pay close attention to equality cases, extremal conditions, and whether a result applies to the full family or only a restricted subfamily.

## Final Answer
- Output the final answer as the single option label only.
