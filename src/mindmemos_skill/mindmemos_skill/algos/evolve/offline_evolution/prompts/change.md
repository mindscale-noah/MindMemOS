You improve agent skills from accepted historical observations and comparisons. The tasks cannot be re-run and have no external scores. Choose changes proportionate to the evidence; preserving the skills unchanged is valid. Use the recorded evidence to guide judgment rather than treating its labels as proof of effectiveness.

## Inputs and scope

- `{report_md}` and `{report_json}`: the collection report. Cite finding ids only from an accepted report. Read complete quotes and related case members where needed; inspect the original inputs when a relationship is unclear.
- `{inputs}/trajectories/` and `{inputs}/skills/`: recorded executions and original resources. Historical environment facts come from these inputs, not the current environment.
- `{edited}`: the editable skill collection. Write edits and `{workspace}/changes.json` only inside `{workspace}`.

An existing skill may change only when cited findings concern it. Descriptions, instructions and bundled code may all be changed regardless of their issue label. A new skill needs a cited uncovered finding and a distinct reusable purpose absent from existing resources. Deleting an entire skill is not allowed. {protected}

Implement an appropriate correction using the cited observations and original resources. Code may be adapted or newly implemented to address the observed need; explain its basis and meaningful adaptations in the rationale. No line-matching percentage or failure/code/success group is required. Keep uncertain conclusions bounded, and leave a finding under `dropped` when it supports no useful edit. A case groups excerpts and establishes no success verdict.

Name each new resource where the skill uses it, and make its execution or reference purpose clear. Scripts must not use the network or delete directories recursively. Cite only accepted findings and actual original-skill quotes. Address each finding with an issue by citing it or explaining its omission under `dropped`; addressing a case member covers the group. Citations of claims for the same file are combined for that file, and shared-file edits pass or revert together.

{lenses}

## Deliverable: `{workspace}/changes.json`

```json
{{
  "changes": [
    {{
      "id": "C1",
      "files": ["<skill>/<path>"],
      "findings": ["F1"],
      "skill_quotes": [{{"path": "<skill>/<path>", "quote": "verbatim original text"}}],
      "rationale": "The observation, its relevant source and how the implementation addresses it."
    }}
  ],
  "dropped": [{{"findings": ["F2"], "reason": "Why no edit is supported or needed."}}],
  "acknowledged": []
}}
```

`files` lists every added, changed or removed file relative to `{edited}`; unlisted changes are reverted. Prefer one claim per file or tightly related group of files. `skill_quotes` is optional; when used, its quotations must match original resources. Change ids contain only letters, digits, `.`, `_` and `-`.

Run `{check_command}` before finishing and address its feedback. For an actionable warning, revise the edit or add `{{"warning": "<key>", "reason": "..."}}` to `acknowledged`. Preflight checks copies; the Host separately accepts or reverts files, without validating the change's task-level effectiveness.
