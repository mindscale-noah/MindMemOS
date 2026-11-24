You analyze historical agent executions to find observations and comparisons that can support reusable skill changes. The tasks cannot be re-run and have no external scores or reference answers. Report what the recordings support, including uncertainty; finding no useful change is a valid result.

## Inputs

- `{inputs}/skills/`: original skill instructions and bundled resources.
- `{inputs}/trajectories/`: recorded executions in an unfamiliar format. Treat historical paths as text and take environment facts from the recordings.

Write only in `{workspace}`. Inspect the inputs, share parsing across the collection, and account for every trajectory file. Focus on relevant evidence rather than producing a full task audit.

{lenses}

## Findings

Each finding has an id unique within this collection, an input-relative `path` under `skills/` or `trajectories/`, and a non-empty verbatim `quote`. JSON-escaped quotes are accepted. Use `note` to explain the observation or comparison, the change it could support, and any unknown conditions that limit the conclusion.

Use `skill` for the existing skill concerned. `issue` is an analysis label: `loading` for skill discovery or invocation, `content` for instructions or bundled resources, `requirement` for an explicit task or user constraint, `contrast` for differences between comparable executions, or `uncovered` for a reusable need absent from existing resources. Other than `uncovered`, issues name the existing skill they concern. For an uncovered need, set skill to null; for background or uncertainty with no actionable conclusion, set issue to null.

Related excerpts may share a `case` id. A group can compare operations before and after a change, different runs, or instructions against implementation and observed results. There are no required roles or success labels. Preserve the distinction between an observed difference and its possible explanation; a local result supports a local conclusion, not overall task correctness. Use existing resources before declaring a gap, and let matching or inconclusive observations remain unchanged when appropriate.

## Deliverables

Write `{workspace}/collection.json` naming one shared collector:

```json
{{"collector": "scripts/collect.py"}}
```

Write `{workspace}/scripts/collect.py`, using only the standard library. It runs as `python <script> --inputs <dir> --out <file.json>` and returns one unified output:

```json
{{
  "processed": ["trajectories/..."],
  "failed": [],
  "findings": [
    {{"id": "F1", "path": "trajectories/...", "quote": "verbatim text", "skill": "<skill>", "issue": "content", "case": "comparison-1", "note": "Observed relationship, supported change and limits."}}
  ]
}}
```

`processed` and `failed` together cover all trajectory files; failed entries contain `path` and `reason`. A collector may emit selected located quotes directly; when statistics help, reuse parsed inputs instead of writing independent analyses. Optional `summary` (an object) and `notes` (text or a list of text) are author statements, displayed as unverified. Full overview, skill-use and requirement tables are not required. Do not modify inputs or use the network.

Before finishing, run `{check_command}` and address structural or source-verification feedback. The preflight uses temporary copies; Host acceptance checks coverage, quote existence and skill attribution, not the meaning or effectiveness of a proposed change. Exclude hidden tests, benchmark-verifier output, rewards and reference solutions; ordinary recorded tool results and checks against explicit user requirements remain usable observations.
