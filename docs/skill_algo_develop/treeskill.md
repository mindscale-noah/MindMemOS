# TreeSkill integration

TreeSkill adds two independent capabilities to `mindmemos_skill`:

- **TreeSkill Evolution** converts one `SKILL.md` into a Markdown heading tree,
  localizes trajectory-derived evidence to existing nodes, and fuses touched
  nodes from the leaves upward.
- **TreeSkill Routing** selects task-relevant subtree roots once per physical
  execution and injects only those subtrees plus their ancestor path context.

The implementation does not add another family runner. It is selected through
the existing `trace2skill` runner and experiment registry.

## Evolution flow

```text
full-Skill training trajectories
-> outcome-aware trajectory analysis
-> atomic evidence localization against the initial tree
-> global evidence grouping by target node
-> bottom-up node fusion against the current reparsed tree
-> SkillCandidate(SKILL.md, resources, metadata["treeskill"])
-> normal immutable MindMemOS Skill persistence
```

The initial operations are `update_node`, `create_child`, and `reject`.
Localization for all records uses one immutable initial tree. Fusion processes
target nodes bottom-up; every accepted edit is applied immediately and the
Markdown is reparsed before the next target. Existing node IDs remain stable
within the evolution run, while newly created nodes receive new IDs.

The algorithm is registered as `treeskill` with capability `optimize` under:

```text
src/mindmemos_skill/mindmemos_skill/algos/trace2skill/treeskill/
```

It accepts either pre-collected `Trajectory` values or tasks plus a
`TaskCollectionConfig`. It returns a `TreeSkillOutput`; the candidate remains
unpersisted until the normal MindMemOS management layer stores it.

## Persisted tree metadata

The complete executable instructions remain in `SKILL.md`. Structural data is
stored under `Skill.metadata["treeskill"]`:

```json
{
  "enabled": true,
  "schema_version": 1,
  "router": "llm_subtree_v1",
  "skill_content_hash": "...",
  "root_ids": ["001"],
  "nodes": [
    {
      "id": "001",
      "level": 1,
      "heading": "Workbook operations",
      "parent_id": null,
      "child_ids": ["002"],
      "ordinal": 0,
      "local_content_hash": "..."
    }
  ]
}
```

Routing validates this metadata against the current `SKILL.md`. Missing,
malformed, or stale metadata produces an auditable full-Skill fallback instead
of failing policy execution.

## Runtime routing flow

```text
AgentExecutionRequest(task, persisted Skill)
-> SkillRuntime.route(request)
-> one LLM subtree-selection call
-> validate IDs and collapse parent/child redundancy
-> render selected full subtrees in source order
-> add ancestor local content as path context
-> ephemeral system-prompt injection
-> bind the original persisted Skill version to the trajectory
```

Use `SkillInjectionMode.TREE_ROUTED_SYSTEM_PROMPT` to activate routing. Routed
content is ephemeral: it never creates a fake persisted Skill version. The
trajectory stores selected IDs, full and routed character counts, context
saving, and fallback state under `metadata["treeskill_routing"]`.

A valid empty selection injects no Skill. Invalid JSON, unknown IDs, model
errors, and stale tree metadata inject the full Skill. Non-Markdown resources
are materialized beside the routed `SKILL.md` and remain available to the
policy.

Generic ReAct execution uses the query-aware runtime directly. SpreadsheetBench
has an explicit bridge because that environment owns its ReAct conversation
loop. In routed mode it does not create or expose the legacy `skill` tool; in
all other modes its previous tool-injection behavior remains unchanged.

## Spreadsheet formula recalculation

The TreeSkill SpreadsheetBench entrypoint enables transactional formula
recalculation for both full-Skill trajectory collection and routed held-out
evaluation. The capability belongs to the environment rather than to a Skill,
so routing cannot accidentally remove it and both conditions receive the same
policy-visible command.

When enabled, the environment removes its legacy literal-values-only guidance
and appends the canonical recalculation instruction to the task prompt. If the
policy creates or changes formulas, it invokes the packaged helper before
completion. The helper:

1. inspects `output.xlsx` and returns `not_needed` without launching
   LibreOffice when no formulas exist;
2. recalculates a temporary workbook through a private headless LibreOffice
   profile;
3. verifies that recalculation preserved the formula count and records cached
   values and formula errors;
4. atomically replaces `output.xlsx` only after validation; and
5. writes `.tree_only_recalc_status.json` beside the workbook.

The experiment performs a real formula preflight before its first model call.
Configure LibreOffice using executable paths in the environment:

```bash
export TREE_ONLY_SOFFICE_PATH=/path/to/libreoffice/program/soffice
export TREE_ONLY_LIBREOFFICE_PYTHON=/path/to/libreoffice/program/python
```

System installations under the standard LibreOffice paths are also detected.
The official cached-cell-value evaluator remains unchanged. Generic MindMemOS
SpreadsheetBench experiments retain their existing behavior unless
`transactional_recalculation` is explicitly enabled in the environment config.

## Run the integrated example

Store endpoint credentials in `.skill.env`, then validate the resolved command:

```bash
UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache \
scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/treeskill/spreadsheetbench/default.yaml \
  --dry-run
```

Run the same command without `--dry-run` to collect training trajectories,
evolve the Skill, and evaluate the candidate with tree routing. Important
outputs are:

```text
<output_dir>/final_skill.md
<output_dir>/final_skill.json
<output_dir>/tree_metadata.json
<output_dir>/result.json
<output_dir>/summary.json
<output_dir>/test/results.jsonl
<output_dir>/test/summary.json
```

`final_skill.json` is the portable Skill package because it includes version
fields, resources, and TreeSkill metadata. `test/summary.json` includes routed
and full character totals when routing records are present.

## Trace2Skill-reference SpreadsheetBench mode

The opt-in reference mode aligns the policy and analysis boundary used for the
TreeSkill SpreadsheetBench comparison while leaving the lightweight default
configuration unchanged. It provides:

- full preloading of the Human-Written `SKILL.md` during trajectory collection;
- the released text ReAct policy contract and bash-only action space;
- the released one-call success-analysis prompt;
- the released agentic, evaluator-gated failure-analysis prompt and tools;
- Qwen3.5 instruct decoding for the policy and thinking decoding for analysis
  and TreeSkill evolution;
- strict JSON-schema output for evidence localization and node fusion; and
- one bounded localization retry with twice the first-attempt output budget.

The Human-Written spreadsheet Skill is not redistributed by this repository.
Place a legally authorized, unmodified copy at:

```text
data/mindmemos_skill/authorized_skills/xlsx/
  SKILL.md
  recalc.py
  LICENSE.txt
```

Reference mode verifies all three files before creating an output directory.
This protects experiments from silently using a partial or modified starting
Skill. The package resources are preserved in the evolved candidate and are
materialized for both full-Skill collection and routed evaluation. Other
MindMemOS experiments continue to load only `SKILL.md` unless resource loading
is explicitly requested by this mode.

Resolve the seed-41 command without starting an experiment:

```bash
UV_CACHE_DIR=/tmp/mindmemos-skill-uv-cache \
scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/treeskill/spreadsheetbench/trace2skill_reference.yaml \
  --dry-run
```

Run seeds 41, 42, and 43 separately with the same configuration:

```bash
scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/treeskill/spreadsheetbench/trace2skill_reference.yaml

scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/treeskill/spreadsheetbench/trace2skill_reference.yaml \
  --set rollout.seed=42

scripts/run_mindmemos_skill_experiment.sh \
  --config config/mindmemos_skill/treeskill/spreadsheetbench/trace2skill_reference.yaml \
  --set rollout.seed=43
```

The reference configuration uses ordered tasks `0:200` for collection and
evolution, ordered tasks `200:400` for routed held-out evaluation, one rollout
per task, `max_turns=100`, 16 policy workers, and the published generation
settings. The 128-way analysis setting controls bounded asynchronous LLM
requests; localization and fusion remain independently bounded.

This mode aligns the implemented P0 and P1 contracts. It does not claim that a
single local run reproduces every paper table, infrastructure condition, or
model-server implementation.

### Recorded Qwen3.5-9B reference run

The completed seed-41 run below used code revision
`d4a8b5a36c9ef5623f132e352d1e36a8fb868869`, the reference configuration
above, Qwen3.5-9B at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, and the ordered `0:200` training
and `200:400` held-out split. Failed held-out outputs were counted as failed
tasks.

| Field | Result |
| --- | ---: |
| Run ID | `mindmemos-treeskill-qwen9b-seed41-23463` |
| Held-out Vrf | **44.00% (88/200)** |
| Completed / failed held-out tasks | 193 / 7 |
| Initial / final tree nodes | 35 / 49 |
| Average full Skill context | 32,209 characters |
| Average routed Skill context | 24,546.95 characters |
| Context saving | 23.79% |
| Full-Skill routing fallbacks | 0 / 200 |

The durable result is stored at:

```text
outputs/treeskill_reference_9b/
  mindmemos-treeskill-qwen9b-seed41-23463/pipeline/summary.json
```

The run completed end to end, but not every training trajectory contributed to
evolution: 65 trajectory analyses failed, followed by five evidence-localization
failures and three node-fusion failures. These counts are retained here so the
44.00% result is not mistaken for a failure-free execution.

## Trace2Skill-compatible SpreadsheetBench split

The repository also packages the ordered task-ID split used by the TreeSkill
experiments:

```text
resources/mindmemos_skill/datasets/spreadsheetbench/trace2skill_200_200/splits/
```

It preserves the released `dataset.json` ordering without shuffling:

- `train`: positions `0:200`, used for trajectory collection and evolution;
- `val`: an explicit alias of `train`, used for training-set validation rather
  than as an independent partition;
- `test`: positions `200:400`, used for held-out evaluation.

Select it with `--split-dir` when invoking the Python entrypoint. The existing
`default.yaml` remains a lightweight MindMemOS integration example. The
`trace2skill_reference.yaml` configuration additionally aligns the full
collection, analysis, generation, and held-out settings described above. Using
the split alone aligns task membership and ordering only.

## Current boundary

The checked-in end-to-end configuration targets SpreadsheetBench. The core
parser, evolution algorithm, and ReAct runtime are environment neutral, but an
environment that owns its conversation loop must explicitly enter
`agent.inject_skill_request(...)` around the complete physical execution, as
the SpreadsheetBench adapter does.
