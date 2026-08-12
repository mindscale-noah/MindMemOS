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

## Current boundary

The checked-in end-to-end configuration targets SpreadsheetBench. The core
parser, evolution algorithm, and ReAct runtime are environment neutral, but an
environment that owns its conversation loop must explicitly enter
`agent.inject_skill_request(...)` around the complete physical execution, as
the SpreadsheetBench adapter does.
