# TreeSkill Integration Plan

## Scope

Integrate TreeSkill Evolution and TreeSkill Routing into the `mindmemos_skill`
package without changing the behavior of existing algorithms or Skill injection
modes. The first supported end-to-end target is the ReAct agent on
SpreadsheetBench. The core tree and evolution modules remain environment
neutral so other environments can be added later.

## Target Flows

### TreeSkill Evolution

```text
offline trajectories
-> success/error trajectory analysis
-> parse the Human-Written SKILL.md into a heading tree
-> localize atomic evidence against the initial tree
-> aggregate evidence by target node
-> fuse touched nodes bottom-up against the current reparsed tree
-> evolved SKILL.md + versioned TreeSkill metadata
-> MindMemOS persists an immutable Skill child version
```

TreeSkill is implemented in the existing `trace2skill` family because it
performs one bounded transformation over offline trajectories. It must not
write repositories, trajectory tables, output directories, or remote state
from inside the algorithm.

### TreeSkill Routing

```text
AgentExecutionRequest(task/query, persisted Skill)
-> query-aware asynchronous Skill runtime callback
-> validate or compile the Markdown tree
-> one LLM call selects necessary subtree roots
-> remove parent/child redundant selections
-> render selected subtrees with ancestor context in source order
-> ephemeral SkillInjection
-> existing Agent execution and Skill binding flow
```

Routing must not create a modified `Skill` while retaining the original
version identity. The original persisted Skill remains attached to the
trajectory; selected nodes and context statistics are execution metadata.

## Implementation Phases

1. Add the pure Markdown tree parser, renderer, stable node contracts, metadata
   compiler, and validators.
2. Add the TreeSkill `optimize` algorithm with typed analysis, localization,
   grouping, bottom-up fusion, and audit reports.
3. Persist final tree metadata under the atomic `metadata["treeskill"]`
   namespace while preserving all existing Skill resources.
4. Add a query-aware asynchronous injection scope and a new
   `tree_routed_system_prompt` mode without changing existing `tool`,
   `system_prompt`, or `filesystem` behavior.
5. Add the ReAct TreeSkill runtime. Route once per physical task, preserve
   source order, inject selected full subtrees plus ancestor local context,
   and fall back to full Skill content for invalid router output or stale
   topology metadata.
6. Bridge SpreadsheetBench, which currently owns its conversation loop and
   bypasses `ReactAgent.execute`, into the same injection scope only for the
   new routed mode. Keep its legacy tool path unchanged.
7. Register `treeskill` as an `optimize` algorithm, add a unified experiment
   adapter/configuration, and keep the existing family runner as the only
   public entrypoint.
8. Add parser, metadata, fusion, runtime, SpreadsheetBench, registry,
   orchestration, and dry-run regression tests.

## Tree Metadata Contract

The complete Markdown remains in `SKILL.md`; metadata stores only validated
structure and hashes:

```json
{
  "treeskill": {
    "enabled": true,
    "schema_version": 1,
    "router": "llm_subtree_v1",
    "skill_content_hash": "...",
    "root_ids": ["001"],
    "nodes": [
      {
        "id": "001",
        "level": 1,
        "heading": "Requirements for Outputs",
        "parent_id": null,
        "child_ids": ["002"],
        "ordinal": 0,
        "local_content_hash": "..."
      }
    ]
  }
}
```

Metadata is validated against `SKILL.md` before routing. Missing TreeSkill
metadata does not implicitly opt a Skill into routing. Stale or malformed
metadata causes an auditable full-Skill fallback.

## Behavioral Constraints

- All evidence localization in one evolution run uses the same initial tree.
- Each atomic evidence item has exactly one target node.
- Fusion groups evidence by target node and processes nodes bottom-up.
- Each accepted node edit is applied immediately and the tree is reparsed.
- The initial operation set is `update_node`, `create_child`, and `reject`.
- Only newly created nodes receive new IDs during fusion.
- Routing uses one LLM call per task and has no fixed `top_k` limit.
- A valid empty router selection injects no Skill content.
- Invalid output, unknown IDs, exceptions, or stale metadata inject the full
  Skill rather than failing the policy rollout.
- Markdown routing never removes non-Markdown Skill resources.
- Existing injection modes and existing experiment configurations remain
  behaviorally unchanged.

## Acceptance Criteria

- Existing MindMemOS Skill tests pass unchanged.
- TreeSkill parser/fusion behavior is checked against fixed fixtures from the
  source implementation.
- A dry-run TreeSkill optimization returns a candidate without mutating the
  base Skill.
- Persisting the candidate creates one immutable child version and retains the
  TreeSkill metadata and resources.
- ReAct routing is called exactly once per task and records selected node IDs,
  full/routed character counts, context saving, and fallback state.
- SpreadsheetBench routed mode uses the routed prompt and does not expose the
  full Skill through the legacy Skill tool.
- Legacy SpreadsheetBench tool injection remains unchanged.
- The unified experiment entrypoint resolves the TreeSkill configuration in
  `--dry-run` mode without requiring a large model or private dataset.

## P0-P1 Reference Alignment

The reference-compatible SpreadsheetBench path is opt-in and is implemented
without changing the default experiment configuration or legacy Skill
injection modes.

### P0: execution and analysis parity

- Training collection preloads the complete Human-Written Skill and exposes
  the released bash-only text ReAct action space.
- The starting package must contain the exact authorized `SKILL.md`,
  `recalc.py`, and `LICENSE.txt`; it is validated locally and is not
  redistributed by MindMemOS.
- Successful trajectories use the released one-call success-analysis prompt.
- Failed trajectories use the released agentic error-analysis prompt with a
  staged trajectory workspace, bash inspection, and official workbook
  comparison before records are admitted.
- The reference configuration fixes ordered train `0:200`, held-out
  `200:400`, seeds 41/42/43, one rollout, 100 policy turns, and the Qwen3.5
  instruct/thinking generation settings. Seed 41 is the checked-in default;
  seeds 42 and 43 are explicit configuration overrides.

### P1: structured TreeSkill evolution

- Evidence localization and node fusion request strict JSON-schema output.
- Localization gets one bounded retry with a doubled output-token budget.
- Localization validates evidence items independently so one malformed item
  does not discard valid items from the same trajectory record.
- The existing TreeSkill localization, bottom-up fusion, metadata, and runtime
  routing contracts remain unchanged outside this structured-output boundary.
