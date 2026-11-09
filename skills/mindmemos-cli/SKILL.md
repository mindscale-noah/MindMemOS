---
name: mindmemos-cli
description: Use the MindMemOS CLI to recall or store project memory, manage remote Skill versions and graphs, or recommend an existing Skill for a user's task. Use for MindMemOS setup, Skill evaluation, and evolution tasks in the configured project.
---

# MindMemOS CLI

## Install the CLI

These instructions target the workspace SDK and CLI 1.0.0. It is not yet published at the time of this update. From the MindMemOS checkout, run `uv sync --all-packages` when setup is needed and use `uv run mindmemos` in place of `mindmemos` below. An installed 1.0.0 build exposes `mindmemos` directly. Check `mindmemos skill --help` if the executable's origin is unclear. The current CLI has `resolve`, `commit`, `graph`, and `evaluations`. If these are missing, use the workspace build or an installed 1.0.0 wheel.

Run `mindmemos auth` if setup is needed. Official mode shares the endpoint and key between Memory and Skill. For a custom deployment with separate services, supply both URLs and enter each key at its prompt:

```bash
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001 \
  --user-id alice
```

Use the user's actual endpoints and identity. For a custom deployment with one endpoint and key, use `--skill-shares-memory`. `mindmemos config show` masks keys and shows the effective profile. Set `MINDMEMOS_HOME` to the intended SDK directory when working across projects. Back up a v1 profile before first use with 1.0.0 because the SDK rewrites it as v2. `mindmemos doctor` checks local setup. Test remote access with `mindmemos skill list --limit 1` or `mindmemos memory get --top-k 1` as appropriate.

## Recommend a Skill

When the user asks which existing Skill fits a task or wants to find one they can reuse, run the recommendation workflow. Use the user's goal, input type, expected output, and relevant constraints to form a short task query. Leave credentials and unnecessary private content out of the query.

1. Use the user's specified graph or the graph configured for this workflow. Otherwise run `mindmemos skill graph list`. Use the one applicable published graph if the scope is clear. If several graphs fit and the choice changes the scope, ask which to use.
2. Run `mindmemos skill graph recommend "TASK" --by GRAPH_REF --top-k 3`. A graph alias selects its current published snapshot. Use an exact snapshot ID when the user specifies one or needs a repeatable result.
3. Use the returned `skills` in rank order. Inspect promising candidates with `mindmemos skill show VERSION_ID` when you need to verify fit. Explain each recommendation against the user's requirements and give its exact version ID. If results are empty or weak, say so. If the graph is missing, unpublished, or unavailable, report that condition before suggesting any graph changes.

Recommendation is read-only. Installation, execution, activation, and graph publication require their own user intent. See [Skill management and recommendation](references/skill-management.md) for result fields, stored evaluations, and version handling.

## Manage Skills or Memory

- **Skills.** Read [Skill management and recommendation](references/skill-management.md) for register, commit, export, activation, graph membership, trajectories, evaluation, and evolution. Resolve an alias or family with `skill resolve REF` and carry forward the exact version ID when inspecting, comparing, exporting, or scoring. Skill commands return JSON by default, except `task progress`, which emits text logs. Check the exit status and read errors from stderr.
- **Memory.** Read [Memory commands](references/memory.md) for recall, writing, and maintenance. Pass `--user-id` explicitly for user-scoped search. Omitting it searches the whole project, even when auth saved a default user for writes.
- **Python apps.** Read [Python SDK usage](references/python-sdk.md) for the matching clients.

## Host integrations

Use these references for automatic Memory hooks. Run Skill commands directly through the CLI.

- [OpenClaw](references/openclaw-plugin.md)
- [DeepSeek Harness](references/deepseek-harness-plugin.md)
