---
name: mindmemos-cli
description: Use MindMemOS to recall project memory, recommend and read reusable Skills, or manage Skill versions, evaluation, and evolution.
---

# MindMemOS CLI

Use Memory for facts, preferences, and project decisions. Use Skills for reusable methods with explicit inputs and checks. Inspect relevant material before applying it, and keep the exact versions of the Skills you load.

## Setup

The plugin workflow requires an SDK 1.0.0 build that exposes `mindmemos trajectory report --help`. From a repository checkout, run `uv sync --all-packages` and use `uv run mindmemos` or the installed `.venv/bin/mindmemos`.

Run `mindmemos auth` when setup is needed. In official mode, omit the URL flags. Memory uses `https://mindmemos.cn/v1/memory/*` and Skill uses `https://mindmemos.cn/v2/skills/*`. An authorized project key can serve both. For a custom deployment, give Memory the service address and Skill the full API root, then enter each key at its prompt.

```bash
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001/v1/skills \
  --user-id alice
mindmemos config show
mindmemos doctor
```

Use the actual service URLs and user identity. Leave the optional Skill address blank for a custom Memory-only deployment. The new SDK saves to settings.v2.json and reads older settings.json files without overwriting them. `doctor` checks local configuration. Verify remote access with `skill list --limit 1` and `memory get --top-k 1`.

## Recall and store Memory

When a task depends on past context and the host has not injected Memory, search for the specific fact or decision. Pass `--user-id` explicitly for a user-scoped search. Without it the search covers the project, even if the profile has a default write user.

```bash
mindmemos memory search "previous CSV schema decision" --user-id alice --top-k 5 --rerank --json
mindmemos memory add --content "The CSV loader rejects unknown columns." --user-id alice --json
```

Use `memory get`, `update`, `delete`, and `feedback` to inspect and maintain Memory. `memory dreaming` starts background consolidation. Check the specific command's `--help` for filters, token budgets, and message inputs.

## Recommend and load a Skill

```bash
mindmemos skill graph list
mindmemos skill graph show GRAPH_REF
mindmemos skill graph recommend "Validate a CSV schema and report invalid rows" --by GRAPH_REF --top-k 3 --include-evaluation
mindmemos skill show VERSION_ID
```

Choose a published graph that covers the task. Its snapshot pins member versions. If no graph is available, use `skill list` and inspect promising versions with `skill show`. Recommendation results are candidates. Load only the relevant Skill documents and adapt their instructions to the task.

Use `skill resolve REF` to pin an alias or family to its active version. If the Skill has scripts, templates, or other resources, export the full version.

```bash
mindmemos skill export VERSION_ID --to DIRECTORY
```

In OpenClaw and DeepSeek Harness, use a standalone CLI call in the host's shell tool. After export, use the host's read tool on the returned directory's `SKILL.md` in the same turn. The plugins can then associate the loaded exact version with the turn. A successful `skill show` through the shell tool also loads an exact version. If you only want to look up the identifier, use `skill resolve`.

## Create, update, and publish

- Use `skill register SOURCE` to capture a reusable method with its inputs, steps, resources, and verification checks.
- Use `skill history REF`, `skill commit REF --from PATH`, and `skill diff OLD NEW` to create and inspect a new version. Evaluate it before using `skill activate VERSION_ID`.
- Use `skill graph publish`, `skill graph add`, and `skill graph remove` to manage published snapshots. Publish a new snapshot when a graph should use updated member versions.

## Report, evaluate, and evolve

OpenClaw and DeepSeek Harness report each captured completed turn once through the unified CLI. When a turn loads several exact versions, one Skill trajectory records all of them. Unattributed turns use an empty version list. When there is no reporting plugin, use `mindmemos trajectory report --file event.json --json` with a stable event ID, the ordered messages, and `skill_version_ids`.

- Read history with `skill trajectory list REF` and `skill trajectory show TRAJECTORY_ID`.
- Score a version or execution with `skill analyze skill REF` or `skill analyze trajectory TRAJECTORY_ID`. Read stored evaluations with `skill evaluations query REF`.
- Use `skill evolve REF_A REF_B` to evolve selected Skills from their evidence and `skill train REF` to train one Skill. Track asynchronous jobs with `skill task show TASK_ID` or `skill task progress TASK_ID --follow`. Inspect and evaluate the resulting versions before activation.

Use each command's `--help` for its exact options.

## References

- [OpenClaw plugin](references/openclaw-plugin.md)
- [DeepSeek Harness plugin](references/deepseek-harness-plugin.md)
- [Python SDK](references/python-sdk.md)
