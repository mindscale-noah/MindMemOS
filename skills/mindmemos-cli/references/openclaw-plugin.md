# OpenClaw integration

The plugin recalls Memory before prompt construction. It reports each successful turn to Memory and Skill through the local CLI. OpenClaw discovers the packaged `mindmemos-cli` guide through the plugin manifest and loads it on demand.

## Install and configure

First authenticate an SDK 1.0.0 build that exposes `mindmemos trajectory report --help`. Configure both service endpoints and a default user through `mindmemos auth`. If the gateway's PATH differs from your terminal's PATH, set `plugins.entries.mindmemos-memory.config.cli` to the executable's absolute path.

For a checkout build, run `npm ci` and `npm run build` in `plugins/openclaw-plugin`, then link the **plugin root**.

```bash
openclaw plugins install --link /path/to/MindMemOS/plugins/openclaw-plugin
openclaw plugins enable mindmemos-memory
openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
openclaw gateway restart
openclaw skills list
```

After npm publication, install `@mindmemos/openclaw-plugin@0.2.0` in place of the linked package. The conversation access setting is required for installed and linked plugins. It allows the `agent_end` hook to read messages for reporting.

Set `plugins.entries.mindmemos-memory.config.userId` for user-scoped recall and reporting. Without it, recall searches the project and reporting uses the CLI profile's default user. The plugin also supports `topK`, `tokenBudget`, `addMode`, `appId`, `sessionId`, `minQueryLength`, and `maxConversationMessages`. Subagent sessions receive memories as background context.

## Verify

- Check that `openclaw skills list` includes `mindmemos-cli`.
- Ask about a stored fact and check the gateway log for the recall hit.
- Complete a turn and look for separate `Memory queued` or `Memory completed` and `Skill stored` lines with the same event ID. Use the returned Skill trajectory ID with `mindmemos skill trajectory show`.
- To verify multi-Skill binding, use the shell and read tools to [load two exact versions](../SKILL.md#recommend-and-load-a-skill) in one turn, then check the trajectory's `skill_version_ids`.

An `agent_end` access warning means the conversation access grant is missing. `ENOENT` points to the CLI executable path. If one destination reports `failed` or `unknown`, its line includes the available error and request identifiers. Memory's `queued` receipt confirms acceptance for extraction.
