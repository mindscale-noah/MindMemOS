# DeepSeek Harness integration

The plugin recalls Memory before the first model step and reports completed turns to Memory through the local CLI. The same call reports to Skill if the turn loads an exact version. The plugin registers the packaged `mindmemos-cli` guide in `ctx.skills`, so the host can discover it and load its Markdown and references on demand.

## Install and configure

Use a dsh profile that exposes the `agents` and `skills` services. The plugin is built against dsh 0.1.0-rc.6. Authenticate an SDK 1.0.0 build that exposes `mindmemos trajectory report --help`, with both service endpoints and a default user.

For a checkout build, run `npm ci` and `npm run build` in `plugins/deepseek-harness-plugin`. Add an entry to the profile's `cordis.patch.yml`.

```yaml
- insert:
    - id: mindmemos-memory
      name: 'file:///path/to/MindMemOS/plugins/deepseek-harness-plugin/dist/index.js'
      config:
        cli: /path/to/MindMemOS/.venv/bin/mindmemos
        userId: alice
```

Use a `file://` URL for the local entry, including on Windows. After publication, install the npm package into the profile and use `@mindmemos/deepseek-harness-plugin` as `name`.

`userId` scopes recall and reporting. Without it, recall searches the project and reporting uses the CLI profile's default user. The plugin also supports `topK`, `tokenBudget`, `addMode`, `appId`, `sessionId`, `minQueryLength`, and `maxConversationMessages`.

## Verify

- Check that `mindmemos-cli` appears in the host's Skill catalog and can load its relative references.
- Ask about a stored fact and look for the recall hit in the plugin log.
- Complete a turn and look for `Memory queued` or `Memory completed`. With an exact Skill version, look for `Skill stored` and use its ID with `mindmemos skill trajectory show`. Otherwise, expect `Skill skipped`.
- To verify multi-Skill binding, use the shell and read tools to [load two exact versions](../SKILL.md#recommend-and-load-a-skill) in one turn, then check the trajectory's `skill_version_ids`.

`ENOENT` points to the CLI executable path. If one destination reports `failed` or `unknown`, its line includes the available error and request identifiers. The report hook runs after `turn/end` with a completed reason. Injected context is excluded from the report.
