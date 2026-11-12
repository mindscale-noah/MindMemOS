# @mindmemos/deepseek-harness-plugin

A DeepSeek Harness (dsh) plugin for automatic Memory recall and unified
trajectory reporting. It recalls Memory before the first model step and reports
completed turns to Memory. The same CLI call reports to Skill when the turn
loads an exact version. The plugin registers the bundled `mindmemos-cli` guide
in the host's Skill catalog.

Use a dsh profile with `agents` and `skills` services. This plugin is built
against dsh 0.1.0-rc.6. Install and authenticate an SDK build that exposes
`skill identify` and `trajectory report` before enabling the plugin.

```bash
mindmemos auth
mindmemos skill identify --help
mindmemos trajectory report --help
```

## Install the plugin

After publication, install the 0.2.0 package in the dsh profile:

```bash
npm install @mindmemos/deepseek-harness-plugin@0.2.0
```

For local development before publishing, build from this repo instead (see
[Build](#build)) and register the built entry by `file://` URL.

## Build

```bash
cd plugins/deepseek-harness-plugin
npm ci
npm run build
```

`npm run typecheck` runs the same compiler pass without emitting — use it to
validate against the dsh types before wiring the plugin up.

## Register

dsh composes plugins through layered `cordis.patch.yml` files. Add an `insert`
entry to your profile patch (`$DSH_HOME/cordis.patch.yml` or the profile's
`cordis.patch.yml`):

```yaml
- insert:
    - id: mindmemos-memory
      name: '@mindmemos/deepseek-harness-plugin'   # once published
      config:
        userId: alice
        appId: deepseek-harness
```

For local testing before publishing, point `name` at the built entry (see
[`cordis.patch.example.yml`](./cordis.patch.example.yml) for the exact form):

```yaml
- insert:
    - id: mindmemos-memory
      name: 'file:///C:/…/plugins/deepseek-harness-plugin/dist/index.js'
      config:
        userId: alice
```

`id` is stable and unique; the plugin's cordis `name` is `mindmemos-memory`.
To disable the plugin, set `disabled: true` on the entry rather than removing
the row.

## Configure

| Option | Default | Meaning |
| --- | --- | --- |
| `cli` | `mindmemos` | Executable used to invoke the CLI — a name on dsh's `PATH` or an absolute path. It is not a shell command, so a wrapper like `uv run mindmemos` will not work; point it at the real executable instead. |
| `topK` | `5` | Number of memories injected per turn. |
| `multimodal` | `true` | Also search stored images by passing `memory search --multimodal`; set `false` for text-only recall. The server ignores the flag when the project's multimodal mode is off. |
| `tokenBudget` | *(none)* | Optional strict token budget for recall; when set, `memory search --token-budget` caps the injected memories by estimated tokens (token-budget retention) in addition to `topK` — whichever limit is tighter applies. |
| `addMode` | `sync` | `sync` blocks until extraction finishes; `async` enqueues and returns. In `async` mode the Memory receipt confirms acceptance for extraction. |
| `userId` | *(none)* | Scopes recall and reporting. Omit for project-wide recall and use the CLI profile's default user for reporting. |
| `appId` | `deepseek-harness` | Application scope for Memory recall and reporting. |
| `sessionId` | *(none)* | Override the harness session id used as the CLI session scope. |
| `minQueryLength` | `2` | Skip recall for prompts shorter than this many characters. |
| `maxConversationMessages` | `80` | Cap on how many trailing messages are persisted per turn. |

## Verify

1. Load the profile and find `mindmemos-cli` in its Skill catalog. The provider
   exposes the guide and the directory containing its references.
2. Ask about a stored fact and look for a recall hit in the plugin log.
3. Finish a turn. The log shows separate Memory and Skill receipts with the
   same event ID. Use `mindmemos skill trajectory show TRAJECTORY_ID` to read the
   Skill receipt's stored trajectory. `Skill skipped` means the turn had no
   identified Skill versions and the SDK sent no Skill request.

The plugin identifies successful `skill show` results and local `SKILL.md` reads.
For a native Skill invocation, it checks the observed instructions against the
host's file-backed Skill before matching the exact source hash through the SDK.
Identified versions share one trajectory. Recommendations and exports alone
leave the version list empty.

If dsh logs `ENOENT`, set `cli` to the absolute path to the executable.
If recall is absent, check `minQueryLength` and the CLI profile. Reporting
runs when `turn/end` has reason `completed`. Injected Memory and Skill
snapshots are excluded from the turn's messages.

To run the shared hook integration tests, build both plugins, then run from the
repository root with the SDK installed.

```bash
MINDMEMOS_TEST_CLI="$PWD/.venv/bin/mindmemos" node --test tests/integration/test_plugin_trajectory.mjs
```
