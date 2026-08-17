# DeepSeek Harness integration

The DeepSeek Harness (dsh) memory plugin wires MindMemOS into dsh automatically:
it `search`es memories before each turn and injects the hits as model context,
and `add`s each completed turn at the end. It is a thin shell-out layer over the
`mindmemos` CLI — the same integration the OpenClaw plugin provides for a
different host.

**Prerequisite:** the `mindmemos` CLI must be installed and authenticated first
(see the main SKILL.md → "Install the CLI"). Installing the plugin alone does
nothing — it spawns the CLI for every operation.

## Install the plugin

The plugin is published to npm as **`@mindmemos/deepseek-harness-plugin`**.
Install it in the project where dsh runs, then register it in a
`cordis.patch.yml` layer:

```bash
npm install @mindmemos/deepseek-harness-plugin
```

dsh composes plugins through layered `cordis.patch.yml` files. Add an `insert`
entry to your profile patch (`$DSH_HOME/cordis.patch.yml` or the profile's
`cordis.patch.yml`):

```yaml
- insert:
    - id: mindmemos-memory
      name: '@mindmemos/deepseek-harness-plugin'
      config:
        userId: alice
        appId: deepseek-harness
```

`id` is stable and unique; the plugin's cordis `name` is `mindmemos-memory`. To
disable the plugin, set `disabled: true` on the entry rather than removing the
row.

For local testing before publishing, point `name` at the built entry instead
(`npm run build` first):

```yaml
- insert:
    - id: mindmemos-memory
      name: 'file:///C:/…/plugins/deepseek-harness-plugin/dist/index.js'
      config:
        userId: alice
```

On Windows the `name` **must** be a `file://` URL — a bare `C:/…` path is read
as a `c:` URL scheme by Node's ESM loader.

## Configure

Optional plugin config:

```yaml
cli: mindmemos
topK: 5
addMode: async
userId: alice
appId: deepseek-harness
sessionId: optional-session-override
minQueryLength: 2
maxConversationMessages: 80
minPythonVersion: '3.11'
maxPythonVersion: '3.14'
```

- `cli` — command used to invoke the CLI. Defaults to `mindmemos`. If dsh does
  not run with the CLI on its PATH, set this to an absolute path or a wrapper
  (e.g. `uv run mindmemos` inside the repo).
- `topK` — number of memories injected per turn.
- `addMode` — `sync` blocks until extraction finishes; `async` (default) enqueues
  and returns. In `async` mode only CLI-level failures are visible to the plugin.
- `userId` — scopes both search and add to one user. If omitted, search is
  project-wide, while add inherits the default user from the local `mindmemos`
  CLI config.
- `appId` / `sessionId` — override the corresponding plugin context values.
- `minQueryLength` — skip recall for very short prompts.
- `maxConversationMessages` — cap on how many trailing messages are persisted per
  turn.
- `minPythonVersion` / `maxPythonVersion` — supported Python range for the CLI,
  matching the SDK's `requires-python = ">=3.11,<3.14"` (min inclusive, max
  exclusive).

## How it works

- **Recall** hooks `agent/pre-step` (step 1 only). It extracts the real human
  prompt, runs `mindmemos memory search`, and injects the hits under a
  `<relevant-memories>` banner. Injected messages are stamped
  `source.kind === "plugin"` so the store step can tell them apart from real
  input.
- **Store** hooks `session/event` and reacts to `turn/end` with
  `reason.kind === "completed"`. It walks that turn's log back to `turn/start`,
  collects the surface messages, and runs `mindmemos memory add`. Plugin-injected
  context is excluded so recalled memories are not re-stored.

## Troubleshooting

- CLI errors surface in the dsh log as `[mindmemos-memory] memory search failed:
  …` / `[mindmemos-memory] memory add failed: …` warnings, including the CLI's
  stderr and exit code.
- `ENOENT` / `command not found` — dsh's `PATH` may not include the CLI (common
  for GUI-launched dsh). Set `cli` to an absolute path from `which mindmemos`, or
  a wrapper that resolves it (`uv run mindmemos` when dsh's working directory is
  this repo).
- **Recall never fires** — check `minQueryLength` and that `cli` resolves
  (`mindmemos config show` from dsh's environment).
- **Recall works but nothing is stored** — the store only runs when a turn
  *completes* (`reason.kind === "completed"`); aborted or errored turns are
  skipped by design. Check the log after a turn that ends cleanly.
