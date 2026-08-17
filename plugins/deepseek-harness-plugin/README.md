# @mindmemos/deepseek-harness-plugin

A [DeepSeek Harness](https://deepseek-harness.github.io/deepseek-harness/) (dsh)
plugin that wires [MindMemOS](https://github.com/mindscale-noah/MindMemOS) long-term
memory into the harness. It:

- **recalls** memories before each turn by running `mindmemos memory search` and
  injecting the hits as model context, and
- **stores** each completed turn by running `mindmemos memory add`.

It is a thin shell-out layer over the `mindmemos` CLI — the same integration the
[OpenClaw plugin](../openclaw-plugin) provides for a different host.

**Prerequisite:** the `mindmemos` CLI must be installed and authenticated
first. Installing this plugin alone does nothing — it spawns the CLI for every
operation.

## Install the CLI

```bash
# whichever installer you use for the MindMemOS CLI; e.g.
pip install mindmemos-sdk
mindmemos auth
```

Confirm it works from the shell dsh will run under:

```bash
mindmemos config show
```

## Build

```bash
cd plugins/deepseek-harness-plugin
npm install
npm run build        # emits dist/index.js + dist/index.d.ts
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
| `cli` | `mindmemos` | Command used to invoke the CLI. Set to an absolute path or a wrapper (`uv run mindmemos`) when it is not on dsh's `PATH`. |
| `topK` | `5` | Number of memories injected per turn. |
| `addMode` | `async` | `sync` blocks until extraction finishes; `async` enqueues and returns. In `async` mode only CLI-level failures are visible to the plugin. |
| `userId` | *(none)* | Scopes both search and add to one user. Omit for project-wide search; add then inherits the CLI's default user. |
| `appId` | `deepseek-harness` | Application scope attached to every search and add. |
| `sessionId` | *(none)* | Override the harness session id used as the CLI session scope. |
| `minQueryLength` | `2` | Skip recall for prompts shorter than this many characters. |
| `maxConversationMessages` | `80` | Cap on how many trailing messages are persisted per turn. |
| `minPythonVersion` | `3.11` | Minimum supported Python version for the `mindmemos` CLI (inclusive). |
| `maxPythonVersion` | `3.14` | Maximum supported Python version for the `mindmemos` CLI (exclusive; the SDK requires `<3.14`). |

## How it works

- **Recall** hooks `agent/pre-step` (step 1 only) as a prepended waterfall
  listener. It extracts the real human prompt, runs `mindmemos memory search`,
  and appends one `createUserMessage` carrying the hits under a
  `<relevant-memories>` banner. The injected message is stamped
  `source.kind === "plugin"` so the store step can tell it apart from real input.
- **Store** hooks the `session/event` firehose and reacts to `turn/end` with
  `reason.kind === "completed"`. It walks that turn's log back to its
  `turn/start`, collecting the surface `user`/`assistant`/`tool` messages, and
  runs `mindmemos memory add --messages-json-file -`. Plugin-injected context
  (`source.kind !== "user"`) is excluded so recalled memories are not re-stored.

## Verify

1. **Types build cleanly**

   ```bash
   npm run typecheck
   ```

2. **Recall fires**

   First store a fact out of band, then ask about it in dsh:

   ```bash
   mindmemos memory add --messages-json-file - --json --user-id alice --app-id deepseek-harness <<'JSON'
   [{"role":"user","content":"My favorite color is teal.","timestamp":0}]
   JSON
   ```

   In dsh, ask *"what is my favorite color?"*. The plugin log should show:

   ```
   [mindmemos-memory] recall hit 1 memories, injected N chars
   ```

   and the model should answer with the stored fact.

3. **Store writes**

   Complete any turn in dsh, then confirm the log shows:

   ```
   [mindmemos-memory] stored N message(s) from turn 1 (session_id=…)
   ```

   and that the memory is searchable afterwards:

   ```bash
   mindmemos memory search "…" --json --user-id alice
   ```

## Troubleshooting

- **`ENOENT` / `command not found`** — dsh's `PATH` may not include the CLI
  (common for GUI-launched dsh). Set `cli` to an absolute path from
  `which mindmemos`, or a wrapper that resolves it.
- **Recall never fires** — check `minQueryLength` and that `cli` resolves
  (`mindmemos config show` from dsh's environment).
- **Recall works but nothing is stored** — the store only runs when a turn
  *completes* (`reason.kind === "completed"`); aborted or errored turns are
  skipped by design. Check the log after a turn that ends cleanly.
- **CLI errors** surface as `[mindmemos-memory] memory search failed: …` /
  `[mindmemos-memory] memory add failed: …` warnings, including the CLI's
  stderr and exit code.
