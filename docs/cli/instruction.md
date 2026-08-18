# MindMemOS CLI Guide


<p align="center">
  <strong><a href="instruction.md">English</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="instruction_ZH.md">简体中文</a></strong>
</p>

## 1. Overview

`mindmemos` is the command-line tool shipped with the MindMemOS Python SDK (`mindmemos_sdk`). It lets you operate the memory service directly from the terminal: write and search memories, manage SDK-registered Skills, and inspect local configuration and connectivity.

It is a thin wrapper over the SDK: every command reads the local config file (`~/.mindmemos/settings.json`) and calls the `mindmemos` service over HTTP, so you neither hand-craft requests nor repeat the service address on the command line.

## 2. Installation

```bash
pip install mindmemos-sdk
```

Verify the command is available:

```bash
mindmemos --help
```

The command is exposed as a global `mindmemos` executable via `project.scripts`.

## 3. Quick Start

### 3.1 Configure authentication

Before first use, run `mindmemos auth` to configure the service address, API key, and default user:

```bash
mindmemos auth
```

You will be prompted for three settings:

| Setting | Local self-hosted service | Official cloud service |
| :--- | :--- | :--- |
| `Base URL` | `http://127.0.0.1:8000` | `https://mindmemos.cn` |
| `API key` | An enabled key from `config/mindmemos/api_keys.yaml` | A key issued via the [website](https://mindmemos.cn) |
| `User id` | A stable identifier, e.g. `u_123` | A stable identifier, e.g. `u_123` |

Or pass them as flags to skip the prompts:

```bash
mindmemos auth --base-url http://127.0.0.1:8000 --api-key dev-api-key-001 --user-id u_123
```

The configuration is saved to `~/.mindmemos/settings.json`. The local service auto-determines `project_id` from the API key, so you do not need to specify it.

### 3.2 Inspect configuration and connectivity

```bash
# Show current configuration (API key is masked by default)
mindmemos config show

# Show the full API key
mindmemos config show --show-secret

# Verify the config is valid and the service is reachable
mindmemos doctor
```

### 3.3 Add a memory

```bash
mindmemos memory add --content "I like iced Americanos."
```

### 3.4 Search memories

```bash
mindmemos memory search "What kind of coffee does the user like?" --top-k 5
```

> See the "Memory commands" section below for more parameters. The CLI only makes the call and displays results; extraction, scoring, and storage all happen on the server.

## 4. Command Overview

```
mindmemos
├── auth                  Configure API key, user, and service address interactively
├── config                Show / reset local configuration
│   ├── show
│   └── reset
├── memory                Memory operations
│   ├── add               Add a dialogue message as memory
│   ├── search            Search memories
│   ├── get               List / filter memories in the current project
│   ├── update            Update a memory's content
│   ├── delete            Delete a memory
│   ├── feedback          Submit explicit / implicit feedback
│   └── dreaming          Trigger the dreaming pipeline
├── skill                 Manage SDK-registered Skills
│   ├── register          Register and upload a local Skill
│   ├── list              List registered Skills
│   ├── show              Show a single Skill
│   ├── evolve            Trigger cloud Skill evolution
│   ├── push              Upload local changes as a new version
│   ├── pull              Pull version metadata (without changing files)
│   ├── update            Update one or all Skills
│   ├── rollback          Roll back to a specific version
│   ├── history           Show version history
│   ├── diff              Show version differences
│   └── unregister        Remove a Skill from management
└── doctor                Check SDK configuration and connectivity
```

Every command supports `--help` for full parameter details:

```bash
mindmemos memory add --help
```

## 5. Memory Commands (`mindmemos memory`)

Memory commands use the credentials configured by `mindmemos auth` — no need to repeat them.

### 5.1 Add memory `add`

The most common form passes a single message:

```bash
mindmemos memory add --content "I like iced Americanos."
```

Specify a role (default `user`):

```bash
mindmemos memory add --content "Remember this preference" --role system
```

Multi-turn messages can be passed as JSON, either inline or from a file (when provided, `--content` / `--role` are ignored):

```bash
# Inline JSON
mindmemos memory add --messages-json \
  '[{"role":"user","content":"I like iced Americanos."},{"role":"assistant","content":"Got it."}]'

# From a file
mindmemos memory add --messages-json-file ./messages.json
```

Async mode (returns a `request_id` immediately, without waiting for extraction):

```bash
mindmemos memory add --content "I like iced Americanos." --async
```

Other options:

| Option | Description |
| :--- | :--- |
| `--user-id` | Override the configured default user |
| `--app-id` / `--agent-id` / `--session-id` | Context identifiers for scoping / filtering |
| `--metadata-json` | Business metadata (JSON object) |
| `--skill-context-json` | Skill context array, overriding SDK auto-detection |
| `--json` | Print the full result as machine-readable JSON |

Example output:

```text
Added 1 memory item(s):
- [did] m_8f3a: I like iced Americanos.
```

### 5.2 Search memories `search`

```bash
mindmemos memory search "What kind of coffee does the user like?" --top-k 5
```

Common options:

| Option | Description |
| :--- | :--- |
| `--top-k` | Number of results, default `10` |
| `--search-strategy` | Search strategy, `fast` (default) or `agentic` |
| `--rerank` | Enable reranking |
| `--score-threshold` | Rerank relevance threshold (0-1); only meaningful with `--rerank` |
| `--filter` | Filter DSL (JSON object string) |
| `--user-id` etc. | Override request context |
| `--json` | Print the full result as JSON |

### 5.3 List and filter `get`

List memories in the current project, optionally filtered:

```bash
# List the 20 most recent
mindmemos memory get --top-k 20

# Filter with a filter DSL
mindmemos memory get --filter '{"field":"value"}'
```

### 5.4 Update and delete

```bash
# Update a memory's content
mindmemos memory update <memory_id> --content "new content"

# Delete a memory (-y skips the confirmation)
mindmemos memory delete <memory_id> --yes
```

### 5.5 Feedback `feedback`

By default this runs implicit feedback, letting the server analyze recent adds; pass `--text` for explicit feedback:

```bash
# Implicit feedback
mindmemos memory feedback

# Explicit feedback, which also requires the messages that produced it
mindmemos memory feedback --text "This memory is inaccurate" \
  --messages-json '[{"role":"user","content":"..."}]'
```

### 5.6 Dreaming `dreaming`

Trigger the dreaming pipeline, either synchronously or asynchronously:

```bash
# Enqueue asynchronously (default)
mindmemos memory dreaming --async

# Wait for completion
mindmemos memory dreaming --sync
```

## 6. Skill Commands (`mindmemos skill`)

`mindmemos skill` manages local Skills that are registered with the SDK and can evolve in the cloud.

Register a local Skill (path may be a directory or a SKILL.md file):

```bash
mindmemos skill register ./my-skill
# Give it an alias for later commands
mindmemos skill register ./my-skill --alias my-skill
```

Common operations:

```bash
# List / show registered Skills
mindmemos skill list
mindmemos skill show <skill-id-or-alias>

# Trigger cloud evolution (sync by default; --async enqueues)
mindmemos skill evolve <skill-id-or-alias> --sync

# Push local changes as a new version; pull version metadata
mindmemos skill push <skill-id-or-alias>
mindmemos skill pull <skill-id-or-alias>

# Update one or all Skills (--all updates everything, -y skips confirmation)
mindmemos skill update <skill-id-or-alias> --yes
mindmemos skill update --all --yes

# Unregister (--delete-files also removes the local directory)
mindmemos skill unregister <skill-id-or-alias> --delete-files --yes
```

Version management:

```bash
# Show version history
mindmemos skill history <skill-id-or-alias>

# Roll back to a specific version
mindmemos skill rollback <skill-id-or-alias> --to <version-id> --yes

# Diff between two versions
mindmemos skill diff <skill-id-or-alias> --from <version-id> --to <version-id>
```

Before applying changes that may touch local files (`update` / `rollback`), the command prints a plan and asks for confirmation.

## 7. Configuration (`mindmemos config`)

```bash
# Show current configuration
mindmemos config show
mindmemos config show --show-secret   # Show the full API key

# Reset and delete the local configuration
mindmemos config reset
mindmemos config reset --yes          # Skip the confirmation
```

The config file lives at `~/.mindmemos/settings.json`; set the `MINDMEMOS_CONFIG_DIR` environment variable to use a different config directory.

> The API key is sensitive and is masked in `config show` by default (use `--show-secret` to reveal it).

## 8. Troubleshooting

| Symptom | Possible cause / fix |
| :--- | :--- |
| No SDK config at … Run `mindmemos auth` first. | Not configured yet; run `mindmemos auth` |
| No api_key configured | API key missing; re-run `mindmemos auth` |
| transport: not ready … | `doctor` found the service unreachable. Confirm the service is up and `base_url` is correct: `http://127.0.0.1:8000` for local, `https://mindmemos.cn` for cloud |
| Authentication error from the service | The API key is invalid or disabled; check the cloud key or an enabled key in `config/mindmemos/api_keys.yaml` |
| `ENOENT` in a plugin environment | A GUI-launched process does not inherit the terminal PATH; configure `mindmemos` as an absolute path or wrap it with `uv run mindmemos`. See [OpenClaw plugin integration](../../skills/mindmemos-cli/references/openclaw-plugin.md) |

Run `mindmemos doctor` anytime as a one-shot config / connectivity check.

## 9. Related Documentation

- [Python SDK integration](../../skills/mindmemos-cli/references/python-sdk.md)
- [OpenClaw plugin integration](../../skills/mindmemos-cli/references/openclaw-plugin.md)
- [Deployment & Configuration Guide](../deploy/instruction.md)
- [API docs](https://mindmemos.cn/api-docs)
