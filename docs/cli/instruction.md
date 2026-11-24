# MindMemOS CLI Guide

English · [简体中文](instruction_ZH.md)

`mindmemos` ships with the Python SDK. Use it to configure service connections, work with memories, and manage remote Skills. Append `--help` to any command for its full options.

## SDK configuration and diagnostics

```bash
pip install mindmemos-sdk
mindmemos --help
```

To use the current checkout, run `uv sync --all-packages` from the repository root and invoke the commands below with `uv run mindmemos`. See the [deployment guide](../deploy/instruction.md) to start self-hosted services.

The CLI checks `MINDMEMOS_HOME`, then the legacy `MINDMEMOS_CONFIG_DIR`, the saved default directory, and finally `~/.mindmemos`. The SDK saves new settings to `settings.v2.json` and can read the legacy `settings.json` as a fallback. The server determines remote project access from the credentials.

### `mindmemos auth`

Choose an official or custom deployment and save the API keys and default user ID. In official mode, omit both base URL flags. Memory uses `https://mindmemos.cn/v1/memory/*` with the existing key. The new Skill API root is `https://mindmemos.cn/v2/skills` and requires a key with Skill permissions.

For a custom deployment, `--memory-base-url` takes the service address before `/v1/memory`. `--skill-base-url` takes the full Skill API root, including `/v1/skills` for a direct connection, and requires the Skill service's key. Existing Skill settings with only the service address still work.

```bash
mindmemos auth
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001/v1/skills \
  --user-id alice
```

For a custom Memory-only connection, leave the optional Skill address blank. Pass `--no-skill` to clear an existing custom Skill connection. By default, enter keys at the prompts. Automation can supply `--memory-api-key` and, for an independent Skill service, `--skill-api-key`.

### `mindmemos config show`

Show effective service addresses, the default user, and the settings path. The command masks API keys by default. `--show-secret` reveals them.

```bash
mindmemos config show
```

### `mindmemos config home set`

Set the default configuration directory for future processes. Restart existing processes to use the new default. The original `settings.json` stays in place.

```bash
mindmemos config home set /absolute/path/to/.mindmemos
```

### `mindmemos config reset`

Clear the current SDK settings while preserving the legacy `settings.json` for older clients. The command prompts for confirmation unless you pass `--yes` or `-y`.

```bash
mindmemos config reset
```

### `mindmemos doctor`

Check local settings, credential presence, and SDK client initialization. `transport: ready` means client initialization succeeded. To check remote connectivity and permissions, run `mindmemos memory get --top-k 1` and `mindmemos skill list --limit 1`.

```bash
mindmemos doctor
```

If settings or keys are missing, check the directory with `config show`, then run `auth`. If a GUI plugin cannot find `mindmemos`, configure the [absolute CLI path](../../skills/mindmemos-cli/references/openclaw-plugin.md).

## Memory commands

`add`, `search`, `feedback`, and `dreaming` accept `--user-id`, `--app-id`, `--agent-id`, and `--session-id`. Replace `alice` and `MEMORY_ID` with actual values.

### `mindmemos memory add`

Add one message or a conversation. The default `--role` is `user`. Pass a nonempty JSON array with `--messages-json` or `--messages-json-file` for multiple messages. Use `--metadata-json` for application metadata.

```bash
mindmemos memory add --content "I like iced Americanos."
mindmemos memory add --messages-json-file ./messages.json --async --json
```

By default, the command processes the request synchronously for the configured user. A successful `--async` submission returns a `request_id`. Use `--json` for the full response.

`--skill-context-json` accepts legacy plugin references as a JSON array. Each item contains `name`, `content_hash`, and optional `base_version_id`, `version_label`, and `usage` (`injected` or `modified`). The original Memory API receives these references; this option does not call the standalone Skill service. Omitting it does not detect or register legacy Skills automatically.

### `mindmemos memory search`

Search by query text and return up to 10 results by default. Pass `--user-id` to scope the search to a user. Omit it to search the current project.

```bash
mindmemos memory search "What kind of coffee do I like?" --user-id alice --top-k 5
```

Use `--search-strategy agentic` to switch from the default `fast` strategy. Pass a JSON filter object with `--filter`, constrain results with `--token-budget`, or use `--json` for the full response. `--score-threshold` applies only with `--rerank`.

### Image add, search, and download

When multimodal mode is enabled for the project, repeat `--image` or `--query-image` for local files and HTTP(S) URLs. Download an original with the hit's `metadata.blob_id` and `metadata.user_id`.

```bash
mindmemos memory add --content "Desk setup" --image ./desk.png
mindmemos memory search "Desk" --query-image ./desk.png --json
mindmemos memory search "Desk" --multimodal --json
mindmemos memory get-image BLOB_ID --user-id alice --output desk.png
```

### `mindmemos memory get`

List memories in the current project. Use `--top-k` to cap the number of results or `--filter` to supply a JSON filter object.

```bash
mindmemos memory get --top-k 20
```

### `mindmemos memory update`

Replace the contents of the memory selected by ID with `--content`.

```bash
mindmemos memory update MEMORY_ID --content "The user likes unsweetened iced Americanos."
```

### `mindmemos memory delete`

Delete a memory by ID. The command prompts for confirmation unless you pass `--yes` or `-y`.

```bash
mindmemos memory delete MEMORY_ID
```

### `mindmemos memory feedback`

Omit `--text` to trigger implicit feedback. For explicit feedback, provide `--text` and the original messages with `--messages-json` or `--messages-json-file`. Attach recalled memories with `--recalled-memories-json` or `--recalled-memories-json-file`.

```bash
mindmemos memory feedback
mindmemos memory feedback --text "The user prefers unsweetened coffee" \
  --messages-json '[{"role":"user","content":"I like unsweetened coffee."}]'
```

### `mindmemos memory dreaming`

Trigger memory evolution. The command queues it by default, and `--async` makes that choice explicit. Use `--sync` to wait for completion.

```bash
mindmemos memory dreaming --sync
```

## Skill commands

A Skill reference can be an alias, a family ID, or a version ID. Aliases and family IDs select the active version. Use a full version ID to pin inputs for scoring, export, and comparison. A graph reference can be an alias, a graph family ID, or a snapshot ID.

Replace `SKILL_VERSION_ID`, `TRAJECTORY_ID`, and `TASK_ID` in the examples with actual IDs. Skill commands return JSON on success. `task progress` prints text logs. Errors go to stderr with a nonzero exit code.

### Basic management

Manage Skill families and versions, execution trajectories, and graph snapshots.

#### `mindmemos skill register`

Register a new family from a local Skill directory or `SKILL.md`. A standard `SKILL.md` needs `name` and `description` in its frontmatter. Local registration defaults to the Skill name as its alias and `v0.0.1` as the first version label.

```bash
mindmemos skill register ./my-skill --alias my-skill
```

Use `--version`, `--message`, and `--metadata-json` to set first-version metadata. `--include` and `--exclude` take comma-separated resource patterns. To supply a complete inline `StructuredSkillInfo` object with `--json-skill`, omit the local path and its options.

#### `mindmemos skill list`

List up to 20 Skill families in the project by default. Filter with `--name`, `--alias`, `--user-id`, or `--agent-id`, or pass a JSON object with `--filter`.

```bash
mindmemos skill list --limit 20
```

#### `mindmemos skill show`

Read Skill family information and the selected version's full content and resources.

```bash
mindmemos skill show my-skill
```

#### `mindmemos skill resolve`

Resolve a Skill reference to its family and version IDs. Read `skill_version_id` to capture the exact selected version.

```bash
mindmemos skill resolve my-skill
```

#### `mindmemos skill export`

Export the selected version's content and resources to a Skill subdirectory under `--to`. `--overwrite` replaces an existing same-named subdirectory and its contents.

```bash
mindmemos skill export my-skill --to ./exports
```

#### `mindmemos skill commit`

Create a child of the referenced version and return its `skill_version_id`. For a local source, pass a directory or `SKILL.md` with `--from` and supply `--message`. Add `--activate` to activate the new version at commit time.

```bash
mindmemos skill commit my-skill --from ./my-skill --message "Add input validation steps"
```

The default is a patch bump with inherited resources. Use `--version` to set a label, or `--bump minor` or `--bump major` to change the increment. Replace resources with `--resources-dir` or clear them with `--clear-resources`. For `--json-skill`, supply a complete JSON object with `commit_message` and omit local source options.

#### `mindmemos skill history`

List version history for the referenced family. Use `--limit` and `--offset` to page through results. The defaults are 20 and 0.

```bash
mindmemos skill history my-skill --limit 20 --offset 0
```

#### `mindmemos skill diff`

Compare the versions selected by two references from left to right. This example compares the active version of `my-skill` with a specified version.

```bash
mindmemos skill diff my-skill SKILL_VERSION_ID
```

#### `mindmemos skill activate`

Switch the family's active version. Use `--version` for a label or `--version-id` for a full version ID from the same family. Use this command to restore an older version as active.

```bash
mindmemos skill activate my-skill --version-id SKILL_VERSION_ID
```

#### `mindmemos skill update-info`

Update family information. Change the name with `--name`, set an alias with `--alias`, clear it with `--clear-alias`, or set metadata with `--metadata-json`.

```bash
mindmemos skill update-info my-skill --alias review-skill
```

#### `mindmemos skill delete`

Soft-delete the entire Skill family. Passing a version ID still targets its owning family. The command prompts for confirmation unless you pass `--yes` or `-y`.

```bash
mindmemos skill delete my-skill
```

#### `mindmemos skill trajectory report`

Report one execution with `user_id`, `agent_id`, `messages`, and `skill_version_ids`. Pass the exact versions loaded during execution as a list, including for a single version. Use `[]` when no exact version is known. The service returns one trajectory ID. Supply `--file` or `--json-data`. Use `--file -` to read from standard input.

```bash
mindmemos skill trajectory report --file ./trajectory.json
```

#### `mindmemos skill trajectory list`

List trajectories for the selected version. Filter with `--status`, `--received-from`, or `--received-to`. The default limit is 20.

```bash
mindmemos skill trajectory list my-skill --limit 20
```

#### `mindmemos skill trajectory show`

Read one complete execution record by trajectory ID.

```bash
mindmemos skill trajectory show TRAJECTORY_ID
```

#### `mindmemos skill graph list`

List graph families in the current project.

```bash
mindmemos skill graph list
```

#### `mindmemos skill graph resolve`

Resolve a graph reference to its graph family ID and selected snapshot ID.

```bash
mindmemos skill graph resolve my-collection
```

#### `mindmemos skill graph history`

List snapshots of the referenced graph family.

```bash
mindmemos skill graph history my-collection
```

#### `mindmemos skill graph show`

Read the published graph snapshot selected by the reference.

```bash
mindmemos skill graph show my-collection
```

#### `mindmemos skill graph preview`

Preview the topology of a published snapshot. This is a read-only command.

```bash
mindmemos skill graph preview my-collection
```

#### `mindmemos skill graph publish`

Publish a graph snapshot. Select the target graph with `--alias` or `--graph-id`. For membership, repeat `--skill`, use `--all-skills`, or use `--empty`. A snapshot pins its members to exact versions and holds at most one version per family.

```bash
mindmemos skill graph publish --alias my-collection --skill my-skill
```

Use `--message` to record a commit message or `--force-extract-scene` to force Scene extraction.

#### `mindmemos skill graph add`

Add selected Skills to a graph and publish a new snapshot. Repeat `--skill` for multiple members. Use `--message` to record a commit message or `--force-extract-scene` to force Scene extraction.

```bash
mindmemos skill graph add my-collection --skill another-skill
```

#### `mindmemos skill graph remove`

Remove selected members from a graph and publish a new snapshot. Their families and versions remain available in the repository. Repeat `--skill` for multiple members or use `--message` to record the reason.

```bash
mindmemos skill graph remove my-collection --skill another-skill
```

### Algorithms and tasks

Scoring and evaluation queries require the server's Analyze pipeline. Scoring also requires the corresponding Judge. If the server returns `pipeline_unavailable`, check whether the required pipeline is enabled. See the [Skill evaluation guide](../evaluation/skill_evaluation.md) for benchmarks.

#### `mindmemos skill analyze skill`

Score the selected version's content and resources. The command reuses a completed result for the same inputs and Judge configuration. `--force-refresh` reruns the Judge and updates that result. Use `--request-id` to attach a trace identifier.

```bash
mindmemos skill analyze skill my-skill
```

#### `mindmemos skill analyze trajectory`

Analyze a stored trajectory. The server sets the attribution default. Override it with `--attribute` or `--no-attribute`. Use `--verifier-score` to supply a verifier score. Supports `--request-id` and `--force-refresh`.

```bash
mindmemos skill analyze trajectory TRAJECTORY_ID --no-attribute --verifier-score 0
```

#### `mindmemos skill evaluations query`

Read completed evaluations for one or more selected versions. Repeat `--judge-type` to select `static_skill` and `trajectory`, or omit it to include both. `--limit` defaults to 200 and accepts up to 1,000. This command reads stored results.

```bash
mindmemos skill evaluations query SKILL_VERSION_ID --judge-type static_skill --limit 20
```

Each static evaluation in `items` includes `skill_version_id`, `score_mean`, `scores`, `summary`, and `findings`. Query by full version ID to keep the selection fixed when the active version changes.

#### `mindmemos skill evolve`

By default, submit an asynchronous evolution task and return its `task_id`; explicit `--async` has the same behavior. Use `--sync` to wait for completion and print the final task snapshot, including its `result`. Failed or cancelled tasks produce a nonzero exit status. You can submit several Skill references, with at most one version per family. Omitting `--pipeline` uses the server default. The repository example configures a pipeline named `offline`.

```bash
mindmemos skill evolve my-skill --pipeline offline
mindmemos skill evolve my-skill another-skill --pipeline offline
mindmemos skill evolve my-skill --pipeline offline --sync
```

Repeat `--editable-skill` to select editable inputs, or use `--read-only` to preserve all input families. Use `--allow-create-skills` or `--no-allow-create-skills` to control creation of new families. Add `--auto-accept` to activate candidate versions of existing families.

The legacy `--sync` and `--async` flags select how the CLI waits; both use the standalone Skill service. References must resolve in the new repository. The command does not read the old SDK registry or fall back to the legacy evolution API. Ctrl+C stops local waiting without cancelling the submitted task.

#### `mindmemos skill train`

Submit an asynchronous training task and return its `task_id`. Supply a configured `--pipeline` and server-side `--work-dir`. Load an environment from a local YAML or JSON file with `--environment`, or pass it inline with `--environment-json`.

```bash
mindmemos skill train my-skill --pipeline TRAINING_PROFILE \
  --work-dir ./runs/train-001 --environment ./environment.yaml \
  --sample-json "$(cat ./train-sample.json)" --train-id SAMPLE_ID
```

Replace `TRAINING_PROFILE` with the server's training profile. The sample file must contain `task` and `reward_data`, and `SAMPLE_ID` must match `task.id`. Repeat `--sample-json`, `--train-id`, `--val-id`, and `--test-id` as needed. IDs must be disjoint across splits. Add `--auto-accept` to activate the trained version.

#### `mindmemos skill task show`

Read task status and its result snapshot. Use the task ID returned by `evolve` or `train`.

```bash
mindmemos skill task show TASK_ID
```

#### `mindmemos skill task progress`

Print task logs as text. Use `--after N` to read events after sequence N. Use `--follow` or `-f` to keep reading until the task ends.

```bash
mindmemos skill task progress TASK_ID --follow
```

#### `mindmemos skill graph recommend`

Recommend Skills for a task from the graph snapshot selected by `--by`. Use a full snapshot ID to pin the scope. `--strategy` accepts `graph` or `vector` and defaults to `graph`.

```bash
mindmemos skill graph recommend "Organize CSV data" --by my-collection --include-evaluation
```

Use `--top-k` to set the result count, which defaults to 5. `--include-scenes` attaches Scenes, and `--include-evaluation` attaches stored evaluations.
