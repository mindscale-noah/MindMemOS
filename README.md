# MindMemOS

**Long-term memory and manageable, trainable, evolving Skills for AI agents.**

<p align="center">
  <img src="./assets/mindmemos-readme-hero.png" alt="MindMemOS Memory For AI Agents">
</p>

<p align="center">
  <a href="https://mindmemos.cn">
    <img src="https://img.shields.io/badge/Website-mindmemos.cn-0A66C2?labelColor=gray&logo=googlechrome&logoColor=white" alt="MindMemOS Website">
  </a>
  <a href="https://mindmemos.cn/#/api-docs">
    <img src="https://img.shields.io/badge/FastAPI-Docs-009688?labelColor=gray&logo=fastapi&logoColor=white" alt="MindMemOS FastAPI Docs">
  </a>
  <a href="https://pypi.org/project/mindmemos-sdk/">
    <img src="https://img.shields.io/pypi/v/mindmemos-sdk?color=%2334D058&label=pypi%20sdk&labelColor=gray&logo=pypi&logoColor=white" alt="MindMemOS SDK PyPI version">
  </a>
  <a href="https://www.npmjs.com/package/@mindmemos/openclaw-plugin">
    <img src="https://img.shields.io/npm/v/%40mindmemos%2Fopenclaw-plugin?label=npm%20plugin&labelColor=gray&logo=npm&logoColor=white" alt="MindMemOS OpenClaw Plugin npm version">
  </a>
  <a href="https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin">
    <img src="https://img.shields.io/npm/v/%40mindmemos%2Fdeepseek-harness-plugin?label=dsh%20plugin&labelColor=gray&logo=npm&logoColor=white" alt="MindMemOS DeepSeek Harness Plugin npm version">
  </a>
  <a href="https://arxiv.org/abs/2608.12428">
    <img src="https://img.shields.io/badge/arXiv-2608.12428-B31B1B?labelColor=gray&logo=arxiv&logoColor=white" alt="MindMemOS arXiv paper">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?labelColor=gray" alt="MIT License">
  </a>
</p>

<p align="center">
  <strong><a href="README_ZH.md">简体中文</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="https://mindmemos.cn">Website</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="https://mindmemos.cn/#/api-docs">API Docs</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="https://pypi.org/project/mindmemos-sdk/">PYPI SDK</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="docs/deploy/instruction.md">Deployment Guide</a></strong>
</p>

<p align="center">
  Store and retrieve user and task context across agents; manage Skill versions and improve Skills through tasks and execution trajectories.
</p>

> ⭐ **Star us on GitHub to automatically upgrade to a Pro quota membership.**

## 📰 News

- **2026-08-18**: We released the [DeepSeek Harness Plugin](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin), letting DeepSeek Harness (dsh) agents automatically recall and write MindMemOS memories.
- **2026-08-14**: We released the [MindMemOS 1.0 technical report](https://arxiv.org/abs/2608.12428), *MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents*.
- **2026-07-17**: MindMemOS integrated with [LLM4AD_NEXT](https://github.com/Optima-CityU/LLM4AD_Next), providing searchable long-term memory for algorithm design tasks and enabling the accumulation and reuse of cross-task experience, domain knowledge, and constraints.
- **2026-06-30**: MindMemOS was officially released!

## 🌟 Core Features

- **Portable across agents**: Persist user profiles, preferences, project facts, tool experience, and skill candidates as reusable assets, allowing OpenClaw, Hermes, Claude Code, OpenHands, and other agents to share or transfer the same long-term memory.
- **Self-evolving memory system**: Continuously improve memory quality through schema learning, dreaming, and feedback by automatically learning frequent memory patterns, consolidating memories offline, and using interaction corrections to optimize add/search workflows.
- **Independent Skill service**: Register and manage Skill versions, save execution trajectories, configure TreeSkill or Skill-Forest training, and evolve new versions from trajectories. Use the Python Service directly or deploy its FastAPI interface. Training and evolution require explicit calls and configured models, data, and environments.
- **Local Web workbench**: Connect to Memory and Skill through the SDK to search and edit memories, manage Skill versions, inspect graphs, preview dynamic Skills, and edit connections. Published graph queries, snapshots, and soft deletion are available; graph construction, recommendation, and member updates await full algorithm integration.
- **Plugin integrations**: Connect MindMemOS to different agents and workflows through plugins that retrieve and inject relevant memories before interactions and automatically write conversations back afterward. The [OpenClaw Plugin](https://www.npmjs.com/package/@mindmemos/openclaw-plugin) and [DeepSeek Harness Plugin](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin) are currently available, with more integrations in progress.

### Packages and services

| Package | Capabilities | Access |
| --- | --- | --- |
| `src/mindmemos` | Memory write, search, query, update, deletion, feedback, and consolidation (dreaming) | Memory HTTP API |
| `src/mindmemos_skill` | Skill registration, versions, trajectories, training, evolution, and published graph queries and management | Python Service, local CLI, or Skill HTTP API |
| `src/mindmemos_sdk` | Call Memory and Skill APIs and manage connections and credentials | Python SDK and `mindmemos` CLI |
| `src/mindmemos_eval` | Memory and Skill evaluation, scores, timings, and model usage | `mindmemos-eval` and evaluation configuration |
| `src/mindmemos_ui` | Local Web workbench for Memory and Skill | Browser UI whose backend calls services through the SDK |

Memory and Skill manage their own business data and can be deployed independently. The default local deployment uses Qdrant and Neo4j for Memory and a separate PostgreSQL database for Skill. Direct Skill Python Service integrations can use SQLite. Both APIs can share Kafka and monitoring infrastructure; the UI accesses business data through the SDK.

Compared with MindMemOS 1.0, the current repository adds Skill training and version management, an independent Skill service, a Web UI, shared model settings through `.env`, and switches for Kafka and monitoring. See the [deployment guide](docs/deploy/instruction.md) for setup details.

## 🚀 Quick Start

Use Memory through the official cloud service or a local deployment, with HTTP, Python SDK / CLI, or agent plugins. The table below lists Memory connections. The independent local Skill API defaults to `http://127.0.0.1:8001` and has its own SDK connection settings.

| Access Method | Use Case | Cloud base_url | Local base_url |
| :--- | :--- | :--- | :--- |
| [HTTP API](https://mindmemos.cn/api-docs) | Call directly from business apps | `https://mindmemos.cn` | `http://127.0.0.1:8000` |
| [Python SDK / CLI](https://pypi.org/project/mindmemos-sdk/) | Integrate into business apps | `https://mindmemos.cn` | `http://127.0.0.1:8000` |
| [OpenClaw Plugin](https://www.npmjs.com/package/@mindmemos/openclaw-plugin) | Agent auto-recalls / writes memory | `https://mindmemos.cn` | `http://127.0.0.1:8000` |
| [DeepSeek Harness Plugin](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin) | Agent auto-recalls / writes memory (dsh) | `https://mindmemos.cn` | `http://127.0.0.1:8000` |

To try memory features without deployment, request a cloud API key on the [website](https://mindmemos.cn). For self-hosting, follow the steps below. For Skill alone, deploy the Skill API independently or call its Python Service using the [local Skill guide](docs/skill/user-guide.md). Prepare model, dataset, and environment dependencies for your use case.

### 1. Local Deployment

MindMemOS uses `uv` to manage dependencies and run local commands. For detailed configuration instructions, see [docs/deploy/instruction.md](docs/deploy/instruction.md).

#### 1.1 Prepare Configuration Files

Install workspace dependencies, create `.env`, fill model connections, and initialize configuration. Skip the copy step if `.env` already exists:

```bash
uv sync --all-packages --extra postgres --extra kafka
cp .env.example .env
# Edit .env: Chat, Embedding, and optional Rerank settings
make init
```

Optionally configure Chat, Embedding, and Rerank API keys, base URLs, and model names in `.env` first. `make init` replaces the corresponding fields in all actual configurations under `config/`; Embedding also requires output dimensions. Complete unset fields manually. See [Initialize local configuration](docs/deploy/instruction.md#12-initialize-local-configuration) for variable names and overwrite behavior.

Complete the following configuration before startup:

| Configuration file | Settings to complete |
|---|---|
| `config/mindmemos/dev.yaml` | `model`, `api_base`, and `api_key` for chat, embedding, and optional rerank routers. The embedding output dimension, `dimensions`, and `database.qdrant.vector_size` must match. |
| `config/mindmemos_skill/dev.yaml` | Specify `model`, `base_url`, and `api_key` for graph management, training, and evolution models as needed. Also specify embedding `dimensions`. |
| `config/presets/local.project.yaml` | **Customize your local API key**: edit `projects.<name>.api_key`. `default_project` selects the default UI/SDK project. Override it with `PROJECT=<name>`. Services identify the project from each request’s API key. |

`make dev` generates local authentication configurations and `.mindmemos/settings.json`. You do not need to specify a Skill token manually. Configure model service keys and local API keys separately. Restart the services after changing the project file. For more project and authentication settings, see the [deployment guide](docs/deploy/instruction.md).

#### 1.2 Startup and Management Commands

You need Python 3.11–3.13, Git, Make, uv, and a running Docker engine. Building the UI also requires Node.js/npm. Before using Memory and the UI for the first time, prepare their resources:

```bash
uv run --no-sync python scripts/install_nlp_assets.py
uv run --no-sync python scripts/check_nlp_assets.py
make ui-build
```

`make dev` does not build the UI automatically. Skill includes Python dependencies for its built-in environments; training may also require datasets, agent programs, and system tools. See [environment requirements](docs/mindmemos_skill/benchmark_environments.md).

```bash
make dev                                # Memory + Skill + UI, all features
make dev FEATURE=core                    # Sync Memory, local async Skill; no Kafka or OTel
make dev FEATURE=async                   # Core + async tasks, without OTel
make dev SERVICE=memory                  # Memory + UI, all features
make dev SERVICE=skill FEATURE=core      # Skill + UI, core features
make dev PLAN=1                          # Validate and preview without starting
```

| Command | Purpose |
| --- | --- |
| `make dev` | Start the selected Docker dependencies, APIs and UI; defaults to all features |
| `make docker` | Start only the selected Docker dependencies |
| `make api` | Start APIs against existing dependencies; UI is off by default |
| `make ui` | Start only the UI using its existing SDK configuration |
| `make status` | Show managed application status, addresses, features and the current Compose project's containers |
| `make stop` | Stop managed APIs/UI and remove Compose containers and networks; preserve volumes |
| `make clean` | List volumes, then stop services and delete Docker volumes after typing `DELETE` |
| `make help` | Show commands and options; bare `make` also shows help |

The singular options `SERVICE` and `FEATURE` apply to `dev`, `docker` and `api`:

| Option | Values and default behavior |
| --- | --- |
| `SERVICE` | Defaults to `all` (Memory + Skill); accepts `memory` or `skill` |
| `FEATURE` | Defaults to `all` (Kafka tasks + OTel); `core` disables Kafka and OTel, with sync Memory and local async Skill tasks; `async` enables Kafka without OTel |
| `UI=0` / `UI=1` | Override the UI switch for `dev` or `api`; `docker` cannot enable UI |
| `PLAN=1` | Validate and preview startup without starting services or writing SDK settings |
| `API_PORT` / `SKILL_PORT` / `UI_PORT` | Override ports; defaults are `8000` / `8001` / `8765` |
| `API_HOST` / `SKILL_HOST` | Override API bind addresses; defaults to `127.0.0.1` |
| `MEMORY_CONFIG` / `SKILL_CONFIG` | Select application YAML files; otherwise use the service's configured path |
| `PROJECT` / `PROJECT_CONFIG` | Select the default client project or project file; default file is `config/presets/local.project.yaml` |
| `SERVICE_AUTH=1` | Keep authentication from each service YAML; skip shared local project settings and automatic UI connection setup |
| `UI_HOME` | Select the UI SDK configuration directory; shared local project mode defaults to the repository's `.mindmemos/` |
| `API_RELOAD=1` / `SKILL_RELOAD=1` | Enable code reload for the corresponding API |

Use matching service and feature options when starting dependencies and APIs separately:

```bash
make docker SERVICE=skill FEATURE=async
make api SERVICE=skill FEATURE=async UI=1
```

Application feature switches and container selection come from the same effective configuration. Local dependency ports and credentials are applied from `.env`, so you do not need to edit matching ports in YAML. Source YAML files stay unchanged; private runtime snapshots are removed on normal exit. SQLite and external dependencies do not provision corresponding local database containers. See [Docker configuration](dockers/README.md) for container and configuration details.

Once all selected applications pass their health checks, the launcher reports addresses and async/OTel status. Default addresses:

| Service | Address | API documentation |
| --- | --- | --- |
| Memory | <http://127.0.0.1:8000> | <http://127.0.0.1:8000/docs> |
| Skill | <http://127.0.0.1:8001> | <http://127.0.0.1:8001/docs> |
| UI | <http://127.0.0.1:8765> | — |

Check that the selected services respond:

```bash
curl -f http://127.0.0.1:8000/healthz
curl -f http://127.0.0.1:8001/health
make status
```

Then make a business request to verify model connections. When running one service, check only its address.

`Ctrl+C` stops the current APIs/UI and leaves containers running. `make stop` stops this workspace's managed applications and the current Compose project's containers. Stop manually launched applications separately. Selecting fewer features does not stop existing extra containers; run `make stop` first for a complete switch.

`make clean` deletes Docker volumes holding database, Kafka and monitoring data. It preserves host SQLite files, YAML, `.env` and SDK settings. Restart after changing `.env` or source YAML.

#### 1.3 Development Tools

| Command | Purpose |
| --- | --- |
| `make setup` | Sync dependencies, install Git hooks, and prepare and check NLP assets |
| `make hook` | Install Git pre-commit hooks |
| `make lint` | Check Python code without changing files |
| `make format` | Fix lint issues and format Python code |
| `make ui-build` | Install frontend dependencies and build UI assets; requires Node.js/npm |
| `make profile` | Record a Memory API CPU flame graph; use `PID` to select a process |

Profiling options default to `PROFILE_SECONDS=60`, `PROFILE_RATE=100` and `PROFILE_OUT=tmp/profile.svg`. Set `UV` to override the uv executable.

### 2. Access Methods

Cloud and local self-hosting use the same access protocol. Local keys come from `config/presets/local.project.yaml`; cloud keys are obtained from the [website](https://mindmemos.cn).

#### 2.1 HTTP API

HTTP is the base access method — the SDK and plugins also talk HTTP underneath. Once the service is up, first use curl to verify the endpoints work, then wire up your business logic. Define the address and key before calling (pick local or cloud):

```bash
export BASE_URL=http://127.0.0.1:8000   # Local self-host; change to https://mindmemos.cn for cloud
export API_KEY=dev-api-key-001          # Local example key; use a website-issued key for cloud
```

Add a memory:

```bash
curl -sS -X POST "$BASE_URL/v1/memory/add" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "messages": [{"role": "user", "content": "I like iced Americanos."}]
  }'
```

Search memories:

```bash
curl -sS -X POST "$BASE_URL/v1/memory/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What kind of coffee does the user like?", "user_id": "u_123", "top_k": 5}'
```

Specify `user_id` to search the same user's memories; omitting it in HTTP searches the entire project. See the [interface usage guide (Chinese)](docs/interfaces/usage_ZH.md) for filters, reranking, token budgets, and response formats, and the [API docs](https://mindmemos.cn/#/api-docs) for other Memory endpoints.

#### 2.2 Configure the SDK

Install the Python SDK:

```bash
pip install mindmemos-sdk
```

Run the authentication command, choose official or custom deployment, and fill the service addresses, API keys, and default user:

```bash
mindmemos auth
```

For two independent local APIs, specify addresses first and enter credentials interactively:

```bash
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001
```

An independent Skill address requires its own API key. Leaving the Skill address empty reuses the Memory address and credentials. The SDK resolves each connection separately; see the [SDK guide](src/mindmemos_sdk/README.md).

| Setting | Local Service | Cloud Service |
| :--- | :--- | :--- |
| `base_url` | `http://127.0.0.1:8000` | `https://mindmemos.cn` |
| `api_key` | An enabled project API key from `config/presets/local.project.yaml` | An API key obtained from the [MindMemOS website](https://mindmemos.cn) |
| `user_id` | A stable identifier for the current end user, such as `u_123` | A stable identifier for the current end user, such as `u_123` |

The configuration is saved to `~/.mindmemos/settings.json`. Check the current configuration with:

```bash
mindmemos config show
```

The local service automatically determines the `project_id` and memory algorithm from the API key, so SDK calls do not need to pass `project_id`. The `user_id` distinguishes users within the same project and can be overridden in an individual `add` or `search` call.

If you prefer not to use the local configuration file, pass connection parameters explicitly when creating the client:

```python
from mindmemos_sdk import MindMemOSClient

with MindMemOSClient(
    base_url="http://127.0.0.1:8000",
    api_key="<api_key>",
    user_id="u_123",
) as client:
    ...
```

Explicit parameters take precedence over values in `~/.mindmemos/settings.json`.

#### 2.3 Add and Search Memories with the SDK

After completing the configuration above, `MindMemOSClient()` automatically reads the service address, API key, and default `user_id`. The SDK adds the authentication header automatically, so there is no need to construct HTTP requests manually:

```python
from mindmemos_sdk import DialogueMessage, MindMemOSClient

with MindMemOSClient() as client:
    add_result = client.memory.add(
        messages=[
            DialogueMessage(
                role="user",
                content="I like iced Americanos.",
            )
        ],
        mode="sync",
    )

    for item in add_result.memories:
        print(item.operation, item.memory_id, item.content)

    search_result = client.memory.search(
        "What kind of coffee does the user like?",
        top_k=5,
        search_strategy="fast",
        # token_budget=2000,  # optional: strict token budget (enables retention)
    )

    for memory in search_result.memories:
        print(memory.id, memory.memory)
```

Skill registration, version commits, and evolution use dedicated request models. Evolution uses a specific version and saved trajectories and returns a background task. See the [SDK guide](src/mindmemos_sdk/README.md) for current examples and the [local Skill guide](docs/skill/user-guide.md) for Python Service training and custom environments.

Use `mindmemos auth` to update service connections and credentials when switching deployments.

#### 2.4 Use the CLI

After running `mindmemos auth`, you can also add and search memories directly with the CLI included in the SDK:

```bash
mindmemos memory add --content "I like iced Americanos"
mindmemos memory search "coffee preferences" --top-k 5
```

The CLI uses the SDK default user when `--user-id` is omitted. Add `--user-id u_123` to match the HTTP example. Search parameters and output are documented in the [interface usage guide (Chinese)](docs/interfaces/usage_ZH.md).

The `memory` subcommand also supports get / update / delete / feedback / dreaming. The `skill` subcommand supports register / commit / export / history / diff / activate and train / evolve / task / trajectory / graph. Check `mindmemos skill --help` for current commands; see the [SDK guide](src/mindmemos_sdk/README.md) and [CLI guide](docs/cli/instruction.md) for details.

#### 2.5 OpenClaw Plugin

**Installing via our [`mindmemos-cli` skill](skills/mindmemos-cli/SKILL.md) is recommended**: deploy `skills/mindmemos-cli/` to your agent's skills directory and let the agent follow its instructions. The skill's [reference docs](skills/mindmemos-cli/references/openclaw-plugin.md) cover installation, permissions, and common troubleshooting.

<details>
<summary><b>Manual installation (not recommended)</b></summary>

**First install the SDK and complete `auth` configuration (required)**: the plugin communicates with the local machine through the `mindmemos` CLI, so you must install the Python SDK first and make sure the `mindmemos` command is available:

```bash
pip install mindmemos-sdk    # or: uv add mindmemos-sdk
mindmemos --version          # confirm the command is available
```

Then configure `base_url`, API key, and `user_id` with `mindmemos auth` (pointing at either the cloud or a local service):

```bash
mindmemos auth
mindmemos config show        # confirm the configuration took effect
```

> Skipping these two steps before installing the plugin causes the logs to error out (`mindmemos` command not found / auth not configured), and the plugin will not be able to read or write memories properly.

Install and enable the plugin:

```bash
openclaw plugins install @mindmemos/openclaw-plugin
openclaw plugins enable mindmemos-memory
```

(`@mindmemos/openclaw-plugin` is the npm package name; `mindmemos-memory` is the plugin id.) Manual installation easily runs into two pitfalls:

- **Write permission (required)**: the plugin's `agent_end` write hook needs `allowConversationAccess`; otherwise everything looks fine, but memories are never actually stored after a turn:
  ```bash
  openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
  openclaw gateway restart
  ```
- **`cli` PATH**: an OpenClaw process launched from the GUI does not inherit your terminal PATH. Configure `mindmemos` as an absolute path or wrap it with `uv run mindmemos`, or the logs will report `ENOENT`.

Once installed, enabled, and the gateway restarted, the plugin recalls and injects relevant memories before each user turn and writes the conversation back automatically when the turn ends.

Full commands, configuration options, and troubleshooting are in the [OpenClaw plugin integration docs](skills/mindmemos-cli/references/openclaw-plugin.md).

</details>

#### 2.6 DeepSeek Harness Plugin

**Installing via our [`mindmemos-cli` skill](skills/mindmemos-cli/SKILL.md) is recommended**: deploy `skills/mindmemos-cli/` to your agent's skills directory and let the agent follow its instructions. The skill's [reference docs](skills/mindmemos-cli/references/deepseek-harness-plugin.md) cover installation and common troubleshooting.

<details>
<summary><b>Manual installation (not recommended)</b></summary>

**First install the SDK and complete `auth` configuration (required)**: the plugin communicates with the local machine through the `mindmemos` CLI, so you must install the Python SDK first and make sure the `mindmemos` command is available:

```bash
pip install mindmemos-sdk    # or: uv add mindmemos-sdk
mindmemos --version          # confirm the command is available
```

Then configure `base_url`, API key, and `user_id` with `mindmemos auth` (pointing at either the cloud or a local service):

```bash
mindmemos auth
mindmemos config show        # confirm the configuration took effect
```

> Skipping these two steps before installing the plugin causes the logs to error out (`mindmemos` command not found / auth not configured), and the plugin will not be able to read or write memories properly.

Install the plugin into a dsh profile (`dsh plugin` forwards to pnpm and installs the package into the profile's `node_modules`):

```bash
dsh plugin --profile <name> add @mindmemos/deepseek-harness-plugin
```

(`@mindmemos/deepseek-harness-plugin` is the npm package name; `mindmemos-memory` is the plugin id.) dsh composes plugins through layered `cordis.patch.yml` files, so register the plugin by adding an `insert` entry to the profile patch (`~/.dsh/profiles/<name>/cordis.patch.yml`):

```yaml
- insert:
    - id: mindmemos-memory
      name: '@mindmemos/deepseek-harness-plugin'
      config:
        userId: alice
        appId: deepseek-harness
```

Restart dsh with that profile. Once registered, the plugin recalls and injects relevant memories before each user turn and writes the conversation back automatically when the turn ends.

Full commands, configuration options, and troubleshooting are in the [DeepSeek Harness plugin integration docs](skills/mindmemos-cli/references/deepseek-harness-plugin.md).

</details>

## 📊 Benchmark

### Dataset downloads

Run `scripts/download_datasets.py` from the repository root to list datasets and download pinned releases for either benchmark group.

```bash
uv sync --all-packages
uv run --no-sync python scripts/download_datasets.py list memory
uv run --no-sync python scripts/download_datasets.py list skill
uv run --no-sync python scripts/download_datasets.py download memory --name locomo
uv run --no-sync python scripts/download_datasets.py download skill --name spreadsheetbench
```

Memory datasets include `locomo`, `personamem` (v1), `memoryagentbench` (all four official partitions),
and `longmemeval` (official cleaned release).
Downloads go to `data/mindmemos_memory/<dataset>/` or `data/mindmemos_skill/<dataset>/`;
use `--data-root` to change the parent directory and `--verify-only` on `download` to verify existing files.
Set evaluation config paths to these downloaded files. Official formats are preserved; MemoryAgentBench
remains Parquet and requires conversion for the current evaluator's local JSON/JSONL input.
Experiment split manifests are maintained in the repository and are not downloaded or generated.


### 💬 Conversational Memory: LoCoMo

- **Benchmark**: [LoCoMo](https://arxiv.org/abs/2402.17753), a mainstream benchmark for long-conversation memory covering single-hop, multi-hop, temporal, and open-domain question answering.

| Method                       | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
| :--------------------------- | :--------: | :-------: | :------: | :---------: | :-----: |
| Mem0                         | 68.97 | 61.70 | 58.26 | 50.00  | 64.20 |
| MemU                         | 74.91 | 72.34 | 43.61 | 54.17  | 66.67 |
| MemOS                        | 85.37 | 79.43 | 75.08 | 64.58  | 80.76 |
| Zep                          | 90.84 | 81.91 | 77.26 | 75.00  | 85.22 |
| EverOS                       | 96.67 | 91.84 | 89.72 | 76.04  | 93.05 |
| **MindMemOS-MindVanilla**    | 92.03 | 85.82 | 83.80 | 66.67  | 87.60 |
| **MindMemOS-MindSchema**     | **96.79** | **93.97** | **90.34** | **82.29** | **94.03** |

### 👤 User Profile Memory: PersonaMem

- **Benchmark**: [PersonaMem](https://arxiv.org/abs/2504.14225), a memory benchmark centered on user profiles and preference understanding that evaluates recall, tracking, revisiting, suggestion, recommendation, and generalization of user traits.

| Method | Recall | Ack. Lat. | Trk. Evo. | Revisit | Suggest | Recom. | General. | Overall |
| :----- | :----: | :-------: | :-------: | :-----: | :-----: | :-----: | :------: | :-----: |
| Mem0 | 46.51 | 41.18 | 65.47 | 90.91 | 12.90 | 34.55 | 43.86 | 51.61 |
| MemU | 64.34 | 64.71 | 66.20 | 87.88 | 31.18 | 67.27 | 84.21 | 65.70 |
| MemOS | 53.49 | 82.35 | 66.91 | 79.80 | 41.94 | 69.09 | 75.44 | 63.67 |
| EverOS | 74.42 | 64.71 | 64.03 | 85.86 | 35.48 | 65.45 | 84.21 | 67.57 |
| **MindMemOS-MindVanilla** | 76.74 | 88.24 | 65.47 | 87.88 | 17.20 | 80.00 | 82.46 | 67.74 |
| **MindMemOS-MindSchema** | 81.40 | 64.71 | 64.75 | 82.83 | 47.31 | 76.36 | 73.68 | 70.63 |

### 🌙 Memory Consolidation: MemoryAgentBench (FactConsolidation)

- **Benchmark**: [MemoryAgentBench](https://arxiv.org/abs/2507.05257) FactConsolidation. Scores in the table are the average Substring Exact Match across four context sizes.

| Method                             | SH score | SH archived | MH score | MH archived |
|:-----------------------------------| :------: | :---------: | :------: | :---------: |
| **GPT-4o-mini**                    |  |  |  |  |
| Mem0                               | 0.180 | — | 0.020 | — |
| MemoRAG                            | 0.270 | — | 0.070 | — |
| HippoRAG-v2                        | 0.540 | — | 0.050 | — |
| MindMemOS-MindVanilla              | 0.635 | — | 0.118 | — |
| **MindMemOS-MindVanilla + Dreaming** | 0.738 | 21.4% | 0.180 | 19.4% |
| **GPT-5-mini**                     |  |  |  |  |
| Infini Memory                      | 0.800 | — | 0.220 | — |
| MindMemOS-MindVanilla              | 0.900 | — | 0.190 | — |
| **MindMemOS-MindVanilla + Dreaming** | 0.920 | 23.5% | 0.250 | 21.5% |

### Skill Benchmark (results pending)

The current Skill service has no complete benchmark results ready for publication. A shared Skill-Forest evaluation entry point covers LiveMath, ALFWorld, SpreadsheetBench, SearchQA, and OfficeQA, comparing `no_skill`, `initial_skill`, and `trained_skill` on fixed splits. Success rates, timings, and token comparisons will be added when results are ready.

After preparing dependencies, model settings, and data, start with a smoke test:

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
uv run --no-sync mindmemos-eval skill evaluate train --algo skill-forest --smoke
```

Model and concurrency settings live in `config/mindmemos_eval/skill_train_skill_forest/dev.yaml`, created by `make init`. Skill-Forest evaluation creates an isolated local service in the evaluation process; it does not require a Memory API or Skill HTTP API. A smoke test checks a small workflow and does not establish benchmark performance. See the [Skill evaluation guide](docs/evaluation/skill_evaluation.md#skill-forest) for full commands, environment requirements, and reports.

## Documentation

| Goal | Guide |
| --- | --- |
| Deploy services, configure models, and select features | [Deployment](docs/deploy/instruction.md) |
| Check Memory search HTTP / CLI parameters and results | [Interface usage (Chinese)](docs/interfaces/usage_ZH.md) |
| Use the full CLI, configure connections, and troubleshoot | [CLI guide](docs/cli/instruction.md) |
| Integrate the Python SDK and independent Skill API | [SDK guide](src/mindmemos_sdk/README.md) |
| Manage, train, and evolve local Skills or add an environment | [Local Skill guide](docs/skill/user-guide.md) |
| Use the local Web workbench | [UI guide](docs/mindmemos_ui/usage.md) |
| Download datasets and reproduce Memory or Skill evaluations | [Memory evaluation](docs/evaluation/memory_evaluation.md) · [Skill evaluation](docs/evaluation/skill_evaluation.md) |

## 🗺️ Coming Features

- **Simpler deployment**: Further reduce dependencies and improve embedded use, building on the existing Kafka / OTel switches, local asynchronous Skill tasks, and SQLite backend.
- **Skill graph organization and recommendation**: Integrate complete graph construction, recommendation, and member management algorithms for larger Skill libraries.
- **File system memory**: Structure scattered knowledge from local files, documents, project artifacts, and agent outputs into searchable and connected file knowledge objects or knowledge graphs, helping agents complete user tasks more effectively.
- **Agent integrations**: Continue expanding support for coding agents, OpenClaw, Codex-style workflows, and long-running multi-agent systems.

## Contributing

Contributions of all kinds are welcome. Please open pull requests against the `develop` branch. After review,
accepted changes will be merged into `develop`; maintainers periodically merge stable `develop` updates into `main`
for release.

## 💬 Community

Join the MindMemOS Feishu group for project updates, usage discussions, and community participation.

<p align="center">
  <img src="./assets/feishu-group-small.png" alt="MindMemOS Feishu group QR code">
</p>

## 📝 Citation

If you find MindMemOS useful in your research, please cite our technical report:

```bibtex
@misc{liang2026mindmemos,
  title        = {MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents},
  author       = {Liang, Kaichao and Cui, Yuqi and Kong, Hao and Huang, Xinyuan and Hou, Guohaotian and Kang, Qingcan and Chen, Liang and Yin, Yiyang and Ye, Ke and Guo, Jiaquan and Chen, Da and Zeng, Lingan and Peng, Yixing and Yao, Rong and Kai, Shixiong and Yuan, Mingxuan},
  year         = {2026},
  eprint       = {2608.12428},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2608.12428},
}
```

## 📄 License

This project is open source under the MIT License.
