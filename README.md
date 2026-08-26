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
  Accurately remember user and task context and reuse it across agents; evolve memory through ongoing interactions, automatically distill Skills, and connect with file-based knowledge systems so experience truly becomes capability.
</p>

> ⭐ **Star us on GitHub to automatically upgrade to a Pro quota membership.**

## 📰 News

- **2026-07-17**: MindMemOS integrated with [LLM4AD_NEXT](https://github.com/Optima-CityU/LLM4AD_Next), providing searchable long-term memory for algorithm design tasks and enabling the accumulation and reuse of cross-task experience, domain knowledge, and constraints.
- **2026-06-30**: MindMemOS was officially released!

## 🌟 Core Features

- **Portable across agents**: Persist user profiles, preferences, project facts, tool experience, and skill candidates as reusable assets, allowing OpenClaw, Hermes, Claude Code, OpenHands, and other agents to share or transfer the same long-term memory.
- **Self-evolving memory system**: Continuously improve memory quality through schema learning, dreaming, and feedback by automatically learning frequent memory patterns, consolidating memories offline, and using interaction corrections to optimize add/search workflows.
- **Memory and Skills integration**: Experience memories can be distilled into skill candidates, while skill execution results, failure traces, and user feedback flow back into the memory system to drive continuous skill evolution.
- **Plugin integrations**: Connect MindMemOS to different agents and workflows through plugins that retrieve and inject relevant memories before interactions and automatically write conversations back afterward. The [OpenClaw Plugin](https://www.npmjs.com/package/@mindmemos/openclaw-plugin) is currently available, with more integrations in progress.

<p align="center">
  <img src="./assets/mindmemos-benchmark-overview.png" alt="MindMemOS benchmark results overview">
</p>

## 🚀 Quick Start

### 1. Local Deployment

MindMemOS uses `uv` to manage dependencies and run local commands. For detailed configuration instructions, see [docs/deploy/instruction.md](docs/deploy/instruction.md).

#### 1.1 Prepare Configuration Files

```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
```

Before startup, configure at least the following three model routers in `config/mindmemos/dev.yaml`:

- `chat_model_router`: supports memory extraction, Skill evolution, and other generation tasks.
- `embed_model_router`: generates semantic embeddings; make sure its dimensions match the Qdrant dimension configuration.
- `rerank_model_router`: optional; reranks memory retrieval results.

Configure an API key and its bound `project_id` in `config/mindmemos/api_keys.yaml`.

#### 1.2 Start the Service

Start the local service:

```bash
make dev
```

`make dev` starts the full Docker dependency stack before starting FastAPI.

To start only core dependencies:

```bash
make dev-core          # Qdrant + Neo4j + Kafka
make db-observability  # Qdrant + Neo4j + Kafka + ClickHouse + OTel + Grafana
```

The default local service port is 8000:

```text
FastAPI:   http://127.0.0.1:8000
```

Stop the local service:

```bash
make dev-down
```

#### 1.3 Configure the SDK

Install the Python SDK:

```bash
pip install mindmemos-sdk
```

Run the authentication command and configure the service address, API key, and default user when prompted:

```bash
mindmemos auth
```

| Setting | Local Service | Cloud Service |
| :--- | :--- | :--- |
| `base_url` | `http://127.0.0.1:8000` | `https://mindmemos.cn` |
| `api_key` | An enabled API key from `config/mindmemos/api_keys.yaml` | An API key obtained from the [MindMemOS website](https://mindmemos.cn) |
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

#### 1.4 Add and Search Memories with the SDK

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

Trigger cloud evolution for a registered Skill:

```python
from mindmemos_sdk import MindMemOSClient

with MindMemOSClient() as client:
    result = client.skills.evolve("my-skill", mode="sync")

    print("evolved:", result.evolved)
    print("pending:", result.pending_count)
    print("threshold:", result.threshold)
    print("new versions:", result.new_version_ids)
```

Local and cloud services use the same SDK call pattern. To switch between them, reconfigure only the `base_url` and corresponding API key.

#### 1.5 Use the CLI

After running `mindmemos auth`, you can also add and search memories directly with the CLI included in the SDK:

```bash
mindmemos memory add --content "I like iced Americanos"
mindmemos memory search "coffee preferences" --top-k 5
```

The CLI can also view, update, and delete memories, submit feedback, or trigger Dreaming:

```bash
mindmemos memory get --top-k 10  # View memories
mindmemos memory update <memory_id> --content "I now prefer lattes"  # Update a memory
mindmemos memory delete <memory_id>  # Delete a memory
mindmemos memory feedback --text "The preference retrieved just now was inaccurate" \
  --messages-json '[{"role":"user","content":"The preference retrieved just now was inaccurate"}]'  # Submit explicit feedback
mindmemos memory feedback  # Submit implicit feedback
mindmemos memory dreaming  # Consolidate memories
```

Use the Skill CLI to register a local Skill and manage it later through the alias set during registration:

```bash
mindmemos skill register ./path/to/skill --alias my-skill
mindmemos skill list
mindmemos skill show my-skill
```

Skill Evolution uses synchronous mode by default. You can also enqueue the evolution task asynchronously:

```bash
mindmemos skill evolve my-skill
mindmemos skill evolve my-skill --async
```

After modifying a local Skill, push it as a new version. You can also retrieve cloud version information and update local files:

```bash
mindmemos skill push my-skill
mindmemos skill pull my-skill
mindmemos skill update my-skill
mindmemos skill update --all
```

`pull` retrieves only cloud version metadata and does not modify local files. `update` first shows an update plan and applies it after confirmation. Use the following commands to view version history, compare versions, or roll back:

```bash
mindmemos skill history my-skill
mindmemos skill diff my-skill --to <version_id>
mindmemos skill rollback my-skill --to <version_id>
```

When a Skill no longer needs to be managed by the SDK, unregister it. Local Skill files are preserved by default:

```bash
mindmemos skill unregister my-skill
```

## 📊 Benchmark

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

### 🧠 Skill Self-Evolution: SpreadsheetBench-Verified

- **Benchmark**: [SpreadsheetBench-Verified](https://huggingface.co/datasets/KAKA22/SpreadsheetBench/blob/main/spreadsheetbench_verified_400.tar.gz), a 400-task verified subset of SpreadsheetBench covering diverse real-world spreadsheet operations.

| Method | Success Rate | Time / Task (s) | Agent Tokens | Evolve Tokens |
| :----- | :----------: | :-------------: | :----------: | :-----------: |
| No-skill | 51.3% ± 0.8% | 11.227 | 10.4M | - |
| Init-skill | 48.0% ± 1.4% | 15.350 | 16.9M | - |
| **MindMemOS-MindEvolve-Unsup.** | **55.3% ± 0.9%** | 15.470 | 27.3M | 5.8M |
| **MindMemOS-MindEvolve-Sup.** | **57.2% ± 2.4%** | 15.631 | 25.2M | 5.5M |

## 🗺️ Coming Features

- **Lite mode**: Designed around low dependencies, replaceable components, and easy embedding, with database backends, async tasks, and log storage decoupled into flexible lightweight components that support in-memory calls and simplified deployment.
- **Skills system**: Govern large and redundant skill libraries and distribute them intelligently; continuously evolve and optimize skills based on real usage; automatically synthesize new skills from frequent user scenarios and refine them through offline simulation.
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

## 📄 License

This project is open source under the MIT License.
