# MindMemOS

**Give AI agents long-term memory and Skills they can reuse and improve.**

<p align="center">
  <img src="./assets/mindmemos-readme-hero.png" alt="MindMemOS — Memory &amp; Skills for AI Agents That Keep Evolving">
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

> ⭐ **Star us on GitHub to automatically upgrade to a Pro quota membership.**

> **Skill SDK and plugin compatibility**
>
> - **Stay on `/v1/skills`.** Keep your current API key and `mindmemos-sdk` `0.1.5`. If you use a plugin, keep OpenClaw `0.1.5` or DeepSeek Harness `0.1.3`.
> - **Switch to `/v2/skills`.** Upgrade to `mindmemos-sdk` `1.0.0+` and, if you use a plugin, upgrade it to `0.2.0+` at the same time. [Enable Skill access on your API key](#grant-a-project-api-key-access-to-the-new-skill-service).

## 📰 News

- **2026-11-09** — We officially released **MindMemOS-Skill**. It is an independently deployable service for managing Skill versions, training and evolving Skills, building Skill graphs, and recommending the best Skill combinations. [Quick start](#1-create-an-account-and-an-api-key)
- **2026-08-18**: We released the [DeepSeek Harness Plugin](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin), letting DeepSeek Harness (dsh) agents automatically recall and write MindMemOS memories.
- **2026-08-14**: We released the [MindMemOS 1.0 technical report](https://arxiv.org/abs/2608.12428), *MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents*.
- **2026-07-17**: MindMemOS integrated with [LLM4AD_NEXT](https://github.com/Optima-CityU/LLM4AD_Next), providing searchable long-term memory for algorithm design tasks and enabling the accumulation and reuse of cross-task experience, domain knowledge, and constraints.
- **2026-06-30**: MindMemOS was officially released!

## 🌟 Core Features

- **Portable across agents**: Persist user profiles, preferences, project facts, tool experience, and skill candidates as reusable assets, allowing OpenClaw, Hermes, Claude Code, OpenHands, and other agents to share or transfer the same long-term memory.
- **Self-evolving memory system**: Continuously improve memory quality through schema learning, dreaming, and feedback by automatically learning frequent memory patterns, consolidating memories offline, and using interaction corrections to optimize add/search workflows.
- **Improve Skills with tasks and past trajectories**. Use MindMemOS-Skill to register and version Skills, save agent execution trajectories, and score both Skills and trajectories. Build and manage Skill graphs to recommend Skill combinations for a given task. Train Skills on tasks with TreeSkill or Skill-Forest, or use saved trajectories to evolve a specific version offline. Deploy the Skill API independently or call the Python service directly.
- **Local Web workbench**: Connect to Memory and Skill through the SDK to search and edit memories, manage Skill versions, inspect graph snapshots, preview dynamic Skills, and edit connections.
- **Plugin integrations**: Connect MindMemOS to different agents and workflows through plugins that retrieve and inject relevant memories before interactions and automatically write conversations back afterward. The [OpenClaw Plugin](https://www.npmjs.com/package/@mindmemos/openclaw-plugin) and [DeepSeek Harness Plugin](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin) are currently available, with more integrations in progress.

- **Multimodal memory**. Vanilla Memory accepts text and images for add and search and stores originals in S3 or MinIO. Enhanced mode adds visual fact extraction and multimodal agentic search. The default mode is text-only.

### Packages and services

![The Web Workbench uses the SDK, which connects to independent Memory and Skill services. Memory evaluation also uses the SDK.](./assets/mindmemos-service-architecture.png)

| Package | Capabilities | Access | Backend dependencies |
| --- | --- | --- | --- |
| `src/mindmemos` | Memory read, write, search, feedback, and consolidation | Memory HTTP API | Qdrant **and** Neo4j, S3 **or** MinIO (for multimodal memory)<br>Kafka (async tasks)<br>OpenTelemetry Collector + ClickHouse + Grafana (optional monitoring) |
| `src/mindmemos_skill` | Skill registration, versions, trajectories, training, evolution, and graphs | Python Service, local CLI, or Skill HTTP API | PostgreSQL (pgvector) **or** SQLite (sqlite-vec)<br>In-process **or** Kafka (async tasks)<br>OpenTelemetry Collector + ClickHouse + Grafana (optional monitoring) |
| `src/mindmemos_sdk` | API calls, connections, and credentials | Python SDK or `mindmemos` CLI | Memory API and Skill API as needed |
| `src/mindmemos_eval` | Memory and Skill evaluation | `mindmemos-eval` | Memory evaluation uses the Memory API, Qdrant and Neo4j (cleanup), and ClickHouse (usage metrics)<br>Skill evaluation uses the in-process Skill service and SQLite |
| `src/mindmemos_ui` | Local Memory and Skill workbench | Browser | Memory API and Skill API via the SDK as needed |

## 🚀 Quick Start

### 1. Create an account and an API key

[Create a MindMemOS account](https://mindmemos.cn/#/register) and sign in to the console. Open or create a project. For a first-time setup, open **API Keys**, choose **Create API key**, and select the project and required scopes. Save the full key after creating it.

#### Grant a project API key access to the new Skill service

1. Sign in to the [MindMemOS console](https://mindmemos.cn/) and open **API Keys**. Find the key your SDK or plugin uses by its name and prefix, then edit its permissions.
2. Set **Skill** to **Read and write** to view and register Skills, write new versions, and report execution trajectories. Choose **Read only** if you only need to view Skills and versions. Keep the **Memory** selections you use.
3. Save the permissions and confirm the Skill access level in the key's **Scopes** column.
4. Rerun `mindmemos auth` with the authorized key, then use the examples below to verify Skill read and write access. [Use the CLI](#2-use-the-cli)

### 2. Use the CLI

With Python 3.11–3.13 and uv installed, run these commands from the repository root. Replace `alice` with your application's user ID and enter your project API key when prompted. The CLI, SDK, and agent plugins share this configuration.

```bash
uv sync --package mindmemos-sdk --no-dev --inexact
source .venv/bin/activate
mindmemos auth --service-type official --user-id alice
```

**Memory · Add, search, and list memories**

```bash
mindmemos memory add --content "I like iced Americanos."
mindmemos memory search "coffee preferences" --top-k 3
mindmemos memory get --top-k 5
```

To update or delete the test memory, replace `MEMORY_ID` with its ID from the `memory get` output.

```bash
mindmemos memory update MEMORY_ID --content "I like unsweetened iced Americanos."
mindmemos memory delete MEMORY_ID
```

**Skill · Register, list, view, and export Skills**

Use a project API key with Skill read and write access. Create a sample `SKILL.md`, then view and export the registered Skill by alias.

```bash
skill_dir=$(mktemp -d)
cat > "$skill_dir/SKILL.md" <<'EOF'
---
name: meeting-action-items
description: Extract action items from meeting notes.
---
List each action item with its owner and deadline.
EOF

mindmemos skill register "$skill_dir/SKILL.md" --alias meeting-action-items
mindmemos skill list --limit 5
mindmemos skill show meeting-action-items
mindmemos skill export meeting-action-items --to "$skill_dir/export"
```

See the [CLI reference](docs/cli/instruction.md) for more commands.

### 3. Connect the Python SDK

After authenticating in step 2, use `python` in the same terminal to run either example.

**Memory — Store and search a memory**

```python
from mindmemos_sdk import DialogueMessage, MindMemOSClient

user_id = "alice"

with MindMemOSClient(user_id=user_id) as client:
    client.memory.add(
        messages=[DialogueMessage(role="user", content="I like iced Americanos.")],
        mode="sync",
    )
    result = client.memory.search("coffee preferences", user_id=user_id, top_k=3)
    for memory in result.memories:
        print(memory.memory)
```

**Skill — Register a Skill and retrieve its first version**

Use a project API key with Skill read and write access.

```python
from mindmemos_sdk import MindMemOSClient
from mindmemos_sdk.skills import RegisterRequest, ShowRequest, StructuredSkillInfo

skill_content = """---
name: meeting-summary
description: Summarize meeting notes into decisions and action items.
---
List decisions first. Then list each action item with its owner and deadline.
"""

with MindMemOSClient() as client:
    created = client.skills.register(
        RegisterRequest(skill=StructuredSkillInfo(content=skill_content, commit_version="v1.0.0"))
    )
    version_id = created["skill_version_id"]
    detail = client.skills.show(ShowRequest(skill_version_id=version_id))
    print(version_id)
    print(detail["version"]["content"])
```

See the SDK guide to [register and version Skills](src/mindmemos_sdk/README.md#python-sdk) or [report one execution record to both APIs](src/mindmemos_sdk/README.md#统一上报执行轨迹).

### 4. Connect OpenClaw

Building the plugins requires Node.js and npm. With OpenClaw installed, run the following from the repository root. The gateway uses the CLI you authenticated in step 2.

```bash
(cd plugins/openclaw-plugin && npm ci && npm run build)
openclaw plugins install --link "$PWD/plugins/openclaw-plugin"
openclaw plugins enable mindmemos-memory
openclaw config set plugins.entries.mindmemos-memory.config.cli "$PWD/.venv/bin/mindmemos"
openclaw config set plugins.entries.mindmemos-memory.config.userId alice
openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
openclaw gateway restart
openclaw skills list
```

The plugin recalls from **Memory** before the model runs and writes completed conversations back. OpenClaw lists the bundled `mindmemos-cli` guide in its Skill catalog. The agent can use the guide to find and load remote Skills. The plugin sends execution trajectories to **Skill** and links the exact versions it detects. See the [OpenClaw guide](skills/mindmemos-cli/references/openclaw-plugin.md).

### 5. Connect DeepSeek Harness

Use a dsh profile with the `agents` and `skills` services. Build from the repository root.

```bash
(cd plugins/deepseek-harness-plugin && npm ci && npm run build)
```

Add this entry to the profile's `cordis.patch.yml`. Replace `/path/to/MindMemOS` with the absolute checkout path, then restart dsh with that profile.

```yaml
- insert:
    - id: mindmemos-memory
      name: 'file:///path/to/MindMemOS/plugins/deepseek-harness-plugin/dist/index.js'
      config:
        cli: /path/to/MindMemOS/.venv/bin/mindmemos
        userId: alice
```

The plugin recalls from **Memory** before the first model step and writes completed conversations back. It registers `mindmemos-cli` in the dsh Skill catalog so the agent can find and load remote Skills. The plugin sends execution trajectories to **Skill** and links the exact versions it detects. See the [DeepSeek Harness guide](skills/mindmemos-cli/references/deepseek-harness-plugin.md).

For both plugins, make `mindmemos` available in the agent's shell and use the same SDK settings directory as in step 2. This also applies when you set a custom `MINDMEMOS_HOME`. After a test turn, check the plugin log for `Memory queued` or `Memory completed`. With a Skill-authorized key, also check for `Skill stored` with the same event ID. See how to [load and report exact Skill versions](skills/mindmemos-cli/SKILL.md#recommend-and-load-a-skill).

### Self-host Memory and Skill

Run the following from the repository root with Python 3.11–3.13, uv, Make, and Docker Compose installed.

```bash
uv sync --all-packages --extra postgres --extra kafka
export UV_NO_SYNC=1
test -f .env || cp .env.example .env
# Set your chat and embedding model connections in .env.
make init
```

Set a local API key in `config/presets/local.project.yaml` for a project with `memory:read`, `memory:write`, `skill:read`, and `skill:write`. Complete the service-specific settings in the [deployment guide](docs/deploy/instruction.md#12-initialize-local-configuration), then start both APIs.

```bash
uv run --no-sync python scripts/install_nlp_assets.py
uv run --no-sync python scripts/check_nlp_assets.py
make dev SERVICE=all FEATURE=async UI=0
```

This starts [Memory](http://127.0.0.1:8000/docs) on port 8000 and [Skill](http://127.0.0.1:8001/docs) on port 8001. The async profile enables the queue used by the plugins' default Memory writes. In a second terminal, configure the SDK for both local endpoints. Enter your local project API key at both prompts.

- `--memory-base-url` takes the service address before `/v1/memory`. The SDK adds the Memory path.
- `--skill-base-url` takes the full Skill API root. For a direct connection, include `/v1/skills`. The SDK appends the operation name. Existing configurations that contain only the Skill service address still work.

```bash
uv run --no-sync mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001/v1/skills \
  --user-id alice
```

For the web workbench and other startup options, see the [deployment guide](docs/deploy/instruction.md).

## 📊 Benchmark

Bold marks the highest score for each metric within its comparison group, including ties.

### 🧠 Memory Benchmarks

#### LoCoMo

Long-conversation question answering across single-hop, multi-hop, temporal, and open-domain tasks. [LoCoMo paper](https://arxiv.org/abs/2402.17753)

| Method | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mem0 | 68.97 | 61.70 | 58.26 | 50.00 | 64.20 |
| MemU | 74.91 | 72.34 | 43.61 | 54.17 | 66.67 |
| MemOS | 85.37 | 79.43 | 75.08 | 64.58 | 80.76 |
| Zep | 90.84 | 81.91 | 77.26 | 75.00 | 85.22 |
| EverOS | 96.67 | 91.84 | 89.72 | 76.04 | 93.05 |
| MindMemOS-MindVanilla | 92.03 | 85.82 | 83.80 | 66.67 | 87.60 |
| MindMemOS-MindSchema | **96.79** | **93.97** | **90.34** | **82.29** | **94.03** |

#### PersonaMem

Accuracy by question type for user profiles and preferences. [PersonaMem paper](https://arxiv.org/abs/2504.14225)

| Method | Recall | Ack. Lat. | Trk. Evo. | Revisit | Suggest | Recom. | General. | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mem0 | 46.51 | 41.18 | 65.47 | **90.91** | 12.90 | 34.55 | 43.86 | 51.61 |
| MemU | 64.34 | 64.71 | 66.20 | 87.88 | 31.18 | 67.27 | **84.21** | 65.70 |
| MemOS | 53.49 | 82.35 | **66.91** | 79.80 | 41.94 | 69.09 | 75.44 | 63.67 |
| EverOS | 74.42 | 64.71 | 64.03 | 85.86 | 35.48 | 65.45 | **84.21** | 67.57 |
| MindMemOS-MindVanilla | 76.74 | **88.24** | 65.47 | 87.88 | 17.20 | **80.00** | 82.46 | 67.74 |
| MindMemOS-MindSchema | **81.40** | 64.71 | 64.75 | 82.83 | **47.31** | 76.36 | 73.68 | **70.63** |

#### MemoryAgentBench

FactConsolidation scores are the mean Substring Exact Match across four context sizes. [MemoryAgentBench paper](https://arxiv.org/abs/2507.05257)

| Model | Method | SH score | SH archived | MH score | MH archived |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-4o-mini | Mem0 | 0.180 | — | 0.020 | — |
|  | MemoRAG | 0.270 | — | 0.070 | — |
|  | HippoRAG-v2 | 0.540 | — | 0.050 | — |
|  | MindMemOS-MindVanilla | 0.635 | — | 0.118 | — |
|  | MindMemOS-MindVanilla + Dreaming | **0.738** | 21.4% | **0.180** | 19.4% |
| GPT-5-mini | Infini Memory | 0.800 | — | 0.220 | — |
|  | MindMemOS-MindVanilla | 0.900 | — | 0.190 | — |
|  | MindMemOS-MindVanilla + Dreaming | **0.920** | 23.5% | **0.250** | 21.5% |

#### MemEye and Mem-Gallery

[MemEye](https://huggingface.co/datasets/MemEyeBench/MemEye) has 742 questions across 8 tasks. It scores multiple-choice answers by exact match (EM) and open-ended answers by BLEU-1 (B-1) and LLM Judge, with macro averages across tasks. [Mem-Gallery](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery) has 1,711 open-ended questions across 20 tasks. It reports token-level F1, BLEU-1, and LLM Judge, with micro averages across questions.

The table reports results from upstream `main`. All methods used `gpt-4.1-mini` to answer. MindMemOS used `vanilla` and local SigLIP2 (`siglip2-base-patch16-384`), averaged over two runs in October 2026. The baselines come from a paper under double-blind review and average three runs with the same answer model. **Compare Judge scores within each group.** MindMemOS used `gpt-4.1-mini` as judge, while the baselines used `GPT-5.4-mini`.

| Method | MemEye EM | MemEye B-1 | MemEye Judge | Mem-Gallery F1 | Mem-Gallery B-1 | Mem-Gallery Judge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MemoryOS | 45.75 | 13.12 | 34.91 | 43.09 | 37.76 | 64.64 |
| A-MEM | 49.73 | 21.11 | 38.27 | 44.08 | 38.37 | 64.35 |
| SimpleMem | 50.34 | 9.42 | 42.59 | 34.73 | 28.90 | 58.33 |
| HyperMem | 47.82 | 19.64 | 36.43 | 44.57 | 33.42 | 66.76 |
| M2A | 52.31 | 6.31 | 44.50 | 15.04 | 10.28 | 51.81 |
| MMA | 50.96 | 16.32 | 43.81 | 26.99 | 20.60 | 60.87 |
| SRAG | 58.36 | 31.48 | 50.54 | 41.07 | 35.03 | 67.06 |
| UniversalRAG | 60.49 | 11.23 | 52.61 | 48.36 | 41.90 | 70.63 |
| MindMemOS b0 (text only) | 43.03 | 20.32 | 26.26 | 38.72 | 32.59 | 70.68 |
| MindMemOS b1 (image similarity) | 59.77 | 27.02 | 47.80 | 54.73 | 47.75 | 85.06 |
| MindMemOS enhanced (VLM visual facts) | 63.45 | 25.49 | 50.80 | 55.66 | 48.98 | 87.06 |
| MindMemOS enhanced_agentic (multimodal agentic retrieval) | 65.89 | 27.09 | 53.30 | 57.06 | 50.45 | 89.14 |

`enhanced_agentic` enables `search_strategy: agentic` in `enhanced` mode. [Reproduction guide](docs/evaluation/memory_evaluation.md#multimodal-memory-evaluation) · [Upstream results](https://github.com/mindscale-noah/MindMemOS/blob/f2d6c3b2e1639e0aab666203791365d295894b52/README.md#L418-L437)

### 🛠️ Skill Benchmarks

#### Skill-Forest

Each benchmark reports the mean Hard score (%) over three runs on the held-out test split. The overall score weights the five benchmarks equally. Select and Merge each use seven trees. [Evaluation setup and results](docs/evaluation/skill_evaluation.md#validation-results)

| Method | LiveMath | ALFWorld | SpreadsheetBench Verified 400 | SearchQA | OfficeQA | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Skill | 27.42 | 56.97 | 80.24 | 76.90 | 28.49 | 54.00 |
| Initial Skill | 28.23 | 70.65 | 79.40 | 76.67 | 43.99 | 59.79 |
| Trace2Skill | 27.69 | 76.62 | 81.79 | 77.50 | 54.84 | 63.69 |
| GEPA | 31.45 | 79.85 | **82.26** | 79.62 | 53.68 | 65.37 |
| SkillOpt | 31.72 | 75.37 | 80.71 | **80.67** | 52.71 | 64.24 |
| Select | **36.02** | 82.34 | 81.67 | 80.21 | 53.88 | 66.82 |
| Merge | 34.68 | **82.84** | 81.07 | 80.52 | **55.23** | **66.87** |

#### TreeSkill

The table reports in-domain SpreadsheetBench scores. Parentheses show the gain in percentage points over the human-written Skill for the same model. The 35B baselines are previously reported scores. The TreeSkill results and the 9B baselines come from our runs. [Evaluation setup and results](docs/evaluation/skill_evaluation.md#in-domain-spreadsheetbench-results)

| Policy model | Method | Vrf (%) | Soft (%) | Hard (%) |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | No Skill | 19.00 | 18.00 | 4.60 |
|  | Human-Written Skill | 9.67 | 13.03 | 3.37 |
|  | Trace2Skill +Combined | 29.67 (+20.00) | 18.80 (+5.77) | 5.73 (+2.36) |
|  | SkillCAT | 55.50 (+45.83) | 39.02 (+25.99) | 20.61 (+17.24) |
|  | TreeSkill (routed) | **66.00 (+56.33)** | **41.72 (+28.69)** | **25.22 (+21.85)** |
| Qwen3.5-9B | No Skill | 25.50 | 21.00 | 9.21 |
|  | Human-Written Skill | 14.00 | 16.76 | 7.24 |
|  | Trace2Skill +Combined | 16.50 (+2.50) | 18.97 (+2.21) | 8.66 (+1.42) |
|  | TreeSkill (routed) | **45.50 (+31.50)** | **30.28 (+13.52)** | **14.58 (+7.34)** |

#### Skill Graph

##### Skill retrieval

In the Skill Graph tables, underlining marks second place, including ties. Asterisks mark scores from the corresponding papers. [Retrieval results](docs/evaluation/skill_evaluation.md#table-1-skill-retrieval)

**ALFWorld**

| Method | Ret@1 | Ret@K | MRR |
| --- | ---: | ---: | ---: |
| Vector Skills | 0.7 | 4.5 | 0.4 |
| SkillRouter (Emb.) | 13.6 | 48.6 | 26.2 |
| SkillRouter (Emb.+RR) | 12.1 | **74.3** | <ins>36.2</ins> |
| AgentSkillOS | 0.7 | 50.7 | 18.5 |
| SkillRAE | 9.3 | 35.6 | 17.5 |
| GoS | 8.6 | 9.3 | 8.9 |
| SkillDAG | <ins>20.0</ins> | 44.8 | 29.2 |
| MindMemOS Skill Graph | **49.3** | <ins>71.4</ins> | **57.9** |

**AgentSkillOS**

| Method | Ret@1 | Ret@K | MRR |
| --- | ---: | ---: | ---: |
| Vector Skills | 56.7 | 83.3 | 66.2 |
| SkillRouter (Emb.) | 50.0 | <ins>86.7</ins> | 64.3 |
| SkillRouter (Emb.+RR) | 56.7 | 70.0 | 61.9 |
| AgentSkillOS | 70.0 | 83.3 | 76.7 |
| SkillRAE | 40.0 | 80.0 | 55.2 |
| GoS | <ins>76.7</ins> | <ins>86.7</ins> | <ins>80.3</ins> |
| SkillDAG | 43.3 | 56.1 | 60.8 |
| MindMemOS Skill Graph | **80.0** | **93.3** | **83.3** |

**SkillsBench**

| Method | Ret@1 | Ret@K | MRR |
| --- | ---: | ---: | ---: |
| Vector Skills | 3.6\* | 10.8\* | 5.8\* |
| SkillRouter (Emb.) | <ins>77.6</ins> | **93.4** | <ins>83.1</ins> |
| SkillRouter (Emb.+RR) | **81.6** | <ins>88.2</ins> | **84.5** |
| AgentSkillOS | 38.2 | 44.7 | 41.0 |
| SkillRAE | 69.7 | 86.8 | 76.8 |
| GoS | 50.6\* | 65.5\* | 57.3\* |
| SkillDAG | 66.7\* | 78.2\* | 71.3\* |
| MindMemOS Skill Graph | 69.2 | 86.8 | 76.8 |

##### End-to-end R scores

Rankings are within each setting. Setting 1 uses MiniMax-M2.7, and the current reproduction guide covers AgentSkillOS for that setting. [End-to-end results](docs/evaluation/skill_evaluation.md#table-2-end-to-end-r-scores)

| Setting | Method | ALFWorld | AgentSkillOS | SkillsBench |
| --- | --- | ---: | ---: | ---: |
| Setting 1 | Vanilla Skills | 47.1\* | 70.8 | 17.2\* |
|  | Vector Skills | 50.7\* | <ins>72.4</ins> | 10.4\* |
|  | GoS | 54.3\* | 68.0 | 18.7\* |
|  | SkillDAG | <ins>67.9</ins> | 54.9 | <ins>27.3</ins>\* |
|  | MindMemOS Skill Graph | **76.4** | **80.9** | **28.0** |
| Setting 2 | Vector Skills | 60.0 | 74.9\* | 27.3 |
|  | GoS | 63.6 | <ins>96.2</ins> | <ins>35.4</ins> |
|  | SkillDAG | <ins>82.0</ins> | 94.2 | 31.4 |
|  | AgentSkillOS | 13.6 | 83.5\* | 17.2\* |
|  | SkillRouter | 29.3 | 82.3\* | 22.0\* |
|  | SkillRAE | 27.1 | 84.6\* | 29.3\* |
|  | MindMemOS Skill Graph | **86.0** | **97.4** | **43.2** |
| Setting 3 | GoS | 79.3 | <ins>96.2</ins> | 40.1 |
|  | SkillDAG | <ins>85.0</ins> | 92.3 | <ins>49.0</ins> |
|  | MindMemOS Skill Graph | **85.7** | **96.5** | **51.0** |

#### Offline Evolution

The SkillsBench comparison is pending. The evaluation guide describes how to collect trajectories with the initial Skill and retest the same tasks with the evolved Skill. [Offline Evolution evaluation](docs/evaluation/skill_evaluation.md#offline-evolution)

### Datasets and reproduction

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

#### Skill-Forest smoke test

Prepare the [Skill-Forest evaluation config](docs/evaluation/skill_evaluation.md#2-configure-the-evaluation), then run a small LiveMath evaluation with model calls.

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
uv run --no-sync mindmemos-eval skill evaluate train --algo skill-forest \
  --config config/mindmemos_eval/skill_train_skill_forest/paper_repro.yaml \
  --env livemath --smoke --output-dir output/skill-forest-smoke
```

## Documentation

| Goal | Guide |
| --- | --- |
| Deploy services, configure models, and select features | [Deployment](docs/deploy/instruction.md) |
| Check Memory search HTTP and CLI parameters | [Memory API](https://mindmemos.cn/#/api-docs) · [CLI search](docs/cli/instruction.md#mindmemos-memory-search) |
| Use the full CLI, configure connections, and troubleshoot | [CLI guide](docs/cli/instruction.md) |
| Integrate the Python SDK and independent Skill API | [SDK guide](src/mindmemos_sdk/README.md) |
| Manage Skills or integrate a custom Python environment | [Skill CLI](docs/cli/instruction.md#skill-commands) · [Skill service](src/mindmemos_skill/README.md) |
| Start and connect the local web workbench | [UI deployment](docs/deploy/instruction.md#2-deploy-the-ui) |
| Download datasets and reproduce Memory or Skill evaluations | [Memory evaluation](docs/evaluation/memory_evaluation.md) · [Skill evaluation](docs/evaluation/skill_evaluation.md) |

## 🗺️ Coming Features

- **Simpler deployment**: Further reduce dependencies and improve embedded use, building on the existing Kafka / OTel switches, local asynchronous Skill tasks, and SQLite backend.
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

## License

This project is open source under the MIT License.
