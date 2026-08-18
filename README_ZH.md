<p align="center">
  <img src="./assets/mindmemos-readme-hero.png" alt="MindMemOS Memory For AI Agents">
</p>

<p align="center">
  <a href="https://mindmemos.cn">
    <img src="https://img.shields.io/badge/Website-mindmemos.cn-0A66C2?labelColor=gray&logo=googlechrome&logoColor=white" alt="MindMemOS 官网">
  </a>
  <a href="https://mindmemos.cn/api-docs">
    <img src="https://img.shields.io/badge/FastAPI-Docs-009688?labelColor=gray&logo=fastapi&logoColor=white" alt="MindMemOS FastAPI 手册">
  </a>
  <a href="https://pypi.org/project/mindmemos-sdk/">
    <img src="https://img.shields.io/pypi/v/mindmemos-sdk?color=%2334D058&label=pypi%20sdk&labelColor=gray&logo=pypi&logoColor=white" alt="MindMemOS SDK PyPI 版本">
  </a>
  <a href="https://www.npmjs.com/package/@mindmemos/openclaw-plugin">
    <img src="https://img.shields.io/npm/v/%40mindmemos%2Fopenclaw-plugin?label=npm%20plugin&labelColor=gray&logo=npm&logoColor=white" alt="MindMemOS OpenClaw 插件 npm 版本">
  </a>
  <a href="https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin">
    <img src="https://img.shields.io/npm/v/%40mindmemos%2Fdeepseek-harness-plugin?label=dsh%20plugin&labelColor=gray&logo=npm&logoColor=white" alt="MindMemOS DeepSeek Harness 插件 npm 版本">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg?labelColor=gray" alt="MIT License">
  </a>
</p>

<p align="center">
  <strong><a href="README.md">English</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="https://mindmemos.cn">官网</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="https://mindmemos.cn/api-docs">API 文档</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="https://pypi.org/project/mindmemos-sdk/">PYPI SDK</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="docs/deploy/instruction_ZH.md">如何部署</a></strong>
</p>

<p align="center">
  精准记忆用户与任务上下文，跨 Agent 迁移复用；在持续交互中演化记忆，自动沉淀 Skills，并联动文件知识系统，让经验真正成为能力。
</p>

> ⭐ **GitHub Star 后自动升级 Pro 额度会员。**

## 📰 News

- **2026-08-18**：我们发布了 [DeepSeek Harness 插件](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin)，让 DeepSeek Harness（dsh）Agent 自动召回并写入 MindMemOS 记忆。
- **2026-08-14**：我们发布了 [MindMemOS 1.0 技术报告](https://arxiv.org/abs/2608.12428)《MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents》。
- **2026-07-17**: MindMemOS 接入 [LLM4AD_NEXT](https://github.com/Optima-CityU/LLM4AD_Next)，为算法设计任务提供可检索的长期记忆能力，实现跨任务经验、领域知识与约束条件的沉淀和复用。
- **2026-06-30**：MindMemOS 正式发布！

## 🌟 Core Features

- **跨 Agent 可迁移**：将用户画像、偏好、项目事实、工具经验和 skill candidates 沉淀为可复用资产，让 OpenClaw、Hermes、Claude Code、OpenHands 等不同 Agent 共享或迁移同一套长期记忆。
- **记忆系统可自主演化**：通过 schema learning、dreaming、feedback 持续优化记忆质量，自动学习高频记忆点、离线巩固合并记忆，并从交互纠错中反向优化 add/search 流程。
- **记忆与 Skills 联动**：经验记忆可以沉淀为 skill candidates；skills 的执行结果、失败轨迹和用户反馈也会回流到记忆系统，推动 skills 持续演进。
- **插件集成能力**：支持通过插件将 MindMemOS 接入不同 Agent 与工作流，在交互前检索并注入相关记忆、回合结束后自动写回对话，当前已提供 [OpenClaw 插件](https://www.npmjs.com/package/@mindmemos/openclaw-plugin) 和 [DeepSeek Harness 插件](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin)，并持续扩展更多集成。

<p align="center">
  <img src="./assets/mindmemos-benchmark-overview.png" alt="MindMemOS 基准测试结果概览">
</p>

## 🚀 快速开始

MindMemOS 有**两种部署方式**（官方云服务、本地自部署）和**三种接入方式**（HTTP 接口、Python SDK / CLI、Agent 插件），可以任意组合，服务端与客户端是同一套协议：

| 接入方式 | 用途 | 云端 base_url | 本地 base_url |
| :--- | :--- | :--- | :--- |
| [HTTP 接口](https://mindmemos.cn/api-docs) | 业务应用直接调用 | `https://mindmemos.cn` | `http://127.0.0.1:8000` |
| [Python SDK / CLI](https://pypi.org/project/mindmemos-sdk/) | 业务应用集成 | `https://mindmemos.cn` | `http://127.0.0.1:8000` |
| [OpenClaw 插件](https://www.npmjs.com/package/@mindmemos/openclaw-plugin) | Agent 自动写入/召回记忆 | `https://mindmemos.cn` | `http://127.0.0.1:8000` |
| [DeepSeek Harness 插件](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin) | Agent 自动写入/召回记忆（dsh） | `https://mindmemos.cn` | `http://127.0.0.1:8000` |

想省去部署直接体验，可以先用官方云服务（在 [官网](https://mindmemos.cn) 申请 API key）；需要私有化或离线使用，按下面的本地部署启动。

### 1. 本地部署

MindMemOS 使用 `uv` 管理依赖和执行本地命令。详细配置方法可以查看[docs/deploy/instruction_ZH.md](docs/deploy/instruction_ZH.md)。

默认端口 8000：FastAPI `http://127.0.0.1:8000`，API 文档 `http://127.0.0.1:8000/docs`。

#### 1.1 准备配置文件
```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
```
启动前，至少需要在 config/mindmemos/dev.yaml 中配置以下三类模型路由：

- `chat_model_router`：支持记忆抽取、Skill演进等。
- `embed_model_router`：语义向量 embedding，维度保证与QDrant维度配置一致。
- `rerank_model_router`：可选，记忆检索结果重排。

在 `config/mindmemos/api_keys.yaml` 中配置 API key 以及绑定的 `project_id`。

#### 1.2 启动服务

启动本地服务（默认端口：8000）：
```bash
make dev  # 会先启动全量 Docker 依赖，再启动 FastAPI。
```

只启动核心依赖时使用：

```bash
make dev-core          # Qdrant + Neo4j + Kafka
make db-observability  # Qdrant + Neo4j + Kafka + ClickHouse + OTel + Grafana
```

#### 1.3 停止服务

```bash
make dev-down
```

### 2. 接入方式

云服务和本地自部署使用同一套接入协议，本地 key 来自 `config/mindmemos/api_keys.yaml`，云端 key 在 [官网](https://mindmemos.cn) 申请。

#### 2.1 HTTP 接口调用

HTTP 接口是基础接入方式，SDK 与插件底层也走 HTTP。先用 curl 调通接口做验证（服务起来后），再接入业务。调用前定义地址与 key，本地或云端二选一：

```bash
export BASE_URL=http://127.0.0.1:8000   # 本地自部署；云端改为 https://mindmemos.cn
export API_KEY=dev-api-key-001          # 本地示例 key；云端改为官网申请的 key
```

写入一条记忆：

```bash
curl -sS -X POST "$BASE_URL/v1/memory/add" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_123",
    "messages": [{"role": "user", "content": "我喜欢喝冰美式。"}]
  }'
```

检索记忆：

```bash
curl -sS -X POST "$BASE_URL/v1/memory/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "用户喜欢喝什么咖啡？", "top_k": 3}'
```

PowerShell 等其他环境 / 语言的 HTTP 调用格式，以及其余接口（get / list / delete / update / feedback / dreaming / skills 等）都见 [API 文档](https://mindmemos.cn/api-docs)。

#### 2.2 配置 SDK

安装 Python SDK：

```bash
pip install mindmemos-sdk
```

运行认证命令，依次配置服务地址、API key 和默认用户：

```bash
mindmemos auth
```

| 配置项 | 本地服务 | 云端服务 |
| :--- | :--- | :--- |
| `base_url` | `http://127.0.0.1:8000` | `https://mindmemos.cn` |
| `api_key` | `config/mindmemos/api_keys.yaml` 中已启用的 API key | 从 [MindMemOS 官网](https://mindmemos.cn) 申请的 API key |
| `user_id` | 当前终端用户的稳定标识，例如 `u_123` | 当前终端用户的稳定标识，例如 `u_123` |

配置会保存到 `~/.mindmemos/settings.json`。可以通过以下命令检查当前配置：
```bash
mindmemos config show
```

本地服务会根据 API key 自动确定 `project_id` 和记忆算法，调用 SDK 时无需再传 `project_id`。`user_id` 用于区分同一项目下的不同用户，也可以在单次 `add` 或 `search` 调用中覆盖。

如果不希望使用本地配置文件，也可以在创建客户端时显式传入连接参数：

```python
from mindmemos_sdk import MindMemOSClient

with MindMemOSClient(
    base_url="http://127.0.0.1:8000",
    api_key="<api_key>",
    user_id="u_123",
) as client:
    ...
```

显式传入的参数优先于 `~/.mindmemos/settings.json` 中的配置。

#### 2.3 通过 SDK 写入和检索记忆

完成上述配置后，`MindMemOSClient()` 会自动读取服务地址、API key 和默认 `user_id`。SDK 会自动添加认证请求头，无需手动拼接 HTTP 请求：

```python
from mindmemos_sdk import DialogueMessage, MindMemOSClient

with MindMemOSClient() as client:
    add_result = client.memory.add(
        messages=[
            DialogueMessage(
                role="user",
                content="我喜欢喝冰美式。",
            )
        ],
        mode="sync",
    )

    for item in add_result.memories:
        print(item.operation, item.memory_id, item.content)

    search_result = client.memory.search(
        "用户喜欢喝什么咖啡？",
        top_k=5,
        search_strategy="fast",
    )

    for memory in search_result.memories:
        print(memory.id, memory.memory)
```

触发已注册 Skill 的云端演进：

```python
from mindmemos_sdk import MindMemOSClient

with MindMemOSClient() as client:
    result = client.skills.evolve("my-skill", mode="sync")

    print("evolved:", result.evolved)
    print("pending:", result.pending_count)
    print("threshold:", result.threshold)
    print("new versions:", result.new_version_ids)
```

本地服务与云端服务使用同一套 SDK 调用逻辑，切换时只需重新配置 `base_url` 和对应的 API key。

#### 2.4 通过 CLI 调用

完成 `mindmemos auth` 后，也可以通过 SDK 提供的 CLI 直接写入和检索记忆：

```bash
mindmemos memory add --content "我喜欢喝冰美式"
mindmemos memory search "咖啡偏好" --top-k 5
```

`memory` 子命令还支持 get / update / delete / feedback / dreaming，`skill` 子命令支持 register / list / evolve / push / pull / history 等，完整命令、参数与故障排查见 [CLI 使用说明](docs/cli/instruction_ZH.md)。

#### 2.5 OpenClaw 插件

**推荐使用我们提供的 [`mindmemos-cli` skill](skills/mindmemos-cli/SKILL.md) 安装**：把 `skills/mindmemos-cli/` 部署到你的 Agent 的 skills 目录后，让 Agent 按其指引操作即可，skill 内的 [参考文档](skills/mindmemos-cli/references/openclaw-plugin.md) 覆盖了安装、授权与常见故障排查。

<details>
<summary><b>直接手动安装（不推荐）</b></summary>

**先安装 SDK 并完成 auth 配置（必须）**：插件通过 `mindmemos` CLI 与本机通信，所以要先安装 Python SDK 并保证 `mindmemos` 命令可用：

```bash
pip install mindmemos-sdk    # 或 uv add mindmemos-sdk
mindmemos --version          # 确认命令已可用
```

然后用 `mindmemos auth` 配置好 base_url、API key 和 user_id（指向云端或本地服务都行）:

```bash
mindmemos auth
mindmemos config show        # 确认配置生效
```

> 没完成这两步就装插件，日志会直接报错（找不到 `mindmemos` 命令 / 未配置认证），插件无法正常读写记忆。

安装并启用插件：

```bash
openclaw plugins install @mindmemos/openclaw-plugin
openclaw plugins enable mindmemos-memory
```

（`@mindmemos/openclaw-plugin` 是 npm 包名，`mindmemos-memory` 是插件 id。）手动安装容易踩下面两个坑，需要注意：

- **写入权限（必配）**：插件的 `agent_end` 写入勾子需要 `allowConversationAccess` 授权，否则看起来一切正常、但回合结束后记忆实际上没有落库：
  ```bash
  openclaw config set plugins.entries.mindmemos-memory.hooks.allowConversationAccess true
  openclaw gateway restart
  ```
- **`cli` 的 PATH**：GUI 方式启动的 OpenClaw 进程不继承终端 PATH，`mindmemos` 命令需显式配置为绝对路径或用 `uv run mindmemos` 包装，否则日志报 `ENOENT`。

安装启用并重启 gateway 后，插件会在每次用户回合前检索并注入相关记忆，回合结束后自动写回对话。

完整命令、配置项与故障排查见 [OpenClaw 插件集成文档](skills/mindmemos-cli/references/openclaw-plugin.md)。

</details>

#### 2.6 DeepSeek Harness 插件

**推荐使用我们提供的 [`mindmemos-cli` skill](skills/mindmemos-cli/SKILL.md) 安装**：把 `skills/mindmemos-cli/` 部署到你的 Agent 的 skills 目录后，让 Agent 按其指引操作即可，skill 内的 [参考文档](skills/mindmemos-cli/references/deepseek-harness-plugin.md) 覆盖了安装与常见故障排查。

<details>
<summary><b>直接手动安装（不推荐）</b></summary>

**先安装 SDK 并完成 auth 配置（必须）**：插件通过 `mindmemos` CLI 与本机通信，所以要先安装 Python SDK 并保证 `mindmemos` 命令可用：

```bash
pip install mindmemos-sdk    # 或 uv add mindmemos-sdk
mindmemos --version          # 确认命令已可用
```

然后用 `mindmemos auth` 配置好 base_url、API key 和 user_id（指向云端或本地服务都行）:

```bash
mindmemos auth
mindmemos config show        # 确认配置生效
```

> 没完成这两步就装插件，日志会直接报错（找不到 `mindmemos` 命令 / 未配置认证），插件无法正常读写记忆。

把插件安装进 dsh profile（`dsh plugin` 会转发给 pnpm，把包装进该 profile 的 `node_modules`）：

```bash
dsh plugin --profile <name> add @mindmemos/deepseek-harness-plugin
```

（`@mindmemos/deepseek-harness-plugin` 是 npm 包名，`mindmemos-memory` 是插件 id。）dsh 通过分层的 `cordis.patch.yml` 组合插件，因此需要在 profile 补丁（`~/.dsh/profiles/<name>/cordis.patch.yml`）里加一条 `insert` 来注册插件：

```yaml
- insert:
    - id: mindmemos-memory
      name: '@mindmemos/deepseek-harness-plugin'
      config:
        userId: alice
        appId: deepseek-harness
```

用该 profile 重启 dsh。注册完成后，插件会在每次用户回合前检索并注入相关记忆，回合结束后自动写回对话。

完整命令、配置项与故障排查见 [DeepSeek Harness 插件集成文档](skills/mindmemos-cli/references/deepseek-harness-plugin.md)。

</details>

## 📊 Benchmark

### 💬 对话记忆：LoCoMo

- **评测基准**：[LoCoMo](https://arxiv.org/abs/2402.17753)，面向长对话记忆的主流基准，覆盖单跳（single-hop）、多跳（multi-hop）、时序（temporal）和开放域（open-domain）问答。

| 方法                       | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
| :------------------------- | :--------: | :-------: | :------: | :---------: | :-----: |
| Mem0                       | 68.97 | 61.70 | 58.26 | 50.00  | 64.20 |
| MemU                       | 74.91 | 72.34 | 43.61 | 54.17  | 66.67 |
| MemOS                      | 85.37 | 79.43 | 75.08 | 64.58  | 80.76 |
| Zep                        | 90.84 | 81.91 | 77.26 | 75.00  | 85.22 |
| EverOS                     | 96.67 | 91.84 | 89.72 | 76.04  | 93.05 |
| **MindMemOS-MindVanilla** | 92.03 | 85.82 | 83.80 | 66.67  | 87.60 |
| **MindMemOS-MindSchema**  | **96.79** | **93.97** | **90.34** | **82.29** | **94.03** |

### 👤 用户画像记忆：PersonaMem

- **评测基准**：[PersonaMem](https://arxiv.org/abs/2504.14225)，以用户画像与喜好理解为中心的记忆基准，评测对用户特征的召回、追踪、重访、建议、推荐与泛化能力。

| Method | Recall | Ack. Lat. | Trk. Evo. | Revisit | Suggest | Recom. | General. | Overall |
| :----- | :----: | :-------: | :-------: | :-----: | :-----: | :-----: | :------: | :-----: |
| Mem0 | 46.51 | 41.18 | 65.47 | 90.91 | 12.90 | 34.55 | 43.86 | 51.61 |
| MemU | 64.34 | 64.71 | 66.20 | 87.88 | 31.18 | 67.27 | 84.21 | 65.70 |
| MemOS | 53.49 | 82.35 | 66.91 | 79.80 | 41.94 | 69.09 | 75.44 | 63.67 |
| EverOS | 74.42 | 64.71 | 64.03 | 85.86 | 35.48 | 65.45 | 84.21 | 67.57 |
| **MindMemOS-MindVanilla** | 76.74 | 88.24 | 65.47 | 87.88 | 17.20 | 80.00 | 82.46 | 67.74 |
| **MindMemOS-MindSchema** | 81.40 | 64.71 | 64.75 | 82.83 | 47.31 | 76.36 | 73.68 | 70.63 |

### 🌙 记忆巩固：MemoryAgentBench（FactConsolidation）

- **评测基准**：[MemoryAgentBench](https://arxiv.org/abs/2507.05257) FactConsolidation，表中分数为四种上下文规模下的平均 Substring Exact Match。

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

### 🧠 Skill 自演进：SpreadsheetBench-Verified

- **评测基准**：[SpreadsheetBench-Verified](https://huggingface.co/datasets/KAKA22/SpreadsheetBench/blob/main/spreadsheetbench_verified_400.tar.gz)，SpreadsheetBench 的 400 题 verified 子集，覆盖多种真实 spreadsheet 操作任务。

| Method | Success Rate | Time / Task (s) | Agent Tokens | Evolve Tokens |
| :----- | :----------: | :-------------: | :----------: | :-----------: |
| No-skill | 51.3% ± 0.8% | 11.227 | 10.4M | - |
| Init-skill | 48.0% ± 1.4% | 15.350 | 16.9M | - |
| **MindMemOS-MindEvolve-Unsup.** | **55.3% ± 0.9%** | 15.470 | 27.3M | 5.8M |
| **MindMemOS-MindEvolve-Sup.** | **57.2% ± 2.4%** | 15.631 | 25.2M | 5.5M |

## 🗺️ Coming Features

- **Lite 模式**：以低依赖、可替换、易嵌入为设计理念，将数据库后端、异步任务和日志存储解耦为灵活的轻量组件，支持 in-memory 调用与简化部署。
- **Skills 系统**：治理庞杂冗余的 skills 并智能分发；根据真实使用持续演化优化；从用户高频场景自动合成新 skills，并通过离线推演不断打磨。
- **文件系统记忆**：将散落在本地文件、文档、项目产物和 Agent 输出中的零碎知识结构化管理，构建可检索、可关联的文件知识对象或知识图谱，帮助 Agent 更好完成用户任务。
- **Agent 集成**：继续增强对代码 Agent、OpenClaw、Codex 风格工作流和长期运行多 Agent 系统的支持。

## 参与贡献

欢迎大家提交各类改进和修复。请将 Pull Request 的目标分支设为 `develop`；
通过审核的改动会合入 `develop`，维护者会定期将 `develop` 的稳定版本合入 `main` 并发布。

## 💬 Community

欢迎加入 MindMemOS 飞书群，获取项目动态、交流使用问题和参与社区讨论。

<p align="center">
  <img src="./assets/feishu-group-small.png" alt="MindMemOS 飞书群二维码">
</p>

## 📝 引用

如果您的研究使用了 MindMemOS，请引用我们的技术报告：

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

本项目采用 MIT License 开源。
