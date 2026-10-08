# MindMemOS 部署&配置说明


<p align="center">
  <strong><a href="instruction.md">English</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="instruction_ZH.md">简体中文</a></strong>
</p>

## 1. Overview

MindMemOS采用 `uv workspace` 管理 3 个核心 Python 包，并按照“服务端、客户端、评测工具”分层：

```text
业务应用 / Agent ───────────────┐
Agent 插件 ─────── CLI ─────────┼──> mindmemos_sdk ── HTTP ──> mindmemos
mindmemos_eval ─────────────────┘
```

- `mindmemos` 是服务端算法核心包，负责 FastAPI 接口、记忆与 Skill 业务流程、模型调用、数据持久化和异步任务。
- `mindmemos_sdk` 是面向业务应用和插件的 Python SDK 与 CLI，通过 HTTP 调用 `mindmemos`，不依赖服务端内部实现。
- `mindmemos_eval` 是独立评测包，依赖 `mindmemos_sdk` 调用服务，负责加载数据集、执行评测流程并统计结果。

运行相关的主要目录如下：

```text
.
├── src/
│   ├── mindmemos/          # 服务端核心包
│   ├── mindmemos_sdk/      # Python SDK 与 mindmemos CLI
│   └── mindmemos_eval/     # Benchmark 评测工具
├── config/
│   ├── mindmemos/          # 服务端运行与认证配置
│   ├── mindmemos_eval/     # 评测任务配置
│   └── presets/            # 算法预设资源
├── dockers/                # Qdrant、Neo4j、Kafka 和观测组件
├── plugins/                # Agent 插件集成
├── Makefile                # 本地服务与依赖启动入口
└── pyproject.toml          # uv workspace 与开发依赖
```

主要配置入口如下：

| 适用范围             | 配置文件                             | 说明 |
|------------------|----------------------------------| --- |
| `mindmemos`      | `.env`                           | Docker 依赖、服务端口和连接地址。 |
| `mindmemos`      | `config/mindmemos/dev.yaml`      | 服务端模型、数据库、Pipeline 和运行配置。 |
| `mindmemos_skill` | `config/mindmemos_skill/dev.yaml` | 从 `dev.example.yaml` 复制的 Skill 服务配置。 |
| `mindmemos`      | `config/mindmemos/api_keys.yaml` | API key、`project_id`、记忆算法和访问权限。 |
| `mindmemos`      | `config/presets/*.json`          | 记忆算法预设。 |
| `mindmemos_sdk`  | `~/.mindmemos/settings.json`     | SDK 与 CLI 的连接信息和默认用户。 |
| `mindmemos_eval` | `config/mindmemos_eval/*/*.yaml` | 评测模型、数据集、并发和算法配置。 |


## 2. 最小启动流程

```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
cp config/mindmemos_skill/dev.example.yaml config/mindmemos_skill/dev.yaml

# 编辑 .env、config/mindmemos/dev.yaml 和 config/mindmemos_skill/dev.yaml 后启动
make setup
make dev
```

`config/mindmemos_skill/dev.example.yaml` 是唯一的 Skill 部署示例。`make dev` 和 `make dev SERVICE=skill` 默认读取用户维护的 `dev.yaml`；缺少文件时在启动 Docker 依赖前报错；先从示例复制并填写配置。

默认地址：

- FastAPI: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- Qdrant: `http://localhost:6333`
- Neo4j Browser: `http://localhost:7474`

`make dev` 默认启用 Memory、Skill、Kafka 异步任务和 OTel 监控，根据最终配置启动依赖，并用 `.env` 自动覆盖连接地址与映射端口。其他入口：

```bash
make dev FEATURE=core                        # 联合服务，仅核心功能
make dev FEATURE=async                       # 联合服务，加异步任务
make dev SERVICE=memory                         # 仅 Memory，全量功能
make dev SERVICE=skill                       # 仅 Skill，全量功能
make dev SERVICE=skill FEATURE=core                  # 仅 Skill，核心功能
make dev PLAN=1          # 校验并预览实际依赖，不启动服务
```

停止本地依赖：

```bash
make stop
```

## 3. 必配环境变量

配置文件选择：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_CONFIG_NAME` | 选择配置名；`dev` 会读取 `config/mindmemos/dev.yaml` | `dev` |
| `MINDMEMOS_CONFIG_PATH` | 直接指定配置文件路径；设置后优先于 `MINDMEMOS_CONFIG_NAME` | 空 |

Qdrant：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_QDRANT_URL` | FastAPI 访问 Qdrant 的 HTTP 地址 | `http://localhost:6333` |
| `MINDMEMOS_QDRANT_HTTP_PORT` | Docker 暴露 Qdrant HTTP 端口 | `6333` |
| `MINDMEMOS_QDRANT_GRPC_PORT` | Docker 暴露 Qdrant gRPC 端口，也会覆盖 config 里的 `database.qdrant.grpc_port` | `6334` |
| `MINDMEMOS_QDRANT_PREFER_GRPC` | Qdrant client 是否优先使用 gRPC | `false` |
| `MINDMEMOS_QDRANT_API_KEY` | Qdrant API key；本地无鉴权可留空 | 空 |
| `MINDMEMOS_GRAFANA_QDRANT_URL` | Grafana 容器访问 Qdrant 的 HTTP 地址 | `http://qdrant:6333` |

Neo4j：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_NEO4J_URI` | FastAPI 访问 Neo4j 的 Bolt 地址 | `bolt://localhost:7687` |
| `MINDMEMOS_NEO4J_HTTP_PORT` | Docker 暴露 Neo4j Browser 端口 | `7474` |
| `MINDMEMOS_NEO4J_BOLT_PORT` | Docker 暴露 Neo4j Bolt 端口 | `7687` |
| `MINDMEMOS_NEO4J_USERNAME` | Neo4j 用户名，也是 Docker `NEO4J_AUTH` 的用户名 | `neo4j` |
| `MINDMEMOS_NEO4J_PASSWORD` | Neo4j 密码，也是 Docker `NEO4J_AUTH` 的密码 | `mindmemos_dev_password` |

可选依赖：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS` | Kafka 地址；只有 config 里 `kafka.enabled=true` 时服务才会启动消费者/生产者 | `localhost:9092` |
| `MINDMEMOS_TELEMETRY_ENDPOINT` | OTel HTTP endpoint；只有 config 里 `telemetry.enabled=true` 时会上报 | `http://localhost:4318` |
| `MINDMEMOS_CLICKHOUSE_USER` / `MINDMEMOS_CLICKHOUSE_PASSWORD` / `MINDMEMOS_CLICKHOUSE_DB` | ClickHouse/Grafana 观测数据配置 | 见 `.env.example` |
| `MINDMEMOS_SKILL_POSTGRES_PORT` | Skill 或联合模式下 pgvector 映射到宿主机的端口 | `55432` |

API 监听地址：

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `MINDMEMOS_API_HOST` | `make dev` / `make api` 启动 FastAPI 的 host | `127.0.0.1` |
| `MINDMEMOS_API_PORT` | `make dev` / `make api` 启动 FastAPI 的 port | `8000` |

## 4. Docker 相关

按所选档位启动 Memory 依赖：

```bash
make docker SERVICE=memory
```

`make docker SERVICE=memory` 默认准备全量依赖；追加 `FEATURE=core` 只准备核心数据库，`FEATURE=async` 增加 Kafka 任务依赖。不会启动 API 或 UI。实际本地容器按最终配置推导，外部依赖不在本地重复启动。详见 [Docker 使用说明](../../dockers/README.md)。

Docker Compose 内的核心服务：

- `qdrant`: 存 memory/entity/source 向量和 payload。
- `neo4j`: 存图关系。
- `kafka`: 异步任务队列；由 `FEATURE` 决定是否启用。
- `clickhouse` + `otel-collector` + `grafana`: 观测链路；使用 `FEATURE=core` 或 `FEATURE=async` 关闭。

本地部署时，`.env` 里的端口变量要和 `config/mindmemos/dev.yaml` 里的连接地址对齐。代码启动时还会用环境变量覆盖这些连接字段：

- `database.qdrant.url`
- `database.qdrant.api_key`
- `database.qdrant.grpc_port`
- `database.qdrant.prefer_grpc`
- `database.neo4j.uri`
- `database.neo4j.username`
- `database.neo4j.password`
- `kafka.bootstrap_servers`
- `telemetry.telemetry_endpoint`

## 5. LLM 配置

LLM 用于记忆抽取、schema 处理、dreaming 等生成任务。需要配置 `chat_model_router`：

```yaml
chat_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/gpt-4.1-mini
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 1200
      temperature: 0.0
      num_retries: 3
      extra_body: {}
```

注意：

- `model` 是 LiteLLM 风格的模型名，OpenAI 兼容接口通常写成 `openai/<model-name>`。
- `api_base` 要包含 `/v1`，除非你的供应商文档明确不是这个形式。
- `api_key` 不要提交到仓库；本地写在未提交的 `config/mindmemos/dev.yaml` 即可。
- 可以配置多个 endpoint，router 会按 `routing_strategy` 路由。

## 6. Embedding 配置

Embedding 是必须配置的；服务启动时会校验 embedding 输出维度和 Qdrant 向量维度是否一致。

```yaml
embed_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/qwen3-embedding-4b
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 600
      num_retries: 3
      dimensions: 2560
      extra_body: {}

database:
  qdrant:
    vector_size: 2560
    semantic_vector_name: semantic
    bm25_vector_name: bm25
```

重点：

- `database.qdrant.vector_size` 必须等于 embedding 模型实际输出维度。
- 如果 embedding 模型支持自定义维度，`dimensions` 和 `vector_size` 也要一致。
- Qdrant collection 已经用旧维度创建后，单纯改 `vector_size` 不会自动迁移旧 collection；本地开发可以 `make clean` 清掉 volume 后重建。

## 7. Rerank 配置（可选）

Rerank 用于检索候选结果重排，提升搜索精度，但不是服务启动的硬依赖。没有外部 rerank endpoint 时，基础 add/search 仍可运行；代码会使用现有召回结果或 fallback 逻辑。

需要外部 reranker 时配置：

```yaml
rerank_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/qwen3-reranker-4b
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 600
      num_retries: 3

algo_config:
  search:
    rerank:
      enabled: true
      max_query_length: 100
      max_doc_length: 5000
      max_batch_size: 20
      max_concurrent_batches: 1
      request_timeout: 5.0
    vanilla:
      use_reranker: true
    schema_search:
      entity:
        use_reranker: true
```

不使用外部 reranker 时可以写成：

```yaml
rerank_model_router:
  routing_strategy: simple-shuffle
  endpoints: []

algo_config:
  search:
    rerank:
      enabled: false
    vanilla:
      use_reranker: false
    schema_search:
      entity:
        use_reranker: false
```

强调：`rerank` 是可选增强项。生产环境建议先把 Docker、LLM、Embedding 跑稳，再接入 rerank。

## 8. 记忆算法版本（v1 / v2）

Schema 记忆抽取有两套可选流程，通过 `algo_config.add.schema.version` 切换：

- `v2`（默认）：规则化图融合，每个 episode 的 LLM 调用更少。
- `v1`：与 develop 对齐的重 LLM 流程，用于基线对比和回退。

写在基础配置里是部署级默认（修改后需重启生效）；写在项目 API key 的覆盖配置里
则只对该项目生效（项目覆盖优先，项目下一个 add 请求即生效，无需重启）。两个
方向的存储都兼容：collection 与 payload 结构相同，v2 可直接读写 v1 数据，反向
亦然，混合历史安全。产出数据带 `mem_extract_version` 标签（`schema_add` /
`schema_add_v1`），可区分来源。v1 保留用于基线对比和回退，直到 v2 在
LoCoMo/PersonaMem 基准上验证完成。

## 9. 认证配置

本地默认使用 API key：

```yaml
auth:
  mode: api_key
  api_key_file: api_keys.yaml
```

`api_key_file` 是相对 config 文件目录解析的，所以默认指向 `config/mindmemos/api_keys.yaml`。本地示例里已有：

- `dev-api-key-001`: vanilla memory
- `dev-api-key-002`: schema memory

调用 API 时使用：

```text
Authorization: Bearer <api_key>
```

## 10. 最小检查清单

启动前至少确认：

- `.env` 里的 Qdrant、Neo4j 端口没有和本机已有服务冲突。
- `config/mindmemos/dev.yaml` 存在。
- `chat_model_router.endpoints[0].api_key` / `api_base` / `model` 可用。
- `embed_model_router.endpoints[0].api_key` / `api_base` / `model` 可用。
- `database.qdrant.vector_size` 等于 embedding 输出维度。
- 不需要 rerank 时，`rerank_model_router.endpoints` 可以留空，并关闭相关 `use_reranker`。

`make dev` 同时启动 UI（默认 `http://127.0.0.1:8765`，可用 `UI_PORT` 覆盖）。启动就绪后会统一列出 Memory、Skill、UI 地址及异步任务和 OTel 状态。UI 的 SDK 连接配置可用 `UI_HOME` 选择。
