# MindMemOS 部署指南

[English](instruction.md) | 简体中文

从源码启动 Memory API、Skill API 和 Web 工作台。API 和 UI 在宿主机上以前台进程运行，Docker Compose 管理按需启动的存储、队列和监控组件。

- [启动本地服务和 UI](#1-启动本地服务和-ui)
- [部署 UI](#2-部署-ui)，随本地 API 启动，或单独连接官方服务和已有 API
- [按需启动和管理](#3-按需启动和管理)，选择单个服务、减少依赖或分开启动

## 1. 启动本地服务和 UI

以下命令均在仓库根目录运行。完整启动需要 Git、Make、uv、Python 3.11–3.13 和运行中的 Docker Compose；构建 UI 还需 Node.js 和 npm。仓库默认使用 Python 3.13。

### 1.1 安装依赖

```bash
uv sync --all-packages --extra postgres --extra kafka
export UV_NO_SYNC=1
```

`postgres` 对应 Skill 示例使用的 PostgreSQL，`kafka` 对应 Skill 的 Kafka 任务队列。`UV_NO_SYNC=1` 让当前终端的 Make 命令沿用已安装的依赖。换终端时重新设置，依赖变更后重新同步。部分 ALFWorld 依赖需要系统编译工具。macOS ARM64 上，安装 TextWorld 时需要允许挂载和卸载 Inform7 安装镜像。

### 1.2 初始化 & 填写配置文件

```bash
test -f .env || cp .env.example .env
# 如需共用模型连接，先编辑 .env，再初始化。
make init
```

`make init` 从示例生成缺失的本地文件。它把 `.env` 中非空的模型参数写入 `config/` 下匹配的现有字段，包括服务、训练和评测配置。留空的变量保留 YAML 原值。对缺少的字段或需要不同供应商的模型，直接补充各自的 YAML。再次执行 `make init` 会重新应用这些非空参数。

| 模型 | API key | 地址 | 模型名 | 输出维度 |
| --- | --- | --- | --- | --- |
| Chat | `MINDMEMOS_CHAT_API_KEY` | `MINDMEMOS_CHAT_BASE_URL` | `MINDMEMOS_CHAT_MODEL` | — |
| Embedding | `MINDMEMOS_EMBEDDING_API_KEY` | `MINDMEMOS_EMBEDDING_BASE_URL` | `MINDMEMOS_EMBEDDING_MODEL` | `MINDMEMOS_EMBEDDING_DIMENSIONS` |
| Rerank | `MINDMEMOS_RERANK_API_KEY` | `MINDMEMOS_RERANK_BASE_URL` | `MINDMEMOS_RERANK_MODEL` | — |

填写任一 Embedding 环境变量时，输出维度必须是正整数。初始化会更新匹配的 Embedding 维度及 Qdrant 的 `vector_size`。Rerank 地址填写供应商要求的完整端点。

启动前检查对应文件：

- `config/mindmemos/dev.yaml`。补齐使用的模型连接，确认 Embedding 的实际输出维度与配置及 `database.qdrant.vector_size` 一致。
- `config/mindmemos_skill/dev.yaml`。检查存储配置，并为要使用的图管理、训练或演化流程补齐模型连接及凭据。
- `config/presets/local.project.yaml`。把 `projects.<项目名>.api_key` 中的占位值改成自己的服务 API key。已有数据时保留项目 ID。需要通过 UI 读写两项服务时，所选项目需包含 `memory:read`、`memory:write`、`skill:read`、`skill:write`。

模型供应商的 key 用于服务调用模型，项目 API key 用于客户端调用 MindMemOS。数据库、端口和容器配置见 [Docker 配置说明](../../dockers/README.md)。训练或评测额外依赖见[任务环境与数据](#任务环境与数据)。

### 1.3 构建并启动

```bash
uv run --no-sync python scripts/install_nlp_assets.py
uv run --no-sync python scripts/check_nlp_assets.py
make ui-build
make dev PLAN=1
make dev
```

NLP 资源供 Memory 使用。若只运行 Skill，可跳过两条 NLP 命令。`make ui-build` 将前端产物写入 UI Python 包的 `static/` 目录。启动器直接使用该目录，前端源码有变更或本地缺少产物时需要重新构建。

`PLAN=1` 校验配置并显示服务地址和容器列表。等待实际启动输出 `Development services ready`，再打开 [Web 工作台](http://127.0.0.1:8765)。

| 服务 | 默认入口 | 健康检查 |
| --- | --- | --- |
| Memory | [API 文档](http://127.0.0.1:8000/docs) | [`/healthz`](http://127.0.0.1:8000/healthz) |
| Skill | [API 文档](http://127.0.0.1:8001/docs) | [`/health`](http://127.0.0.1:8001/health) |
| UI | [工作台](http://127.0.0.1:8765) | 打开页面，确认可加载 **Settings** |
| Kafka UI | [消息和消费者](http://127.0.0.1:8080) | — |
| Grafana | [监控面板](http://127.0.0.1:3000) | — |

`make dev` 默认启用两个 API、UI、Kafka 任务和 OTLP 监控。Memory 使用 Qdrant 和 Neo4j，Skill 示例使用独立的 PostgreSQL。启动器按最终配置的连接地址选择本地容器。

## 2. 部署 UI

### 随本地 API 启动

默认鉴权模式下，`make dev` 和 `make api UI=1` 会把本次启动的 API 地址和项目凭据写入 `.mindmemos/settings.v2.json`。用 `UI_HOME` 指定其他配置目录。用 `PROJECT=<项目名>` 覆盖项目文件中的 `default_project`。两个 API 仍接受项目文件中各自有权访问的项目凭据。

### 连接官方服务或已有 API

已有 Python 依赖时，直接从构建步骤开始。仅运行 UI 的新环境可以按下方命令安装：

```bash
uv sync --package mindmemos-ui --inexact
make ui-build
UV_NO_SYNC=1 make ui
```

`make ui` 只启动工作台，读取现有的 SDK 配置。它默认使用仓库中的 `.mindmemos/`，也可以通过 `UI_HOME` 或 `MINDMEMOS_HOME` 指定配置目录。要隔离不同环境的连接，可为每个环境使用独立目录。例如：

```bash
UV_NO_SYNC=1 make ui UI_HOME="$PWD/.mindmemos/cloud" UI_PORT=8865
```

打开 [工作台](http://127.0.0.1:8765)，或使用启动时输出的地址，进入 **Settings → SDK configuration**。

- **MindMemOS** 使用各自的官方服务地址。Skill 入口需要带 Skill 权限的项目 Key，可单独填写。
- **Self-hosted** 填写已有 Memory API 的根地址和服务 API key。连接独立 Skill API 时，另填 Skill 根地址和对应凭据。留空可仅配置 Memory。只连接独立 Skill 时，Memory key 可以留空。

点击 **Save configuration**，在侧栏 **Service availability** 查看状态。**Available** 表示健康接口已响应。打开 **Memories** 或 **Skills** 加载列表，验证业务凭据与权限。模型连接需通过实际调用模型的业务请求验证。

UI 固定监听 `127.0.0.1`，供运行 UI 的机器本地访问。UI 后端通过 SDK 连接配置的 API。

## 3. 按需启动和管理

### 选择服务和功能

`SERVICE=all|memory|skill` 默认选中两个服务。Skill API 可以单独启动。下表适用于 `make dev`、`make docker` 和 `make api`。

| `FEATURE` | Memory 任务 | Skill 任务 | OTLP 监控 |
| --- | --- | --- | --- |
| `core` | 同步 | 本地异步 | 关闭 |
| `async` | 同步 + Kafka 异步 | Kafka | 关闭 |
| `all`（默认） | 同步 + Kafka 异步 | Kafka | 开启 |

```bash
make dev SERVICE=skill FEATURE=core         # Skill API 和 UI
make dev SERVICE=memory FEATURE=core UI=0   # Memory API，关闭 UI
```

`UI=0` 或 `UI=1` 可覆盖 `make dev` 和 `make api` 的 UI 默认值。选项覆盖写入启动时使用的临时配置，源 YAML 保留原值。

### 分开启动或使用已有基础设施

```bash
make docker FEATURE=core             # 仅启动所需的本地容器
make api FEATURE=core UI=1           # 使用相同范围启动 API，并自动配置 UI
```

`make api` 要求依赖已就绪，默认只启动 API。使用外部数据库、Kafka 或 OTLP 时，在 YAML 或 `.env` 填写连接地址；启动器只为本机回环地址选择容器。使用 `SERVICE_AUTH=1` 保留各服务 YAML 中的鉴权设置，并在 UI 的 **Settings** 中单独配置连接。

| 参数 | 用途 |
| --- | --- |
| `API_PORT`、`SKILL_PORT`、`UI_PORT` | 覆盖应用端口 |
| `MEMORY_CONFIG`、`SKILL_CONFIG` | 指定服务 YAML |
| `PROJECT_CONFIG` | 指定本地项目 YAML |
| `API_RELOAD=1`、`SKILL_RELOAD=1` | 开发时开启 API 热重载 |

### 查看状态和停止

- `make status` 查看受管理的应用、地址和容器。
- `Ctrl+C` 停止当前终端启动的 API 和 UI，保留容器。
- `make stop` 停止受管理的应用和当前 Compose 项目的容器，保留数据。需要缩小启动范围时先执行它，再用新参数启动。
- `make clean` 在确认后删除当前 Compose 项目的 Docker 数据卷。日常重启使用 `make stop`。

## 任务环境与数据

训练和评测按任务准备数据集、Agent 程序及系统依赖。下载 Hugging Face 受限数据前，先取得数据集访问权限，再把读取令牌填入 `.env` 的 `HF_TOKEN`。下载和运行命令见 [Skill 评测指南](../evaluation/skill_evaluation_ZH.md)。

SpreadsheetBench 启用公式重算时需要 LibreOffice。macOS 可安装到 `/Applications`，Ubuntu/Debian 可运行 `sudo apt install libreoffice-calc`。可在 `.env` 中将 `SPREADSHEETBENCH_SOFFICE_PATH` 设为 `soffice` 的绝对路径，例如 `/Applications/LibreOffice.app/Contents/MacOS/soffice` 或 `/usr/bin/soffice`。运行该可执行文件的 `--version` 确认安装。

直接从 Python 使用 Skill 时，参见 [Skill 服务文档](../../src/mindmemos_skill/README.md)。
