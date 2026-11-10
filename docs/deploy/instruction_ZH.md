# MindMemOS 部署指南

[English](instruction.md) | 简体中文

从源码启动 Memory API、Skill API 和 Web 工作台。API 和 UI 在宿主机上以前台进程运行，Docker Compose 管理按需启动的存储、队列和监控组件。

- [启动本地服务和 UI](#1-启动本地服务和-ui)
- [部署 UI](#2-部署-ui)，随本地 API 启动，或单独连接官方服务和已有 API
- [按需启动和管理](#3-按需启动和管理)，选择单个服务、减少依赖或分开启动
- [迁移历史 Skill 和轨迹](#4-迁移历史-skill-和轨迹)

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

### 1.4 多模态 Memory

在 `config/mindmemos/dev.yaml` 中将 `algo_config.multimodal.mode` 设为 `basic` 或 `enhanced`。默认值 `"off"` 保留纯文本模式。两种模式均需 `blob_store` 指向 S3 或 MinIO。配置 `mm_embed_model_router` 后可按图文相似度检索，`enhanced` 还需要支持图片输入的聊天模型。配置和评测示例见[多模态评测](../evaluation/memory_evaluation_ZH.md#多模态记忆评测)。

`make dev SERVICE=memory UI=0` 在全局开启多模态、且存储地址指向本机时启动 MinIO 并初始化桶。默认端口为 9100。远端存储使用已有的 S3 地址、桶和凭据。若只对部分项目开启多模态，可用以下命令启动本地 MinIO：

```bash
docker compose --env-file .env -f dockers/compose.yaml up -d --wait minio
docker compose --env-file .env -f dockers/compose.yaml run --rm minio-init
```

生产安装需要 `mindmemos[s3]`。通过环境变量 `MINDMEMOS_S3_ENDPOINT`、`MINDMEMOS_S3_BUCKET`、`MINDMEMOS_S3_ACCESS_KEY`、`MINDMEMOS_S3_SECRET_KEY` 提供连接。若客户端无法访问内部地址，用 `MINDMEMOS_S3_PRESIGN_ENDPOINT` 指定外部签名地址。使用代理签名链接时，网关需转发 `GET /v1/memory/images/{blob_id}` 和原始查询参数。

已有部署可先保持 `"off"`，用现有集合名、项目 ID 和向量维度滚动升级 Memory API 和 Kafka worker。启用 `database.qdrant.auto_create` 时，启动会在已有 Add Record 集合中补齐溯源索引，已有记录会直接参与查询。开启多模态前确认新版 API 和 worker 全部就绪。需要回滚时先关闭多模态并让已排队的图片任务处理完，再回滚进程，保留 Qdrant 和对象存储数据。控制台的[生命周期与关系图接口](internal-memory-api.md)使用项目网关令牌。

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

## 4. 迁移历史 Skill 和轨迹

### Add 触发轨迹上报

给旧 Memory API 设置 `MINDMEMOS_LEGACY_SKILL_INGEST_URL`，填独立 Skill 服务的私有 HTTP 根地址。两端设置相同的 `MINDMEMOS_LEGACY_SKILL_INGEST_SECRET`。该凭据只用于内部轨迹入口，用户 Key 的权限仍由网关控制。

旧 Add 保存了 Skill 绑定后，Memory 把原项目和记录 ID 交给异步后台任务。请求无需等待新版服务。后台读取原记录并上报；旧版本补绑后会再次上报。新版使用原记录 ID 合并轨迹和版本索引。

保留下面的对账进程，用于导入历史数据、同步 Skill 版本，以及补齐进程重启或重试耗尽时尚未送达的轨迹。新版暂未导入关联版本时，内部入口返回 `202`，等待版本同步后重试。

### 持续单向同步

用独立运维进程从旧 Memory 同步到新版 Skill。旧 Qdrant 继续提供旧协议的数据。进程只读取 Platform 中仍有效的项目 namespace，每轮扫描旧集合并提交增量。Key 的权限变更只影响用户请求。

1. 先部署同一版本的独立 Skill 服务，初始化新版目标表，并部署支持 `skill.migration_pending` 的网关。
2. 为进程配置 `MINDMEMOS_QDRANT_API_KEY`、`MINDMEMOS_SKILL_MIGRATION_DATABASE_URL` 和 `MINDMEMOS_SKILL_MIGRATION_PLATFORM_DATABASE_URL`。Platform 连接只需要读取 `memory_projects.namespace` 和 `memory_projects.active`。旧 Qdrant 使用只读身份。
3. 由运维进程管理器启动下列命令并启用自动重启。实际轮询间隔为一轮耗时加上配置的等待时间。

```bash
uv run --no-sync python -m scripts.gateway.skill_migration \
  --qdrant-url "$MEMORY_QDRANT_URL" --apply --watch 30
```

在向旧用户开放新版入口前，核对 `legacy_skill_sync_projects` 已覆盖预期项目。首轮成功后 `ready` 为真。每轮输出项目计数和失败记录 ID。中断后重启相同命令，轨迹扫描从已提交游标继续。突然退出时，替代进程等到当前项目租约到期后接管。

新版服务复用原 Key。无新版权限返回 `403`。已发现但首次回填未就绪的项目，授权后返回 `503` 和 `skill.migration_pending`。已接管的家族保留新版激活指针和删除状态。源摘要和消费标记作为旧来源元数据保留。

停止同步时，停止该独立进程。旧 Qdrant 可继续处理请求，已写入新版的业务数据保留。恢复时使用相同的项目映射和目标数据库。旧 Skill 部署从未使用新版目标库，开发库结构按当前版本初始化。已有测试库需单独备份并使用独立的新库验证，避免清理其他开发进程正在使用的数据。

### 单次预演与导入

在网关的私有运维环境运行迁移任务。使用与独立 Skill 服务同一版本的代码。先启动独立 Skill 服务，初始化并备份目标数据库。为任务配置旧 Memory Qdrant 的只读连接，以及目标 Skill 数据库的读写连接。PostgreSQL 使用 `postgresql+psycopg` 和 `postgres` 依赖。测试用 SQLite 需提供已初始化的绝对文件路径。

脚本从旧库的注册版本自动发现全部有 Skill 的项目，并保留各自的 `project_id`。通过运行环境设置 `MINDMEMOS_QDRANT_API_KEY` 和 `MINDMEMOS_SKILL_MIGRATION_DATABASE_URL`。以下命令使用默认集合名。部署改过集合名时，用 `--version-collection`、`--blob-collection` 和 `--add-record-collection` 指定。

```bash
uv run --no-sync python -m scripts.gateway.skill_migration \
  --qdrant-url "$MEMORY_QDRANT_URL"
```

1. 用旧库的固定快照预演，核对各项目的家族、版本、轨迹和排除计数。
2. 对同一份快照加 `--apply` 写入。每个家族及其版本一起提交，每条轨迹与版本索引一起提交。
3. 再跑预演，确认 `would_migrate`、`failed` 和 `unprocessed` 都为零，且报告状态为 `completed`。使用网关的 v2 Skill 接口核对版本历史和轨迹。

迁移按项目和旧 ID 生成稳定的 `sk-` 家族 ID 和 `skv-` 版本 ID。父版本、当前版本和轨迹关联统一使用新 ID。旧 ID 保存在版本的来源信息中。正文保留原样，缺失的展示字段单独补齐。轨迹只保留同项目已注册版本的关联，正文和版本索引均写入独立 Skill 服务的轨迹库。普通记忆、待注册队列和演化摘要留在 Memory。旧的 `/v1/skills` 和 Memory 请求仍走原服务。迁移覆盖所选快照，快照之后写入旧入口的数据留在旧服务。

单个 Skill 或轨迹失败时，脚本继续处理其他记录。引用了失败 Skill 版本的轨迹整条记为失败，以便修复后重跑。无已注册关联的轨迹计入跳过。进程中断后，用同一份快照重跑，已提交记录会复用。

结束时，终端输出项目数、Skill 数、轨迹数、迁移结果和失败原因计数。预演的待写数量计入 `would_migrate`，实际新增计入 `migrated`，已存在计入 `already_migrated`。轨迹的 `scanned` 含跳过记录，`total` 为尝试迁移的记录数。

默认在当前目录保存带 UTC 时间戳的 `skill-migration-report-*.json`。可用 `--report PATH` 指定新的文件路径。报告的 `failed_skills` 和 `failed_trajectories` 包含旧记录 ID、项目和失败原因。Skill 条目也列出相关旧版本 ID。存在记录级失败时，命令完成并返回 0，报告状态为 `completed_with_failures`。源库扫描或迁移启动未完成时，报告状态为 `incomplete`，命令返回 1。报告写入失败时命令也返回 1。

运行 `uv run --no-sync pytest tests/scripts/test_gateway_skill_migration.py` 验证迁移。

## 任务环境与数据

训练和评测按任务准备数据集、Agent 程序及系统依赖。下载 Hugging Face 受限数据前，先取得数据集访问权限，再把读取令牌填入 `.env` 的 `HF_TOKEN`。下载和运行命令见 [Skill 评测指南](../evaluation/skill_evaluation_ZH.md)。

SpreadsheetBench 启用公式重算时需要 LibreOffice。macOS 可安装到 `/Applications`，Ubuntu/Debian 可运行 `sudo apt install libreoffice-calc`。可在 `.env` 中将 `SPREADSHEETBENCH_SOFFICE_PATH` 设为 `soffice` 的绝对路径，例如 `/Applications/LibreOffice.app/Contents/MacOS/soffice` 或 `/usr/bin/soffice`。运行该可执行文件的 `--version` 确认安装。

直接从 Python 使用 Skill 时，参见 [Skill 服务文档](../../src/mindmemos_skill/README.md)。
