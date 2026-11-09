# MindMemOS 部署指南

[English](instruction.md) | 简体中文

本文说明如何本地拉起MindMemOS的记忆和Skill服务。

## 0. Overview：包与服务架构

MindMemOS 当前包含五个 Python 包。其中`mindmemos` 和 `mindmemos_skill` 分别提供 Memory 和 Skill 两个算法核心服务。SDK、评测工具和 UI 负责接入及使用这些服务。

### 0.1 包与功能


| 包 | 主要功能 |
| --- | --- |
| `src/mindmemos` | 提供 Memory 服务。支持记忆写入、检索、查询、修改、删除、反馈和记忆巩固（dreaming）。支持Fastapi方式部署和使用。 |
| `src/mindmemos_skill` | 提供独立 Skill 服务。支持 Skill 注册、版本管理、轨迹保存、训练、自动演化、图组织和推荐。 支持fastapi和python直接调用。|
| `src/mindmemos_sdk` | 提供接入Memory、Skill的fastapi服务的SDK接口和CLI命令，并支持管理服务连接地址、凭据和鉴权。 |
| `src/mindmemos_eval` | 执行 Memory 和 Skill 评测，汇总评分、耗时和模型用量。服务调用通过 SDK 完成。 |
| `src/mindmemos_ui` | 提供本机 Web 工作台，通过 SDK 调用对接 Memory 和 Skill 服务。支持 Memory管理和搜索，Skill 和图谱管理，动态 Skill 预览，以及 SDK 配置编辑。|

### 0.2 服务启动后的架构

| 层次 | 服务连接与职责 |
| --- | --- |
| 接入 | 业务应用通过 HTTP 或 SDK 调用两个 API。浏览器连接 UI，UI 后端通过 SDK 调用两个 API。默认地址为 Memory `:8000`、Skill `:8001`、UI `:8765`。 |
| Memory 服务 | Memory API 执行记忆 pipeline，调用模型服务，并访问 Qdrant 和 Neo4j。Qdrant 提供向量存储与检索，Neo4j 保存图数据。 |
| Skill 服务 | Skill API 执行版本管理、训练、演化和图管理流程。默认部署使用独立 PostgreSQL，保存 Skill 正文、辅助文件、版本和轨迹等数据。本地目录保存运行产物。单独使用 Python Service 时也可配置 SQLite。 |
| 异步任务 | 两个服务使用同一套 Kafka 基础设施，各自处理任务消息。默认启动会启用相应消费者，Skill worker 随 Skill 服务启动。Worker 是任务执行组件，不是另一个 HTTP API。Kafka UI 用于查看消息。 |
| 监控 | 两个服务将监控数据发送到 OpenTelemetry Collector，再写入 ClickHouse，由 Grafana 展示。Kafka Exporter 提供 Kafka 指标。 |
| 外部能力 | 按配置连接 Chat、Embedding、可选的 Rerank 服务。Skill 训练还可能需要数据集、Agent 程序及任务环境依赖。 |

两个服务各自管理业务数据，UI 不直接访问业务数据库。共享 Kafka 和监控组件不意味着共享业务存储。

独立 Skill 服务可单独部署，不要求先启动 Memory API。


### 0.3 与 MindMemOS 1.0 的服务架构差异

1. 新增 skill 全链路算法能力。从原有的只支持 根据轨迹自动演化skill，变为支持skill训练、演化、组织的全流程算法。
2. 独立部署、独立存储。对于只需要skill服务的用户，支持本地独立部署skill服务，支持自定义环境和数据在自己的场景下使用算法训练skill。
3. 新增 UI 操作界面，支持切换对接 我们的官方服务和本地部署服务，方便记忆和Skill的查看、管理和编辑。
4. 降低部署难度，配置env后可自动分发模型api到config，不再需要配置繁复的config文件；支持关闭kafka、otel监控等额外功能。


## 1. 通过FastAPI启动服务

我们建议使用FastAPI启动服务。同时考虑到Skill训练能力需要本地部署，`mindmemos_skill` 还提供代码访问服务（见xxx节）。

### 1.1 软件与Python依赖

部署前需要安装以下软件：
- Git、Make 和 uv
- Docker 和 Docker Compose
- Node.js 和 npm，用于构建 UI

安装全量Python依赖：

```bash
uv sync --all-packages --extra postgres --extra kafka
```

其中：
- kafka负责记忆、Skill服务的异步任务管理
- postgres为Skill服务的默认后端数据库

### 1.2 初始化 & 填写配置文件

```bash
cp .env.example .env
```
并填写必要的ENV信息，包括：

| ENV 名字 | 用途 |
| --- | --- |
| `MINDMEMOS_CHAT_API_KEY` | Chat 模型提供商的 API key。 |
| `MINDMEMOS_CHAT_BASE_URL` | Chat 模型 API 的 Base URL。 |
| `MINDMEMOS_CHAT_MODEL` | Chat 模型名称，用于聊天、训练和测评中的聊天模型配置。 |
| `MINDMEMOS_EMBEDDING_API_KEY` | Embedding 模型提供商的 API key。 |
| `MINDMEMOS_EMBEDDING_BASE_URL` | Embedding 模型 API 的 Base URL。 |
| `MINDMEMOS_EMBEDDING_MODEL` | Embedding 模型名称，用于生成文本向量。 |
| `MINDMEMOS_EMBEDDING_DIMENSIONS` | Embedding 输出维度，必须为正整数；配置任意 Embedding 参数时必填，同时用于设置 Qdrant 的 `vector_size`。 |
| `MINDMEMOS_RERANK_API_KEY` | Rerank 模型提供商的 API key（可选）。 |
| `MINDMEMOS_RERANK_BASE_URL` | Rerank API 的完整端点地址，按提供商要求填写（可选）。 |
| `MINDMEMOS_RERANK_MODEL` | Rerank 模型名称，用于对召回结果重新排序（可选）。 |

执行以下命令从模板生成具体的配置脚本：

```bash
make init
```

该命令会从 `.example` 模板创建缺失的本地文件。已有文件不重新复制；随后会将 `.env` 中非空的模型参数写入 `config/` 下所有实际 YAML 配置，包括已有文件、Memory/Skill 服务、训练和测评配置。`.example.yaml` 模板保留公开示例值。

如果没有配置 CHAT、EMBEDDING、RERANK模型，或需要配置多种API供应商，需要在生成的config文件里自行添加。

额外依赖：

- `HF_TOKEN`：数据集下载脚本访问 Hugging Face 上的受限（gated）或私有数据集时使用的访问令牌。登录 Hugging Face，在 [Access Tokens](https://huggingface.co/settings/tokens) 中创建具有目标数据集读取权限的令牌，将完整值填入 `.env`，例如 `HF_TOKEN=hf_...`。受限数据集还需要先在数据集页面接受使用条款或申请访问权限；令牌本身不会授予数据集访问资格。下载允许匿名访问的公开数据集时可留空。详见 [Hugging Face 令牌说明](https://huggingface.co/docs/hub/security-tokens)。
- `SPREADSHEETBENCH_SOFFICE_PATH`：指定 LibreOffice 的 `soffice` 可执行文件，用于 SpreadsheetBench 启用公式重算时重新计算工作簿并保存公式结果缓存。取值为可执行文件的绝对路径，不是安装目录；只在需要重算的任务中使用。

安装 LibreOffice：macOS 可从 [LibreOffice 官网](https://www.libreoffice.org/installation-instructions/) 下载对应芯片架构的安装包，将应用安装到 `/Applications`；Ubuntu/Debian 可执行 `sudo apt update && sudo apt install libreoffice-calc`。

安装后，在 `.env` 中填写当前系统的实际路径，例如：

```dotenv
# macOS
SPREADSHEETBENCH_SOFFICE_PATH=/Applications/LibreOffice.app/Contents/MacOS/soffice
# Linux 常见路径（与上面的 macOS 配置二选一）
# SPREADSHEETBENCH_SOFFICE_PATH=/usr/bin/soffice
```

Linux 可通过 `command -v soffice` 查找路径。执行 `"<soffice 的绝对路径>" --version`，确认能输出 LibreOffice 版本。该变量留空时，程序会尝试查找打包的 LibreOffice、PATH 中的 `soffice` 和常见安装路径；填写绝对路径后，无需再将 `soffice` 加入 PATH。

### 1.3 准备 NLP 和 UI 资源

配置文件准备完成后，安装并检查 NLP 模型：

```bash
uv run --no-sync python scripts/install_nlp_assets.py
uv run --no-sync python scripts/check_nlp_assets.py
```

安装器读取 Memory 配置中的 spaCy 模型名称。
构建 UI：

```bash
make ui-build
```

注意：`make dev` 启动命令不执行 UI 构建。缺少前端资源时，UI 页面可能无法打开。

### 1.4 启动部署

执行：

```bash
make dev
```

该命令等价于：

```bash
make dev SERVICE=all FEATURE=all UI=1
```

启动内容如下：

| 类别 | 内容 |
| --- | --- |
| Python 服务 | Memory API、Skill API、UI |
| Memory 存储 | Qdrant、Neo4j |
| Skill 存储 | PostgreSQL，由 Skill 的数据库配置决定 |
| 异步任务 | Kafka、Kafka UI，Skill 同时启用任务 worker |
| 监控 | OpenTelemetry Collector、ClickHouse、Grafana、Kafka Exporter |

### 1.5 检查服务并停止

默认地址如下。修改端口后，使用启动计划中的实际地址。

| 服务 | 地址 |
| --- | --- |
| Memory 健康检查 | `http://127.0.0.1:8000/healthz` |
| Memory API 文档 | `http://127.0.0.1:8000/docs` |
| Skill 健康检查 | `http://127.0.0.1:8001/health` |
| Skill API 文档 | `http://127.0.0.1:8001/docs` |
| UI | `http://127.0.0.1:8765` |
| Kafka UI | `http://127.0.0.1:8080` |
| Grafana | `http://127.0.0.1:3000` |

在另一个终端检查健康状态：

```bash
curl -f http://127.0.0.1:8000/healthz
curl -f http://127.0.0.1:8001/health
make status
```

健康检查通过只说明服务已响应。用配置的模型执行一次业务请求，才能检查模型连接。
停止方式如下：

| 操作 | 结果 |
| --- | --- |
| 前台终端按 `Ctrl+C` | 停止此次启动的 API 和 UI，保留运行中的容器 |
| `make stop` | 停止工作区管理的应用和当前 Compose 项目的容器，保留数据 |
| `make clean` | 确认后删除当前 Compose 项目的 Docker 数据卷及其中的数据 |


#### 1.6 精简部署范围

全量部署依赖Kafka、OpenTelemtry、Clickhouse等异步任务管理、监控等功能，本地部署可简化。我们同样提供部署范围控制：


| 参数 | 值 | Memory | Skill |
| --- | --- | --- | --- |
| `SERVICE` | `all` | 启动 Memory API | 启动 Skill API |
| `SERVICE` | `memory` | 启动 Memory API | 不启动 Skill API |
| `SERVICE` | `skill` | 不启动 Memory API | 启动 Skill API |
| `FEATURE` | `core` | 启用同步功能；关闭 Kafka 异步任务和 OTLP 导出 | 启用同步功能和本地异步任务管理；关闭 Kafka 和 OTLP 导出 |
| `FEATURE` | `async` | 启用同步功能和 Kafka 异步任务；关闭 OTLP 导出 | 启用同步功能和 Kafka 异步任务管理；关闭 OTLP 导出 |
| `FEATURE` | `all` | 启用同步功能、Kafka 异步任务和 OTLP 导出 | 启用同步功能、Kafka 异步任务管理和 OTLP 导出 |
| `UI` | `0`、`1` | 关闭或启动共用 UI；`make dev` 默认启动 | 关闭或启动共用 UI；`make dev` 默认启动 |

切换 `FEATURE` 不会修改 `config/` 中的原始 YAML 文件，只会生成临时 YAML，供服务启动时读取。临时配置位于操作系统临时目录下的 `mindmemos-dev-<随机字符>/memory.yaml` 和 `mindmemos-dev-<随机字符>/skill.yaml`，具体路径每次启动都会变化，启动器退出后会清理该目录。

`FEATURE` 在运行配置中进行以下调整：

- `core`：关闭 Kafka，关闭 OTLP 监控数据导出。此时 Memory 服务只支持同步接口，Skill 服务采用 Python 原生异步任务管理。
- `async`：启用 Kafka 异步任务，关闭 OTLP 监控数据导出。
- `all`：启用 Kafka 异步任务，开启 OTLP 监控数据导出。此时可以通过 Grafana 面板监控 Memory 和 Skill 服务日志。

常用命令如下：

```bash
# 两个 API、UI；Memory 同步功能，Skill 本地异步任务。
make dev SERVICE=all FEATURE=core

# 仅 Memory API，不启动 UI。
make dev SERVICE=memory FEATURE=core UI=0

# 仅 Skill API，不启动 UI。
make dev SERVICE=skill FEATURE=core UI=0

# 两个 API 和异步任务，不启动 OTLP 监控。
make dev SERVICE=all FEATURE=async
```

一次只执行一个启动命令。切换范围前，先执行 `make stop`。
较小的启动范围不会自动停止以前启动的额外容器。

### 2.2 分开启动容器、API 和 UI

先启动同步功能所需的容器：

```bash
make docker SERVICE=all FEATURE=core
```

再启动 API：

```bash
make api SERVICE=all FEATURE=core
```

最后，在另一个终端启动 UI：

```bash
make ui
```
