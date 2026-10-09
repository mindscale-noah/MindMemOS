# MindMemOS SDK CLI：当前命令、接口、参数与默认值

核对日期：2026-10-08。本文以当前工作区源码为准，包括尚未提交的实现。SDK 包声明版本为 `0.1.4`。本文不保证 PyPI 上的同名版本包含当前工作区的全部行为。

## 1. 入口与实现范围

Python 包名为 `mindmemos-sdk`，导入名为 `mindmemos_sdk`。命令入口为 `mindmemos_sdk.cli:main`，安装后使用 `mindmemos`。

```bash
pip install mindmemos-sdk
mindmemos --help
mindmemos memory search --help
```

当前 CLI 只注册以下命令。这些命令已有处理逻辑。`cli.py` 顶部仍有“placeholder”说明，该说明与当前处理函数不一致。

| 命令 | 当前行为 | HTTP 接口 |
| --- | --- | --- |
| `auth` | 保存 Memory 和 Skill 连接配置 | 无 |
| `config show` | 显示本地连接配置 | 无 |
| `config reset` | 删除本地 `settings.json` | 无 |
| `config home set` | 保存默认配置目录 | 无 |
| `doctor` | 检查本地配置及客户端能否初始化 | 无 |
| `memory add` | 写入消息并提取记忆 | `POST /v1/memory/add` |
| `memory search` | 按查询检索记忆 | `POST /v1/memory/search` |
| `memory get` | 列出或筛选当前项目的记忆 | `POST /v1/memory/get` |
| `memory update` | 按 ID 更新内容 | `POST /v1/memory/update` |
| `memory delete` | 按 ID 删除记忆 | `POST /v1/memory/delete` |
| `memory feedback` | 提交显式反馈或触发隐式反馈 | `POST /v1/memory/feedback` |
| `memory dreaming` | 触发记忆整理流程 | `POST /v1/memory/dreaming` |

当前已接入 `mindmemos skill ...` 命令组，通过 `client.skills` 对接独立 Skill FastAPI；使用单数 `skill`。Python SDK 的 `memory.list()` 和 `memory.scroll()` 也没有 CLI 入口。独立服务的 `mindmemos-skill` 不属于本文的 SDK CLI。

Skill 的 25 个命令、SDK/FastAPI 对照、参数默认值及 JSON 输入见 [Skill CLI 当前实现](sdk_skill_cli_current.md)。此处记录当前工作区实现，不代表同版本 PyPI 包已发布。

每一级命令都支持 `-h` / `--help`。CLI 没有全局 `--base-url`、`--api-key`、`--timeout`、`--max-retries`、`--project-id` 或 `--version` 参数。连接参数只在 `auth` 下可用。

## 2. 配置来源与网络默认值

Memory 命令读取所选目录中的 `settings.json`。目录按以下优先级选择，使用第一个有效来源：

1. 环境变量 `MINDMEMOS_HOME`。
2. 旧环境变量 `MINDMEMOS_CONFIG_DIR`。
3. `~/.config/mindmemos/home.json` 中保存的 `home`。
4. `~/.mindmemos`。

目录环境变量支持 `~` 展开。CLI 不读取 `.env` 来获取 API key 或服务地址。Memory 命令要求配置文件已存在，且 Memory API key 非空。没有配置时，先执行 `mindmemos auth`。

| 配置字段 | 新配置默认值 | 实际含义 |
| --- | --- | --- |
| `version` | `2` | 本地配置格式版本 |
| `service_type` | `official` | 官方服务或自部署服务 |
| `memory_service.base_url` | `None` | 官方模式解析为 `https://mindmemos.cn` |
| `memory_service.auth.api_key` | `None` | Memory Bearer key，使用前必须配置 |
| `skill_service.base_url` | `None` | 复用 Memory 地址和 key |
| `skill_service.auth` | `None` | 独立 Skill 服务才保存独立认证 |
| `defaults.user_id` | `None` | 部分操作的默认业务用户 ID |
| `network.timeout_seconds` | `30` | HTTP 请求超时，单位为秒 |
| `network.max_retries` | `2` | 最多重试 2 次，总共最多尝试 3 次 |

已有配置中保存的网络值优先于新配置默认值。CLI 没有修改网络值的命令。Memory 请求使用同步 HTTP 客户端，即使业务参数选择 `--async`，CLI 仍会等待服务器返回排队结果。

请求 URL 为解析后的 Memory 根地址加上表中的接口路径。所有 Memory 操作都使用 POST 和 JSON 请求体，包括删除操作。请求头携带 `Authorization: Bearer <api_key>`。CLI 不发送 `project_id`，项目范围由服务端认证确定。

传输层只对超时和网络传输异常重试。默认两次重试前分别等待 1 秒和 2 秒。HTTP 错误状态和业务错误响应不触发该重试。

配置读取支持将旧 v1 配置升级为 v2 并保存。读取 v2 配置时，如果发现旧 `storage` 字段，也会移除该字段并保存。因此，部分读取命令可能更新旧配置文件。

## 3. `auth`：配置连接

```text
mindmemos auth [连接选项] [--user-id USER_ID]
```

| 参数 | 类型 / 取值 | 省略时的行为 |
| --- | --- | --- |
| `--service-type` | `official` 或 `custom` | 交互选择。已有配置默认使用原模式，新配置默认 `official` |
| `--memory-base-url` / `--base-url` | HTTP(S) URL | 自部署模式交互输入。已有自部署 Memory 地址可作为默认值 |
| `--memory-api-key` / `--api-key` | 字符串 | 隐藏输入。目标地址未变时，回车保留原 key |
| `--skill-base-url` | HTTP(S) URL | 自部署模式交互输入。直接回车表示复用 Memory |
| `--skill-shares-memory` | 布尔开关，默认 `False` | 未指定时仍需在自部署模式询问 Skill 地址 |
| `--skill-api-key` | 字符串 | 独立 Skill 服务隐藏输入。目标地址未变时，可保留原独立 key |
| `--user-id` | 非空字符串 | 交互输入。已有用户 ID 可作为默认值 |

`auth` 只校验和保存本地配置，不调用登录接口，也不验证 key 是否被服务器接受。

官方模式固定使用 `https://mindmemos.cn`。Skill 复用 Memory 的地址和 key。官方模式拒绝自定义 Memory 地址、Skill 地址、`--skill-shares-memory` 和独立 `--skill-api-key`。

自部署模式必须有 Memory 地址。`--skill-base-url` 与 `--skill-shares-memory` 互斥。复用 Memory 时不能传独立 Skill key。独立 Skill 地址必须有独立 key。URL 必须使用 HTTP 或 HTTPS，且不能包含账号、密码、查询参数或片段。保存前会删除末尾 `/`。

更新配置时，`auth` 保留原网络设置，但重新建立 Skill 连接设置。省略 `--skill-base-url` 不会自动保留原独立 Skill 地址，交互时回车会改为复用 Memory。

以下示例提供全部必需值，不需要交互输入：

```bash
mindmemos auth --service-type official --api-key YOUR_API_KEY --user-id u_123

mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --memory-api-key YOUR_MEMORY_KEY \
  --skill-shares-memory --user-id u_123

mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --memory-api-key YOUR_MEMORY_KEY \
  --skill-base-url http://127.0.0.1:8001 \
  --skill-api-key YOUR_SKILL_KEY --user-id u_123
```

## 4. 本地配置与检查命令

### 4.1 `config show`

| 参数 | 必填 | 默认值与行为 |
| --- | --- | --- |
| `--show-secret` | 否 | `False`。默认隐藏 key，只保留末尾 4 个字符。长度不超过 4 时全部隐藏 |

`config show` 显示配置路径、部署模式、Memory/Skill 地址和 key、默认 `user_id`。它不输出完整配置 JSON，也不显示网络字段。

### 4.2 `config reset`

| 参数 | 必填 | 默认值与行为 |
| --- | --- | --- |
| `--yes` / `-y` | 否 | `False`。默认询问确认，回车取消。传开关后直接删除 `settings.json` |

`config reset` 只删除所选目录中的 `settings.json`，保留其他文件和默认目录偏好。配置文件不存在时返回成功。

### 4.3 `config home set`

| 参数 | 必填 | 默认值与行为 |
| --- | --- | --- |
| `DIRECTORY` | 是 | 无默认值。必须为绝对路径，支持 `~` 展开 |

`config home set` 写入 `~/.config/mindmemos/home.json`。它不移动旧配置，也不复制 `settings.json`。环境变量仍优先于保存的默认目录。运行中的进程需要重启以使用新默认值。

### 4.4 `doctor`

`doctor` 无专用参数。它检查配置是否存在、能否解析、Memory key 是否配置及客户端能否初始化。

`doctor` 不发送 HTTP 请求。因此，`transport: ready` 只表示客户端能够初始化，不能证明服务器可访问、API key 有效或 Skill 接口可用。

## 5. Memory 身份参数

`add`、`search`、`feedback`、`dreaming` 支持以下参数。它们不是全局参数，不能用于 `get`、`update` 或 `delete`。

| 参数 | 类型 | CLI 默认值 | 请求体字段与回退行为 |
| --- | --- | --- | --- |
| `--user-id` | 字符串 | `None` | `user_id`。`add`、`feedback`、`dreaming` 回退到 `defaults.user_id`，`search` 不回退 |
| `--app-id` | 字符串 | `None` | `app_id`。CLI 没有持久化默认值，省略时不发送 |
| `--agent-id` | 字符串 | `None` | `agent_id`。省略时不发送 |
| `--session-id` | 字符串 | `None` | `session_id`。省略时不发送 |

`add` 最终必须有 `user_id`，否则 SDK 拒绝请求。`feedback` 和 `dreaming` 可以在没有默认用户时省略该字段。`search` 只有显式提供非空 `--user-id` 才发送 `user_id`。需要按用户检索时，应明确传入该参数。

请求构造会省略空身份字符串。`add`、`feedback`、`dreaming` 的空用户字符串会触发默认用户回退，不能用于清除默认用户。

## 6. `memory search`

```text
mindmemos memory search QUERY [选项]
```

接口：`POST /v1/memory/search`。

| 参数 | 类型 / 取值 | 必填 | 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `QUERY` | 字符串 | 是 | 无 | `query` |
| `--top-k` | 整数 | 否 | `10` | `top_k`，CLI 始终发送 |
| `--search-strategy` | `fast` / `agentic` | 否 | `fast` | `search_strategy` |
| `--rerank` | 布尔开关 | 否 | `False` | `rerank`，CLI 始终发送 |
| `--score-threshold` | 浮点数 | 否 | `None` | `score_threshold`，省略时不发送，只在 rerank 开启时有效 |
| `--token-budget` | 整数 | 否 | `None` | `token_budget`，省略时不发送，指定后启用严格 token 预算处理 |
| `--filter` | JSON 对象字符串 | 否 | `None` | `filters`，非空对象才发送 |
| `--json` | 布尔开关 | 否 | `False` | 只控制输出，不发送到接口 |
| 身份参数 | 见第 5 节 | 否 | `None` | 特别注意 `user_id` 不使用配置默认值 |

CLI 只检查整数或浮点数能否解析。当前仓库服务端要求 `top_k >= 1`、`0 <= score_threshold <= 1`、`token_budget >= 1`。空查询和范围错误由服务端校验。

CLI 无法把 `top_k` 设置为 JSON `null`，也没有 `--max-rounds`。当前仓库服务端的 agentic `max_rounds` 默认值为 `3`，CLI 不发送该字段。自部署服务可以有不同实现。

默认请求体如下，不包含配置中的默认用户：

```json
{"query":"tea","top_k":10,"search_strategy":"fast","rerank":false}
```

```bash
mindmemos memory search "tea" --user-id u_123 --top-k 5 --json
mindmemos memory search "tea" --rerank --score-threshold 0.5 --token-budget 1000
```

默认输出记忆序号、ID、内容及更新时间。`--json` 输出 SDK 的 `SearchResult`，主要字段为 `request_id` 和 `memories`，不保留完整服务器响应信封。

## 7. `memory add`

```text
mindmemos memory add [消息输入] [选项]
```

接口：`POST /v1/memory/add`。

| 参数 | 类型 / 取值 | 必填 | 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `--content` | 字符串 | 无 JSON 输入时必填 | `None` | 构造一条 `messages` 对话消息 |
| `--role` | `user` / `assistant` / `system` / `tool` | 否 | `user` | 仅用于 `--content` 构造的消息 |
| `--messages-json` | JSON 数组字符串 | 与文件选项二选一 | `None` | `messages`，覆盖 `--content` 和 `--role` |
| `--messages-json-file` | 文件路径或 `-` | 与内联选项二选一 | `None` | 读取 UTF-8 JSON 数组。`-` 表示标准输入 |
| `--metadata-json` | JSON 对象字符串 | 否 | `None` | `metadata`，空对象不发送 |
| `--async` | 布尔开关 | 否 | `False` | 未指定时发送 `mode="sync"`，指定后发送 `mode="async"` |
| `--json` | 布尔开关 | 否 | `False` | 只控制输出 |
| 身份参数 | 见第 5 节 | 用户最终必需 | `None` | 显式用户或配置默认用户 |

使用 `--content` 时，CLI 自动设置当前毫秒时间戳 `timestamp`。使用 JSON 输入时，CLI 不自动补充时间戳。消息数组必须非空，且每一项必须为 JSON 对象。CLI 不完整校验各对象的消息字段，服务端继续校验内容。

JSON 消息可使用 SDK 支持的结构：对话 `role/content/timestamp`、文本 `text`、URL `url`、文件 `file_name/file_path/file_type`。`timestamp` 可省略，`file_type` 可为空。文件消息传递的是引用信息，CLI 不上传文件内容。

```bash
mindmemos memory add --content "I prefer tea." --user-id u_123
mindmemos memory add --messages-json '[{"role":"user","content":"I prefer tea."}]' --async --json
mindmemos memory add --messages-json-file messages.json --metadata-json '{"source":"chat"}'
```

同步输出提取结果。异步返回 `queued` 时，CLI 输出请求 ID 并结束，不轮询完成状态。`--json` 输出 `AddResult`，包含 `code`、`request_id`、`memories`。

## 8. `memory get`

```text
mindmemos memory get [--filter FILTER_JSON] [--top-k TOP_K]
```

接口：`POST /v1/memory/get`。

| 参数 | 类型 | 必填 | 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `--filter` | JSON 对象字符串 | 否 | `None` | 非空对象发送为 `filters` |
| `--top-k` | 整数 | 否 | `None` | 省略时不发送 `top_k` |

默认请求体为 `{}`。当前仓库服务端对省略的 `top_k` 使用读取器默认页大小，不能将其理解为返回全部记忆。显式 `top_k` 要求至少为 `1`。

该命令不发送默认用户 ID，不支持身份参数或 `--json`。输出包含记忆序号、ID、内容及更新时间。

```bash
mindmemos memory get --filter '{"app_id":"a1"}' --top-k 20
```

## 9. `memory update`

```text
mindmemos memory update MEMORY_ID --content CONTENT
```

接口：`POST /v1/memory/update`。

| 参数 | 类型 | 必填 | 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `MEMORY_ID` | 字符串 | 是 | 无 | `memory_id` |
| `--content` | 字符串 | 是 | 无 | `content` |

请求体为 `{"memory_id":"...","content":"..."}`。该命令不发送默认用户 ID，不支持身份参数或 `--json`。输出包含状态及可用的请求 ID。

```bash
mindmemos memory update MEMORY_ID --content "I prefer coffee."
```

## 10. `memory delete`

```text
mindmemos memory delete MEMORY_ID [--yes]
```

接口：`POST /v1/memory/delete`。

| 参数 | 类型 | 必填 | 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `MEMORY_ID` | 字符串 | 是 | 无 | `memory_id` |
| `--yes` / `-y` | 布尔开关 | 否 | `False` | 默认询问确认，回车取消。该开关不进入请求体 |

请求体为 `{"memory_id":"..."}`。该命令不发送默认用户 ID，不支持身份参数或 `--json`。输出包含状态及可用的请求 ID。

```bash
mindmemos memory delete MEMORY_ID --yes
```

## 11. `memory feedback`

接口：`POST /v1/memory/feedback`。

| 参数 | 类型 | 必填 | 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `--text` | 字符串 | 否 | `None` | `feedback`，省略时触发隐式反馈 |
| `--messages-json` | JSON 对象数组 | 有非空 `--text` 时与文件二选一 | `None` | `messages` |
| `--messages-json-file` | 路径或 `-` | 有非空 `--text` 时与内联二选一 | `None` | 从 UTF-8 文件或标准输入读取 `messages` |
| `--recalled-memories-json` | JSON 对象数组 | 否 | `None` | `recalled_memories` |
| `--recalled-memories-json-file` | 路径或 `-` | 否 | `None` | 从文件或标准输入读取召回记忆 |
| 身份参数 | 见第 5 节 | 否 | `None` | 用户可回退到配置默认值 |

显式反馈要求非空消息数组。每一项必须为对象。召回记忆数组可以为空，每一项也必须为对象。同类内联选项和文件选项不能同时传入。省略 `--text` 时不能提供消息或召回记忆上下文。不要把两种文件选项都设为 `-`，它们会依次读取同一个标准输入流。

召回记忆可传最小结构 `{"id":"...","memory":"..."}`。SDK 会补充 `memory_type="fact"`、`last_update_at=""`、空 metadata 及其他可选上下文字段。该转换不替代服务端字段校验。

CLI 不提供 `--sync`、`--async` 或 `--json`。它不发送 `mode`，当前仓库服务端使用默认 `sync`。CLI 输出状态和可用的请求 ID。

```bash
mindmemos memory feedback --user-id u_123
mindmemos memory feedback --text "The recalled preference is incorrect." \
  --messages-json '[{"role":"user","content":"I prefer coffee."}]' \
  --recalled-memories-json '[{"id":"MEMORY_ID","memory":"I prefer tea."}]'
```

## 12. `memory dreaming`

接口：`POST /v1/memory/dreaming`。

| 参数 | 类型 | 必填 | 默认值 | 行为 |
| --- | --- | --- | --- | --- |
| `--sync` | 布尔开关 | 否 | `False` | 指定后发送 `mode="sync"` |
| `--async` | 布尔开关 | 否 | `False` | 指定后发送 `mode="async"` |
| 身份参数 | 见第 5 节 | 否 | `None` | 用户可回退到配置默认值 |

两个模式开关互斥。都不传时，CLI 发送 `mode="async"`。这与 `memory add` 默认同步的行为不同。CLI 没有 `--json`，也不轮询异步任务完成状态。

```bash
mindmemos memory dreaming --user-id u_123
mindmemos memory dreaming --sync --user-id u_123
```

## 13. JSON、响应与退出码

`--filter` 和 `--metadata-json` 必须为 JSON 对象。CLI 只解析 JSON 和检查顶层类型，不完整校验过滤 DSL。空对象通常不进入请求体。消息文件选项接受 UTF-8 文件，`-` 表示读取标准输入。包含空格的 JSON 或查询应使用 shell 引号。

传输层解析 `code`、`message`、`request_id`、`data` 响应信封。成功业务码为 `ok` 或 `queued`。HTTP 状态至少为 400，或响应存在其他业务码时，SDK 报错。返回内容不是 JSON 时也报错。

| 退出码 | 常见情形 |
| --- | --- |
| `0` | 命令完成、帮助输出、查询无结果、异步任务已排队、重置时配置不存在 |
| `1` | 配置缺失或无效、API/网络错误、用户取消删除或重置 |
| `2` | 参数解析错误、缺少子命令、JSON 输入错误、模式冲突、输入遇到 EOF |

无参数执行 `mindmemos` 会显示帮助并返回 `0`。只执行 `config`、`config home` 或 `memory` 会报告缺少子命令并返回 `2`。以上是显式处理的常见退出码，不表示所有操作系统异常都会被转换为这些码。

## 14. 维护依据

更新文档时，先检查以下源码，再核对命令帮助。旧教程与当前实现冲突时，本文记录的当前源码行为应优先核对。

| 源码 | 核对内容 |
| --- | --- |
| [SDK 包定义](../../src/mindmemos_sdk/pyproject.toml) | 版本、Python 要求及命令入口 |
| [CLI 实现](../../src/mindmemos_sdk/mindmemos_sdk/cli.py) | 命令、别名、CLI 默认值、交互、输出及退出码 |
| [根客户端](../../src/mindmemos_sdk/mindmemos_sdk/client.py) | 配置如何组装 Memory/Skill 客户端 |
| [Memory 请求核心](../../src/mindmemos_sdk/mindmemos_sdk/memory/core.py) | 接口路径和身份回退规则 |
| [Memory 模型与请求构造](../../src/mindmemos_sdk/mindmemos_sdk/memory/models.py) | 请求体字段、省略规则和结果结构 |
| [配置模型](../../src/mindmemos_sdk/mindmemos_sdk/config/models.py) | 服务解析和网络默认值 |
| [配置路径](../../src/mindmemos_sdk/mindmemos_sdk/config/paths.py) | 目录选择优先级 |
| [配置管理](../../src/mindmemos_sdk/mindmemos_sdk/config/manager.py) | 保存、重置及旧配置升级 |
| [HTTP 传输](../../src/mindmemos_sdk/mindmemos_sdk/transport.py) | 认证、重试和响应处理 |
| [当前仓库服务端 schema](../../src/mindmemos/mindmemos/api/schemas.py) | 服务端范围校验和省略字段默认值 |

本文区分 CLI 默认值、SDK 请求构造和当前仓库服务端默认值。部署版本变化后，应重新核对服务端行为。
