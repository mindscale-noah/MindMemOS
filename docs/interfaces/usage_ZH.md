# MindMemOS 接口使用指南

本文介绍通过 CLI 和 HTTP 使用 MindMemOS 服务的方法，按功能组织调用示例、参数和返回结果。

CLI 随 Python SDK 发布，安装 `mindmemos-sdk` 后可使用 `mindmemos` 命令。首次使用前，运行 `mindmemos auth`，按提示配置服务地址、API key 和默认用户。

HTTP 调用直接填写服务地址和凭据。本地 Memory 服务默认地址为 `http://127.0.0.1:8000`，部署方法见[部署指南](../deploy/instruction_ZH.md)。

独立 Skill 服务默认地址为 `http://127.0.0.1:8001`，CLI 使用 SDK 中的 Skill 连接配置。下文 `YOUR_API_KEY`、`YOUR_SKILL_API_KEY` 分别替换为目标服务凭据。Skill 写权限 `skill:write` 同时允许读取。

示例中的 ID、文件路径、模型连接和返回值用于说明格式，需替换为实际值。CLI 默认值使用 Python 写法；实际 JSON 使用 `null`、`false`、`true`。Skill CLI 除任务进度外默认输出 JSON，无需添加输出用的 `--json`。

评测命令由 `mindmemos-eval` 包提供，入口为 `mindmemos-eval`，也可用 `python -m mindmemos_eval`。仓库环境先执行 `uv sync --all-packages`，命令可通过 `uv run mindmemos-eval` 调用；配置模板与数据准备见[评测指南](../eval/README_ZH.md)。

**功能索引**

| 范围 | 功能 |
| --- | --- |
| Memory | [检索记忆](#api-1)、[写入记忆](#api-2)、[流式写入记忆](#api-3)、[读取记忆](#api-4)、[分页列出记忆](#api-5)、[游标遍历记忆](#api-6)、[修改记忆](#api-7)、[删除记忆](#api-8)、[反馈修正记忆](#api-9)、[巩固记忆](#api-10) |
| 独立 Skill | [注册 Skill](#api-11)、[列出 Skill](#api-12)、[查看 Skill](#api-13)、[提交 Skill 新版本](#api-14)、[查看版本历史](#api-15)、[比较 Skill 版本](#api-16)、[激活 Skill 版本](#api-17)、[修改 Skill 信息](#api-18)、[删除 Skill](#api-19)、[解析 Skill 引用](#api-20)、[解析 Skill 内容版本](#api-21)、[上报执行轨迹](#api-22)、[查看执行轨迹](#api-23)、[列出执行轨迹](#api-24)、[统计执行轨迹](#api-25)、[演化 Skill](#api-26)、[训练 Skill](#api-27)、[查看异步任务](#api-28)、[查看任务进度](#api-29)、[组织 Skill 图](#api-30)、[列出 Skill 图](#api-31)、[解析 Skill 图引用](#api-32)、[查看图快照历史](#api-33)、[读取图快照](#api-34)、[预览 Skill 图](#api-35)、[添加图成员](#api-36)、[移除图成员](#api-37)、[删除 Skill 图](#api-38)、[推荐 Skill](#api-39) |
| 服务与身份 | [检查 Memory 服务](#api-48)、[检查 Skill 服务](#api-49)、[查看凭据身份](#api-50) |
| 评测 CLI | [Memory 评测](#eval-memory)、[评测环境列表](#eval-envs)、[固定 Skill 测试](#eval-test)、[TreeSkill 评测](#eval-treeskill)、[Skill-Forest 评测](#eval-run)、[报告重建](#eval-report) |

<a id="memory"></a>

## 1. Memory


<a id="api-1"></a>

### 1.1. 检索记忆（`POST /v1/memory/search`，权限：`memory:read`）

根据查询内容检索相关记忆，支持限定用户、应用、Agent 和会话，通过过滤条件筛选记忆，并控制返回条数、检索策略、重排和 token 预算。

#### 1.1.1 HTTP

**调用示例**

检索用户 `u_123` 的咖啡偏好，最多返回 5 条记忆。将示例地址替换为目标服务地址，将 `YOUR_API_KEY` 替换为有效 API key。

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/search" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "用户喜欢喝什么咖啡？",
    "user_id": "u_123",
    "top_k": 5
  }'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

参数通过 JSON 请求体传入。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | string | 是 | — | 查询内容。<br>例如 `"用户喜欢喝什么咖啡？"`。 |
| `user_id` | string / null | 否 | `None` | 限定查询的用户。省略时查询整个项目的记忆，不限定用户。 |
| `top_k` | integer / null | 否 | `10` | 最多返回的记忆条数，必须大于等于 `1` |
| `search_strategy` | string | 否 | `"fast"` | `fast`：单轮检索。<br>`agentic`：多轮检索。 |
| `max_rounds` | integer | 否 | `3` | 多轮检索的最大轮数，必须大于等于 `1`。<br>仅在 `search_strategy=agentic` 时生效。 |
| `rerank` | boolean | 否 | `False` | 是否对召回结果重新排序，需要服务配置可用的 Rerank 模型。 |
| `score_threshold` | number / null | 否 | `None` | 最低重排相关性分数，范围为 `0`–`1`。<br>仅在 `rerank=true` 时生效。 |
| `token_budget` | integer / null | 否 | `None` | 返回记忆的 token 预算，整数值必须大于等于 `1`。<br>设置后按预算保留结果。 |
| `filters` | object / null | 否 | `None` | 结构化过滤条件，直接传入 JSON 对象。<br><br>**单个条件**：`{"session_id":"s_001"}`，仅检索会话 `s_001` 的记忆。<br><br>**组合条件**：限定会话为 `s_001`，且应用为 `app_a` 或 `app_b`。<br><code>{</code><br>&nbsp;&nbsp;<code>"AND": [</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>{"session_id": "s_001"},</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>{"app_id": {"in": ["app_a", "app_b"]}}</code><br>&nbsp;&nbsp;<code>]</code><br><code>}</code><br>`AND`：条件数组，所有条件都必须满足。<br>`in`：非空字符串数组，匹配其中任意一个值。 |
| `app_id` | string / null | 否 | `None` | 限定查询的应用。 |
| `agent_id` | string / null | 否 | `None` | 限定查询的 Agent。 |
| `session_id` | string / null | 否 | `None` | 限定查询的会话。 |

**返回结果**

成功时返回 HTTP `200`，响应中的 `code` 为 `ok`，`data.memories` 为记忆列表。以下是没有匹配记忆时的响应示例，`request_id` 为示例值：

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": {
    "memories": []
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `code` | 业务结果。`ok` 表示成功。 |
| `message` | 结果说明，可能为空。 |
| `request_id` | 请求标识，用于定位本次调用。 |
| `data.memories` | 匹配的记忆列表。没有匹配结果时为空数组。 |

</details>

#### 1.1.2 CLI

**调用示例**

检索用户 `u_123` 的咖啡偏好，最多返回 5 条记忆。服务地址和 API key 从 SDK 配置读取。

```bash
mindmemos memory search "用户喜欢喝什么咖啡？" \
  --user-id u_123 \
  --top-k 5
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `query` | string | 是 | — | 查询内容。<br>例如 `"用户喜欢喝什么咖啡？"`。 |
| `--user-id` | string | 否 | `None` | 限定查询的用户。省略时查询 SDK 配置中的默认用户的记忆。 |
| `--top-k` | integer | 否 | `10` | 最多返回的记忆条数，必须大于等于 `1`。 |
| `--search-strategy` | string | 否 | `"fast"` | `fast`：单轮检索。<br>`agentic`：多轮检索。 |
| `--rerank` | boolean | 否 | `False` | 是否对召回结果重新排序，需要服务配置可用的 Rerank 模型。<br>添加 `--rerank` 即启用。 |
| `--score-threshold` | number | 否 | `None` | 最低重排相关性分数，范围为 `0`–`1`。<br>仅在启用 `--rerank` 时生效。 |
| `--token-budget` | integer | 否 | `None` | 返回记忆的 token 预算，必须大于等于 `1`。<br>设置后按预算保留结果。 |
| `--filter` | string | 否 | `None` | 结构化过滤条件，传入 JSON 对象字符串。<br><br>**单个条件**：`--filter '{"session_id":"s_001"}'`，仅检索会话 `s_001` 的记忆。<br><br>**组合条件**：限定会话为 `s_001`，且应用为 `app_a` 或 `app_b`。<br><code>{</code><br>&nbsp;&nbsp;<code>"AND": [</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>{"session_id": "s_001"},</code><br>&nbsp;&nbsp;&nbsp;&nbsp;<code>{"app_id": {"in": ["app_a", "app_b"]}}</code><br>&nbsp;&nbsp;<code>]</code><br><code>}</code><br>`AND`：条件数组，所有条件都必须满足。<br>`in`：非空字符串数组，匹配其中任意一个值。<br>将对象写成一行，用单引号包裹，作为 `--filter` 的值。 |
| `--app-id` | string | 否 | `None` | 限定查询的应用。 |
| `--agent-id` | string | 否 | `None` | 限定查询的 Agent。 |
| `--session-id` | string | 否 | `None` | 限定查询的会话。 |
| `--json` | boolean | 否 | `False` | 是否以 JSON 格式输出结果。<br>添加 `--json` 即启用。 |

**返回结果**

默认逐条显示记忆编号、ID、内容和可用的更新时间。检索到记忆时，输出示例如下，ID、内容和时间均为示例值：

```text
1. [mem_example_001] 用户喜欢喝冰美式。  (2026-10-12 10:30:00)
2. [mem_example_002] 用户喝咖啡时通常不加糖。  (2026-10-12 10:35:00)
```

没有匹配结果时显示：

```text
No memories found.
```

添加 `--json` 可获取结构化结果，便于脚本读取：

```json
{
  "request_id": "req_example",
  "memories": [
    {
      "id": "mem_example_001",
      "memory": "用户喜欢喝冰美式。",
      "user_id": "u_123",
      "memory_type": "fact",
      "last_update_at": "2026-10-12 10:30:00",
      "event_time": null,
      "source_timestamp": null,
      "lineage": null,
      "metadata": {}
    },
    {
      "id": "mem_example_002",
      "memory": "用户喝咖啡时通常不加糖。",
      "user_id": "u_123",
      "memory_type": "fact",
      "last_update_at": "2026-10-12 10:35:00",
      "event_time": null,
      "source_timestamp": null,
      "lineage": null,
      "metadata": {}
    }
  ]
}
```
</details>


<a id="api-2"></a>

### 1.2. 写入记忆（`POST /v1/memory/add`，权限：`memory:write`）

从消息中提取并保存记忆，支持用户与会话范围、业务元数据和同步或异步执行。

#### 1.2.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/add" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "u_123",
  "messages": [
    {
      "role": "user",
      "content": "我喜欢喝冰美式。"
    }
  ]
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string / null | 是 | — | 记忆所属用户，必须提供；CLI 可使用 SDK 默认用户。 |
| `app_id` | string / null | 否 | `None` | 限定应用。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `agent_id` | string / null | 否 | `None` | 限定 Agent。 |
| `messages` | array | 是 | — | 非空消息数组。<br>对话：`[{"role":"user","content":"我喜欢喝冰美式。","timestamp":1791772200000}]`，`timestamp` 可省略，单位为毫秒。<br>文本：`[{"text":"我喜欢喝冰美式。"}]`。<br>URL：`[{"url":"https://example.com/notes"}]`。<br>文件：`[{"file_name":"notes.txt","file_path":"/service/path/notes.txt","file_type":"txt"}]`，路径须由服务访问，`file_type` 可省略。 |
| `mode` | string | 否 | `"sync"` | `"sync"`：等待执行完成。<br>`"async"`：提交后台任务，需要启用 Kafka。 |
| `metadata` | object | 否 | `{}` | 业务元数据。<br>例如 `{"category":"coffee"}`。 |
| `prompt_language` | string / null | 否 | `None` | 提取提示词语言，可选 `"EN"` 或 `"ZH"`。省略时由服务选择。 |
| `skill_context` | array / null | 否 | `None` | 旧版 Skill 的轨迹绑定信息。<br>例如 `[{"name":"coffee-helper","content_hash":"实际内容哈希","base_version_id":"旧版版本ID"}]`。<br>每项必填 `name`、`content_hash`，可选 `base_version_id`（默认 `""`）、`version_label`（默认 `None`）和 `usage`（默认 `None`，值为 `"injected"` 或 `"modified"`）。 |
| `score` | number / null | 否 | `None` | 本次轨迹的评分，供旧版 Skill 演化使用。 |
| `task_id` | string / null | 否 | `None` | 任务 ID，例如 `"task_example_001"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": {
    "memories": [
      {
        "operation": "add",
        "content": "用户喜欢喝冰美式。",
        "memory_id": "mem_example_001",
        "mem_type": "fact",
        "memory_type": "fact",
        "confidence": 0.95,
        "related_memory_ids": [],
        "graph_edge_count": 0
      }
    ]
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.memories` | 记忆数组。读取结果含 id、memory、user_id、memory_type、last_update_at 及可选的来源、谱系和元数据；写入结果含 operation、memory_id、content 和提取信息。 |

</details>

#### 1.2.2 CLI

**调用示例**

```bash
mindmemos memory add --content "我喜欢喝冰美式。" --user-id u_123
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--content` | string | 否 | `None` | 消息或记忆内容，例如 `"我喜欢喝冰美式。"`。 |
| `--messages-json` | string | 否 | `None` | 消息数组的 JSON 字符串。<br>例如 `'[ {"role":"user","content":"我喜欢喝冰美式。"} ]'`。 |
| `--messages-json-file` | string | 否 | `None` | 消息 JSON 数组文件路径，例如 `messages.json`。`-` 表示从标准输入读取。与 `--messages-json` 互斥。 |
| `--role` | string | 否 | `"user"` | 单条消息的角色：`user`、`assistant`、`system` 或 `tool`。 |
| `--user-id` | string | 否 | `None` | 限定用户。省略时使用 SDK 配置中的默认用户。 |
| `--app-id` | string | 否 | `None` | 限定应用。 |
| `--agent-id` | string | 否 | `None` | 限定 Agent。 |
| `--session-id` | string | 否 | `None` | 限定会话。 |
| `--metadata-json` | string | 否 | `None` | 元数据 JSON 对象。<br>例如 `'{"category":"coffee"}'`。 |
| `--async` | boolean | 否 | `False` | 提交后台任务，添加 `--async` 即启用，需要服务启用 Kafka。 |
| `--json` | boolean | 否 | `False` | 以 JSON 输出结果，添加 `--json` 即启用。 |

至少提供 --content、--messages-json、--messages-json-file 中的一种消息输入；消息数组优先于 --content 和 --role。

**返回结果**

```text
Added 1 memory item(s):
- [add] mem_example_001: 用户喜欢喝冰美式。
```

添加 `--json` 后的正常返回示例：

```json
{
  "code": "ok",
  "request_id": "req_example",
  "memories": [
    {
      "operation": "add",
      "content": "用户喜欢喝冰美式。",
      "memory_id": "mem_example_001",
      "mem_type": "fact",
      "confidence": 0.95,
      "related_memory_ids": [],
      "graph_edge_count": 0
    }
  ]
}
```

</details>


<a id="api-3"></a>

### 1.3. 流式写入记忆（`POST /v1/memory/add/stream`，权限：`memory:write`）

同步提取并保存记忆，通过 SSE 返回处理进度和最终写入结果。

#### 1.3.1 HTTP

**调用示例**

```bash
curl -N -X POST "http://127.0.0.1:8000/v1/memory/add/stream" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "u_123",
  "messages": [
    {
      "role": "user",
      "content": "我喜欢喝冰美式。"
    }
  ],
  "mode": "sync"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string / null | 是 | — | 记忆所属用户，必须提供。 |
| `app_id` | string / null | 否 | `None` | 限定应用。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `agent_id` | string / null | 否 | `None` | 限定 Agent。 |
| `messages` | array | 是 | — | 消息数组，每项包含角色和内容。<br>例如 `[{"role":"user","content":"我喜欢喝冰美式。"}]`。 |
| `mode` | string | 否 | `"sync"` | 仅支持 `"sync"`，不接受异步写入。 |
| `metadata` | object | 否 | `{}` | 业务元数据。<br>例如 `{"category":"coffee"}`。 |
| `prompt_language` | string / null | 否 | `None` | 提取提示词语言，可选 `"EN"` 或 `"ZH"`。省略时由服务选择。 |
| `skill_context` | array / null | 否 | `None` | 旧版 Skill 的轨迹绑定信息。<br>例如 `[{"name":"coffee-helper","content_hash":"实际内容哈希","base_version_id":"旧版版本ID"}]`。<br>每项必填 `name`、`content_hash`，可选 `base_version_id`（默认 `""`）、`version_label`（默认 `None`）和 `usage`（默认 `None`，值为 `"injected"` 或 `"modified"`）。 |
| `score` | number / null | 否 | `None` | 本次轨迹的评分，供旧版 Skill 演化使用。 |
| `task_id` | string / null | 否 | `None` | 任务 ID，例如 `"task_example_001"`。 |

响应为 `text/event-stream`，使用 curl 的 `-N` 选项即时显示事件。仅支持 `mode="sync"`。进度事件为 `progress`，完成事件为 `completed`，失败事件为 `error`。

**返回结果**

返回 SSE 事件，以下为成功完成的示例：

```text
event: progress
data: {"stage":"accepted","message":"Streaming add accepted."}

event: completed
data: {"stage": "completed", "message": "Add completed.", "data": {"memories": [{"operation": "add", "content": "用户喜欢喝冰美式。", "memory_id": "mem_example_001", "mem_type": "fact", "memory_type": "fact", "confidence": 0.95, "related_memory_ids": [], "graph_edge_count": 0}]}}
```

`progress` 表示处理中，`completed` 的 data.memories 为写入结果。发生错误时返回 error 事件及 stage、message。

</details>

#### 1.3.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="api-4"></a>

### 1.4. 读取记忆（`POST /v1/memory/get`，权限：`memory:read`）

按结构化条件读取项目内的记忆，支持限定返回条数。

#### 1.4.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/get" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "filters": {
    "user_id": "u_123"
  },
  "top_k": 5
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `filters` | object / null | 否 | `None` | 结构化过滤条件。<br>例如 `{"session_id":"s_001"}`，仅匹配该会话的记忆。<br>组合示例：`{"AND":[{"user_id":"u_123"},{"app_id":{"in":["app_a","app_b"]}}]}`。<br>`AND` 要求所有条件满足，`in` 匹配数组中的任意一个值。 |
| `top_k` | integer / null | 否 | `None` | 最多返回的记忆条数，必须大于等于 `1`。省略时使用读取流程的默认页大小。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": {
    "memories": [
      {
        "id": "mem_example_001",
        "memory": "用户喜欢喝冰美式。",
        "user_id": "u_123",
        "memory_type": "fact",
        "last_update_at": "2026-10-12 10:30:00",
        "event_time": null,
        "source_timestamp": null,
        "lineage": null,
        "metadata": {}
      }
    ]
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.memories` | 记忆数组。读取结果含 id、memory、user_id、memory_type、last_update_at 及可选的来源、谱系和元数据；写入结果含 operation、memory_id、content 和提取信息。 |

</details>

#### 1.4.2 CLI

**调用示例**

```bash
mindmemos memory get --filter '{"user_id":"u_123"}' --top-k 5
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--filter` | string | 否 | `None` | 过滤条件 JSON 对象字符串。<br>例如 `'{"session_id":"s_001"}'`。<br>组合例子：`'{"AND":[{"user_id":"u_123"},{"app_id":{"in":["app_a","app_b"]}}]}'`。 |
| `--top-k` | integer | 否 | `None` | 最多返回的条数，必须大于等于 `1`。 |

**返回结果**

```text
1. [mem_example_001] 用户喜欢喝冰美式。  (2026-10-12 10:30:00)
```

</details>


<a id="api-5"></a>

### 1.5. 分页列出记忆（`POST /v1/memory/list`，权限：`memory:read`）

分页查看项目记忆，支持身份筛选、结构化过滤、总条数和非激活记忆。

#### 1.5.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/list" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "u_123",
  "page": 1,
  "page_size": 20
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string / null | 否 | `None` | 限定业务用户。 |
| `app_id` | string / null | 否 | `None` | 限定应用。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `agent_id` | string / null | 否 | `None` | 限定 Agent。 |
| `filters` | object / null | 否 | `None` | 结构化过滤条件。<br>例如 `{"session_id":"s_001"}`，仅匹配该会话的记忆。<br>组合示例：`{"AND":[{"user_id":"u_123"},{"app_id":{"in":["app_a","app_b"]}}]}`。<br>`AND` 要求所有条件满足，`in` 匹配数组中的任意一个值。 |
| `page` | integer | 否 | `1` | 页码，从 `1` 开始。 |
| `page_size` | integer | 否 | `20` | 每页记忆条数，必须大于等于 `1`。 |
| `include_total` | boolean | 否 | `True` | 是否返回匹配记忆的总条数。 |
| `include_inactive` | boolean | 否 | `False` | 是否包含非 active 状态的记忆。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": {
    "memories": [
      {
        "id": "mem_example_001",
        "memory": "用户喜欢喝冰美式。",
        "user_id": "u_123",
        "memory_type": "fact",
        "last_update_at": "2026-10-12 10:30:00",
        "event_time": null,
        "source_timestamp": null,
        "lineage": null,
        "metadata": {}
      }
    ],
    "page": 1,
    "page_size": 20,
    "total": 1,
    "has_more": false
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.memories` | 记忆数组。读取结果含 id、memory、user_id、memory_type、last_update_at 及可选的来源、谱系和元数据；写入结果含 operation、memory_id、content 和提取信息。 |
| `data.page` | 当前页码。 |
| `data.page_size` | 每页条数。 |
| `data.total` | 匹配的总条数，未计算时为 None。 |
| `data.has_more` | 是否还有下一页。 |

</details>

#### 1.5.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="api-6"></a>

### 1.6. 游标遍历记忆（`POST /v1/memory/scroll`，权限：`memory:read`）

使用游标连续读取记忆，支持身份筛选、结构化过滤和每批条数。

#### 1.6.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/scroll" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "u_123",
  "limit": 100
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string / null | 否 | `None` | 限定业务用户。 |
| `app_id` | string / null | 否 | `None` | 限定应用。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `agent_id` | string / null | 否 | `None` | 限定 Agent。 |
| `filters` | object / null | 否 | `None` | 结构化过滤条件。<br>例如 `{"session_id":"s_001"}`，仅匹配该会话的记忆。<br>组合示例：`{"AND":[{"user_id":"u_123"},{"app_id":{"in":["app_a","app_b"]}}]}`。<br>`AND` 要求所有条件满足，`in` 匹配数组中的任意一个值。 |
| `limit` | integer | 否 | `100` | 最多返回的条数，整数值必须大于等于 `1`。 |
| `cursor` | string / null | 否 | `None` | 上一页返回的游标。首次调用省略，后续原样传入 `next_cursor`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": {
    "memories": [
      {
        "id": "mem_example_001",
        "memory": "用户喜欢喝冰美式。",
        "user_id": "u_123",
        "memory_type": "fact",
        "last_update_at": "2026-10-12 10:30:00",
        "event_time": null,
        "source_timestamp": null,
        "lineage": null,
        "metadata": {}
      }
    ],
    "next_cursor": null
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.memories` | 记忆数组。读取结果含 id、memory、user_id、memory_type、last_update_at 及可选的来源、谱系和元数据；写入结果含 operation、memory_id、content 和提取信息。 |
| `data.next_cursor` | 下一页游标，None 表示没有后续游标。 |

</details>

#### 1.6.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="api-7"></a>

### 1.7. 修改记忆（`POST /v1/memory/update`，权限：`memory:write`）

更新指定记忆的内容、元数据或状态。

#### 1.7.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/update" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "memory_id": "mem_example_001",
  "content": "用户现在更喜欢喝拿铁。"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string / null | 否 | `None` | 限定业务用户。 |
| `app_id` | string / null | 否 | `None` | 限定应用。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `agent_id` | string / null | 否 | `None` | 限定 Agent。 |
| `memory_id` | string | 是 | — | 目标记忆 ID，例如 `"mem_example_001"`。 |
| `content` | string / null | 否 | `None` | 新的记忆内容。 |
| `metadata_patch` | object | 否 | `{}` | 合并到现有记忆中的元数据。<br>例如 `{"category":"coffee"}`。 |
| `status` | string / null | 否 | `None` | 记忆状态，可选 `"active"` 或 `"archived"`。省略时保持原状态。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": null
}
```

| 字段 | 说明 |
| --- | --- |
| `data` | 本接口无业务返回数据，值为 null。 |

</details>

#### 1.7.2 CLI

**调用示例**

```bash
mindmemos memory update mem_example_001 --content "用户现在更喜欢喝拿铁。"
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `memory_id` | string | 是 | — | 目标记忆 ID，例如 `"mem_example_001"`。 |
| `--content` | string | 是 | — | 消息或记忆内容，例如 `"我喜欢喝冰美式。"`。 |

**返回结果**

```text
Updated mem_example_001. (request_id=req_example)
```

</details>


<a id="api-8"></a>

### 1.8. 删除记忆（`POST /v1/memory/delete`，权限：`memory:write`）

删除指定 ID 的记忆。

#### 1.8.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/delete" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "memory_id": "mem_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `memory_id` | string | 是 | — | 目标记忆 ID，例如 `"mem_example_001"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": null
}
```

| 字段 | 说明 |
| --- | --- |
| `data` | 本接口无业务返回数据，值为 null。 |

</details>

#### 1.8.2 CLI

**调用示例**

```bash
mindmemos memory delete mem_example_001 --yes
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `memory_id` | string | 是 | — | 目标记忆 ID，例如 `"mem_example_001"`。 |
| `--yes / -y` | boolean | 否 | `False` | 跳过删除确认，添加 `--yes` 或 `-y` 即启用。 |

**返回结果**

```text
Deleted mem_example_001. (request_id=req_example)
```

</details>


<a id="api-9"></a>

### 1.9. 反馈修正记忆（`POST /v1/memory/feedback`，权限：`memory:write`）

根据用户反馈和对话上下文修正记忆，也可分析近期写入记录，支持同步或异步执行。

#### 1.9.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/feedback" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "u_123",
  "feedback": "我现在更喜欢拿铁。",
  "messages": [
    {
      "role": "user",
      "content": "我现在更喜欢拿铁。"
    }
  ]
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string / null | 是 | — | 反馈所属用户，必须提供；CLI 可使用 SDK 默认用户。 |
| `app_id` | string / null | 否 | `None` | 限定应用。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `agent_id` | string / null | 否 | `None` | 限定 Agent。 |
| `feedback` | string / null | 否 | `None` | 反馈内容，例如 `"我现在更喜欢拿铁。"`。省略时分析近期写入记录。 |
| `messages` | array | 否 | `[]` | 消息数组，每项包含角色和内容。<br>例如 `[{"role":"user","content":"我喜欢喝冰美式。"}]`。 |
| `recalled_memories` | array | 否 | `[]` | 本轮召回的记忆数组。<br>例如 `[{"id":"mem_example_001","memory":"用户喜欢喝冰美式。","last_update_at":"2026-10-12 10:30:00"}]`。 |
| `mode` | string | 否 | `"sync"` | `"sync"`：等待执行完成。<br>`"async"`：提交后台任务，需要启用 Kafka。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": null
}
```

| 字段 | 说明 |
| --- | --- |
| `data` | 本接口无业务返回数据，值为 null。 |

</details>

#### 1.9.2 CLI

**调用示例**

```bash
mindmemos memory feedback --user-id u_123 --text '我现在更喜欢拿铁。' \
  --messages-json '[{"role":"user","content":"我现在更喜欢拿铁。"}]'
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--text` | string | 否 | `None` | 显式反馈内容。提供时必须同时提供消息数组。省略时分析近期写入记录。 |
| `--messages-json` | string | 否 | `None` | 消息数组的 JSON 字符串。<br>例如 `'[ {"role":"user","content":"我喜欢喝冰美式。"} ]'`。 |
| `--messages-json-file` | string | 否 | `None` | 消息 JSON 数组文件路径，例如 `messages.json`。`-` 表示从标准输入读取。与 `--messages-json` 互斥。 |
| `--recalled-memories-json` | string | 否 | `None` | 召回记忆数组的 JSON 字符串。<br>例如 `'[{"id":"mem_example_001","memory":"用户喜欢喝冰美式。","last_update_at":"2026-10-12 10:30:00"}]'`。 |
| `--recalled-memories-json-file` | string | 否 | `None` | 召回记忆数组的 JSON 文件。`-` 表示标准输入；与内联数组互斥。 |
| `--user-id` | string | 否 | `None` | 限定用户。省略时使用 SDK 配置中的默认用户。 |
| `--app-id` | string | 否 | `None` | 限定应用。 |
| `--agent-id` | string | 否 | `None` | 限定 Agent。 |
| `--session-id` | string | 否 | `None` | 限定会话。 |

提供 --text 时必须提供消息数组；省略 --text 时不提供上下文选项。

**返回结果**

```text
Feedback accepted. (request_id=req_example)
```

</details>


<a id="api-10"></a>

### 1.10. 巩固记忆（`POST /v1/memory/dreaming`，权限：`memory:write`）

整理和巩固已有记忆，支持身份范围及同步或异步执行。

#### 1.10.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8000/v1/memory/dreaming" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "u_123",
  "mode": "sync"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string / null | 否 | `None` | 限定业务用户。 |
| `app_id` | string / null | 否 | `None` | 限定应用。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `agent_id` | string / null | 否 | `None` | 限定 Agent。 |
| `mode` | string | 否 | `"async"` | `"sync"`：等待执行完成。<br>`"async"`：提交后台任务，需要启用 Kafka。 |

默认异步执行时 `code="queued"`，表示已入队；需要服务启用 Kafka。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "message": "",
  "request_id": "req_example",
  "data": null
}
```

| 字段 | 说明 |
| --- | --- |
| `data` | 本接口无业务返回数据，值为 null。 |

</details>

#### 1.10.2 CLI

**调用示例**

```bash
mindmemos memory dreaming --user-id u_123 --sync
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--sync` | boolean | 否 | `False` | 同步等待完成，添加 `--sync` 即启用，与 `--async` 互斥。 |
| `--async` | boolean | 否 | `False` | 提交后台任务，添加 `--async` 即启用，需要服务启用 Kafka。 |
| `--user-id` | string | 否 | `None` | 限定用户。省略时使用 SDK 配置中的默认用户。 |
| `--app-id` | string | 否 | `None` | 限定应用。 |
| `--agent-id` | string | 否 | `None` | 限定 Agent。 |
| `--session-id` | string | 否 | `None` | 限定会话。 |

**返回结果**

```text
Dreaming triggered. (request_id=req_example)
```

</details>


<a id="api-48"></a>

### 1.11. 检查 Memory 服务（`GET /healthz`，无需鉴权）

检查 Memory API 进程是否响应。

#### 1.11.1 HTTP

**调用示例**

```bash
curl -X GET "http://127.0.0.1:8000/healthz"
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

无请求参数。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "status": "ok"
}
```

| 字段 | 说明 |
| --- | --- |
| `status` | 任务或执行状态。 |

</details>

#### 1.11.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="api-50"></a>

### 1.12. 查看凭据身份（`GET /v1/auth/identity`，有效 Memory 凭据）

返回当前 Memory 凭据对应的项目和权限。

#### 1.12.1 HTTP

**调用示例**

```bash
curl -X GET "http://127.0.0.1:8000/v1/auth/identity" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

无请求参数。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "project_id": "default_project",
  "scopes": [
    "memory:read",
    "memory:write"
  ]
}
```

| 字段 | 说明 |
| --- | --- |
| `project_id` | 所属项目。 |
| `scopes` | 当前凭据授权的权限列表。 |

</details>

#### 1.12.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="eval-memory"></a>

### 1.13. 评测 Memory（`mindmemos-eval memory`）

执行配置中的记忆写入、检索、回答和评分流程，支持基准选择、算法、阶段开关、并发和模型参数。

**调用示例**

```bash
mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list locomo \
  --algorithm vanilla \
  --manifest-output output/memory_eval/manifest.jsonl \
  --api-key-output output/memory_eval/api_keys.yaml \
  --server-config config/mindmemos/dev.yaml \
  --limit 1
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--benchmark-config` | string | 是 | — | 评测矩阵 YAML，包含 runner、benchmarks 和 algorithm_profiles。<br>例如 `config/mindmemos_eval/memory_evaluation/locomo.yaml`。 |
| `--benchmark-list` | string | 是 | — | 要运行的基准名称，用逗号分隔。<br>例如 `locomo,longmemeval`，名称必须同时存在于配置和适配器中。 |
| `--manifest-output` | string | 是 | — | 本次评测的 JSONL 清单路径，例如 `output/memory_eval/manifest.jsonl`。 |
| `--api-key-output` | string | 条件必填 | `None` | 新评测生成的凭据 YAML 路径，须与 Memory 服务的 auth.api_key_file 一致。<br>新评测必须提供；使用 --reuse-api-key 时不使用此参数。 |
| `--algorithm` | string | 否 | `None` | 统一选择算法配置，可选 `vanilla` 或 `schema`。省略时按每个基准的 default_algorithm 选择。 |
| `--memory-algorithm` | string | 否 | `None` | 仅覆盖记忆算法绑定，可选 `vanilla` 或 `schema`。一般使用 --algorithm。 |
| `--base-url` | string | 否 | `None` | Memory API 地址。省略时使用评测 YAML 的 runner.base_url。 |
| `--timeout-seconds` | number | 否 | `None` | Memory HTTP 请求超时，单位秒。 |
| `--limit` | integer | 否 | `None` | 每个基准最多评测的样本数。省略时不限制。 |
| `--session-limit` | integer | 否 | `None` | 每个 LongMemEval 样本最多写入的会话数。 |
| `--top-k` | integer | 否 | `None` | 每次检索最多返回的记忆条数。 |
| `--search-strategy` | string | 否 | `None` | 检索策略，例如 `fast` 或 `agentic`。兼容的 vanilla、schema 值会映射为 fast。 |
| `--rerank`<br>`--no-rerank` | boolean | 否 | `None` | 是否请求重排，使用 --rerank 启用，--no-rerank 禁用。 |
| `--max-conv-concurrency` | integer | 否 | `None` | 同时执行的对话或样本构建任务数。 |
| `--max-qa-concurrency` | integer | 否 | `None` | 同时执行的问答任务数。 |
| `--max-search-concurrency` | integer | 否 | `None` | 同时执行的记忆检索请求数。 |
| `--max-score-concurrency` | integer | 否 | `None` | 同时执行的模型评分任务数。 |
| `--judge-runs` | integer | 否 | `None` | 每个问题的独立评分次数，多数票决定最终评分。 |
| `--add`<br>`--no-add` | boolean | 否 | `None` | 是否执行记忆写入阶段，使用 --add 启用，--no-add 跳过。 |
| `--score`<br>`--no-score` | boolean | 否 | `None` | 是否执行模型评分阶段，使用 --score 启用，--no-score 跳过。 |
| `--show-progress`<br>`--no-show-progress` | boolean | 否 | `None` | 是否显示进度，使用 --show-progress 启用，--no-show-progress 禁用。 |
| `--reuse-api-key` | string | 否 | `None` | 复用已有凭据 YAML 的第一个 API key，仅支持一个基准。<br>例如 `output/memory_eval/api_keys.yaml`。配合 --no-add 可复评已有记忆。 |
| `--qdrant-url` | string | 否 | `None` | 写入前清理评测项目所用的 Qdrant 地址。省略时读取 MINDMEMOS_QDRANT_URL，再回退到 http://localhost:6333。 |
| `--qdrant-api-key` | string | 否 | `None` | 项目清理所用的 Qdrant 凭据。省略时读取 MINDMEMOS_QDRANT_API_KEY。 |
| `--neo4j-uri` | string | 否 | `None` | 项目清理所用的 Neo4j Bolt 地址。省略时读取 MINDMEMOS_NEO4J_URI，再回退到 bolt://localhost:7687。 |
| `--neo4j-username` | string | 否 | `None` | 项目清理所用的 Neo4j 用户名。省略时读取 MINDMEMOS_NEO4J_USERNAME，再回退到 neo4j。 |
| `--neo4j-password` | string | 否 | `None` | 项目清理所用的 Neo4j 密码。省略时读取 MINDMEMOS_NEO4J_PASSWORD。 |
| `--neo4j-database` | string | 否 | `None` | 项目清理所用的 Neo4j 数据库。省略时读取 MINDMEMOS_NEO4J_DATABASE，再回退到 neo4j。 |
| `--skip-clean` | boolean | 否 | `False` | 跳过写入前的评测项目清理，添加即启用。默认执行写入时会清理本次 project_id 的记忆；--no-add 不清理。 |
| `--server-config` | string | 否 | `None` | Memory 服务 YAML，用于读取清理所需的实际 collection 名称。<br>例如 `config/mindmemos/dev.yaml`，省略时使用默认名称。 |
| `--llm-model` | string | 否 | `None` | 回答和评分共用的默认模型名称。 |
| `--llm-api-key` | string | 否 | `None` | 回答和评分共用的默认模型凭据。 |
| `--llm-base-url` | string | 否 | `None` | 回答和评分共用的 OpenAI 兼容模型地址。 |
| `--llm-temperature` | number | 否 | `None` | 回答和评分共用的采样温度。 |
| `--llm-max-tokens` | integer | 否 | `None` | 回答和评分共用的最大输出 token 数。 |
| `--llm-timeout` | number | 否 | `None` | 回答和评分共用的模型超时，单位秒。 |
| `--answer-llm-model` | string | 否 | `None` | 回答模型名称，覆盖通用 LLM 模型。 |
| `--answer-llm-api-key` | string | 否 | `None` | 回答模型凭据，覆盖通用 LLM 凭据。 |
| `--answer-llm-base-url` | string | 否 | `None` | 回答模型地址，覆盖通用 LLM 地址。 |
| `--judge-llm-model` | string | 否 | `None` | 评分模型名称，覆盖通用 LLM 模型。 |
| `--judge-llm-api-key` | string | 否 | `None` | 评分模型凭据，覆盖通用 LLM 凭据。 |
| `--judge-llm-base-url` | string | 否 | `None` | 评分模型地址，覆盖通用 LLM 地址。 |

CLI 中的 None 表示未覆盖该选项，运行参数从评测 YAML 合并；数据库清理连接按对应环境变量回退。

首次评测需准备基准数据及模型配置，并让服务的 auth.api_key_file 指向 --api-key-output 的同一文件。命令会生成该凭据文件，启用写入阶段时默认先清理本次评测项目，再写入数据。仅复评已有记忆时，可将 --api-key-output 替换为 --reuse-api-key output/memory_eval/api_keys.yaml，并添加 --no-add；此时只选择一个基准。


**返回结果**

命令执行时显示评测进度，结束后将每个基准的运行结果写为 --manifest-output 指定文件中的一行 JSON。下面为清单结构示例，eval_result 只展示指标部分，数值为示例值：

```json
{
  "benchmark": "locomo",
  "run_id": "eval_example_001",
  "key_id": "key_example_001",
  "project_id": "eval_project_001",
  "memory_algorithm": "vanilla",
  "api_key_file": "output/memory_eval/api_keys.yaml",
  "request_ids": {
    "add": [
      "req_add_example"
    ],
    "search": [
      "req_search_example"
    ]
  },
  "request_metadata": {},
  "eval_result": {
    "metrics": {
      "overall_accuracy": 1.0,
      "total_questions": 1,
      "total": 1,
      "correct": 1
    }
  },
  "started_at": "2026-10-12T02:30:00Z",
  "finished_at": "2026-10-12T02:31:00Z"
}
```

| 字段 | 说明 |
| --- | --- |
| `benchmark` | 本行结果对应的基准。 |
| `run_id` | 本次运行 ID。 |
| `key_id` | 评测凭据记录 ID。 |
| `project_id` | 本次评测的项目。 |
| `memory_algorithm` | 记忆算法绑定。 |
| `api_key_file` | 新生成或复用的服务凭据文件。 |
| `request_ids` | 写入和检索等调用的请求 ID 数组。 |
| `request_metadata` | 按请求 ID 保存的业务元数据。 |
| `eval_result` | 基准的原始结果与指标，结构由基准定义。 |
| `started_at` | 开始时间。 |
| `finished_at` | 结束时间。 |

正常完成时退出码为 0。评测项目清理失败时退出码为 1，其他未捕获异常也会以非零状态结束。


</details>


### 1.14. 通过 MindMemOSClient 操作 Memory

以下示例覆盖写入、检索、读取、分页与游标查询、修改、反馈、巩固和删除；服务需已配置可用的模型。

```python
from uuid import uuid4

from mindmemos_sdk import MindMemOSClient

user_id = "u_sdk_demo"
session_id = f"memory-demo-{uuid4().hex}"

with MindMemOSClient(
    base_url="http://127.0.0.1:8000",
    api_key="YOUR_API_KEY",
    user_id=user_id,
    session_id=session_id,
) as client:
    # Add messages and keep the returned memory identity.
    added = client.memory.add(
        messages=[{"role": "user", "content": "我喜欢喝冰美式，不加糖。"}],
        metadata={"category": "coffee"},
        mode="sync",
    )
    print(added.model_dump())
    memory_id = next(
        (item.memory_id for item in added.memories if item.memory_id), None
    )
    if memory_id is None:
        raise RuntimeError("No memory was extracted from the input.")

    # Search and read memories from this demonstration session.
    found = client.memory.search("用户喜欢喝什么咖啡？", top_k=5)
    print(found.model_dump())
    selected = {"user_id": user_id, "session_id": session_id}
    print(client.memory.get(filters=selected, top_k=10).model_dump())
    page = client.memory.list(page=1, page_size=20, include_total=True)
    print(page.model_dump())

    cursor = None
    while True:
        batch = client.memory.scroll(limit=100, cursor=cursor)
        print(batch.model_dump())
        cursor = batch.next_cursor
        if cursor is None:
            break

    # Update the selected memory and provide explicit feedback.
    print(
        client.memory.update(
            memory_id, "用户现在更喜欢拿铁，不加糖。"
        ).model_dump()
    )
    print(
        client.memory.feedback(
            feedback="我现在更喜欢拿铁，请更新咖啡偏好。",
            messages=[
                {"role": "user", "content": "我现在更喜欢拿铁，不加糖。"}
            ],
            recalled_memories=found.memories,
            mode="sync",
        ).model_dump()
    )
    print(client.memory.dreaming(mode="sync").model_dump())

    # Collect current session IDs before deleting demonstration records.
    memory_ids = []
    cursor = None
    while True:
        batch = client.memory.scroll(limit=100, cursor=cursor)
        memory_ids.extend(item.id for item in batch.memories)
        cursor = batch.next_cursor
        if cursor is None:
            break
    for current_id in memory_ids:
        print(client.memory.delete(current_id).model_dump())
```

Memory 方法返回带类型的结果，可用 `.model_dump()` 转为字典。流式写入和服务身份没有对应的 MemoryClient 方法，使用前面的 HTTP 接口。


<a id="skill"></a>

## 2. Skill


<a id="api-11"></a>

### 2.1. 注册 Skill（`POST /v1/skills/register`，权限：`skill:write`）

注册完整 Skill 正文和辅助文件，支持设置别名、元数据和初始版本标签。

#### 2.1.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/register" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "alias": "coffee",
  "skill": {
    "content": "---\nname: coffee-helper\ndescription: Answer questions about coffee preferences.\n---\n\nUse the user preferences to answer coffee questions.\n",
    "resources": {}
  }
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill` | object | 是 | — | 完整 Skill 内容和提交信息，子字段如下。<br>例如 `{"content":"---\nname: coffee-helper\ndescription: Answer coffee questions.\n---\n\nUse the user preferences.","resources":{}}`。 |
| `skill.metadata` | object | 否 | `{}` | Skill 家族元数据。<br>例如 `{"category":"coffee"}`。 |
| `skill.commit_message` | string / null | 否 | `None` | 本次提交说明，例如 `"Improve coffee answers"`。 |
| `skill.commit_version` | string / null | 否 | `None` | 用户指定的版本标签，例如 `"v0.0.1"`。省略时自动编号。 |
| `skill.version_bump` | string | 否 | `"patch"` | 自动编号增量：`"patch"`、`"minor"` 或 `"major"`。 |
| `skill.commit_source` | string | 否 | `"user"` | 提交来源，`"user"` 或 `"algo"`。 |
| `skill.content` | string | 是 | — | 完整 SKILL.md 文本，标准格式包含 `name` 和 `description` frontmatter。 |
| `skill.resources` | object | 否 | `{}` | 相对文件路径到 Base64 内容的映射。<br>例如 `{"references/notes.txt":"SGVsbG8="}`，表示文件内容为 `Hello`。 |
| `skill.name` | string / null | 否 | `None` | 显示名称。标准格式时可从 frontmatter 读取。 |
| `skill.description` | string / null | 否 | `None` | Skill 描述。标准格式时可从 frontmatter 读取。 |
| `skill.standard_format` | boolean | 否 | `True` | 是否要求正文带有包含 `name` 和 `description` 的标准 frontmatter。 |
| `alias` | string / null | 否 | `None` | 项目内唯一别名，例如 `"coffee"`。 |

本地目录需包含标准 SKILL.md。HTTP 传入正文与辅助文件内容，CLI 读取本地目录后上传；目录注册的默认别名为 frontmatter 中的 name。

**返回结果**

成功时返回 HTTP `201`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill_id": "sk_example_001",
    "skill_version_id": "v_example_001",
    "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "skill_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill_id` | Skill 家族 ID。 |
| `data.skill_version_id` | 不可变版本 ID。 |
| `data.content_hash` | SKILL.md 内容哈希。 |
| `data.skill_hash` | 完整 Skill 内容哈希。 |

</details>

#### 2.1.2 CLI

**调用示例**

```bash
mindmemos skill register ./skills/coffee-helper --alias coffee
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `source` | string | 二选一 | `None` | 本地 Skill 目录或 SKILL.md 路径，与 `--json-skill` 严格互斥。 |
| `--json-skill` | string | 二选一 | `None` | 完整 StructuredSkillInfo JSON 对象，与 `source` 严格互斥。 |
| `--alias` | string | 否 | `None` | Skill 或图的别名。 |
| `--version` | string | 否 | `"v0.0.1"` | 版本标签，例如 `v0.0.2`。 |
| `--message` | string | 否 | `"Initial registration"` | 提交说明。 |
| `--metadata-json` | string | 否 | `{}` | 元数据 JSON 对象。<br>例如 `'{"category":"coffee"}'`。 |
| `--include` | string | 否 | `None` | 辅助文件的 glob 包含规则，以逗号分隔，例如 `"references/**,assets/**"`。 |
| `--exclude` | string | 否 | `None` | 辅助文件的 glob 排除规则，以逗号分隔，例如 `"**/*.tmp"`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_001",
  "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "skill_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
}
```

也可在同一个注册命令中使用结构化内容。路径和 `--json-skill` 必须且只能提供一个。JSON 模式不接受 `--version`、`--message`、`--metadata-json`、`--include` 或 `--exclude`；提交信息和资源全部由 JSON 提供，不套用路径模式的默认值，`--alias` 仍可单独指定：

```bash
mindmemos skill register --alias coffee --json-skill '{"content": "---\nname: coffee-helper\ndescription: Answer questions about coffee preferences.\n---\n\nUse the user preferences to answer coffee questions.\n", "resources": {}}'
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--json-skill` | string | 是（JSON 模式） | — | StructuredSkillInfo JSON 对象。<br>例如 `'{"content":"---\nname: coffee-helper\ndescription: Answer coffee questions.\n---\nUse preferences.","resources":{}}'`。此处 `--json-skill` 是完整 Skill 输入。 |
| `--alias` | string | 否 | `None` | Skill 或图的别名。 |

返回 JSON 与目录注册相同。

</details>


<a id="api-12"></a>

### 2.2. 列出 Skill（`POST /v1/skills/list`，权限：`skill:read`）

列出项目内的 Skill，支持名称、别名、使用者筛选和游标分页。

#### 2.2.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/list" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "filters": {
    "name": "coffee*"
  },
  "limit": 20
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `filters` | object | 否 | `{}` | 筛选对象，条件同时满足。<br>例如 `{"name":"coffee*","alias":"coffee*","user_id":"u_123","agent_id":"agent_001"}`。<br>`name`、`alias` 支持区分大小写的通配符；`user_id`、`agent_id` 精确匹配同一条使用轨迹。 |
| `limit` | integer / null | 否 | `None` | 每页最多返回的 Skill 数，必须大于等于 `1`。省略时不设条数上限。 |
| `cursor` | string / null | 否 | `None` | 上一页返回的游标。首次调用省略，后续原样传入 `next_cursor`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "items": [
      {
        "skill_id": "sk_example_001",
        "name": "coffee-helper",
        "alias": "coffee",
        "metadata": {},
        "active_skill_version_id": "v_example_001",
        "created_at": "2026-10-12T02:30:00Z",
        "updated_at": "2026-10-12T02:30:00Z"
      }
    ],
    "limit": 20,
    "next_cursor": null,
    "has_more": false
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.items` | 当前页记录数组，具体记录字段见示例。 |
| `data.limit` | 本页条数上限。 |
| `data.next_cursor` | 下一页游标，None 表示没有后续游标。 |
| `data.has_more` | 是否还有下一页。 |

</details>

#### 2.2.2 CLI

**调用示例**

```bash
mindmemos skill list --name "coffee*" --limit 20
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--filter` | string | 否 | `{}` | Skill 筛选 JSON 对象。<br>例如 `'{"name":"coffee*","user_id":"u_123"}'`。<br>支持 `name`、`alias`、`user_id`、`agent_id`，与单独选项不能重复指定同一字段。 |
| `--name` | string | 否 | `None` | 名称筛选模式，例如 `"coffee*"`；更新信息时为新的名称。 |
| `--alias` | string | 否 | `None` | Skill 或图的别名。 |
| `--user-id` | string | 否 | `None` | 限定用户。省略时使用 SDK 配置中的默认用户。 |
| `--agent-id` | string | 否 | `None` | 限定 Agent。 |
| `--limit` | integer | 否 | `20` | 最多返回的条数，必须大于等于 `1`。 |

CLI 默认 limit=20，仅返回本次列表页；需要继续游标分页时使用 HTTP。

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "items": [
    {
      "skill_id": "sk_example_001",
      "name": "coffee-helper",
      "alias": "coffee",
      "metadata": {},
      "active_skill_version_id": "v_example_001",
      "created_at": "2026-10-12T02:30:00Z",
      "updated_at": "2026-10-12T02:30:00Z"
    }
  ],
  "limit": 20,
  "next_cursor": null,
  "has_more": false
}
```

</details>


<a id="api-13"></a>

### 2.3. 查看 Skill（`POST /v1/skills/show`，权限：`skill:read`）

查看 Skill 家族信息及完整版本内容，支持选择指定历史版本。

#### 2.3.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/show" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string / null | 否 | `None` | Skill ID，例如 `"sk_example_001"`。 |
| `skill_version_id` | string / null | 否 | `None` | 不可变 Skill 版本 ID，例如 `"v_example_001"`。 |

HTTP 必须提供 skill_id 或 skill_version_id；仅提供 skill_id 时读取激活版本。CLI 可直接使用别名或版本引用。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill": {
      "skill_id": "sk_example_001",
      "name": "coffee-helper",
      "alias": "coffee",
      "metadata": {},
      "active_skill_version_id": "v_example_001",
      "created_at": "2026-10-12T02:30:00Z",
      "updated_at": "2026-10-12T02:30:00Z"
    },
    "version": {
      "skill_version_id": "v_example_001",
      "version_label": "v0.0.1",
      "parent_skill_version_ids": [],
      "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "skill_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "name": "coffee-helper",
      "description": "Answer questions about coffee preferences.",
      "standard_format": true,
      "metadata": {},
      "commit_message": "Initial registration",
      "implementation": "static",
      "config_schema_version": 1,
      "config": {},
      "content": "---\nname: coffee-helper\ndescription: Answer questions about coffee preferences.\n---\n\nUse the user preferences to answer coffee questions.\n",
      "resources": {},
      "created_at": "2026-10-12T02:30:00Z"
    }
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill` | Skill 家族信息，包含 ID、名称、别名、元数据和激活版本。 |
| `data.version` | 版本内容和元数据；辅助文件 resources 使用 Base64 映射。 |

</details>

#### 2.3.2 CLI

**调用示例**

```bash
mindmemos skill show coffee
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "skill": {
    "skill_id": "sk_example_001",
    "name": "coffee-helper",
    "alias": "coffee",
    "metadata": {},
    "active_skill_version_id": "v_example_001",
    "created_at": "2026-10-12T02:30:00Z",
    "updated_at": "2026-10-12T02:30:00Z"
  },
  "version": {
    "skill_version_id": "v_example_001",
    "version_label": "v0.0.1",
    "parent_skill_version_ids": [],
    "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "skill_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "name": "coffee-helper",
    "description": "Answer questions about coffee preferences.",
    "standard_format": true,
    "metadata": {},
    "commit_message": "Initial registration",
    "implementation": "static",
    "config_schema_version": 1,
    "config": {},
    "content": "---\nname: coffee-helper\ndescription: Answer questions about coffee preferences.\n---\n\nUse the user preferences to answer coffee questions.\n",
    "resources": {},
    "created_at": "2026-10-12T02:30:00Z"
  }
}
```

导出选定版本到父目录下的 `<skill-name>/` 子目录。名称来自该版本的 `SKILL.md`；非标准格式使用版本的显式名称。例如名称为 `coffee-helper` 时：

```bash
mindmemos skill export coffee --to ./output
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--to` | string | 是 | — | 导出父目录，例如 `./output`；最终目录为 `./output/<skill-name>/`。 |
| `--overwrite` | boolean | 否 | `False` | 替换同名 Skill 子目录并移除旧文件；父目录中的其他内容保持不变。 |

返回的 `directory` 是最终 Skill 目录，可直接用于重新注册。UI 和 Python `Skill.export(directory)` 使用相同规则。始终追加名称：旧用法 `--to ./output/coffee-helper` 现在会写入 `./output/coffee-helper/coffee-helper/`，应改为 `--to ./output`。非法目录名会在写入前报错。

</details>


<a id="api-14"></a>

### 2.4. 提交 Skill 新版本（`POST /v1/skills/commit`，权限：`skill:write`）

保存完整 Skill 内容为新版本，支持父版本、辅助文件继承和是否激活。

#### 2.4.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/commit" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "base_version_id": "v_example_001",
  "skill": {
    "content": "---\nname: coffee-helper\ndescription: Answer questions about coffee preferences.\n---\n\nUse the user preferences to answer coffee questions.\n",
    "resources": {},
    "commit_message": "Improve coffee answers"
  },
  "active": false
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string | 是 | — | Skill ID，例如 `"sk_example_001"`。 |
| `skill` | object | 是 | — | 完整 Skill 内容和提交信息，子字段如下。<br>例如 `{"content":"---\nname: coffee-helper\ndescription: Answer coffee questions.\n---\n\nUse the user preferences.","resources":{}}`。 |
| `skill.metadata` | object | 否 | `{}` | 业务元数据。<br>例如 `{"category":"coffee"}`。 |
| `skill.commit_message` | string / null | 否 | `None` | 本次提交说明，例如 `"Improve coffee answers"`。 |
| `skill.commit_version` | string / null | 否 | `None` | 用户指定的版本标签，例如 `"v0.0.1"`。省略时自动编号。 |
| `skill.version_bump` | string | 否 | `"patch"` | 自动编号增量：`"patch"`、`"minor"` 或 `"major"`。 |
| `skill.commit_source` | string | 否 | `"user"` | 提交来源，`"user"` 或 `"algo"`。 |
| `skill.content` | string | 是 | — | 本次新版本的完整 SKILL.md 文本。 |
| `skill.resources` | object | 否 | `{}` | 相对文件路径到 Base64 内容的映射。<br>例如 `{"references/notes.txt":"SGVsbG8="}`，表示文件内容为 `Hello`。 |
| `skill.name` | string / null | 否 | `None` | 显示名称。 |
| `skill.description` | string / null | 否 | `None` | Skill 描述。标准格式时可从 frontmatter 读取。 |
| `skill.standard_format` | boolean | 否 | `True` | 是否要求正文带有包含 `name` 和 `description` 的标准 frontmatter。 |
| `base_version_id` | string / null | 否 | `None` | 父版本 ID。省略时使用当前激活版本。 |
| `active` | boolean | 否 | `True` | 是否将本次提交的新版本设为激活版本。 |
| `inherit_resources` | boolean | 否 | `False` | 是否继承父版本的辅助文件。启用时 `skill.resources` 必须为空。 |

HTTP 的 active 默认 True，CLI 仅在添加 --activate 时激活。CLI 默认继承父版本辅助文件；使用 --resources-dir 替换，使用 --clear-resources 清空。

**返回结果**

成功时返回 HTTP `201`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill_id": "sk_example_001",
    "skill_version_id": "v_example_002",
    "version_label": "v0.0.2",
    "active": false,
    "content_hash": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "skill_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
    "active_skill_version_id": "v_example_001"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill_id` | Skill 家族 ID。 |
| `data.skill_version_id` | 不可变版本 ID。 |
| `data.version_label` | 版本标签。 |
| `data.active` | 本次新版本是否激活。 |
| `data.content_hash` | SKILL.md 内容哈希。 |
| `data.skill_hash` | 完整 Skill 内容哈希。 |
| `data.active_skill_version_id` | 当前激活版本 ID。 |

</details>

#### 2.4.2 CLI

**调用示例**

```bash
mindmemos skill commit coffee --from ./skills/coffee-helper --message "Improve coffee answers"
```

路径与 JSON 输入必须且只能提供一个。JSON 模式不接受 `--message`、`--version`、`--bump`、`--resources-dir` 或 `--clear-resources`；非空提交说明放在 `commit_message` 中，版本信息和资源也由 JSON 提供。JSON 资源完整替换父版本资源，省略 `resources` 或传 `{}` 都表示清空；路径模式继续默认继承父资源。两种模式都可通过 `--activate` 激活新版本，默认不激活。

```bash
mindmemos skill commit coffee --json-skill '{"content":"---\nname: coffee-helper\ndescription: Answer coffee questions.\n---\nUse preferences.\n","resources":{},"commit_message":"Improve coffee answers","version_bump":"minor"}'
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--from` | string | 二选一 | `None` | 本地 Skill 目录或 SKILL.md 路径，与 `--json-skill` 严格互斥。 |
| `--json-skill` | string | 二选一 | `None` | 完整 StructuredSkillInfo JSON 对象，与 `--from` 严格互斥。 |
| `--message` | string | 路径模式必填 | `None` | 非空提交说明；JSON 模式使用对象中的 `commit_message`。 |
| `--version` | string | 否 | `None` | 版本标签，例如 `v0.0.2`。 |
| `--bump` | string | 否 | `None` | 自动版本增量：`patch`、`minor` 或 `major`。与 `--version` 互斥，省略时按 patch 编号。 |
| `--resources-dir` | string | 否 | `None` | 新的辅助文件目录。与 `--clear-resources` 互斥；两者均省略时继承父版本辅助文件。 |
| `--clear-resources` | boolean | 否 | `False` | 清空辅助文件，添加即启用。 |
| `--activate` | boolean | 否 | `False` | 将提交的新版本设为激活版本，添加即启用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_002",
  "version_label": "v0.0.2",
  "active": false,
  "content_hash": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "skill_hash": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "active_skill_version_id": "v_example_001"
}
```

</details>


<a id="api-15"></a>

### 2.5. 查看版本历史（`POST /v1/skills/history`，权限：`skill:read`）

读取 Skill 的版本历史，支持条数、偏移量和游标分页。

#### 2.5.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/history" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "limit": 20,
  "offset": 0
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string | 是 | — | Skill ID，例如 `"sk_example_001"`。 |
| `limit` | integer / null | 否 | `None` | 最多返回的条数，整数值必须大于等于 `1`。省略时不设条数上限。 |
| `offset` | integer | 否 | `0` | 跳过的版本条数，必须大于等于 `0`。不能与非空游标同时使用。 |
| `cursor` | string / null | 否 | `None` | 上一页返回的游标。首次调用省略，后续原样传入 `next_cursor`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "items": [
      {
        "skill_version_id": "v_example_001",
        "version_label": "v0.0.1",
        "commit_message": "Initial registration",
        "parent_skill_version_ids": [],
        "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "skill_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "implementation": "static",
        "status": "active",
        "origin": "user",
        "created_at": "2026-10-12T02:30:00Z"
      }
    ],
    "limit": 20,
    "offset": 0,
    "has_more": false,
    "next_cursor": null
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.items` | 当前页记录数组，具体记录字段见示例。 |
| `data.limit` | 本页条数上限。 |
| `data.offset` | 本页起始偏移。 |
| `data.has_more` | 是否还有下一页。 |
| `data.next_cursor` | 下一页游标，None 表示没有后续游标。 |

</details>

#### 2.5.2 CLI

**调用示例**

```bash
mindmemos skill history coffee --limit 20 --offset 0
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--limit` | integer | 否 | `20` | 最多返回的条数，必须大于等于 `1`。 |
| `--offset` | integer | 否 | `0` | 跳过的版本条数，必须大于等于 `0`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "items": [
    {
      "skill_version_id": "v_example_001",
      "version_label": "v0.0.1",
      "commit_message": "Initial registration",
      "parent_skill_version_ids": [],
      "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "skill_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "implementation": "static",
      "status": "active",
      "origin": "user",
      "created_at": "2026-10-12T02:30:00Z"
    }
  ],
  "limit": 20,
  "offset": 0,
  "has_more": false,
  "next_cursor": null
}
```

</details>


<a id="api-16"></a>

### 2.6. 比较 Skill 版本（`POST /v1/skills/diff`，权限：`skill:read`）

比较两个 Skill 版本的文件差异和实现类型，支持跨 Skill 对比。

#### 2.6.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/diff" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "from_skill_version_id": "v_example_001",
  "to_skill_version_id": "v_example_002"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string | 是 | — | Skill ID，例如 `"sk_example_001"`。 |
| `from_skill_version_id` | string / null | 否 | `None` | 对比的起始版本。省略时使用起始 Skill 的激活版本。 |
| `to_skill_id` | string / null | 否 | `None` | 目标 Skill ID。省略时对比同一 Skill 的版本。 |
| `to_skill_version_id` | string | 是 | — | 对比的目标版本 ID。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "from_skill_id": "sk_example_001",
    "from_skill_version_id": "v_example_001",
    "to_skill_id": "sk_example_001",
    "to_skill_version_id": "v_example_002",
    "files": [],
    "implementation_changed": false
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.from_skill_id` | 起始 Skill ID。 |
| `data.from_skill_version_id` | 起始版本 ID。 |
| `data.to_skill_id` | 目标 Skill ID。 |
| `data.to_skill_version_id` | 目标版本 ID。 |
| `data.files` | 文件差异数组，包含文件路径、变更类型、文本 diff 或二进制标记。 |
| `data.implementation_changed` | 实现类型是否发生变化。 |

</details>

#### 2.6.2 CLI

**调用示例**

```bash
mindmemos skill diff v_example_001 v_example_002
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref_a` | string | 是 | — | 对比的起始 Skill 或版本引用。 |
| `skill_ref_b` | string | 是 | — | 对比的目标 Skill 或版本引用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "from_skill_id": "sk_example_001",
  "from_skill_version_id": "v_example_001",
  "to_skill_id": "sk_example_001",
  "to_skill_version_id": "v_example_002",
  "files": [],
  "implementation_changed": false
}
```

</details>


<a id="api-17"></a>

### 2.7. 激活 Skill 版本（`POST /v1/skills/activate`，权限：`skill:write`）

将已有版本设为当前激活版本，支持检查调用前的激活版本。

#### 2.7.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/activate" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string / null | 否 | `None` | Skill ID，例如 `"sk_example_001"`。 |
| `skill_version_id` | string | 是 | — | 不可变 Skill 版本 ID，例如 `"v_example_001"`。 |
| `expected_active_version_id` | string / null | 否 | `None` | 调用前预期的激活版本 ID。与服务当前值不一致时拒绝修改。 |

CLI 的 --version 与 --version-id 必须二选一；版本标签应在该 Skill 中唯一。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill_id": "sk_example_001",
    "active_skill_version_id": "v_example_001"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill_id` | Skill 家族 ID。 |
| `data.active_skill_version_id` | 当前激活版本 ID。 |

</details>

#### 2.7.2 CLI

**调用示例**

```bash
mindmemos skill activate coffee --version-id v_example_001
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--version` | string | 否 | `None` | 版本标签，例如 `v0.0.2`。 |
| `--version-id` | string | 否 | `None` | 目标版本的精确 ID，与 `--version` 二选一。 |

--version 和 --version-id 必须二选一。

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "skill_id": "sk_example_001",
  "active_skill_version_id": "v_example_001"
}
```

</details>


<a id="api-18"></a>

### 2.8. 修改 Skill 信息（`POST /v1/skills/update-info`，权限：`skill:write`）

修改 Skill 的名称、别名或元数据，不创建新内容版本。

#### 2.8.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/update-info" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "name": "coffee-assistant"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string | 是 | — | Skill ID，例如 `"sk_example_001"`。 |
| `name` | string / null | 否 | `None` | 新的显示名称，省略时保持原值，不能传入 null。 |
| `alias` | string / null | 否 | `None` | 新的别名。显式传入 null 时清除别名，省略时保持原值。 |
| `metadata` | object / null | 否 | `None` | 替换家族元数据。<br>例如 `{"category":"coffee"}`，`{}` 清空元数据，不能传入 null。 |

至少提供 name、alias、metadata 中的一项。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill_id": "sk_example_001",
    "name": "coffee-assistant",
    "alias": "coffee",
    "metadata": {},
    "active_skill_version_id": "v_example_001",
    "created_at": "2026-10-12T02:30:00Z",
    "updated_at": "2026-10-12T02:30:00Z"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill_id` | Skill 家族 ID。 |
| `data.name` | Skill 显示名称。 |
| `data.alias` | 项目内别名。 |
| `data.metadata` | 业务元数据。 |
| `data.active_skill_version_id` | 当前激活版本 ID。 |
| `data.created_at` | 创建时间。 |
| `data.updated_at` | 更新时间。 |

</details>

#### 2.8.2 CLI

**调用示例**

```bash
mindmemos skill update-info coffee --name coffee-assistant
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--name` | string | 否 | `None` | 名称筛选模式，例如 `"coffee*"`；更新信息时为新的名称。 |
| `--alias` | string | 否 | `None` | Skill 或图的别名。 |
| `--clear-alias` | boolean | 否 | `False` | 清除别名，添加即启用，与 `--alias` 互斥。 |
| `--metadata-json` | string | 否 | `None` | 元数据 JSON 对象。<br>例如 `'{"category":"coffee"}'`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "skill_id": "sk_example_001",
  "name": "coffee-assistant",
  "alias": "coffee",
  "metadata": {},
  "active_skill_version_id": "v_example_001",
  "created_at": "2026-10-12T02:30:00Z",
  "updated_at": "2026-10-12T02:30:00Z"
}
```

</details>


<a id="api-19"></a>

### 2.9. 删除 Skill（`POST /v1/skills/delete`，权限：`skill:write`）

软删除整个 Skill 家族，并保留历史记录。

#### 2.9.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/delete" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string | 是 | — | Skill ID，例如 `"sk_example_001"`。 |

仍被保留轨迹或图数据引用的 Skill 可能返回 skill_in_use。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "deleted": true
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.deleted` | 是否已完成软删除。 |

</details>

#### 2.9.2 CLI

**调用示例**

```bash
mindmemos skill delete coffee --yes
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--yes / -y` | boolean | 否 | `False` | 跳过删除确认，添加 `--yes` 或 `-y` 即启用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "deleted": true
}
```

</details>


<a id="api-20"></a>

### 2.10. 解析 Skill 引用（`POST /v1/skills/skill-resolve`，权限：`skill:read`）

将 Skill 别名、ID 或唯一前缀解析为确定的 Skill 和版本。

#### 2.10.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/skill-resolve" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_ref": "coffee"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID、版本 ID 或唯一 ID 前缀，例如 `"coffee"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill_id": "sk_example_001",
    "skill_version_id": "v_example_001",
    "alias": "coffee",
    "is_active": true,
    "version_label": "v0.0.1"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill_id` | Skill 家族 ID。 |
| `data.skill_version_id` | 不可变版本 ID。 |
| `data.alias` | 项目内别名。 |
| `data.is_active` | 解析出的版本是否为当前激活版本。 |
| `data.version_label` | 版本标签。 |

</details>

#### 2.10.2 CLI

**调用示例**

```bash
mindmemos skill resolve coffee
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_001",
  "alias": "coffee",
  "is_active": true,
  "version_label": "v0.0.1"
}
```

</details>


<a id="api-21"></a>

### 2.11. 解析 Skill 内容版本（`POST /v1/skills/version-resolve`，权限：`skill:read`）

根据精确版本 ID 或内容哈希查找版本，支持限定所属 Skill。

#### 2.11.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/version-resolve" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_version_id": "v_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `content_hash` | string / null | 否 | `None` | SKILL.md 原文的 SHA-256，64 位小写十六进制字符串。 |
| `skill_hash` | string / null | 否 | `None` | 正文与辅助文件的完整 Skill 哈希，64 位小写十六进制字符串。 |
| `skill_id` | string / null | 否 | `None` | Skill ID，例如 `"sk_example_001"`。 |
| `skill_version_id` | string / null | 否 | `None` | 不可变 Skill 版本 ID，例如 `"v_example_001"`。 |

至少提供 content_hash、skill_hash、skill_version_id 中的一项。哈希对应多个版本时，需要进一步限定 Skill 或版本。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill_id": "sk_example_001",
    "skill_version_id": "v_example_001",
    "alias": "coffee",
    "commit_message": "Initial registration",
    "content_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "skill_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "version": "v0.0.1",
    "is_active": true
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill_id` | Skill 家族 ID。 |
| `data.skill_version_id` | 不可变版本 ID。 |
| `data.alias` | 项目内别名。 |
| `data.commit_message` | 版本提交说明。 |
| `data.content_hash` | SKILL.md 内容哈希。 |
| `data.skill_hash` | 完整 Skill 内容哈希。 |
| `data.version` | 版本内容和元数据；辅助文件 resources 使用 Base64 映射。 |
| `data.is_active` | 解析出的版本是否为当前激活版本。 |

</details>

#### 2.11.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="api-22"></a>

### 2.12. 上报执行轨迹（`POST /v1/skills/trajectories/report`，权限：`skill:write`）

保存 Agent 实际使用版本的执行轨迹，支持记录用户、会话、状态和执行时间。

#### 2.12.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/trajectories/report" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "user_id": "u_123",
  "agent_id": "agent_001",
  "skill_version_id": "v_example_001",
  "messages": [
    {
      "role": "user",
      "content": "我喜欢喝什么咖啡？"
    },
    {
      "role": "assistant",
      "content": "你喜欢喝冰美式。"
    }
  ]
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `user_id` | string | 是 | — | 执行任务的业务用户 ID。 |
| `agent_id` | string | 是 | — | 执行任务的 Agent ID。 |
| `messages` | array | 是 | — | 非空交互消息数组，每项须包含 role 和 content。<br>例如 `[{"role":"user","content":"我喜欢什么咖啡？"},{"role":"assistant","content":"你喜欢冰美式。"}]`。 |
| `skill_version_id` | string | 是 | — | 不可变 Skill 版本 ID，例如 `"v_example_001"`。 |
| `session_id` | string / null | 否 | `None` | 限定会话。 |
| `started_at` | string / null | 否 | `None` | 实际开始时间，使用带时区的 ISO 8601 字符串，例如 `"2026-10-12T02:30:00Z"`。 |
| `finished_at` | string / null | 否 | `None` | 实际结束时间，不能早于开始时间。<br>例如 `"2026-10-12T02:31:00Z"`。 |
| `status` | string | 否 | `"completed"` | 执行状态，例如 `"completed"` 或 `"failed"`。 |

**返回结果**

成功时返回 HTTP `201`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "trajectory_id": "traj_example_001",
    "skill_version_id": "v_example_001",
    "stored_at": "2026-10-12T02:30:00Z"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.trajectory_id` | 执行轨迹 ID。 |
| `data.skill_version_id` | 不可变版本 ID。 |
| `data.stored_at` | 服务接收轨迹的时间。 |

</details>

#### 2.12.2 CLI

**调用示例**

```bash
mindmemos skill trajectory report --json-data '{"user_id": "u_123", "agent_id": "agent_001", "skill_version_id": "v_example_001", "messages": [{"role": "user", "content": "我喜欢喝什么咖啡？"}, {"role": "assistant", "content": "你喜欢喝冰美式。"}]}'
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--json-data` | string | 否 | `None` | 完整轨迹请求 JSON。<br>例如 `'{"user_id":"u_123","agent_id":"agent_001","skill_version_id":"v_example_001","messages":[{"role":"user","content":"我喜欢什么咖啡？"}]}'`。 |
| `--file` | string | 否 | `None` | 轨迹 JSON 对象文件，`-` 表示从标准输入读取，与 `--json-data` 二选一。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "trajectory_id": "traj_example_001",
  "skill_version_id": "v_example_001",
  "stored_at": "2026-10-12T02:30:00Z"
}
```

</details>


<a id="api-23"></a>

### 2.13. 查看执行轨迹（`POST /v1/skills/trajectories/show`，权限：`skill:read`）

读取指定轨迹的交互内容、版本绑定、状态和执行时间。

#### 2.13.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/trajectories/show" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "trajectory_id": "traj_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `trajectory_id` | string | 是 | — | 轨迹 ID，例如 `"traj_example_001"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "trajectory_id": "traj_example_001",
    "task_id": "task_example_001",
    "source": "reported",
    "skill_id": "sk_example_001",
    "skill_version_id": "v_example_001",
    "skill_bindings": [],
    "user_id": "u_123",
    "agent_id": "agent_001",
    "session_id": null,
    "task_instruction": null,
    "task_system_prompt": null,
    "environment": null,
    "trajectory": [
      {
        "role": "user",
        "content": "我喜欢喝什么咖啡？"
      },
      {
        "role": "assistant",
        "content": "你喜欢喝冰美式。"
      }
    ],
    "status": "completed",
    "started_at": null,
    "finished_at": null,
    "received_at": "2026-10-12T02:30:00Z"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.trajectory_id` | 执行轨迹 ID。 |
| `data.task_id` | 异步任务 ID。 |
| `data.source` | 轨迹来源。 |
| `data.skill_id` | Skill 家族 ID。 |
| `data.skill_version_id` | 不可变版本 ID。 |
| `data.skill_bindings` | 执行时使用的版本绑定。 |
| `data.user_id` | 执行用户。 |
| `data.agent_id` | 执行 Agent。 |
| `data.session_id` | 执行会话。 |
| `data.task_instruction` | 任务指令。 |
| `data.task_system_prompt` | 任务系统提示词。 |
| `data.environment` | 执行环境引用。 |
| `data.trajectory` | 完整交互记录。 |
| `data.status` | 任务或执行状态。 |
| `data.started_at` | 开始时间。 |
| `data.finished_at` | 结束时间。 |
| `data.received_at` | 服务接收时间。 |

</details>

#### 2.13.2 CLI

**调用示例**

```bash
mindmemos skill trajectory show traj_example_001
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `trajectory_id` | string | 是 | — | 轨迹 ID，例如 `"traj_example_001"`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "trajectory_id": "traj_example_001",
  "task_id": "task_example_001",
  "source": "reported",
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_001",
  "skill_bindings": [],
  "user_id": "u_123",
  "agent_id": "agent_001",
  "session_id": null,
  "task_instruction": null,
  "task_system_prompt": null,
  "environment": null,
  "trajectory": [
    {
      "role": "user",
      "content": "我喜欢喝什么咖啡？"
    },
    {
      "role": "assistant",
      "content": "你喜欢喝冰美式。"
    }
  ],
  "status": "completed",
  "started_at": null,
  "finished_at": null,
  "received_at": "2026-10-12T02:30:00Z"
}
```

</details>


<a id="api-24"></a>

### 2.14. 列出执行轨迹（`POST /v1/skills/trajectories/list`，权限：`skill:read`）

分页读取指定 Skill 的轨迹摘要，支持版本、状态和接收时间范围。

#### 2.14.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/trajectories/list" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_001",
  "limit": 20
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string | 是 | — | Skill ID，例如 `"sk_example_001"`。 |
| `skill_version_id` | string / null | 否 | `None` | 不可变 Skill 版本 ID，例如 `"v_example_001"`。 |
| `status` | string / null | 否 | `None` | 执行状态，例如 `"completed"` 或 `"failed"`。 |
| `received_at_from` | string / null | 否 | `None` | 轨迹接收时间的下限（含边界）。<br>例如 `"2026-10-01T00:00:00Z"`。 |
| `received_at_to` | string / null | 否 | `None` | 轨迹接收时间的上限（含边界）。<br>例如 `"2026-10-12T23:59:59Z"`。 |
| `limit` | integer / null | 否 | `None` | 最多返回的条数，整数值必须大于等于 `1`。省略时不设条数上限。 |
| `cursor` | string / null | 否 | `None` | 上一页返回的游标。首次调用省略，后续原样传入 `next_cursor`。 |

CLI 先解析 skill_ref，再按解析出的版本筛选；HTTP 省略 skill_version_id 时查询该 Skill 的所有版本。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "items": [
      {
        "trajectory_id": "traj_example_001",
        "task_id": "task_example_001",
        "skill_id": "sk_example_001",
        "skill_version_id": "v_example_001",
        "user_id": "u_123",
        "agent_id": "agent_001",
        "status": "completed",
        "started_at": null,
        "finished_at": null,
        "received_at": "2026-10-12T02:30:00Z"
      }
    ],
    "next_cursor": null,
    "has_more": false
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.items` | 当前页记录数组，具体记录字段见示例。 |
| `data.next_cursor` | 下一页游标，None 表示没有后续游标。 |
| `data.has_more` | 是否还有下一页。 |

</details>

#### 2.14.2 CLI

**调用示例**

```bash
mindmemos skill trajectory list coffee --status completed --limit 20
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--status` | string | 否 | `None` | 状态筛选或状态值。 |
| `--received-from` | string | 否 | `None` | 接收时间下限，带时区的 ISO 8601 字符串。<br>例如 `2026-10-01T00:00:00Z`。 |
| `--received-to` | string | 否 | `None` | 接收时间上限，带时区的 ISO 8601 字符串。<br>例如 `2026-10-12T23:59:59Z`。 |
| `--limit` | integer | 否 | `20` | 最多返回的条数，必须大于等于 `1`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "items": [
    {
      "trajectory_id": "traj_example_001",
      "task_id": "task_example_001",
      "skill_id": "sk_example_001",
      "skill_version_id": "v_example_001",
      "user_id": "u_123",
      "agent_id": "agent_001",
      "status": "completed",
      "started_at": null,
      "finished_at": null,
      "received_at": "2026-10-12T02:30:00Z"
    }
  ],
  "next_cursor": null,
  "has_more": false
}
```

</details>


<a id="api-25"></a>

### 2.15. 统计执行轨迹（`POST /v1/skills/trajectories/statistics`，权限：`skill:read`）

统计指定版本的轨迹总量、已消费量和未消费量，并返回固定 30 天的每日统计。

#### 2.15.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/trajectories/statistics" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_id": "sk_example_001",
  "skill_version_id": "v_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_id` | string | 是 | — | Skill ID，例如 `"sk_example_001"`。 |
| `skill_version_id` | string | 是 | — | 不可变 Skill 版本 ID，例如 `"v_example_001"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skill_version_id": "v_example_001",
    "timezone": "Asia/Shanghai",
    "totals": {
      "total": 1,
      "consumed": 0,
      "unconsumed": 1
    },
    "days": [
      {
        "day": "2026-09-13",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-14",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-15",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-16",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-17",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-18",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-19",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-20",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-21",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-22",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-23",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-24",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-25",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-26",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-27",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-28",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-29",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-09-30",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-01",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-02",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-03",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-04",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-05",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-06",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-07",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-08",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-09",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-10",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-11",
        "total": 0,
        "consumed": 0,
        "unconsumed": 0
      },
      {
        "day": "2026-10-12",
        "total": 1,
        "consumed": 0,
        "unconsumed": 1
      }
    ],
    "last_evolved_at": null
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skill_version_id` | 不可变版本 ID。 |
| `data.timezone` | 统计使用的日历时区。 |
| `data.totals` | 全量计数：total、consumed、unconsumed。 |
| `data.days` | 固定 30 个连续日历日的计数，包括零计数日期。 |
| `data.last_evolved_at` | 最近一次演化开始时间。 |

</details>

#### 2.15.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="api-26"></a>

### 2.16. 演化 Skill（`POST /v1/skills/algos/evolve`，权限：`skill:write`）

基于已有轨迹提交 Skill 演化任务，支持控制新版本是否自动激活。

#### 2.16.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/algos/evolve" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_version_id": "v_example_001",
  "auto_accept": false
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_version_id` | string | 是 | — | 不可变 Skill 版本 ID，例如 `"v_example_001"`。 |
| `auto_accept` | boolean | 否 | `True` | 是否将算法产生的新版本设为激活版本。 |

服务须配置任务管理器和唯一的演化 pipeline。HTTP 的 auto_accept 默认 True，CLI 默认 False；提交成功不代表演化完成。

**返回结果**

成功时返回 HTTP `202`。HTTP `202` 表示任务已接收，完成情况通过任务接口查询。

```json
{
  "code": "accepted",
  "request_id": "req_example",
  "data": {
    "task_id": "task_example_001",
    "project_id": "default_project",
    "kind": "skill.evolve",
    "status": "queued",
    "result": null,
    "error_type": null,
    "attempt": 0,
    "accepted_at": "2026-10-12T02:30:00Z",
    "started_at": null,
    "finished_at": null,
    "waiting_ms": 0.0,
    "execution_ms": null,
    "end_to_end_ms": 0.0
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.task_id` | 异步任务 ID。 |
| `data.project_id` | 所属项目。 |
| `data.kind` | 任务类型，例如 skill.train 或 skill.evolve。 |
| `data.status` | 任务或执行状态。 |
| `data.result` | 任务成功后的结果，未完成时为 None。 |
| `data.error_type` | 失败异常类型。 |
| `data.attempt` | 已开始的执行次数。 |
| `data.accepted_at` | 任务首次接收时间。 |
| `data.started_at` | 开始时间。 |
| `data.finished_at` | 结束时间。 |
| `data.waiting_ms` | 等待耗时，单位毫秒。 |
| `data.execution_ms` | 执行耗时，单位毫秒。 |
| `data.end_to_end_ms` | 从接收到结束的总耗时，单位毫秒。 |

</details>

#### 2.16.2 CLI

**调用示例**

```bash
mindmemos skill evolve coffee
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--auto-accept` | boolean | 否 | `False` | 自动激活算法产生的新版本，添加即启用。省略时不自动激活。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "task_id": "task_example_001",
  "project_id": "default_project",
  "kind": "skill.evolve",
  "status": "queued",
  "result": null,
  "error_type": null,
  "attempt": 0,
  "accepted_at": "2026-10-12T02:30:00Z",
  "started_at": null,
  "finished_at": null,
  "waiting_ms": 0.0,
  "execution_ms": null,
  "end_to_end_ms": 0.0
}
```

</details>


<a id="api-27"></a>

### 2.17. 训练 Skill（`POST /v1/skills/algos/train`，权限：`skill:write`）

使用指定版本和环境提交训练任务，支持任务划分、输出目录和自动激活。

#### 2.17.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/algos/train" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "algo": "treeskill",
  "env": {
    "class_name": "searchqa",
    "config": {
      "agent": {
        "model": "YOUR_CHAT_MODEL",
        "api_key": "YOUR_MODEL_API_KEY",
        "base_url": "https://YOUR_MODEL_ENDPOINT/v1"
      }
    }
  },
  "init_skill_version_id": "v_example_001",
  "work_dir": "/absolute/path/output/train-coffee",
  "tasks": [
    {
      "task": {
        "id": "t1",
        "instruction": "What coffee does the user prefer?",
        "context": "The user prefers iced Americano."
      },
      "reward_data": {
        "answers": [
          "iced Americano"
        ]
      }
    }
  ],
  "split": {
    "train": [
      "t1"
    ],
    "val": [],
    "test": []
  },
  "auto_accept": false
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `algo` | string | 是 | — | 服务已配置的训练算法实例名，例如 `"treeskill"`。 |
| `env` | object | 是 | — | 本次训练的环境实现及配置。<br>格式：`{"class_name":"searchqa","config":{"agent":{"model":"YOUR_CHAT_MODEL","api_key":"YOUR_MODEL_API_KEY","base_url":"https://YOUR_MODEL_ENDPOINT/v1"}}}`。 |
| `project_id` | string / null | 否 | `None` | 项目提示字段。训练接口以已鉴权的项目覆盖此值。 |
| `init_skill_version_id` | string | 是 | — | 训练的初始 Skill 版本 ID。 |
| `auto_accept` | boolean | 否 | `True` | 是否将算法产生的新版本设为激活版本。 |
| `work_dir` | string | 是 | — | 训练服务可访问的独立输出目录，例如 `"/absolute/path/output/train-coffee"`。 |
| `tasks` | array | 是 | — | 环境定义的样本数组，每项包含 `task` 和 `reward_data`。<br>SearchQA 示例：`[{"task":{"id":"t1","instruction":"What coffee does the user prefer?","context":"The user prefers iced Americano."},"reward_data":{"answers":["iced Americano"]}}]`。 |
| `split` | object | 是 | — | 训练、验证和测试任务 ID 划分。<br>例如 `{"train":["t1"],"val":[],"test":[]}`。<br>`train` 至少包含一个 ID，各组不得重叠，ID 必须对应 `tasks` 中的任务。 |

模型、环境和目录使用示例值，请替换为训练服务可用的配置。算法可能要求非空验证集，需按所选算法补充 val。HTTP 的 auto_accept 默认 True，CLI 默认 False。

**返回结果**

成功时返回 HTTP `202`。HTTP `202` 表示任务已接收，完成情况通过任务接口查询。

```json
{
  "code": "accepted",
  "request_id": "req_example",
  "data": {
    "task_id": "task_example_001",
    "project_id": "default_project",
    "kind": "skill.train",
    "status": "queued",
    "result": null,
    "error_type": null,
    "attempt": 0,
    "accepted_at": "2026-10-12T02:30:00Z",
    "started_at": null,
    "finished_at": null,
    "waiting_ms": 0.0,
    "execution_ms": null,
    "end_to_end_ms": 0.0
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.task_id` | 异步任务 ID。 |
| `data.project_id` | 所属项目。 |
| `data.kind` | 任务类型，例如 skill.train 或 skill.evolve。 |
| `data.status` | 任务或执行状态。 |
| `data.result` | 任务成功后的结果，未完成时为 None。 |
| `data.error_type` | 失败异常类型。 |
| `data.attempt` | 已开始的执行次数。 |
| `data.accepted_at` | 任务首次接收时间。 |
| `data.started_at` | 开始时间。 |
| `data.finished_at` | 结束时间。 |
| `data.waiting_ms` | 等待耗时，单位毫秒。 |
| `data.execution_ms` | 执行耗时，单位毫秒。 |
| `data.end_to_end_ms` | 从接收到结束的总耗时，单位毫秒。 |

</details>

#### 2.17.2 CLI

**调用示例**

```bash
mindmemos skill train coffee \
  --pipeline treeskill \
  --work-dir /absolute/path/output/train-coffee \
  --environment-json '{"class_name": "searchqa", "config": {"agent": {"model": "YOUR_CHAT_MODEL", "api_key": "YOUR_MODEL_API_KEY", "base_url": "https://YOUR_MODEL_ENDPOINT/v1"}}}' \
  --sample-json '{"task": {"id": "t1", "instruction": "What coffee does the user prefer?", "context": "The user prefers iced Americano."}, "reward_data": {"answers": ["iced Americano"]}}' \
  --train-id t1
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_ref` | string | 是 | — | Skill 别名、ID 或版本引用，例如 `coffee`。 |
| `--work-dir` | string | 是 | — | 训练服务可访问的独立输出目录。 |
| `--pipeline` | string | 是 | — | 服务已配置的训练算法实例名，对应 HTTP 的 `algo`。 |
| `--environment` | string | 否 | `None` | 环境配置 YAML 或 JSON 文件，例如 `environment.yaml`，与 `--environment-json` 二选一。 |
| `--environment-json` | string | 否 | `None` | 环境配置 JSON 对象。<br>格式：`'{"class_name":"searchqa","config":{"agent":{"model":"YOUR_CHAT_MODEL","api_key":"YOUR_MODEL_API_KEY","base_url":"https://YOUR_MODEL_ENDPOINT/v1"}}}'`。 |
| `--sample-json` | string | 是 | — | 单个任务样本 JSON，可重复指定。<br>例如 `'{"task":{"id":"t1","instruction":"What coffee does the user prefer?","context":"The user prefers iced Americano."},"reward_data":{"answers":["iced Americano"]}}'`。 |
| `--train-id` | string | 是 | — | 训练任务 ID，可重复指定，至少一个。 |
| `--val-id` | string | 否 | `[]` | 验证任务 ID，可重复指定。 |
| `--test-id` | string | 否 | `[]` | 测试任务 ID，可重复指定。 |
| `--auto-accept` | boolean | 否 | `False` | 自动激活算法产生的新版本，添加即启用。省略时不自动激活。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "task_id": "task_example_001",
  "project_id": "default_project",
  "kind": "skill.train",
  "status": "queued",
  "result": null,
  "error_type": null,
  "attempt": 0,
  "accepted_at": "2026-10-12T02:30:00Z",
  "started_at": null,
  "finished_at": null,
  "waiting_ms": 0.0,
  "execution_ms": null,
  "end_to_end_ms": 0.0
}
```

</details>


<a id="api-28"></a>

### 2.18. 查看异步任务（`POST /v1/skills/tasks/show`，权限：`skill:read`）

查询训练或演化任务的状态、结果、错误类型和耗时。

#### 2.18.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/tasks/show" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "task_id": "task_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task_id` | string | 是 | — | 任务 ID，例如 `"task_example_001"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "task_id": "task_example_001",
    "project_id": "default_project",
    "kind": "skill.evolve",
    "status": "queued",
    "result": null,
    "error_type": null,
    "attempt": 0,
    "accepted_at": "2026-10-12T02:30:00Z",
    "started_at": null,
    "finished_at": null,
    "waiting_ms": 0.0,
    "execution_ms": null,
    "end_to_end_ms": 0.0
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.task_id` | 异步任务 ID。 |
| `data.project_id` | 所属项目。 |
| `data.kind` | 任务类型，例如 skill.train 或 skill.evolve。 |
| `data.status` | 任务或执行状态。 |
| `data.result` | 任务成功后的结果，未完成时为 None。 |
| `data.error_type` | 失败异常类型。 |
| `data.attempt` | 已开始的执行次数。 |
| `data.accepted_at` | 任务首次接收时间。 |
| `data.started_at` | 开始时间。 |
| `data.finished_at` | 结束时间。 |
| `data.waiting_ms` | 等待耗时，单位毫秒。 |
| `data.execution_ms` | 执行耗时，单位毫秒。 |
| `data.end_to_end_ms` | 从接收到结束的总耗时，单位毫秒。 |

</details>

#### 2.18.2 CLI

**调用示例**

```bash
mindmemos skill task show task_example_001
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task_id` | string | 是 | — | 任务 ID，例如 `"task_example_001"`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "task_id": "task_example_001",
  "project_id": "default_project",
  "kind": "skill.evolve",
  "status": "queued",
  "result": null,
  "error_type": null,
  "attempt": 0,
  "accepted_at": "2026-10-12T02:30:00Z",
  "started_at": null,
  "finished_at": null,
  "waiting_ms": 0.0,
  "execution_ms": null,
  "end_to_end_ms": 0.0
}
```

</details>


<a id="api-29"></a>

### 2.19. 查看任务进度（`POST /v1/skills/tasks/progress`，权限：`skill:read`）

按日志序号增量读取任务日志，CLI 可持续跟踪直到任务结束。

#### 2.19.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/tasks/progress?after=0" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "task_id": "task_example_001"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task_id` | string | 是 | — | 任务 ID，例如 `"task_example_001"`。 |
| `after（查询参数）` | integer | 否 | 0 | 仅返回序号大于 after 的日志，必须大于等于 `0`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": [
    {
      "sequence": 1,
      "task_id": "task_example_001",
      "project_id": "default_project",
      "attempt": 1,
      "occurred_at": "2026-10-12T02:30:00Z",
      "payload": {
        "level": "INFO",
        "name": "mindmemos_skill",
        "message": "Training started."
      },
      "log_id": "log_example_001"
    }
  ]
}
```

| 字段 | 说明 |
| --- | --- |
| `data[]` | 日志记录，包含 sequence、log_id、project_id、task_id、attempt、occurred_at、payload。payload 包含日志级别、名称、消息及可选 traceback。 |

</details>

#### 2.19.2 CLI

**调用示例**

```bash
mindmemos skill task progress task_example_001 --after 0
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task_id` | string | 是 | — | 任务 ID，例如 `"task_example_001"`。 |
| `--after` | integer | 否 | `0` | 仅读取大于该序号的日志，必须大于等于 `0`。 |
| `--follow / -f` | boolean | 否 | `False` | 持续读取新日志直到任务结束，添加 `--follow` 或 `-f` 即启用。 |

**返回结果**

```text
[2026-10-12T02:30:00.000000Z] [INFO] [mindmemos_skill] | project_id=default_project task_id=task_example_001 attempt=1 | Training started.
```

添加 --follow 持续跟踪；进度命令输出普通文本日志，失败日志可附带多行 traceback，没有日志时不输出内容。

</details>


<a id="api-30"></a>

### 2.20. 组织 Skill 图（`POST /v1/skills/algos/organize`，权限：`skill:write`）

将指定 Skill 版本组织为图并发布快照，支持新建图、更新已有图和重新提取 Scene。

#### 2.20.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/algos/organize" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "skill_refs": [
    "v_example_001"
  ],
  "alias": "coffee-skills"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `skill_refs` | array | 是 | — | Skill 或版本引用数组，同一图中每个 Skill 只保留一个版本。<br>例如 `["coffee","v_example_002"]`。 |
| `graph_id` | string / null | 否 | `None` | 已有图的精确 ID。省略时创建新图。 |
| `alias` | string / null | 否 | `None` | 新图别名，例如 `"coffee-skills"`。 |
| `commit_message` | string / null | 否 | `None` | 本次提交说明，例如 `"Improve coffee answers"`。 |
| `force_extract_scene` | boolean | 否 | `False` | 是否重新提取 Scene，忽略已有缓存。 |

服务须配置可用的图组织算法。CLI 的 --alias 与 --graph-id 二选一，--skill、--all-skills、--empty 三选一。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "graph_id": "skg-example",
    "graph_version_id": "skgv-11111111111111111111111111111111",
    "skill_version_ids": [
      "v_example_001"
    ]
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.graph_id` | 图 ID。 |
| `data.graph_version_id` | 所选图快照 ID。 |
| `data.skill_version_ids` | 本次图快照中的成员版本 ID。 |

</details>

#### 2.20.2 CLI

**调用示例**

```bash
mindmemos skill graph publish --alias coffee-skills --skill coffee
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--alias` | string | 否 | `None` | Skill 或图的别名。 |
| `--graph-id` | string | 否 | `None` | 已有图的精确 ID，与 `--alias` 二选一。 |
| `--skill` | string | 否 | `None` | 图成员 Skill 或版本引用，可重复指定，例如 `--skill coffee --skill v_example_002`。 |
| `--all-skills` | boolean | 否 | `False` | 将当前项目所有 Skill 的激活版本加入图，与 `--skill`、`--empty` 互斥。 |
| `--empty` | boolean | 否 | `False` | 发布空图，与 `--skill`、`--all-skills` 互斥。 |
| `--message` | string | 否 | `None` | 提交说明。 |
| `--force-extract-scene` | boolean | 否 | `False` | 重新提取 Scene，添加即启用。 |

--alias 与 --graph-id 二选一；--skill、--all-skills、--empty 三选一。

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "graph_id": "skg-example",
  "graph_version_id": "skgv-11111111111111111111111111111111",
  "skill_version_ids": [
    "v_example_001"
  ]
}
```

</details>


<a id="api-31"></a>

### 2.21. 列出 Skill 图（`POST /v1/skills/graphs/list`，权限：`skill:read`）

列出当前项目的 Skill 图及其当前版本。

#### 2.21.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graphs/list" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

无请求参数。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "items": [
      {
        "project_id": "default_project",
        "graph_id": "skg-example",
        "alias": "coffee-skills",
        "name": "Coffee skills",
        "description": null,
        "current_version_id": "skgv-11111111111111111111111111111111",
        "created_at": "2026-10-12T02:30:00Z",
        "updated_at": "2026-10-12T02:30:00Z",
        "is_deleted": false,
        "deleted_at": null
      }
    ]
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.items` | 当前页记录数组，具体记录字段见示例。 |

</details>

#### 2.21.2 CLI

**调用示例**

```bash
mindmemos skill graph list
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

无命令参数。

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "items": [
    {
      "project_id": "default_project",
      "graph_id": "skg-example",
      "alias": "coffee-skills",
      "name": "Coffee skills",
      "description": null,
      "current_version_id": "skgv-11111111111111111111111111111111",
      "created_at": "2026-10-12T02:30:00Z",
      "updated_at": "2026-10-12T02:30:00Z",
      "is_deleted": false,
      "deleted_at": null
    }
  ]
}
```

</details>


<a id="api-32"></a>

### 2.22. 解析 Skill 图引用（`POST /v1/skills/graph-resolve`，权限：`skill:read`）

将图别名、图 ID 或快照引用解析为确定的图和快照。

#### 2.22.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graph-resolve" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_ref": "coffee-skills"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或快照 ID，例如 `"coffee-skills"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "graph_id": "skg-example",
    "graph_version_id": "skgv-11111111111111111111111111111111",
    "alias": "coffee-skills",
    "is_current": true,
    "status": "ready"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.graph_id` | 图 ID。 |
| `data.graph_version_id` | 所选图快照 ID。 |
| `data.alias` | 项目内别名。 |
| `data.is_current` | 是否为当前图快照。 |
| `data.status` | 任务或执行状态。 |

</details>

#### 2.22.2 CLI

**调用示例**

```bash
mindmemos skill graph resolve coffee-skills
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或图快照引用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "graph_id": "skg-example",
  "graph_version_id": "skgv-11111111111111111111111111111111",
  "alias": "coffee-skills",
  "is_current": true,
  "status": "ready"
}
```

</details>


<a id="api-33"></a>

### 2.23. 查看图快照历史（`POST /v1/skills/graphs/snapshots`，权限：`skill:read`）

列出指定图的不可变快照和发布修订。

#### 2.23.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graphs/snapshots" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_id": "skg-example"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_id` | string | 是 | — | 图 ID。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "items": [
      {
        "project_id": "default_project",
        "graph_id": "skg-example",
        "graph_version_id": "skgv-11111111111111111111111111111111",
        "status": "ready",
        "node_count": 1,
        "edge_count": 0,
        "vector_count": 1,
        "snapshot_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "metadata": {},
        "request_hash": null,
        "actor_id": null,
        "message": null,
        "created_at": "2026-10-12T02:30:00Z",
        "ready_at": "2026-10-12T02:30:00Z"
      }
    ]
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.items` | 当前页记录数组，具体记录字段见示例。 |

</details>

#### 2.23.2 CLI

**调用示例**

```bash
mindmemos skill graph history coffee-skills
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或图快照引用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "items": [
    {
      "project_id": "default_project",
      "graph_id": "skg-example",
      "graph_version_id": "skgv-11111111111111111111111111111111",
      "status": "ready",
      "node_count": 1,
      "edge_count": 0,
      "vector_count": 1,
      "snapshot_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "metadata": {},
      "request_hash": null,
      "actor_id": null,
      "message": null,
      "created_at": "2026-10-12T02:30:00Z",
      "ready_at": "2026-10-12T02:30:00Z"
    }
  ]
}
```

</details>


<a id="api-34"></a>

### 2.24. 读取图快照（`POST /v1/skills/graphs/snapshot`，权限：`skill:read`）

读取指定图快照的 Skill、Scene 节点和关系。

#### 2.24.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graphs/snapshot" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_id": "skg-example",
  "graph_version_id": "skgv-11111111111111111111111111111111"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_id` | string | 是 | — | 图 ID。 |
| `graph_version_id` | string | 是 | — | 不可变图快照 ID，例如 `"skgv-11111111111111111111111111111111"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "graph_id": "skg-example",
    "graph_version_id": "skgv-11111111111111111111111111111111",
    "nodes": [
      {
        "node_id": "node_example_001",
        "node_type": "skill",
        "owner_node_id": null,
        "skill_version_id": "v_example_001",
        "skill_id": "sk_example_001",
        "skill_name": "coffee-helper",
        "name": "coffee-helper",
        "version_label": "v0.0.1",
        "status": "draft",
        "description": "Answer coffee questions.",
        "one_line_capability": "Recall coffee preferences.",
        "inputs": [
          "question"
        ],
        "outputs": [
          "answer"
        ],
        "domain_tags": [
          "coffee"
        ],
        "tooling": [],
        "scene_metadata": {}
      }
    ],
    "edges": []
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.graph_id` | 图 ID。 |
| `data.graph_version_id` | 所选图快照 ID。 |
| `data.nodes` | Skill 和 Scene 节点数组，包含 node_id、node_type、skill_version_id、skill_id 和节点描述。 |
| `data.edges` | 关系数组，包含 edge_id、source_node_id、target_node_id、edge_type、weight，可带 condition 和 evidence。 |

</details>

#### 2.24.2 CLI

**调用示例**

```bash
mindmemos skill graph show coffee-skills
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或图快照引用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "graph_id": "skg-example",
  "graph_version_id": "skgv-11111111111111111111111111111111",
  "nodes": [
    {
      "node_id": "node_example_001",
      "node_type": "skill",
      "owner_node_id": null,
      "skill_version_id": "v_example_001",
      "skill_id": "sk_example_001",
      "skill_name": "coffee-helper",
      "name": "coffee-helper",
      "version_label": "v0.0.1",
      "status": "draft",
      "description": "Answer coffee questions.",
      "one_line_capability": "Recall coffee preferences.",
      "inputs": [
        "question"
      ],
      "outputs": [
        "answer"
      ],
      "domain_tags": [
        "coffee"
      ],
      "tooling": [],
      "scene_metadata": {}
    }
  ],
  "edges": []
}
```

</details>


<a id="api-35"></a>

### 2.25. 预览 Skill 图（`POST /v1/skills/graphs/preview`，权限：`skill:read`）

读取图引用所选快照的拓扑结构，不重新提取 Scene。

#### 2.25.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graphs/preview" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_ref": "coffee-skills"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或快照 ID，例如 `"coffee-skills"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "graph_id": "skg-example",
    "graph_version_id": "skgv-11111111111111111111111111111111",
    "nodes": [
      {
        "node_id": "node_example_001",
        "node_type": "skill",
        "owner_node_id": null,
        "skill_version_id": "v_example_001",
        "skill_id": "sk_example_001",
        "skill_name": "coffee-helper",
        "name": "coffee-helper",
        "version_label": "v0.0.1",
        "status": "draft",
        "description": "Answer coffee questions.",
        "one_line_capability": "Recall coffee preferences.",
        "inputs": [
          "question"
        ],
        "outputs": [
          "answer"
        ],
        "domain_tags": [
          "coffee"
        ],
        "tooling": [],
        "scene_metadata": {}
      }
    ],
    "edges": []
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.graph_id` | 图 ID。 |
| `data.graph_version_id` | 所选图快照 ID。 |
| `data.nodes` | Skill 和 Scene 节点数组，包含 node_id、node_type、skill_version_id、skill_id 和节点描述。 |
| `data.edges` | 关系数组，包含 edge_id、source_node_id、target_node_id、edge_type、weight，可带 condition 和 evidence。 |

</details>

#### 2.25.2 CLI

**调用示例**

```bash
mindmemos skill graph preview coffee-skills
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或图快照引用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "graph_id": "skg-example",
  "graph_version_id": "skgv-11111111111111111111111111111111",
  "nodes": [
    {
      "node_id": "node_example_001",
      "node_type": "skill",
      "owner_node_id": null,
      "skill_version_id": "v_example_001",
      "skill_id": "sk_example_001",
      "skill_name": "coffee-helper",
      "name": "coffee-helper",
      "version_label": "v0.0.1",
      "status": "draft",
      "description": "Answer coffee questions.",
      "one_line_capability": "Recall coffee preferences.",
      "inputs": [
        "question"
      ],
      "outputs": [
        "answer"
      ],
      "domain_tags": [
        "coffee"
      ],
      "tooling": [],
      "scene_metadata": {}
    }
  ],
  "edges": []
}
```

</details>


<a id="api-36"></a>

### 2.26. 添加图成员（`POST /v1/skills/graphs/members/add`，权限：`skill:write`）

向已有图添加 Skill 成员并发布新快照，支持提交说明和 Scene 重新提取。

#### 2.26.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graphs/members/add" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_ref": "coffee-skills",
  "skill_refs": [
    "coffee"
  ]
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或快照 ID，例如 `"coffee-skills"`。 |
| `skill_refs` | array | 是 | — | Skill 或版本引用数组，同一图中每个 Skill 只保留一个版本。<br>例如 `["coffee","v_example_002"]`。 |
| `commit_message` | string / null | 否 | `None` | 本次提交说明，例如 `"Improve coffee answers"`。 |
| `force_extract_scene` | boolean | 否 | `False` | 是否重新提取 Scene，忽略已有缓存。 |

新增成员的 Skill 家族不能与图中已有成员重复。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "graph_id": "skg-example",
    "graph_version_id": "skgv-11111111111111111111111111111111",
    "skill_version_ids": [
      "v_example_001"
    ]
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.graph_id` | 图 ID。 |
| `data.graph_version_id` | 所选图快照 ID。 |
| `data.skill_version_ids` | 本次图快照中的成员版本 ID。 |

</details>

#### 2.26.2 CLI

**调用示例**

```bash
mindmemos skill graph add coffee-skills --skill coffee
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或图快照引用。 |
| `--skill` | string | 是 | — | 图成员 Skill 或版本引用，可重复指定，例如 `--skill coffee --skill v_example_002`。 |
| `--message` | string | 否 | `None` | 提交说明。 |
| `--force-extract-scene` | boolean | 否 | `False` | 重新提取 Scene，添加即启用。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "graph_id": "skg-example",
  "graph_version_id": "skgv-11111111111111111111111111111111",
  "skill_version_ids": [
    "v_example_001"
  ]
}
```

</details>


<a id="api-37"></a>

### 2.27. 移除图成员（`POST /v1/skills/graphs/members/remove`，权限：`skill:write`）

从图中移除指定成员并发布新快照，保留原始 Skill 版本。

#### 2.27.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graphs/members/remove" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_ref": "coffee-skills",
  "skill_refs": [
    "coffee"
  ]
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或快照 ID，例如 `"coffee-skills"`。 |
| `skill_refs` | array | 是 | — | Skill 或版本引用数组，同一图中每个 Skill 只保留一个版本。<br>例如 `["coffee","v_example_002"]`。 |
| `commit_message` | string / null | 否 | `None` | 本次提交说明，例如 `"Improve coffee answers"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "graph_id": "skg-example",
    "graph_version_id": "skgv-11111111111111111111111111111111",
    "skill_version_ids": []
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.graph_id` | 图 ID。 |
| `data.graph_version_id` | 所选图快照 ID。 |
| `data.skill_version_ids` | 本次图快照中的成员版本 ID。 |

</details>

#### 2.27.2 CLI

**调用示例**

```bash
mindmemos skill graph remove coffee-skills --skill coffee
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或图快照引用。 |
| `--skill` | string | 是 | — | 图成员 Skill 或版本引用，可重复指定，例如 `--skill coffee --skill v_example_002`。 |
| `--message` | string | 否 | `None` | 提交说明。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "graph_id": "skg-example",
  "graph_version_id": "skgv-11111111111111111111111111111111",
  "skill_version_ids": []
}
```

</details>


<a id="api-38"></a>

### 2.28. 删除 Skill 图（`POST /v1/skills/graphs/delete`，权限：`skill:write`）

软删除指定图，不删除成员 Skill 的版本。

#### 2.28.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/graphs/delete" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_ref": "coffee-skills"
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或快照 ID，例如 `"coffee-skills"`。 |

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "deleted": true
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.deleted` | 是否已完成软删除。 |

</details>

#### 2.28.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="api-39"></a>

### 2.29. 推荐 Skill（`POST /v1/skills/algos/recommend`，权限：`skill:write`）

从指定图快照推荐适合任务的 Skill 版本，支持图传播或向量策略、条数及 Scene 匹配。

#### 2.29.1 HTTP

**调用示例**

```bash
curl -X POST "http://127.0.0.1:8001/v1/skills/algos/recommend" \
  -H "Authorization: Bearer YOUR_SKILL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "graph_ref": "coffee-skills",
  "query": "回答用户的咖啡偏好",
  "top_k": 5
}'
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `graph_ref` | string | 是 | — | 图别名、图 ID 或快照 ID，例如 `"coffee-skills"`。 |
| `graph_version_id` | string / null | 否 | `None` | 指定图快照。省略时使用该图的当前已发布快照。 |
| `query` | string | 是 | — | 查询或任务内容。 |
| `top_k` | integer / null | 否 | `None` | 最多推荐的 Skill 数，必须大于等于 `1`。省略或传入 null 时按 `5` 处理。 |
| `strategy` | string | 否 | `"graph"` | `"graph"`：图传播推荐。<br>`"vector"`：向量相似度推荐。 |
| `include_scenes` | boolean | 否 | `False` | 向量推荐时是否同时匹配 Scene 节点。仅在 `strategy="vector"` 时可启用。 |

服务须配置推荐算法。strategy="graph" 时不能启用 include_scenes。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "code": "ok",
  "request_id": "req_example",
  "data": {
    "skills": [
      {
        "project_id": "default_project",
        "version_id": "v_example_001",
        "skill_id": "sk_example_001",
        "name": "coffee-helper",
        "description": "Answer questions about coffee preferences.",
        "standard_format": true,
        "parent_version_ids": [],
        "content": "---\nname: coffee-helper\ndescription: Answer questions about coffee preferences.\n---\n\nUse the user preferences to answer coffee questions.\n",
        "resources": {},
        "content_hash": "04830d5953e58ea7094d02686b01f485de4bfebe0ff31145553bae7860484bf3",
        "skill_hash": "8e6503ee2e1a9eb8fd784ec8d9e077ceb6a87da18b39d1c3c5f290e16263b135",
        "implementation": "static",
        "config_schema_version": 1,
        "config": {},
        "status": "draft",
        "version_revision": 0,
        "version_label": "v0.0.1",
        "commit_message": null,
        "metadata": {},
        "local_metadata": {},
        "origin": "local",
        "created_at": "2026-10-12T02:30:00Z",
        "updated_at": "2026-10-12T02:30:00Z"
      }
    ],
    "graph_version_id": "skgv-11111111111111111111111111111111",
    "strategy": "graph",
    "score_type": "graph_components",
    "scores": [
      {
        "skill_version_id": "v_example_001",
        "ppr_score": 0.7,
        "scene_boost": 0.1,
        "graph_score": 0.8,
        "rerank_score": 0.9
      }
    ],
    "include_scenes": false
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.skills` | 推荐的完整 Skill 版本或旧版 Skill 摘要数组。 |
| `data.graph_version_id` | 所选图快照 ID。 |
| `data.strategy` | 本次推荐策略。 |
| `data.score_type` | graph_components 或 cosine_similarity。 |
| `data.scores` | 与推荐版本对应的评分数组。graph 策略包含 skill_version_id、ppr_score、scene_boost、graph_score、rerank_score；vector 策略包含 skill_version_id、similarity、node_id、node_type。 |
| `data.include_scenes` | 本次向量推荐是否包含 Scene。 |

</details>

#### 2.29.2 CLI

**调用示例**

```bash
mindmemos skill graph recommend "回答用户的咖啡偏好" --by coffee-skills --top-k 5
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `task` | string | 是 | — | 任务内容，作为位置参数传入。 |
| `--by` | string | 是 | — | 用于推荐的图引用，例如 `coffee-skills`。 |
| `--top-k` | integer | 否 | `5` | 最多返回的条数，必须大于等于 `1`。 |
| `--strategy` | string | 否 | `"graph"` | `graph`：图传播推荐。<br>`vector`：向量相似度推荐。 |
| `--include-scenes` | boolean | 否 | `False` | 同时匹配 Scene 节点，添加即启用，仅支持 `--strategy vector`。 |

**返回结果**

命令直接输出业务 data 的 JSON，示例如下：

```json
{
  "skills": [
    {
      "project_id": "default_project",
      "version_id": "v_example_001",
      "skill_id": "sk_example_001",
      "name": "coffee-helper",
      "description": "Answer questions about coffee preferences.",
      "standard_format": true,
      "parent_version_ids": [],
      "content": "---\nname: coffee-helper\ndescription: Answer questions about coffee preferences.\n---\n\nUse the user preferences to answer coffee questions.\n",
      "resources": {},
      "content_hash": "04830d5953e58ea7094d02686b01f485de4bfebe0ff31145553bae7860484bf3",
      "skill_hash": "8e6503ee2e1a9eb8fd784ec8d9e077ceb6a87da18b39d1c3c5f290e16263b135",
      "implementation": "static",
      "config_schema_version": 1,
      "config": {},
      "status": "draft",
      "version_revision": 0,
      "version_label": "v0.0.1",
      "commit_message": null,
      "metadata": {},
      "local_metadata": {},
      "origin": "local",
      "created_at": "2026-10-12T02:30:00Z",
      "updated_at": "2026-10-12T02:30:00Z"
    }
  ],
  "graph_version_id": "skgv-11111111111111111111111111111111",
  "strategy": "graph",
  "score_type": "graph_components",
  "scores": [
    {
      "skill_version_id": "v_example_001",
      "ppr_score": 0.7,
      "scene_boost": 0.1,
      "graph_score": 0.8,
      "rerank_score": 0.9
    }
  ],
  "include_scenes": false
}
```

</details>


<a id="api-49"></a>

### 2.30. 检查 Skill 服务（`GET /health`，无需鉴权）

检查独立 Skill API 进程是否响应。

#### 2.30.1 HTTP

**调用示例**

```bash
curl -X GET "http://127.0.0.1:8001/health"
```

<details>
<summary>参数和返回值详情</summary>

**请求参数**

无请求参数。

**返回结果**

成功时返回 HTTP `200`。

```json
{
  "status": "ok"
}
```

| 字段 | 说明 |
| --- | --- |
| `status` | 任务或执行状态。 |

</details>

#### 2.30.2 CLI

当前 `mindmemos` CLI 没有对应命令，可使用上面的 HTTP 示例。


<a id="eval-envs"></a>

### 2.31. 列出评测环境（`mindmemos-eval skill envs`）

列出已注册的 Skill 执行环境，不执行任务或调用模型。

**调用示例**

```bash
mindmemos-eval skill envs
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

无命令参数。



**返回结果**

当前环境列表如下，退出码为 0：

```text
agentskillos
alfworld_bounded_history
livemath
officeqa
searchqa
spreadsheetbench_text_action
spreadsheetbench_tool_calling
```

</details>


<a id="eval-test"></a>

### 2.32. 测试环境与固定 Skill（`mindmemos-eval skill test`）

在指定环境执行无 Skill 或固定静态 Skill 测试，支持样本、划分、条数和并发。

**调用示例**

```bash
mindmemos-eval skill test \
  --config config/mindmemos_eval/env_test/dev.yaml \
  --no-skill \
  --limit 2 \
  --concurrency 1 \
  --output-dir output/env_test
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--config` | string | 否 | `"config/mindmemos_eval/env_test/dev.yaml"` | 评测 YAML 配置文件。默认路径相对于仓库根目录；显式传入的相对路径按当前工作目录解析。 |
| `--env` | string | 否 | `None` | 覆盖环境的注册名。<br>例如 `livemath`、`searchqa`、`spreadsheetbench_text_action`。 |
| `--no-skill` | boolean | 否 | `False` | 不加载 Skill，添加即启用，与 --skills 互斥。省略两者时按 YAML 的 skills 配置执行。 |
| `--skills` | array | 否 | `None` | 一起加载的静态 Skill 路径。<br>例如 `--skills ./skills/coffee-helper ./skills/search-helper`，与 --no-skill 互斥。 |
| `--samples` | string | 否 | `None` | 样本 JSON 数组文件，替代配置中的数据加载器。<br>每项格式：`{"task":{"id":"t1","instruction":"问题"},"reward_data":{}}`，具体 task 和 reward_data 字段由环境定义。 |
| `--split-file` | string | 否 | `None` | 任务划分 JSON 文件。<br>例如 `{"train":["t1"],"val":["t2"],"test":["t3"]}`。 |
| `--split` | string | 否 | `None` | 选择任务划分名称，例如 `test`。 |
| `--limit` | integer | 否 | `None` | 最多执行的样本数，必须大于等于 `1`。 |
| `--concurrency` | integer | 否 | `None` | 任务执行并发数，必须大于等于 `1`。 |
| `--output-dir` | string | 否 | `None` | 新的输出目录。省略时生成独立目录；指定的目录须为空。 |

None 表示不覆盖 YAML 配置。测试在评测进程中直接运行环境和 Agent，不提交 Skill 服务的训练任务。

**返回结果**

正常完成时输出 JSON 摘要，再显示报告路径；以下为格式示例，数值为示例值：

```json
{
  "environment": "livemath",
  "mode": "no_skill",
  "planned": 2,
  "attempted": 2,
  "completed": 2,
  "not_started": 0,
  "errors": 0,
  "successful": 1,
  "success_rate": 0.5,
  "wall_seconds": 12.0,
  "input_tokens": 1200,
  "output_tokens": 200,
  "total_tokens": 1400,
  "reported_usage": {
    "input_tokens": 1200,
    "output_tokens": 200,
    "total_tokens": 1400
  },
  "usage_missing": 0,
  "trajectory_file": "/absolute/path/output/env_test/tasks/trajectories.jsonl"
}
```

```text
Report: /absolute/path/output/env_test/report.md
```

| 字段 | 说明 |
| --- | --- |
| `environment` | 使用的注册环境。 |
| `mode` | no_skill、static_skill 或 treeskill。 |
| `planned` | 选中的任务数。 |
| `attempted` | 实际开始执行的任务数。 |
| `completed` | 正常得到评分的任务数，包括零分任务。 |
| `not_started` | 未开始的任务数。 |
| `errors` | 执行或评分失败的任务数。 |
| `successful` | 环境判定成功的任务数。 |
| `success_rate` | successful / completed；没有正常完成任务时为 None。 |
| `wall_seconds` | 本次评测耗时，单位秒。 |
| `input_tokens` | 输入 token 总量，有缺失时为 None。 |
| `output_tokens` | 输出 token 总量，有缺失时为 None。 |
| `total_tokens` | 总 token 量，有缺失时为 None。 |
| `reported_usage` | 已知的 token 用量汇总。 |
| `usage_missing` | 未报告完整用量的任务数。 |
| `trajectory_file` | JSONL 执行轨迹文件。 |

输出目录保存 summary.json、results.json、report.md 和 tasks/trajectories.jsonl。没有执行错误时退出码为 0；出现执行错误或配置失败时为 1。任务得分低不等同于命令执行失败。

</details>


<a id="eval-treeskill"></a>

### 2.33. 评测 TreeSkill（`mindmemos-eval skill treeskill`）

通过已配置的 Skill 服务执行 TreeSkill 训练与测试，支持输入检查、环境覆盖和独立输出目录。

**调用示例**

```bash
mindmemos-eval skill treeskill \
  --config config/mindmemos_eval/treeskill/dev.yaml \
  --output-dir output/treeskill_eval
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--config` | string | 否 | `"config/mindmemos_eval/treeskill/dev.yaml"` | 评测 YAML 配置文件。默认路径相对于仓库根目录；显式传入的相对路径按当前工作目录解析。 |
| `--check` | boolean | 否 | `False` | 只检查输入、参考协议和服务配置，不调用模型或提交训练，添加即启用。 |
| `--env` | string | 否 | `None` | 覆盖环境的注册名。<br>例如 `livemath`、`searchqa`、`spreadsheetbench_text_action`。 |
| `--output-dir` | string | 否 | `None` | 新的输出目录。省略时生成独立目录；指定的目录须为空。 |

配置包含 reference_run、service_config、sdk_home、data_root、split_file、initial_skill 和 profile。参考 SpreadsheetBench 协议要求与记录的训练输入一致；其他环境可以运行，但不保证与参考分数可比。--check 可在训练前核对输入，不要求启动 UI 页面；实际训练要求 SDK 指向可用的 Skill 服务。

**返回结果**

添加 --check 时，只输出输入检查结果，参考协议配置通过时的格式如下：

```text
Prepared spreadsheetbench_text_action: 200 train / 200 test.
Reference aligned: max_turns=100; recalculation=true; concurrency=32.
```

未添加 --check 时执行训练和测试，输出 EnvTestSummary JSON，格式如下，数值为示例值：

```json
{
  "environment": "spreadsheetbench_text_action",
  "mode": "treeskill",
  "planned": 2,
  "attempted": 2,
  "completed": 2,
  "not_started": 0,
  "errors": 0,
  "successful": 1,
  "success_rate": 0.5,
  "wall_seconds": 12.0,
  "input_tokens": 1200,
  "output_tokens": 200,
  "total_tokens": 1400,
  "reported_usage": {
    "input_tokens": 1200,
    "output_tokens": 200,
    "total_tokens": 1400
  },
  "usage_missing": 0,
  "trajectory_file": "/absolute/path/output/treeskill_eval/test/tasks/trajectories.jsonl"
}
```

| 字段 | 说明 |
| --- | --- |
| `environment` | 使用的注册环境。 |
| `mode` | no_skill、static_skill 或 treeskill。 |
| `planned` | 选中的任务数。 |
| `attempted` | 实际开始执行的任务数。 |
| `completed` | 正常得到评分的任务数，包括零分任务。 |
| `not_started` | 未开始的任务数。 |
| `errors` | 执行或评分失败的任务数。 |
| `successful` | 环境判定成功的任务数。 |
| `success_rate` | successful / completed；没有正常完成任务时为 None。 |
| `wall_seconds` | 本次评测耗时，单位秒。 |
| `input_tokens` | 输入 token 总量，有缺失时为 None。 |
| `output_tokens` | 输出 token 总量，有缺失时为 None。 |
| `total_tokens` | 总 token 量，有缺失时为 None。 |
| `reported_usage` | 已知的 token 用量汇总。 |
| `usage_missing` | 未报告完整用量的任务数。 |
| `trajectory_file` | JSONL 执行轨迹文件。 |

目录包含 config.json、split.json、registration.json、task.json、runtime.json，以及测试的 summary.json、results.json、report.md 和轨迹文件。--check 检查通过时退出码为 0；实际运行有错误时为 1。

</details>


<a id="eval-run"></a>

### 2.34. 评测 Skill-Forest（`mindmemos-eval skill run`）

对比无 Skill、初始 Skill 与训练后 Skill 的效果，支持基准选择、精简运行和多次测试。

**调用示例**

```bash
mindmemos-eval skill run \
  --config config/mindmemos_eval/dev.yaml \
  --benchmark livemath \
  --smoke \
  --output-dir output/skill_forest_eval
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `--config` | string | 否 | `"config/mindmemos_eval/dev.yaml"` | 评测 YAML 配置文件。默认路径相对于仓库根目录；显式传入的相对路径按当前工作目录解析。 |
| `--algorithm` | string | 否 | `"skill_forest"` | 训练算法，当前仅支持 `skill_forest`。 |
| `--benchmark` | string | 否 | `"livemath"` | 选择 livemath、alfworld、spreadsheetbench、searchqa、officeqa 中的一个，或 all。 |
| `--smoke` | boolean | 否 | `False` | 使用固定划分的 4/2/2 前缀和精简 Skill-Forest 设置，添加即启用。 |
| `--test-repeats` | integer | 否 | `1` | 对相同测试任务和固定版本独立重复评测的次数，必须大于等于 `1`。 |
| `--arms` | string | 否 | `"no_skill,initial_skill,trained_skill"` | 评测组，用逗号分隔，不可重复。<br>例如 `no_skill,initial_skill,trained_skill`。选择 trained_skill 时才执行训练。 |
| `--data-root` | string | 否 | `"data/mindmemos_skill"` | 基准数据的父目录，包含各基准子目录。 |
| `--output-dir` | string | 否 | `None` | 新的输出目录。省略时生成独立目录；指定的目录须为空。 |

每次 run 是新的评测，不能用已有非空目录续跑。--arms 中的 no_skill、initial_skill、trained_skill 分别表示无 Skill、初始 Skill、训练后 Skill；只有选择 trained_skill 才运行训练。

**返回结果**

执行后显示报告路径，示例如下：

```text
Report: /absolute/path/output/skill_forest_eval/report.md
```

| 文件 | 说明 |
| --- | --- |
| `summary.json` | 基准、各组结果、覆盖率、重复统计和相对基线增益。 |
| `summary.csv` | 每个基准、评测组和重复轮次的指标表。 |
| `report.md` | 汇总报告。 |

全部计划任务完成且结果完整时退出码为 0，存在配置错误、运行错误或不完整结果时为 1。不完整组不生成总体准确率。


</details>


<a id="eval-report"></a>

### 2.35. 重建评测报告（`mindmemos-eval skill report`）

根据已保存的 Skill-Forest 结果重新生成报告，不调用模型或重新执行任务。

**调用示例**

```bash
mindmemos-eval skill report output/skill_forest_eval
```

<details>
<summary>参数和返回值详情</summary>

**命令参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `run_dir` | string | 是 | — | 已有 Skill-Forest 运行目录，包含完整执行计划和结果文件。 |

run_dir 应来自 skill run。此命令读取计划与任务结果并重建 summary.json、summary.csv 和 report.md，不恢复或补跑未完成任务。

**返回结果**

执行后显示报告路径，示例如下：

```text
Report: /absolute/path/output/skill_forest_eval/report.md
```

| 文件 | 说明 |
| --- | --- |
| `summary.json` | 基准、各组结果、覆盖率、重复统计和相对基线增益。 |
| `summary.csv` | 每个基准、评测组和重复轮次的指标表。 |
| `report.md` | 汇总报告。 |

全部计划任务完成且结果完整时退出码为 0，存在配置错误、运行错误或不完整结果时为 1。不完整组不生成总体准确率。


</details>


### 2.36. 通过 MindMemOSClient 操作 Skill

以下示例覆盖注册、查询、版本提交、对比、激活、轨迹、图组织与推荐、演化、训练、任务进度和删除；服务需已配置版本仓库、图管理算法、唯一演化 pipeline、训练实例和任务管理器。

示例使用独立 Skill 地址，Memory 服务不参与这些操作。将模型配置、训练目录和样本替换为自己的环境与数据；`work_dir` 必须由 Skill 服务访问。带有轨迹或图引用的 Skill 保留，删除操作用一个独立的无引用 Skill 演示。

```python
import time
from uuid import uuid4

from mindmemos_sdk import MindMemOSClient
from mindmemos_sdk.config import SDKConfig
from mindmemos_sdk.skills import (
    ActivateRequest,
    AddGraphMembersRequest,
    CommitRequest,
    DeleteRequest,
    DiffRequest,
    EnvironmentConfig,
    EvolveRequest,
    GraphResolveRequest,
    GraphSnapshotRequest,
    GraphSnapshotsRequest,
    HistoryRequest,
    ListRequest,
    OrganizeRequest,
    RecommendRequest,
    RegisterRequest,
    RemoveGraphMembersRequest,
    ShowRequest,
    SkillResolveRequest,
    StructuredSkillInfo,
    TaskProgressRequest,
    TaskShowRequest,
    TrainingRequest,
    TrainingSplitRequest,
    TrajectoryListRequest,
    TrajectoryReportRequest,
    TrajectoryShowRequest,
    TrajectoryStatisticsRequest,
    UpdateInfoRequest,
    VersionResolveRequest,
)

config = SDKConfig(
    service_type="custom",
    memory_service={
        "base_url": "http://127.0.0.1:8000",
        "auth": {"api_key": "YOUR_API_KEY"},
    },
    skill_service={
        "base_url": "http://127.0.0.1:8001",
        "auth": {"api_key": "YOUR_SKILL_API_KEY"},
    },
)
alias = f"coffee-demo-{uuid4().hex}"
content = """---
name: coffee-helper
description: Answer coffee preference questions.
---

Use the user's preferences to answer coffee questions.
"""


def wait_for_task(client, task_id, timeout_s=600):
    """Read task logs and wait for a persisted terminal result."""
    deadline = time.monotonic() + timeout_s
    after = 0
    while time.monotonic() < deadline:
        logs = client.skills.task_progress(
            TaskProgressRequest(task_id=task_id, after=after)
        )
        for event in logs:
            print(event["payload"])
            after = max(after, event["sequence"])
        task = client.skills.show_task(TaskShowRequest(task_id=task_id))
        if task["status"] == "succeeded":
            return task["result"]
        if task["status"] in {"failed", "cancelled"}:
            raise RuntimeError(f"Task ended: {task['status']} {task['error_type']}")
        time.sleep(2)
    raise TimeoutError(f"Task is still pending: {task_id}")


with MindMemOSClient(config=config) as client:
    # Register, resolve, and inspect the first version.
    registered = client.skills.register(
        RegisterRequest(
            alias=alias,
            skill=StructuredSkillInfo(content=content, commit_version="v0.0.1"),
        )
    )
    skill_id = registered["skill_id"]
    first_version = registered["skill_version_id"]
    print(client.skills.list(ListRequest(filters={"alias": alias}, limit=20)))
    print(client.skills.skill_resolve(SkillResolveRequest(skill_ref=alias)))
    print(
        client.skills.version_resolve(
            VersionResolveRequest(skill_version_id=first_version)
        )
    )
    print(client.skills.show(ShowRequest(skill_id=skill_id)))

    # Publish a child version, compare it, and activate it explicitly.
    committed = client.skills.commit(
        CommitRequest(
            skill_id=skill_id,
            base_version_id=first_version,
            skill=StructuredSkillInfo(
                content=content + "\nMention whether the user adds sugar.\n",
                commit_message="Include sugar preferences",
            ),
            active=False,
            inherit_resources=True,
        )
    )
    second_version = committed["skill_version_id"]
    print(client.skills.history(HistoryRequest(skill_id=skill_id, limit=20)))
    print(
        client.skills.diff(
            DiffRequest(
                skill_id=skill_id,
                from_skill_version_id=first_version,
                to_skill_version_id=second_version,
            )
        )
    )
    print(
        client.skills.activate(
            ActivateRequest(
                skill_id=skill_id,
                skill_version_id=second_version,
                expected_active_version_id=first_version,
            )
        )
    )
    print(
        client.skills.update_info(
            UpdateInfoRequest(skill_id=skill_id, metadata={"category": "coffee"})
        )
    )

    # Report the version actually used by an Agent and read its evidence.
    reported = client.skills.report_trajectory(
        TrajectoryReportRequest(
            user_id="u_sdk_demo",
            agent_id="agent_001",
            skill_version_id=second_version,
            messages=[
                {"role": "user", "content": "我喜欢什么咖啡？"},
                {"role": "assistant", "content": "你喜欢冰美式，不加糖。"},
            ],
        )
    )
    print(
        client.skills.show_trajectory(
            TrajectoryShowRequest(trajectory_id=reported["trajectory_id"])
        )
    )
    print(
        client.skills.list_trajectories(
            TrajectoryListRequest(skill_id=skill_id, limit=20)
        )
    )
    print(
        client.skills.trajectory_statistics(
            TrajectoryStatisticsRequest(
                skill_id=skill_id, skill_version_id=second_version
            )
        )
    )

    # Publish a graph, inspect a pinned snapshot, and retrieve recommendations.
    graph = client.skills.organize(
        OrganizeRequest(skill_refs=[second_version], alias=f"{alias}-graph")
    )
    graph_id = graph["graph_id"]
    graph_version = graph["graph_version_id"]
    print(client.skills.list_graphs())
    print(client.skills.graph_resolve(GraphResolveRequest(graph_ref=graph_id)))
    print(client.skills.graph_snapshots(GraphSnapshotsRequest(graph_id=graph_id)))
    print(
        client.skills.graph_snapshot(
            GraphSnapshotRequest(
                graph_id=graph_id, graph_version_id=graph_version
            )
        )
    )
    print(client.skills.preview_graph(GraphResolveRequest(graph_ref=graph_id)))
    print(
        client.skills.recommend(
            RecommendRequest(
                graph_ref=graph_id,
                graph_version_id=graph_version,
                query="回答用户的咖啡偏好",
                top_k=5,
            )
        )
    )
    extra = client.skills.register(
        RegisterRequest(
            alias=f"{alias}-extra", skill=StructuredSkillInfo(content=content)
        )
    )
    print(
        client.skills.add_graph_members(
            AddGraphMembersRequest(
                graph_ref=graph_id, skill_refs=[extra["skill_version_id"]]
            )
        )
    )
    print(
        client.skills.remove_graph_members(
            RemoveGraphMembersRequest(
                graph_ref=graph_id, skill_refs=[extra["skill_version_id"]]
            )
        )
    )

    # Submit evolution and training sequentially, without auto-activation.
    evolution = client.skills.evolve(
        EvolveRequest(skill_version_id=second_version, auto_accept=False)
    )
    print(wait_for_task(client, evolution["task_id"]))
    environment = EnvironmentConfig(
        class_name="searchqa",
        config={
            "agent": {
                "model": "YOUR_CHAT_MODEL",
                "api_key": "YOUR_MODEL_API_KEY",
                "base_url": "https://YOUR_MODEL_ENDPOINT/v1",
            }
        },
    )
    samples = [
        {
            "task": {
                "id": task_id,
                "instruction": "What coffee does the user prefer?",
                "context": "The user prefers iced Americano.",
            },
            "reward_data": {"answers": ["iced Americano"]},
        }
        for task_id in ("t1", "t2", "t3")
    ]
    training = client.skills.train(
        TrainingRequest(
            algo="treeskill",
            env=environment,
            init_skill_version_id=second_version,
            tasks=samples,
            split=TrainingSplitRequest(train=["t1"], val=["t2"], test=["t3"]),
            work_dir=f"/absolute/path/output/{alias}",
            auto_accept=False,
        )
    )
    print(wait_for_task(client, training["task_id"]))

    # Delete the demonstration graph and an unreferenced temporary Skill.
    print(client.skills.delete_graph(GraphResolveRequest(graph_ref=graph_id)))
    temporary = client.skills.register(
        RegisterRequest(
            alias=f"{alias}-temporary", skill=StructuredSkillInfo(content=content)
        )
    )
    print(client.skills.delete(DeleteRequest(skill_id=temporary["skill_id"])))
```

Skill 方法直接返回业务数据字典。算法任务提交后，通过 `show_task()` 和 `task_progress()` 获取状态与日志；超时只结束示例的等待，不会取消服务任务。此示例调用独立 Skill API。
