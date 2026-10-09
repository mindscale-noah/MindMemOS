# MindMemOS SDK Skill CLI：当前命令、接口、参数与默认值

核对日期：2026-10-08。本文以当前工作区源码为准，包括尚未提交的实现。SDK 包声明版本为 `0.1.4`；本文不保证 PyPI 上的同名版本具有相同行为。文档结构参照 [SDK CLI 当前命令](sdk_cli_current.md)。

**当前工作区已实现 `mindmemos skill ...` 的 25 个命令，并通过 SDK 对接独立 Skill FastAPI。** 本文记录本地源码中的命令行为；使用前需安装或运行当前工作区 SDK，并配置兼容的 Skill 服务。算法执行依赖服务配置，本次未发布 PyPI 或部署线上服务。

## 1. 入口与实现范围

SDK 包名为 `mindmemos-sdk`，导入名为 `mindmemos_sdk`，命令入口为 `mindmemos_sdk.cli:main`。安装后使用 `mindmemos`。当前 CLI 只有 `auth`、`config`、`memory`、`doctor`；本地服务的 `mindmemos-skill` 是另一套入口。

```bash
mindmemos --help
```

以下命令均已接入 CLI。表中的 SDK 方法在 `SkillClient` 和 `AsyncSkillClient` 中均存在；HTTP 路径统一以 `/v1/skills/` 开头。CLI 负责引用解析、组合调用、本地输入与默认 JSON 输出。

最后一列给出对应 FastAPI 接口的 JSON 输入；多次调用按“解析 → 读取/提交”列出。示例中的精确 ID 应替换为前序响应或实际已有对象的 ID。所有 Skill CLI 命令默认输出 JSON，不需要输出格式开关。认证放在请求头，确认开关和本机文件路径不进入请求体；`register-structured --json` 接收结构化输入，包装为请求体的 `skill`。`task progress` 的 `after` 单独放在 URL 查询参数。Graph 推荐同时列出 HTTP 支持的精确快照格式与当前 SDK 可构造的格式。

| 命令 | 当前 SDK 方法 | 当前 HTTP 接口 | 实现边界 | FastAPI JSON 输入（请求体示例） |
| --- | --- | --- | --- | --- |
| `skill register` | `register()` | `POST /v1/skills/register` | 标准目录注册；自动资源收集和 include/exclude 筛选待 CLI 实现 | `{"alias":"demo-skill","skill":{"content":"---\nname: demo-skill\ndescription: Example skill\n---\n# Demo\n","commit_version":"v0.0.1","commit_message":"Initial registration","resources":{},"metadata":{}}}` |
| `skill register-structured` | `register()` | `POST /v1/skills/register` | 已实现 StructuredSkillInfo JSON 输入和非标准内容校验 | `{"alias":"custom-skill","skill":{"content":"Follow the supplied task instructions.","name":"custom-skill","description":"Nonstandard instruction document","standard_format":false,"resources":{},"metadata":{},"commit_version":"v0.0.1","commit_message":"Initial registration"}}` |
| `skill list` | `list()` | `POST /v1/skills/list` | 已实现家族过滤与 cursor 分页；CLI 使用 `--filter`，不提供 cursor 参数 | `{"filters":{"name":"demo-*"},"limit":20}` |
| `skill show` | `skill_resolve()`、`show()` | `POST /v1/skills/skill-resolve` → `show` | 已实现引用解析、精确版本读取与 JSON 输出 | 解析：`{"skill_ref":"demo-skill"}`<br>读取：`{"skill_id":"sk-family-id","skill_version_id":"skv-version-id"}` |
| `skill resolve` | `skill_resolve()` | `POST /v1/skills/skill-resolve` | 已实现，Skill 返回 `is_active` | `{"skill_ref":"demo-skill"}` |
| `skill export` | `skill_resolve()`、`show()` | `POST /v1/skills/skill-resolve` → `show` | CLI 已实现本地导出、overwrite 和失败恢复 | 解析：`{"skill_ref":"demo-skill"}`<br>读取：`{"skill_id":"sk-family-id","skill_version_id":"skv-version-id"}`<br>导出目录仅用于本机，不进入 JSON |
| `skill commit` | `skill_resolve()`、`commit()` | `POST /v1/skills/commit`，提交前解析 | 已实现结构化提交、资源继承与激活参数 | 解析：`{"skill_ref":"demo-skill"}`<br>提交：`{"skill_id":"sk-family-id","skill":{"content":"---\nname: demo-skill\ndescription: Example skill\n---\n# Demo\n","commit_message":"Explain the change","version_bump":"patch"},"base_version_id":"skv-version-id","active":false,"inherit_resources":true}` |
| `skill history` | `history()` | `POST /v1/skills/history` | 已实现 offset/cursor 分页；CLI 仅使用 offset | 解析：`{"skill_ref":"demo-skill"}`<br>历史：`{"skill_id":"sk-family-id","limit":20,"offset":0}` |
| `skill diff` | `diff()` | `POST /v1/skills/diff` | 已实现同项目内跨家族比较 | 源解析：`{"skill_ref":"demo-skill"}`<br>目标解析：`{"skill_ref":"skv-target-id"}`<br>比较：`{"skill_id":"sk-family-id","from_skill_version_id":"skv-source-id","to_skill_id":"sk-target-family-id","to_skill_version_id":"skv-target-id"}` |
| `skill activate` | `activate()` | `POST /v1/skills/activate` | CLI 支持标签或 ID 选择，API 接收精确版本 ID | 家族解析：`{"skill_ref":"demo-skill"}`<br>标签查询：`{"skill_id":"sk-family-id","limit":20,"offset":0}`<br>激活：`{"skill_id":"sk-family-id","skill_version_id":"skv-target-id"}` |
| `skill update-info` | `update_info()` | `POST /v1/skills/update-info` | 已实现，可显式清除 alias | 解析：`{"skill_ref":"demo-skill"}`<br>更新：`{"skill_id":"sk-family-id","alias":"new-alias"}`<br>清除 alias：`{"skill_id":"sk-family-id","alias":null}` |
| `skill delete` | `delete()` | `POST /v1/skills/delete` | 已实现软删除，保留历史与引用记录 | 解析：`{"skill_ref":"demo-skill"}`<br>删除：`{"skill_id":"sk-family-id"}` |
| `skill evolve` | `evolve()` | `POST /v1/skills/algos/evolve` | 已实现异步受理；SDK 正确识别 `accepted` | 解析：`{"skill_ref":"demo-skill"}`<br>演进：`{"skill_version_id":"skv-version-id","auto_accept":false}` |
| `skill train` | `train()` | `POST /v1/skills/algos/train` | 已实现异步受理；SDK 正确识别 `accepted` | 解析：`{"skill_ref":"demo-skill"}`<br>训练：`{"algo":"algo-name","env":{"class_name":"demo_offline","config":{}},"init_skill_version_id":"skv-version-id","work_dir":"/server/run-1","tasks":[{"task":{"id":"task-1","instruction":"完成任务"},"reward_data":{}}],"split":{"train":["task-1"],"val":[],"test":[]},"auto_accept":false}` |
| `skill task show` | `show_task()` | `POST /v1/skills/tasks/show` | 已实现任务查询 | `{"task_id":"task-id"}` |
| `skill task progress` | `task_progress()` | `POST /v1/skills/tasks/progress?after=…` | 已实现单次事件数组和 follow JSON Lines | `{"task_id":"task-id"}`<br>查询参数：`?after=0`，不放入 JSON |
| `skill trajectory report` | `report_trajectory()` | `POST /v1/skills/trajectories/report` | 已实现 JSON 参数、文件和 stdin 输入 | `{"user_id":"user-1","agent_id":"agent-1","messages":[{"role":"user","content":"执行输入"}],"skill_version_id":"skv-version-id","status":"completed"}` |
| `skill trajectory list` | `list_trajectories()` | `POST /v1/skills/trajectories/list` | 已实现过滤与 cursor 分页；CLI 无 cursor，按解析版本过滤 | 解析：`{"skill_ref":"demo-skill"}`<br>列表：`{"skill_id":"sk-family-id","skill_version_id":"skv-version-id","limit":20}` |
| `skill trajectory show` | `show_trajectory()` | `POST /v1/skills/trajectories/show` | 已实现完整轨迹读取 | `{"trajectory_id":"trajectory-id"}` |
| `skill graph list` | `list_graphs()` | `POST /v1/skills/graphs/list` | 已实现，不接受筛选请求体 | `{}`（SDK 发送空对象；FastAPI 无请求体参数） |
| `skill graph resolve` | `graph_resolve()` | `POST /v1/skills/graph-resolve` | 已实现家族与快照解析 | `{"graph_ref":"graph-alias"}` |
| `skill graph history` | `graph_snapshots()` | `POST /v1/skills/graphs/snapshots` | 已实现快照列表 | 解析：`{"graph_ref":"graph-alias"}`<br>历史：`{"graph_id":"skg-graph-id"}` |
| `skill graph show` | `graph_snapshot()` | `POST /v1/skills/graphs/snapshot` | 已实现精确快照读取 | 解析：`{"graph_ref":"graph-alias"}`<br>快照：`{"graph_id":"skg-graph-id","graph_version_id":"skgv-snapshot-id"}` |
| `skill graph publish` | `organize()` | `POST /v1/skills/algos/organize` | 已实现同步发布，输入为 `skill_refs` | 新建：`{"alias":"graph-alias","skill_refs":["skv-version-1","skv-version-2"],"commit_message":"Publish graph"}`<br>更新：`{"graph_id":"skg-graph-id","skill_refs":["skv-version-1"],"commit_message":"Update graph"}` |
| `skill graph recommend` | `recommend()` | `POST /v1/skills/algos/recommend` | CLI/SDK 已支持精确 graph_version_id | 解析：`{"graph_ref":"graph-alias"}`<br>HTTP 精确快照：`{"graph_ref":"skg-graph-id","graph_version_id":"skgv-snapshot-id","query":"任务描述","top_k":5}`<br>当前 SDK 可用：`{"graph_ref":"skgv-snapshot-id","query":"任务描述","top_k":5}` |

本文共实现 25 个 CLI 命令；两个注册命令复用同一个 `register()` 方法和 HTTP 路由。SDK 共覆盖 25 个 Skill HTTP 操作。除表中的远程操作，还包含 `version_resolve()` 和 `delete_graph()`。前者保留为 SDK/Harness 能力，不恢复此前删除的源码哈希 CLI 命令；后者暂不新增 CLI 命令。`export` 不对应独立 HTTP 路由。

## 2. 配置来源与网络默认值

Skill CLI 复用 `mindmemos auth` 保存的配置。所选配置目录按 `MINDMEMOS_HOME`、旧 `MINDMEMOS_CONFIG_DIR`、`~/.config/mindmemos/home.json`、`~/.mindmemos` 的顺序解析；具体配置命令见 [SDK CLI 文档第 2–4 节](sdk_cli_current.md)。

| 配置 / 行为 | 当前 SDK 默认值 | 实际含义 |
| --- | --- | --- |
| `skill_service.base_url` | `None` | 复用解析后的 Memory 地址与 key |
| `skill_service.auth` | `None` | 独立地址可保存独立 key |
| `network.timeout_seconds` | `30` | HTTP 请求超时，单位为秒 |
| Skill transport `max_retries` | `0` | 根客户端为 Skill 单独关闭网络重试，不使用 Memory 的默认 2 次重试 |
| Skill transport `require_api_key` | `False` | 底层支持本地无认证服务；是否需要认证由服务端决定 |
| HTTP 方法 | `POST` | 读取、删除和任务进度同样使用 POST |
| 项目身份 | 认证身份 / 本地绑定范围 | CLI 不提供 `--project-id`；HTTP 不信任调用方的项目提示 |

SDK 的独立 Skill 客户端对应 `src/mindmemos_skill/` 的 API。仓库中 `src/mindmemos/` 的旧云 Skill 路由和请求结构不同；共用 Memory 地址并不保证目标服务兼容这套 API。线上是否接入独立服务须按实际部署核对。

当前 `auth` 可配置独立 Skill 地址，`doctor` 只检查本地初始化，不探测接口兼容性。以下是已经实现的配置命令示例：

```bash
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --memory-api-key YOUR_MEMORY_KEY \
  --skill-base-url http://127.0.0.1:8001 \
  --skill-api-key YOUR_SKILL_KEY --user-id u_123
```

## 3. 引用、参数与公共行为

以下为当前 CLI 公共行为：

| 参数 / 输入 | 类型 | CLI 默认值 | 当前接口语义 |
| --- | --- | --- | --- |
| `SKILL_REF` | alias、家族 ID/唯一前缀、版本 ID/唯一前缀 | 无 | `skill-resolve` 选择当前或精确历史版本 |
| `GRAPH_REF` | alias、Graph ID/唯一前缀、快照 ID/唯一前缀 | 无 | `graph-resolve` 选择当前发布或精确历史快照 |
| `--json` | JSON 字符串，仅 `register-structured` | 必填，无默认值 | 接收 StructuredSkillInfo 并包装为 `skill`；用于输入 |
| `--yes` / `-y` | 布尔开关 | `False` | 删除默认询问确认，回车取消；不进入请求体 |

所有接收 `SKILL_REF` 的命令统一先调用 `skill-resolve`：alias、家族 ID 或唯一家族前缀没有指定明确版本，选择当前生效（激活）版本；版本 ID 或唯一版本前缀选择给定版本。CLI 使用解析返回的家族和版本 ID；`activate` 另外要求通过 `--version` 或 `--version-id` 明确指定激活目标，详见第 9.3 节。歧义引用报错，不自动选择候选。历史查询、信息更新和删除作用于所属家族；提交使用选定版本作为父版本；轨迹列表按选定版本过滤。服务端仍负责最终项目归属与权限校验。

所有 CLI 命令均不提供 cursor 参数，也不接受用户传入续页 token。history 保留 offset 分页；list 和 trajectory list 单次只请求第一页，不自动续页。SDK/FastAPI 的 cursor 能力及响应中的 `next_cursor` 属于当前接口事实，保留原样。任务进度的 `--after` 是事件序号，不是分页 token。

当前 SDK 使用 `exclude_unset=True` 序列化。只在模型中存在默认值而调用方没有赋值的字段通常不发送；服务端使用自己的默认值。显式传入 `alias=None` 会发送 JSON `null`，可用于清除别名。CLI 默认值不能等同于当前 SDK/API 默认值。

参数示例沿用 `--alias x`。argparse 解析器同时接受 `--alias=x`。CLI 不新增全局服务地址或 key 参数，连接仍通过 `auth` 配置。

所有 Skill 命令默认返回 JSON。普通 API 操作输出服务返回的 data；展示选项可缩减字段，但输出仍为 JSON。CLI 不提供 JSON 输出开关，现有 auth/config/memory/doctor 的输出行为不属于本次调整范围。

## 4. 注册与提交共用的结构化内容

当前 `register` 和 `commit` 使用嵌套 `skill` 对象，不再使用旧的顶层 `bundle`。辅助资源值为原始字节的标准 Base64，正文保留原始 UTF-8 文本与换行。

| `skill` 字段 | 类型 / 取值 | 必填 | 当前 SDK/API 默认值 | 实际含义 |
| --- | --- | --- | --- | --- |
| `content` | 字符串 | 是 | 无 | 完整 `SKILL.md` 原文 |
| `resources` | 路径到 Base64 的对象 | 否 | `{}` | 辅助资源；提交时默认为替换为空资源 |
| `name` | 字符串 | 条件必填 | `None` | 标准格式由服务端从 frontmatter 解析；非标准格式需提供 |
| `description` | 字符串 | 条件必填 | `None` | 标准格式由服务端解析；非标准格式需提供 |
| `standard_format` | 布尔值 | 否 | `True` | 校验标准 frontmatter |
| `metadata` | JSON 对象 | 否 | `{}` | 用户元数据；标准 frontmatter 的额外字段也由服务端合并 |
| `commit_message` | 字符串或 null | 否 | `None` | 版本说明；HTTP 现在不要求非空说明 |
| `commit_version` | `v主.次.修订` 或 null | 否 | `None` | 显式版本标签；省略时由服务端分配 |
| `version_bump` | `patch` / `minor` / `major` | 否 | `patch` | 自动编号的递增分量 |
| `commit_source` | `user` / `algo` | 否 | `user` | 提交来源；目录注册使用 user，结构化输入按现有模型校验该字段 |

SDK 已实现 Base64 编解码和资源路径校验。读取目录、阻止符号链接越界及导出写目录仍需 CLI 文件处理实现；不能仅凭资源字符串校验宣称目录操作已完成。

## 5. Skill 注册

### 5.1 `skill register`：标准目录输入

命令语法：

```text
mindmemos skill register SOURCE [选项]
```

接口：`POST /v1/skills/register`。当前 SDK：`register(RegisterRequest(...))`。

| 参数 | 类型 / 取值 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `SOURCE` | 目录或 `SKILL.md` | 是 | 无 | 目录的根 `SKILL.md` 或指定文件作为正文，所在目录为资源根目录 |
| `--alias` | 字符串 | 否 | frontmatter name | `alias`；这是 CLI 便利默认值，API 默认 null |
| `--version` | `v主.次.修订` | 否 | `v0.0.1` | 显式发送 `skill.commit_version`；API 省略时初始版本为 `v1.0.0` |
| `--message` | 字符串 | 否 | `Initial registration` | `skill.commit_message`；API 本身允许省略 |
| `--metadata-json` | JSON 对象 | 否 | `{}` | `skill.metadata` |
| `--exclude` | 逗号分隔的通配符字符串 | 否 | `None` | 从 include 选出的候选资源中排除命中文件或目录；优先级高于 include |
| `--include` | 逗号分隔的通配符字符串 | 否 | `None` | 选择候选资源范围；省略时递归收集源目录内全部辅助文件 |

执行步骤：

1. 解析 `SOURCE`。路径须存在；目录须有根 `SKILL.md`，指定文件须为 `SKILL.md`。解析并校验标准 frontmatter 的 name/description，不合法时在上传前报错。
2. 递归收集源目录内的辅助文件，根 `SKILL.md` 已作为正文，不重复加入 resources。用户不需要指定 `--resources-dir`。
3. 先按 `--include` 选择候选资源，再按 `--exclude` 排除资源；同一文件同时命中时排除优先。模式相对于源目录，路径分隔符使用 `/`；命中目录时作用于其子树。逗号两侧空白去除，显式提供空模式时报参数错误。
4. 校验资源相对路径与符号链接边界，拒绝源目录外文件。读取保留文件的原始字节并编码为 Base64，填入 `skill.resources`；原始正文填入 `skill.content`，使用 `standard_format=true`。
5. 补充 alias、版本、说明与 metadata，构造 `RegisterRequest(skill=StructuredSkillInfo(...), alias=...)`，调用现有 `register()` 上传；校验失败时不发送请求。

include/exclude 只筛选辅助文件，必需的根 `SKILL.md` 始终作为正文上传。筛选模式由调用方指定，不执行敏感内容自动识别；需要排除的文件应明确写入 exclude，或通过 include 限定上传范围。`--include` 为参数的正确拼写。

命令示例：

```bash
mindmemos skill register ./demo-skill \
  --include 'scripts/**,references/**,assets/**' \
  --exclude '**/.env,**/.env.*,**/*.pem,**/*.key,private/**'
```

```json
{"alias":"demo-skill","skill":{"content":"---\nname: demo-skill\ndescription: Example skill\n---\n# Demo\n","commit_version":"v0.0.1","commit_message":"Initial registration","resources":{},"metadata":{}}}
```

返回包含 `skill_id`、`skill_version_id` 和版本标签等信息。注册生成家族与初始当前版本，不维护客户端注册表。

### 5.2 `skill register-structured`：直接输入 StructuredSkillInfo

命令语法：

```text
mindmemos skill register-structured --json STRUCTURED_SKILL_INFO_JSON [--alias ALIAS]
```

接口同样为 `POST /v1/skills/register`，复用当前 SDK 的 `register(RegisterRequest(...))`。该命令直接接收第 4 节定义的 `StructuredSkillInfo` 对象，支持设置 `standard_format=false` 注册非标准正文；也支持传入标准内容。

| 参数 | 类型 / 取值 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `--json` | 单个 JSON 对象字符串 | 是 | 无 | 解析为 StructuredSkillInfo，整体包装为请求体的 `skill` |
| `--alias` | 非空字符串 | 否 | `None` | 请求体顶层 `alias`，省略时不自动使用 name |

`--json` 的值是 StructuredSkillInfo 本身，不带外层 `skill` 或 `alias`。提供非标准 Skill 时，必须显式设置 `standard_format=false` 并填写 `name`、`description`；省略 standard_format 时使用模型默认 true 并校验标准 frontmatter。

执行步骤：

1. 解析 `--json`，要求顶层为单个对象，并用当前 SDK 的 `StructuredSkillInfo` 校验；未知字段、非法 JSON 或资源编码/路径错误在上传前报错。
2. 按当前服务端规则校验内容与显示信息：标准内容校验 frontmatter，非标准内容要求显式 name/description。CLI 已在请求前完成内容校验；服务端保留最终校验。
3. resources 由 JSON 直接提供，相对路径对应标准 Base64 原始字节。该命令不读取本地资源目录，因此没有 SOURCE、include/exclude 或 resources-dir 参数。
4. 保留 JSON 中的正文、资源和提交信息，按模型/服务端默认值处理省略字段：resources/metadata 为 `{}`、commit_message/commit_version 为 `None`、version_bump 为 patch、commit_source 为 user。不套用目录注册的别名、v0.0.1 或 Initial registration 默认值；commit_version 省略时由 API 分配初始 v1.0.0。
5. 将对象包装为 `RegisterRequest(skill=..., alias=...)`，调用现有 register 接口，默认输出返回的 data JSON，包含家族与版本标识。`--json` 在此命令中专用于输入。

非标准 Skill 的 CLI 输入示例：

```bash
mindmemos skill register-structured \
  --alias custom-skill \
  --json '{"content":"Follow the supplied task instructions.","name":"custom-skill","description":"Nonstandard instruction document","standard_format":false,"resources":{},"metadata":{},"commit_version":"v0.0.1","commit_message":"Initial registration"}'
```

对应的 FastAPI JSON 输入为：

```json
{
  "alias": "custom-skill",
  "skill": {
    "content": "Follow the supplied task instructions.",
    "name": "custom-skill",
    "description": "Nonstandard instruction document",
    "standard_format": false,
    "resources": {},
    "metadata": {},
    "commit_version": "v0.0.1",
    "commit_message": "Initial registration"
  }
}
```

CLI、SDK 和 FastAPI 均已支持上述非标准注册输入，复用现有 HTTP 路由。

## 6. `skill list`

```text
mindmemos skill list [--filter FILTER_JSON] [过滤参数] [--limit LIMIT]
```

接口：`POST /v1/skills/list`。当前 SDK：`list(ListRequest(...))`。

| 参数 | 类型 / 取值 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `--filter` | JSON 对象字符串 | 否 | `{}` | 解析后发送到 `filters`；支持 `name`、`alias`、`user_id`、`agent_id`，字段值为非空字符串 |
| `--name` / `--alias` | 通配模式 | 否 | `None` | `filters.name` / `filters.alias` |
| `--user-id` / `--agent-id` | 字符串 | 否 | `None` | 轨迹中的执行者过滤；不自动使用配置默认用户 |
| `--limit` | 正整数 | 否 | `20` | `limit`；当前 SDK/HTTP 未定义旧文档所述的 100 上限 |

`--filter` 接收过滤对象本身，不包含外层 `filters`。多个字段按 AND 组合；`name` 和 `alias` 使用区分大小写的完整名称通配匹配，`user_id` 和 `agent_id` 使用精确匹配，二者同时指定时须匹配同一条轨迹。不支持任意过滤 DSL 或额外字段。单字段选项与 `--filter` 合并；同一字段重复指定时报参数错误。

当前 API 只返回家族摘要和当前版本指针，不展开版本历史。SDK/API `limit` 默认 `None`；CLI 显式发送 20。CLI 不提供 `--cursor`，只请求第一页，不自动续页，也不在请求体中发送 `cursor`。当前 SDK/FastAPI 仍支持 cursor 分页，默认 JSON 输出保留服务返回的 `has_more`、`next_cursor` 等字段。

CLI 示例：

```bash
mindmemos skill list --filter '{"name":"demo-*","user_id":"user-001"}' --limit 20
```

对应 FastAPI JSON 输入：

```json
{"filters":{"name":"demo-*","user_id":"user-001"},"limit":20}
```

## 7. `skill show`、`resolve` 与 `export`

### 7.1 `skill show`

```text
mindmemos skill show SKILL_REF
```

接口：先 `POST /v1/skills/skill-resolve`，再 `POST /v1/skills/show`。

| 参数 | 类型 | 必填 | CLI 默认值 | 行为 |
| --- | --- | --- | --- | --- |
| `SKILL_REF` | alias、家族 ID/唯一前缀、版本 ID/唯一前缀 | 是 | 无 | 家族引用选择当前激活版本；版本引用选择指定历史版本，并解析其所属家族 |

命令只接收 `SKILL_REF`。当前 `skill-resolve` 已支持版本 ID 及唯一版本前缀，因此查看指定版本时直接将其作为 `SKILL_REF`，无需单独的版本参数。先调用 `skill-resolve`，再将返回的 `skill_id` 和 `skill_version_id` 传给 `show`；版本前缀存在歧义时报错。

默认输出完整 data JSON，包含 `skill` 家族信息与 `version`，版本正文在 `version.content`，资源在 `version.resources`。当前 HTTP `show` 只接收 ID，引用解析由 CLI 组合调用完成。

查看指定版本的 CLI 示例：

```bash
mindmemos skill show skv-version-id
```

对应 FastAPI JSON 输入，第一步解析版本引用：

```json
{"skill_ref":"skv-version-id"}
```

第二步读取解析结果中的家族与版本：

```json
{"skill_id":"sk-family-id","skill_version_id":"skv-version-id"}
```

### 7.2 `skill resolve`

```text
mindmemos skill resolve SKILL_REF
```

唯一必填参数为 `SKILL_REF`。请求体为 `{"skill_ref":"demo-skill"}`。当前返回 `skill_id`、`skill_version_id`、`alias`、`version_label`、`is_active`；旧文档的 Skill `is_current` 应改为 `is_active`。该命令只读。

### 7.3 `skill export`

```text
mindmemos skill export SKILL_REF --to DIRECTORY [--overwrite]
```

| 参数 | 类型 | 必填 | CLI 默认值 | 行为 |
| --- | --- | --- | --- | --- |
| `SKILL_REF` | alias、家族 ID/唯一前缀、版本 ID/唯一前缀 | 是 | 无 | 家族引用选择当前激活版本；版本引用选择指定历史版本，先解析再读取完整内容 |
| `--to` | 本地目录 | 是 | 无 | 本地输出位置，不进入 HTTP 请求体 |
| `--overwrite` | 布尔开关 | 否 | `False` | 目标目录非空时显式允许覆盖；仅控制本地文件写入，不进入 HTTP 请求体 |

与 `skill show` 使用相同的引用解析流程：直接将版本 ID 或唯一版本前缀作为 `SKILL_REF`，再使用解析返回的 `skill_id` 和 `skill_version_id` 读取该版本，不提供单独的版本参数。目标目录非空且未传 `--overwrite` 时，报错并保留已有文件。

导出指定版本的 CLI 示例：

```bash
mindmemos skill export skv-version-id --to ./exported-skill --overwrite
```

对应 FastAPI JSON 输入，第一步调用 `POST /v1/skills/skill-resolve`：

```json
{"skill_ref":"skv-version-id"}
```

第二步将解析结果传给 `POST /v1/skills/show`：

```json
{"skill_id":"sk-family-id","skill_version_id":"skv-version-id"}
```

当前 `SkillClient` 没有 `export()` 方法；本地文件导出由 CLI 文件模块实现。独立服务的 Python `export_skill()` 是服务进程内能力，不是远程 SDK 的本机导出流程。CLI 根据 `show` 的 `content/resources` 写出 `SKILL.md` 和解码资源，验证目标路径，并使用暂存目录与替换失败回滚保留原目录。

## 8. `skill commit`

```text
mindmemos skill commit SKILL_REF --from SOURCE --message TEXT [选项]
```

接口：解析 `SKILL_REF` 后，调用 `POST /v1/skills/commit`。

| 参数 | 类型 / 取值 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `SKILL_REF` | 家族或版本引用 | 是 | 无 | 解析出所属家族 `skill_id` 和选定版本 |
| `--from` | 目录或 `SKILL.md` | 是 | 无 | 本地读取 `skill.content` |
| `--message` | 非空字符串 | 是 | 无 | `skill.commit_message`；非空是 CLI 设计要求，不是当前 HTTP 要求 |
| `--version` | 版本标签 | 否 | `None` | `skill.commit_version`，与 `--bump` 互斥 |
| `--bump` | `patch` / `minor` / `major` | 否 | `patch` | `skill.version_bump`；显式标签时不发送自定义 bump |
| `--resources-dir` | 目录 | 否 | `None` | 替换资源，发送编码后的 `skill.resources`，`inherit_resources=false` |
| `--clear-resources` | 布尔开关 | 否 | `False` | 清空资源，发送 `{}` 且 `inherit_resources=false` |
| `--activate` | 布尔开关 | 否 | `False` | CLI 显式发送 `active=false`，指定后为 true |

资源目录和清空开关互斥。都不传时，CLI 保持旧版本资源，因此必须显式发送 `inherit_resources=true`。当前 SDK/API 默认 `inherit_resources=False`、`active=True`，不能依靠字段省略来实现 CLI 默认行为。

CLI 将解析选定的版本作为 `base_version_id`。该字段表示新版本的父版本；当前实现允许从历史父版本建分支，不能将它描述为客户端读取后到提交之间的严格当前版本比较锁。服务端激活写入有自身的并发校验。

```json
{"skill_id":"sk-family-id","skill":{"content":"---\nname: demo-skill\ndescription: Updated skill\n---\n# Demo\n","commit_message":"Explain the change","version_bump":"patch"},"base_version_id":"skv-parent-id","active":false,"inherit_resources":true}
```

## 9. 版本历史、比较与激活

### 9.1 `skill history`

```text
mindmemos skill history SKILL_REF [--limit LIMIT] [--offset OFFSET]
```

接口：`POST /v1/skills/history`。先解析引用，使用所属家族 `skill_id` 查询整个家族历史；传入版本引用也不将历史限定为该版本。

| 参数 | 类型 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `SKILL_REF` | 家族或版本引用 | 是 | 无 | 解析出所属家族 `skill_id` 和选定版本 |
| `--limit` | 正整数 | 否 | `20` | `limit`，当前 SDK/HTTP 无 100 上限 |
| `--offset` | 非负整数 | 否 | `0` | `offset` |

CLI 只发送 `limit` 和 `offset` 分页参数，不发送 cursor。返回 `items`、`has_more`、`next_cursor`，保持默认 JSON 输出。当前 API 的版本摘要不包含完整正文；CLI 若标记当前版本，需要结合家族当前版本指针。

### 9.2 `skill diff`

```text
mindmemos skill diff SKILL_REF_A SKILL_REF_B
```

接口：分别调用 `POST /v1/skills/skill-resolve` 解析源、目标引用，再调用 `POST /v1/skills/diff`。

| 参数 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- |
| `SKILL_REF_A` | 是 | 无 | 源引用解析结果映射为 `skill_id`、`from_skill_version_id` |
| `SKILL_REF_B` | 是 | 无 | 目标引用解析结果映射为 `to_skill_id`、`to_skill_version_id`，支持同项目跨家族比较 |

两个位置参数均按统一规则选择版本：家族引用选择当前生效版本，版本引用选择给定版本。比较方向为 A → B；不提供 `--from`、`--to` 或 `--to-skill`，明确发送两侧解析出的版本 ID。

```json
{"skill_id":"sk-family-id","from_skill_version_id":"skv-source-id","to_skill_id":"sk-target-family-id","to_skill_version_id":"skv-target-id"}
```

### 9.3 `skill activate`

```text
mindmemos skill activate SKILL_REF (--version VERSION | --version-id VERSION_ID)
```

接口：`POST /v1/skills/activate`。

| 参数 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- |
| `SKILL_REF` | 是 | 无 | 解析出所属家族 `skill_id`，限定目标版本的家族范围 |
| `--version` | 与 `--version-id` 二选一，必填 | 无 | 版本标签，如 `v1.2.3`；在该家族历史中匹配 `version_label`，将匹配版本的 ID 发送为 `skill_version_id` |
| `--version-id` | 与 `--version` 二选一，必填 | 无 | 精确版本 ID；校验属于解析出的家族，发送为 `skill_version_id` |

两个目标版本选项互斥，必须指定其中一个；未指定或同时指定时，CLI 报参数错误。`SKILL_REF` 用于确定所属家族，激活目标由显式版本选项决定；若引用本身包含版本，也仅使用其所属家族。目标不存在、版本标签匹配不唯一或版本不属于该家族时，报错。

当前 SDK/FastAPI 的 activate 请求只接受版本 ID，没有直接按版本标签激活的字段。`--version` 需要先调用 `POST /v1/skills/history`，使用 `limit` 和 `offset` 遍历家族历史以匹配标签；`--version-id` 可通过现有解析接口校验家族归属。最终统一调用 activate，不将 CLI 的 `version` 标签直接发到该接口。

CLI 示例：

```bash
mindmemos skill activate demo-skill --version v1.2.3
mindmemos skill activate demo-skill --version-id skv-target-id
```

对应 FastAPI JSON 输入，先确定家族：

```json
{"skill_ref":"demo-skill"}
```

按标签选择时，调用 history 并按需增加 offset：

```json
{"skill_id":"sk-family-id","limit":20,"offset":0}
```

最终 activate 请求，两种选择方式使用相同的请求格式：

```json
{"skill_id":"sk-family-id","skill_version_id":"skv-target-id"}
```

## 10. 家族信息与删除

### 10.1 `skill update-info`

```text
mindmemos skill update-info SKILL_REF [信息选项]
```

接口：`POST /v1/skills/update-info`。

| 参数 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- |
| `SKILL_REF` | 是 | 无 | 解析后使用所属家族 `skill_id`，操作作用于整个家族 |
| `--name` | 否 | 不修改 | `name`，不允许显式 null |
| `--alias` | 否 | 不修改 | `alias` |
| `--clear-alias` | 否 | `False` | 显式发送 `alias=null`，与 `--alias` 互斥 |
| `--metadata-json` | 否 | 不修改 | 替换 `metadata`，`{}` 清空，null 不合法 |

至少提供一项变更。当前 SDK 允许构造空变更模型，服务端会拒绝；CLI 应在请求前校验。

```json
{"skill_id":"sk-family-id","alias":null}
```

### 10.2 `skill delete`

```text
mindmemos skill delete SKILL_REF [--yes]
```

必填 `SKILL_REF` 解析后使用所属家族 `skill_id`；版本引用也删除整个所属家族；`--yes` / `-y` 默认 false，仅跳过 CLI 确认。请求为 `POST /v1/skills/delete`，请求体为 `{"skill_id":"sk-family-id"}`。

当前为软删除：家族不再作为正常可用对象列出，但历史版本和引用记录保留。旧文档中“删除家族及版本”的表述已过时。

## 11. 演进与训练

### 11.1 `skill evolve`

```text
mindmemos skill evolve SKILL_REF [--auto-accept]
```

接口：解析选定版本后，调用 `POST /v1/skills/algos/evolve`。

| 参数 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- |
| `SKILL_REF` | 是 | 无 | 家族选当前版本，版本引用选精确历史版本，发送 `skill_version_id` |
| `--auto-accept` | 否 | `False` | 显式发送 false，启用时发送 true；SDK/API 默认 true |

```json
{"skill_version_id":"skv-version-id","auto_accept":false}
```

当前 HTTP 入口要求 HTTP 策略中恰好允许一个 evolution pipeline，并由服务端选择。SDK/API 不接受演进 `pipeline` 请求字段。服务需配置持久化、任务执行和演进流程。

HTTP 受理返回 `202`、`code=accepted` 和 `task_id`。SDK 已正确识别成功受理，CLI 返回任务 JSON，可通过 `task show/progress` 跟踪。

### 11.2 `skill train`

设计保留逐项传参，不使用 `--request-file`：

```text
mindmemos skill train SKILL_REF --work-dir SERVER_DIR \
  --pipeline NAME --environment PATH --sample-json JSON --train-id TASK_ID [选项]
```

接口：`POST /v1/skills/algos/train`。当前 SDK：`train(TrainingRequest(...))`。

| 参数 | 类型 / 取值 | 必填 | CLI 默认值 | 当前请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `SKILL_REF` | 家族或版本引用 | 是 | 无 | 先解析引用，将选定版本发送为 **`init_skill_version_id`** |
| `--work-dir` | 服务端目录 | 是 | 无 | `work_dir`，不是本机输出目录 |
| `--pipeline` | 已配置训练流程名 | 是 | 无 | **`algo`**；当前 SDK/API 要求显式指定 |
| `--environment` | 环境配置 YAML/JSON 文件路径 | 与 `--environment-json` 二选一 | 无 | **`env`**；文件内容为 `class_name/config` |
| `--environment-json` | 内联环境配置 JSON | 与 `--environment` 二选一 | 无 | **`env`**；由服务端环境工厂校验 |
| `--sample-json` | 单个样本 JSON，可重复 | 是，至少一项 | 无 | 追加到 `tasks[]`，包含 `task.id`、`task.instruction` 和 `reward_data` |
| `--train-id` | 样本 ID，可重复 | 是，至少一项 | 无 | `split.train[]` |
| `--val-id` | 样本 ID，可重复 | 否 | 空数组 | `split.val[]` |
| `--test-id` | 样本 ID，可重复 | 否 | 空数组 | `split.test[]` |
| `--auto-accept` | 布尔开关 | 否 | `False` | CLI 显式发送 false；SDK/API 默认 true |

训练初始版本统一由 `SKILL_REF` 选择，不再提供 `--base-version-id`。HTTP 仍使用 `init_skill_version_id`，不接受旧顶层 `base_version_id`；没有旧字段兼容层。算法、环境与任务结构由当前服务校验。样本 ID 须唯一，划分不能重叠且只能引用提供的样本；`reward_data` 结构由环境决定。test 只评估最终选定候选，不参与训练选优。

```json
{"algo":"algo-name","env":{"class_name":"demo_offline","config":{}},"init_skill_version_id":"skv-version-id","work_dir":"/server/run-1","tasks":[{"task":{"id":"task-1","instruction":"完成任务"},"reward_data":{}}],"split":{"train":["task-1"],"val":[],"test":[]},"auto_accept":false}
```

命令示例：

```bash
mindmemos skill train skv-version-id \
  --work-dir /server/runs/run-1 \
  --pipeline algo-name --environment-json '{"class_name":"demo_offline","config":{}}' \
  --sample-json '{"task":{"id":"t1","instruction":"完成训练任务"},"reward_data":{}}' \
  --sample-json '{"task":{"id":"t2","instruction":"完成验证任务"},"reward_data":{}}' \
  --train-id t1 --val-id t2
```

当前训练 Pipeline、任务提交、进度与结果发布已有实现，但执行依赖服务配置。SDK 已正确识别 `202/accepted`，CLI 返回任务 JSON。

## 12. 异步任务查询与进度

### 12.1 `skill task show`

```text
mindmemos skill task show TASK_ID
```

`TASK_ID` 必填。接口为 `POST /v1/skills/tasks/show`，请求体为 `{"task_id":"task-id"}`。查询凭据所属项目的任务；未配置任务存储时返回能力不可用，任务不存在时返回 404。

### 12.2 `skill task progress`

```text
mindmemos skill task progress TASK_ID [--after SEQUENCE] [-f | --follow]
```

| 参数 | 必填 | CLI 默认值 | 行为 |
| --- | --- | --- | --- |
| `TASK_ID` | 是 | 无 | JSON 请求体 `task_id` |
| `--after` | 否 | `0` | 非负整数，作为 URL 查询参数发送 |
| `-f` / `--follow` | 否 | `False` | CLI 持续读取新事件，当前 SDK 仅提供单次读取 |

接口为 `POST /v1/skills/tasks/progress?after=0`，请求体只有 `{"task_id":"task-id"}`。SDK 已正确分开 body 和查询参数。默认单次输出事件数组 JSON；follow 持续输出时采用 JSON Lines，每行一个事件 JSON。follow 根据最新事件序号续读，每秒检查一次；任务 succeeded、failed 或 cancelled 后补读最终事件并停止。

## 13. 执行轨迹

### 13.1 `skill trajectory report`

```text
mindmemos skill trajectory report (--json-data JSON | --file JSON_PATH)
```

两种输入互斥且必须选一项，默认均未提供；`--file -` 从标准输入读取 UTF-8 JSON 对象。它们仅用于 CLI 输入处理，不是 HTTP 字段。

接口：`POST /v1/skills/trajectories/report`。

| JSON 字段 | 类型 | 必填 | 当前默认值 | 行为 |
| --- | --- | --- | --- | --- |
| `user_id` / `agent_id` | 非空字符串 | 是 | 无 | 本次实际执行身份，不自动补配置默认用户 |
| `messages` | 对象数组 | 是 | 无 | 至少一条，每条须有非空 role 和 content 字段 |
| `skill_version_id` | 精确版本 ID | 是 | 无 | 实际加载版本 |
| `session_id` | 字符串 | 否 | `None` | Harness 会话标识 |
| `started_at` / `finished_at` | 带时区时间 | 否 | `None` | 结束不早于开始；SDK 的宽松 datetime 不能替代服务端校验 |
| `status` | 非空字符串 | 否 | `completed` | 执行状态 |

```json
{"user_id":"user-1","agent_id":"agent-1","messages":[{"role":"user","content":"执行输入"}],"skill_version_id":"skv-version-id","status":"completed"}
```

### 13.2 `skill trajectory list`

```text
mindmemos skill trajectory list SKILL_REF [过滤参数] [--limit LIMIT]
```

接口：`POST /v1/skills/trajectories/list`。

| 参数 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- |
| `SKILL_REF` | 是 | 无 | 解析返回的家族和选定版本分别发送为 `skill_id`、`skill_version_id` |
| `--status` | 否 | `None` | `status` |
| `--received-from` / `--received-to` | 否 | `None` | `received_at_from` / `received_at_to`，带时区 |
| `--limit` | 否 | `20` | 正整数，当前 SDK/HTTP 无 100 上限 |

```json
{"skill_id":"sk-family-id","skill_version_id":"skv-version-id","limit":20}
```

家族引用默认只查询当前生效版本的轨迹；版本引用查询给定版本的轨迹。CLI 不提供独立版本参数或 cursor 参数，只请求第一页，不自动续页。当前 HTTP 可省略 `skill_version_id` 查询家族全部版本轨迹，但本 CLI 显式发送解析选定的版本。

### 13.3 `skill trajectory show`

```text
mindmemos skill trajectory show TRAJECTORY_ID
```

`TRAJECTORY_ID` 必填。接口为 `POST /v1/skills/trajectories/show`，请求体为 `{"trajectory_id":"trajectory-id"}`，返回完整轨迹和精确版本绑定。

## 14. Graph 管理

保留 `mindmemos skill graph` 命令组，不恢复旧 `skill set` 名称。Graph 特指管理流程发布的图与不可变快照。

### 14.1 `skill graph list`

```text
mindmemos skill graph list
```

无专用参数，默认输出返回的 data JSON。SDK 调用 `list_graphs()`，向 `POST /v1/skills/graphs/list` 发送 `{}`；当前 HTTP 路由未定义请求体，不提供 Graph 引用过滤。

### 14.2 `skill graph resolve`

```text
mindmemos skill graph resolve GRAPH_REF
```

唯一必填参数为 `GRAPH_REF`。向 `POST /v1/skills/graph-resolve` 发送 `{"graph_ref":"graph-alias"}`。返回 `graph_id`、`graph_version_id`、`alias`、`status`、`is_current`。未发布图可能没有版本 ID；Graph 无独立数字版本标签。

### 14.3 `skill graph history`

```text
mindmemos skill graph history GRAPH_REF
```

必填家族引用。先 resolve，然后向 `POST /v1/skills/graphs/snapshots` 发送 `{"graph_id":"skg-graph-id"}`。当前接口没有 limit/cursor 参数。

### 14.4 `skill graph show`

```text
mindmemos skill graph show GRAPH_REF
```

必填家族或快照引用。先 resolve，然后向 `POST /v1/skills/graphs/snapshot` 发送以下请求。未发布且没有版本时应明确报错。

```json
{"graph_id":"skg-graph-id","graph_version_id":"skgv-snapshot-id"}
```

### 14.5 `skill graph publish`

```text
mindmemos skill graph publish (--alias ALIAS | --graph-id ID) \
  (--skill SKILL_REF | --all-skills | --empty) [--message TEXT]
```

接口：`POST /v1/skills/algos/organize`。同步返回发布的 Graph 与版本 ID，不返回 task_id。

| 参数 | 类型 | 必填 | CLI 默认值 | 当前请求体字段 / 行为 |
| --- | --- | --- | --- | --- |
| `--alias` | 字符串 | 新建时必填 | 无 | 新图别名，发送 `alias` |
| `--graph-id` | 现有 Graph 家族 ID | 更新时必填 | 无 | `graph_id`；与 `--alias` 互斥，不接受快照 ID |
| `--skill` | 引用，可重复 | 成员输入三选一 | 无 | 组成 `skill_refs`；CLI 先解析为精确版本，API 也支持直接解析引用 |
| `--all-skills` | 布尔开关 | 成员输入三选一 | `False` | 使用无 limit 的 list 请求列出全部家族，再解析为精确版本；不接受 cursor 参数 |
| `--empty` | 布尔开关 | 成员输入三选一 | `False` | 显式发送 `skill_refs=[]` |
| `--message` | 字符串 | 否 | `None` | **`commit_message`** |

当前公开接口不接受旧 `skill_version_ids`、`message` 或 `scene_records`。旧 `--scenes-file` 暂不列为可对接参数；未来若恢复该功能，需要先增加明确的接口契约。每个 Skill 家族只能选择一个版本。

```json
{"alias":"graph-alias","skill_refs":["skv-version-1","skv-version-2"],"commit_message":"Publish graph"}
```

`--graph-id` 已对接更新已有图；新建图要求显式别名。当前 Graph 管理流程仍要求成员有标准 frontmatter；非标准注册不代表该流程已支持非标准内容。

### 14.6 `skill graph recommend`

```text
mindmemos skill graph recommend TASK --by GRAPH_REF [--top-k K]
```

| 参数 | 必填 | CLI 默认值 | 请求体字段 / 行为 |
| --- | --- | --- | --- |
| `TASK` | 是 | 无 | `query` |
| `--by` | 是 | 无 | Graph 引用；先解析出精确快照 |
| `--top-k` | 否 | `5` | 正整数，发送 `top_k` |

HTTP 与 SDK `RecommendRequest` 均支持以下精确快照请求，CLI 显式发送解析出的快照 ID：

```json
{"graph_ref":"skg-graph-id","graph_version_id":"skgv-snapshot-id","query":"任务描述","top_k":5}
```

当前 SDK 可将精确 `skgv-…` 版本引用直接作为 `graph_ref`，由服务端解析为选定快照；这不等于 SDK 已支持上述双字段形式。接口当前要求 `skill:write`，不能将其描述为只读权限接口。

## 15. JSON、响应与退出码

当前 SDK 的返回值是响应信封中的 `data` 字典，不保留完整信封。`show` 的字段为 `content/resources`，不是旧 `bundle`；Skill resolve 的当前状态字段为 `is_active`，Graph resolve 为 `is_current`。

当前传输层成功码只有 `ok` 和 `queued`。独立 Skill 服务的 evolve/train 返回 `accepted`，同步和异步客户端都会将 HTTP 202 误判为 `ApiError`。任务可能已受理，不应把该异常解释为任务没有提交。

当前 Skill CLI 的输出与退出码：

| 行为 | 约定 |
| --- | --- |
| 默认输出 | 成功结果在 stdout 输出 JSON；普通 API 操作输出 data，不需要格式开关 |
| `register-structured --json` | 仅接收 StructuredSkillInfo 输入，不控制输出格式 |
| `task progress --follow` | JSON Lines，每行一个进度事件；单次读取输出 JSON 数组 |
| 交互确认 | 提示写入 stderr，避免混入 stdout 的结果 JSON |
| 错误输出 | 在 stderr 输出 JSON，包含错误码、说明和可用的 request_id，不输出凭据 |
| 退出码 `0` | 成功、无结果或任务成功受理 |
| 退出码 `1` | 配置、网络、服务错误或用户取消删除 |
| 退出码 `2` | 参数/JSON 错误、缺少子命令或输入冲突 |

## 16. 当前实现与验证范围

| 项目 | 当前实现 | 验证或边界 |
| --- | --- | --- |
| CLI 入口 | 25 个 Skill 命令已接入 SDK CLI | 命令解析、SDK 请求和 JSON 输出已覆盖 |
| 标准注册 | 自动收集资源，先 include 后 exclude，编码 Base64 | 验证排除根目录 .env、保留原始换行及二进制资源 |
| 结构化注册 | 支持标准与非标准 StructuredSkillInfo | 校验内容和资源，不套用目录注册默认值 |
| 引用与版本 | 家族引用选生效版本，版本引用选给定版本 | show、export、commit、diff、train、evolve、轨迹列表复用解析流程 |
| activate | --version 与 --version-id 互斥必选 | 标签通过 offset 历史查询匹配，校验版本归属 |
| 分页 | 所有 CLI 均不接受 cursor | history 使用 offset；list/轨迹列表只请求第一页 |
| 导出 | 暂存写入、overwrite、替换失败回滚 | 资源路径和符号链接校验，禁止目录穿越 |
| 异步受理 | 同步/异步 transport 均接受 accepted | evolve/train 受理后返回任务 JSON，不等待算法完成 |
| 任务进度 | SDK 保留数组和空数组；CLI 支持 follow | 按事件序号续读，终态后补读最终事件 |
| 图管理 | 新建/更新图、快照读取和精确快照推荐；修复重复发布历史集合未切换生效指针 | 实际 FastAPI 联调使用标准 Skill；管理流程仍要求标准 frontmatter |
| SDK 推荐 | 新增可选 graph_version_id | CLI 显式固定解析出的快照 |
| 配置 | 独立 Skill URL/key，复用现有配置 | Skill 调用不要求先具备 Memory key；服务端负责认证 |
| 发布与部署 | 本地工作区实现 | 未发布 PyPI，未部署或验证线上服务 |

本次 287 项测试通过，包括 SDK 全量回归、Skill HTTP 契约、任务管理、图管理/软删除，以及新增 CLI 跨模块联调。验证使用隔离的本地数据库、实际 FastAPI 应用和离线 HTTP 响应；图联调使用固定 embedding，训练/演进联调替换后台执行方法，仅验证 HTTP 契约与任务受理。Ruff 检查与格式验证通过；SDK 源码包和 wheel 构建成功，并验证从 wheel 加载 CLI。存在一条现有 Starlette/httpx 弃用警告。真实模型执行依赖服务配置，本次不启动在线模型实验。

## 17. 维护依据

| 源码 / 文档 | 核对内容 |
| --- | --- |
| [SDK CLI 当前命令](sdk_cli_current.md) | 已实现入口、配置命令与文档格式 |
| [SDK 包定义](../../src/mindmemos_sdk/pyproject.toml) | 版本与命令入口 |
| [SDK CLI](../../src/mindmemos_sdk/mindmemos_sdk/cli.py) | 根命令接入与参数错误 JSON |
| [Skill CLI](../../src/mindmemos_sdk/mindmemos_sdk/skills/cli.py) | 25 个命令、引用解析和组合调用 |
| [Skill CLI 文件处理](../../src/mindmemos_sdk/mindmemos_sdk/skills/cli_files.py) | 本地资源收集与导出 |
| [根客户端](../../src/mindmemos_sdk/mindmemos_sdk/client.py) | 独立 Skill transport、认证与重试 |
| [配置模型](../../src/mindmemos_sdk/mindmemos_sdk/config/models.py) | 连接解析与默认值 |
| [同步 Skill 客户端](../../src/mindmemos_sdk/mindmemos_sdk/skills/client.py) | 当前公开方法 |
| [异步 Skill 客户端](../../src/mindmemos_sdk/mindmemos_sdk/skills/async_client.py) | 异步对应方法 |
| [Skill 请求核心](../../src/mindmemos_sdk/mindmemos_sdk/skills/core.py) | 路由与 body/query 分离 |
| [Skill SDK 模型](../../src/mindmemos_sdk/mindmemos_sdk/skills/typing.py) | 字段、默认值及缺失字段 |
| [SDK 内容处理](../../src/mindmemos_sdk/mindmemos_sdk/skills/content.py) | Base64、资源路径与哈希 |
| [HTTP 传输](../../src/mindmemos_sdk/mindmemos_sdk/transport.py) | 响应信封与成功码 |
| [独立 Skill FastAPI](../../src/mindmemos_skill/mindmemos_skill/api/entries/fastapi.py) | HTTP 模型、校验、权限、响应 |
| [Skill Service](../../src/mindmemos_skill/mindmemos_skill/api/service.py) | 实际委托、任务执行与配置边界 |
| [中心仓 Pipeline](../../src/mindmemos_skill/mindmemos_skill/pipelines/builtin_pipelines/repository/repository.py) | 导出、资源继承、版本与软删除 |
| [Graph Pipeline](../../src/mindmemos_skill/mindmemos_skill/pipelines/builtin_pipelines/manage/graph_manage.py) | 引用解析和同步发布 |
| [版本标签](../../src/mindmemos_skill/mindmemos_skill/persistence/db_services/skills/versions.py) | 省略标签时初始版本为 v1.0.0 |
| [旧云 Skill 路由](../../src/mindmemos/mindmemos/api/skill_routes.py) | 与独立 API 的差异 |

后续变更需同步更新本文件并核对服务配置。本地实现、服务能力、PyPI 发布和线上部署状态分别记录。
