# MindMemOS CLI 使用说明


<p align="center">
  <strong><a href="instruction.md">English</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="instruction_ZH.md">简体中文</a></strong>
</p>

## 1. 简介

`mindmemos` 是随 MindMemOS Python SDK（`mindmemos_sdk`）一起发布的命令行工具，用于在终端中直接操作记忆服务：写入与检索记忆、管理 SDK 注册的 Skill、检查本地配置与连通性。

它是 SDK 的薄封装：所有命令都读取本地配置文件（`~/.mindmemos/settings.json`），通过 HTTP 调用 `mindmemos` 服务，因此使用时无需手动拼接请求、也不需要在命令中重复填写服务地址。

## 2. 安装

```bash
pip install mindmemos-sdk
```

安装后确认命令可用：

```bash
mindmemos --help
```

命令通过 `project.scripts` 暴露为全局可用的 `mindmemos` 可执行文件。

## 3. 快速上手

### 3.1 配置认证

首次使用前，运行 `mindmemos auth` 配置服务地址、API key 与默认用户：

```bash
mindmemos auth
```

交互式依次输入三项配置：

| 配置项 | 本地自部署服务 | 官方云服务 |
| :--- | :--- | :--- |
| `Base URL` | `http://127.0.0.1:8000` | `https://mindmemos.cn` |
| `API key` | `config/mindmemos/api_keys.yaml` 中已启用的 key | 从 [官网](https://mindmemos.cn) 申请的 key |
| `User id` | 当前用户的稳定标识，例如 `u_123` | 当前用户的稳定标识，例如 `u_123` |

也可以一次性传入参数跳过交互：

```bash
mindmemos auth --base-url http://127.0.0.1:8000 --api-key dev-api-key-001 --user-id u_123
```

配置保存到 `~/.mindmemos/settings.json`。本地服务会根据 API key 自动确定 `project_id`，无需在命令中指定。

### 3.2 检查配置与连通性

```bash
# 查看当前配置（API key 默认打码）
mindmemos config show

# 完整显示 API key
mindmemos config show --show-secret

# 检查配置是否有效、服务是否连通
mindmemos doctor
```

### 3.3 写入一条记忆

```bash
mindmemos memory add --content "我喜欢喝冰美式。"
```

### 3.4 检索记忆

```bash
mindmemos memory search "用户喜欢喝什么咖啡？" --top-k 5
```

> 写入、检索的更多参数见下文「记忆命令」；CLI 仅做调用与结果展示，记忆的完整提取、打分、存储逻辑由服务端完成。

## 4. 命令总览

```
mindmemos
├── auth                  交互式配置 API key、用户与服务地址
├── config                查看 / 重置本地配置
│   ├── show
│   └── reset
├── memory                记忆相关操作
│   ├── add               写入一条对话消息作为记忆
│   ├── search            检索记忆
│   ├── get               列出 / 过滤当前项目下的记忆
│   ├── update            更新指定记忆内容
│   ├── delete            删除指定记忆
│   ├── feedback          提交显式 / 隐式反馈
│   └── dreaming          触发记忆演进管线
├── skill                 管理 SDK 注册的 Skill
│   ├── register          注册并上传本地 Skill
│   ├── list              列出已注册 Skill
│   ├── show              查看单个 Skill
│   ├── evolve            触发云端 Skill 演进
│   ├── push              上传本地改动为新版本
│   ├── pull              拉取版本元数据（不改动文件）
│   ├── update            更新一个或全部 Skill
│   ├── rollback          回滚到指定版本
│   ├── history           查看版本历史
│   ├── diff              查看版本差异
│   └── unregister        移除注册（可同时删除文件）
└── doctor                检查 SDK 配置与连通性
```

每个命令都支持 `--help` 查看完整参数说明：

```bash
mindmemos memory add --help
```

## 5. 记忆命令（`mindmemos memory`）

记忆命令使用 `mindmemos auth` 配置好的凭据，无需重复输入。

### 5.1 写入记忆 `add`

最常用的方式是传入单条消息内容：

```bash
mindmemos memory add --content "我喜欢喝冰美式。"
```

指定消息角色（默认 `user`）：

```bash
mindmemos memory add --content "记住这个偏好" --role system
```

多轮消息以 JSON 传入（此时 `--content` / `--role` 被忽略）。可内联或从文件读取：

```bash
# 内联 JSON
mindmemos memory add --messages-json \
  '[{"role":"user","content":"我喜欢喝冰美式。"},{"role":"assistant","content":"好的，记住了。"}]'

# 从文件读取
mindmemos memory add --messages-json-file ./messages.json
```

异步模式（立即返回 `request_id`，不等待提取完成）：

```bash
mindmemos memory add --content "我喜欢喝冰美式。" --async
```

其他可选参数：

| 参数 | 说明 |
| :--- | :--- |
| `--user-id` | 覆盖配置中的默认用户 |
| `--app-id` / `--agent-id` / `--session-id` | 上下文标识，供细分与过滤使用 |
| `--metadata-json` | 业务元数据（JSON 对象） |
| `--skill-context-json` | Skill 上下文数组，覆盖 SDK 自动检测 |
| `--json` | 以机器可读 JSON 输出完整结果 |

示例输出：

```text
Added 1 memory item(s):
- [did] m_8f3a: 我喜欢喝冰美式。
```

### 5.2 检索记忆 `search`

```bash
mindmemos memory search "用户喜欢喝什么咖啡？" --top-k 5
```

常用参数：

| 参数 | 说明 |
| :--- | :--- |
| `--top-k` | 返回结果条数，默认 `10` |
| `--search-strategy` | 检索策略，`fast`（默认）或 `agentic` |
| `--rerank` | 开启重排 |
| `--score-threshold` | 重排相关度阈值（0-1），需配合 `--rerank` |
| `--filter` | 过滤 DSL（JSON 对象字符串） |
| `--user-id` 等 | 覆盖请求上下文 |
| `--json` | 以 JSON 输出完整结果 |

### 5.3 列出与过滤 `get`

列出当前项目下的记忆，可选过滤：

```bash
# 列出最近 20 条
mindmemos memory get --top-k 20

# 按过滤 DSL 过滤
mindmemos memory get --filter '{"field":"value"}'
```

### 5.4 更新与删除

```bash
# 更新指定记忆的内容
mindmemos memory update <memory_id> --content "新内容"

# 删除指定记忆（-y 跳过二次确认）
mindmemos memory delete <memory_id> --yes
```

### 5.5 反馈 `feedback`

默认运行隐式反馈，由服务端分析近期写入并生成反馈；也可通过 `--text` 提交显式反馈：

```bash
# 隐式反馈
mindmemos memory feedback

# 显式反馈，需同时给出产生该反馈的消息
mindmemos memory feedback --text "这条记忆不正确" \
  --messages-json '[{"role":"user","content":"..."}]'
```

### 5.6 记忆演进 `dreaming`

触发记忆演进（dreaming）管线，选择同步或异步：

```bash
# 异步排队（默认）
mindmemos memory dreaming --async

# 同步等待完成
mindmemos memory dreaming --sync
```

## 6. Skill 命令（`mindmemos skill`）

`mindmemos skill` 管理已在 SDK 注册、可由云端演进（evolve）的本地 Skill。

注册一个本地 Skill（路径可为目录或 SKILL.md 文件）：

```bash
mindmemos skill register ./my-skill
# 指定别名，便于后续命令使用
mindmemos skill register ./my-skill --alias my-skill
```

常用操作：

```bash
# 列出 / 查看已注册 Skill
mindmemos skill list
mindmemos skill show <skill-id-or-alias>

# 触发云端演进（默认同步，--async 改为排队）
mindmemos skill evolve <skill-id-or-alias> --sync

# 推送本地改动为新版本；拉取版本元数据
mindmemos skill push <skill-id-or-alias>
mindmemos skill pull <skill-id-or-alias>

# 更新一个或全部 Skill（--all 更新全部，-y 跳过确认）
mindmemos skill update <skill-id-or-alias> --yes
mindmemos skill update --all --yes

# 撤销注册（--delete-files 同时删除本地文件）
mindmemos skill unregister <skill-id-or-alias> --delete-files --yes
```

版本管理：

```bash
# 查看版本历史
mindmemos skill history <skill-id-or-alias>

# 回滚到指定版本
mindmemos skill rollback <skill-id-or-alias> --to <version-id> --yes

# 查看两个版本间的差异
mindmemos skill diff <skill-id-or-alias> --from <version-id> --to <version-id>
```

在提交 `update` / `rollback` 等可能改动本地文件的操作前，命令会先展示变更计划并请求确认。

## 7. 配置管理（`mindmemos config`）

```bash
# 查看当前配置
mindmemos config show
mindmemos config show --show-secret   # 显示完整 API key

# 重置并删除本地配置
mindmemos config reset
mindmemos config reset --yes          # 跳过确认
```

配置文件位于 `~/.mindmemos/settings.json`，也可通过环境变量 `MINDMEMOS_CONFIG_DIR` 指定其他配置目录。

> API key 属于敏感信息，默认在 `config show` 中打码显示（`config show --show-secret` 才会完整展示）。

## 8. 故障排查

| 现象 | 可能原因与处理 |
| :--- | :--- |
| No SDK config at … Run `mindmemos auth` first. | 尚未配置认证，先运行 `mindmemos auth` |
| No api_key configured | API key 缺失，重新运行 `mindmemos auth` |
| transport: not ready … | `doctor` 检测到服务不可达。确认服务已启动，且 `base_url` 正确：本地自部署为 `http://127.0.0.1:8000`，云端为 `https://mindmemos.cn` |
| 服务返回认证错误 | API key 无效或未启用，核对云端的 key 或 `config/mindmemos/api_keys.yaml` 中的本地 key |
| 插件环境报 `ENOENT` | GUI 启动的进程不继承终端 PATH，`mindmemos` 需显式配置绝对路径或用 `uv run mindmemos` 包装，详见 [OpenClaw 插件集成](../../skills/mindmemos-cli/references/openclaw-plugin.md) |

可随时运行 `mindmemos doctor` 一键检查配置与连通性。

## 9. 相关文档

- [Python SDK 集成](../../skills/mindmemos-cli/references/python-sdk.md)
- [OpenClaw 插件集成](../../skills/mindmemos-cli/references/openclaw-plugin.md)
- [部署 & 配置说明](../deploy/instruction_ZH.md)
- [API 文档](https://mindmemos.cn/api-docs)
