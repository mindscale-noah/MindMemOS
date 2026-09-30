<h1>
  <img src="https://raw.githubusercontent.com/mindscale-noah/MindMemOS/main/assets/mindmemos-logo-small.png" alt="MindMemOS logo" width="40" height="40" align="absmiddle" style="vertical-align: middle;" />
  mindmemos-sdk
</h1>

![MindMemOS Memory For AI Agents](https://raw.githubusercontent.com/mindscale-noah/MindMemOS/main/assets/mindmemos-hero.png)

<p align="center">
  <a href="https://github.com/mindscale-noah/MindMemOS">
    <img src="https://img.shields.io/badge/GitHub-MindMemOS-181717?logo=github&logoColor=white" alt="MindMemOS GitHub">
  </a>
  <a href="https://mindmemos.cn">
    <img src="https://img.shields.io/badge/Website-mindmemos.cn-0A66C2?logo=googlechrome&logoColor=white" alt="MindMemOS Website">
  </a>
  <a href="https://mindmemos.cn/#/api-docs">
    <img src="https://img.shields.io/badge/FastAPI-Docs-009688?logo=fastapi&logoColor=white" alt="MindMemOS FastAPI Docs">
  </a>
  <a href="https://pypi.org/project/mindmemos-sdk/">
    <img src="https://img.shields.io/pypi/v/mindmemos-sdk?color=%2334D058&label=pypi%20sdk" alt="mindmemos-sdk PyPI version">
  </a>
  <a href="https://pypi.org/project/mindmemos-sdk/">
    <img src="https://img.shields.io/pypi/dm/mindmemos-sdk?label=pypi%20downloads" alt="mindmemos-sdk PyPI downloads">
  </a>
</p>

Python SDK and CLI for MindMemOS, a long-term memory system for AI agents and applications.

## Install

```bash
pip install mindmemos-sdk
```

The package also installs the `mindmemos` command.

## Configure

```bash
mindmemos auth
```

默认将本地状态保存在 `~/.mindmemos`。在启动 CLI 或创建 SDK 客户端前设置
`MINDMEMOS_HOME`，即可为不同项目使用独立的配置、凭据、Skill 注册信息、历史记录、
待上传记录，以及默认缓存和备份目录：

```bash
# Run in project A's terminal.
export MINDMEMOS_HOME="/path/to/project-a/.mindmemos"
mindmemos auth

# Run in project B's terminal.
export MINDMEMOS_HOME="/path/to/project-b/.mindmemos"
mindmemos auth
```

目录选择优先级：显式 `ConfigManager(config_dir=...)` > 非空 `MINDMEMOS_HOME` >
非空 `MINDMEMOS_CONFIG_DIR`（兼容旧配置）> `~/.mindmemos`。支持 `~` 展开。
默认缓存和备份位于所选目录的 `skills/cache` 和 `skills/backups` 下。
已有 `settings.json` 中显式设置的存储路径保持有效；复制旧配置时，如需完全隔离，
请删除 `storage` 中的旧路径以使用新默认值，或将其改为项目独立路径。
切换环境变量后请重新创建客户端；已有配置管理器保持创建时选定的目录。
该变量隔离本地 SDK 状态，远端数据的项目归属仍由服务端和 API 凭据决定。

You can also pass `base_url`, `api_key`, and `user_id` directly when creating a client. The default `user_id`
is used by operations that require a user, such as `memory.add`. Search is different: `memory.search` only uses
the `user_id` passed to that call. Omit it for a project-wide search, or pass it explicitly for a user-scoped search.

## Python SDK

```python
from mindmemos_sdk import DialogueMessage, MindMemOSClient

with MindMemOSClient(user_id="alice", app_id="my-agent") as client:
    client.memory.add(
        messages=[
            DialogueMessage(role="user", content="I prefer iced Americano."),
        ],
    )

    result = client.memory.search(
        "What coffee does the user prefer?", top_k=5, user_id="alice", token_budget=2000
    )
    for memory in result.memories:
        print(memory.memory)
```

## CLI

```bash
mindmemos memory add --content "I prefer iced Americano" --user-id alice
mindmemos memory search "coffee preference" --top-k 5 --user-id alice --token-budget 2000
```
