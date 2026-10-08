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
mindmemos config show
```

`auth` 会先询问使用官方服务还是 custom 部署。官方模式只需填写 Memory API Key
与默认用户，Skill 复用同一地址和 Key。custom 模式先询问 Memory 地址，再询问 Skill 地址；
Skill 地址直接回车即复用 Memory 地址和 Key，即使先前配置过独立地址也是如此。填写独立 Skill 地址时必须填写自己的 Key；
本机无鉴权服务也可以填写任意非空 Key。重新配置时，只有服务地址未变才可留空保留已有 Key。
`config show` 显示部署类型及两个服务的生效地址，默认遮住 Key。

也可以先指定部署类型和地址，再交互输入 Key，避免把 Key 放进终端历史：

```bash
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001
mindmemos config show
```

`settings.json` 使用 v2 结构。顶层 `service_type` 为 `official` 时，Memory 和 Skill
都使用 `https://mindmemos.cn`，两个服务配置都不需要填写 `base_url`，Skill 复用 Memory 的凭据。
自部署时设置为 `custom`，在 `memory_service.base_url` 中填写 Memory 地址；
`skill_service.base_url` 留空时复用 Memory，填写独立地址时还需填写独立的
`skill_service.auth.api_key`。`auth` 会按上述选择写入这些字段。例如：

```json
{
  "version": 2,
  "service_type": "custom",
  "memory_service": {
    "base_url": "https://memory.example.com",
    "auth": {"api_key": "memory-key"}
  },
  "skill_service": {
    "base_url": "https://skill.example.com",
    "auth": {"api_key": "skill-key"}
  }
}
```

首次读取 v1 `settings.json` 时，SDK 会升级连接配置到 v2。`MindMemOSClient.memory`
使用 `memory_service`，`MindMemOSClient.skills` 使用 `skill_service` 解析出的连接。
独立 Skill 地址在配置中必须填写自己的 API Key；本机服务未启用鉴权时，该值仍可填写任意非空字符串。

默认将本地状态保存在 `~/.mindmemos`。在启动 CLI 或创建 SDK 客户端前设置
`MINDMEMOS_HOME`，即可为不同项目使用独立的连接配置和凭据：

```bash
# Run in project A's terminal.
export MINDMEMOS_HOME="/path/to/project-a/.mindmemos"
mindmemos auth

# Run in project B's terminal.
export MINDMEMOS_HOME="/path/to/project-b/.mindmemos"
mindmemos auth
```

`mindmemos config home set "/absolute/path/to/.mindmemos"` 可为当前用户保存长期使用的默认目录。
这个命令不修改操作系统环境变量；需要同时运行多个应用分身时，仍可在各自的启动环境中设置 `MINDMEMOS_HOME`。
目录选择优先级：显式 `ConfigManager(config_dir=...)` > 非空 `MINDMEMOS_HOME` >
非空 `MINDMEMOS_CONFIG_DIR`（兼容旧配置）> 本机保存的默认目录 > `~/.mindmemos`。支持 `~` 展开。
切换环境变量后请重新创建客户端；已有配置管理器保持创建时选定的目录。
切换默认目录后也需要重新启动已有进程。命令不会移动原目录的 `settings.json`。
该变量只选择本地配置，远端数据的项目归属由服务端和 API 凭据决定。

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

Skill 接口直接调用独立服务的 `/v1/skills/*`，请求类型位于
`mindmemos_sdk.skills`。客户端不维护本地 Skill 注册表，也不会在 `memory.add`
时检测或上传本地 Skill：

```python
from mindmemos_sdk import MindMemOSClient
from mindmemos_sdk.skills import ListRequest, RegisterRequest, StructuredSkillInfo

with MindMemOSClient() as client:
    created = client.skills.register(
        RegisterRequest(
            alias="demo",
            skill=StructuredSkillInfo(
                content="---\nname: demo\ndescription: Demo\n---\nUse concise answers.\n",
                commit_version="v1.0.0",
                commit_message="Initial version",
            ),
        )
    )
    print(created["skill_id"])
    print(client.skills.list(ListRequest(limit=20))["items"])
```

Skill HTTP 路由提供同步 `SkillClient` 和异步 `AsyncSkillClient` 方法。
注册与提交都接收 `skill: StructuredSkillInfo`；其中 `content` 保留原文及换行，
`resources` 是相对路径到标准 Base64 字符串的映射，默认完整替换。
提交只有显式 `inherit_resources=True` 才继承父资源，此时不能传入非空资源。
`base_version_id` 指定父版本；省略时选择 active 版本。自动版本号使用 `skill.version_bump`，
手动版本号使用 `skill.commit_version`。

注册结果、提交结果和版本详情均返回 `content_hash`、`skill_hash`。
SDK 的 `content_hash(text)` 与 `skill_hash(text, resource_bytes)` 独立实现相同哈希协议；
不再提供 bundle JSON、旧 bundle 哈希或隐式换行归一化工具。
`activate(ActivateRequest(...))` 激活已有版本，可传 `expected_active_version_id` 检查过期状态。
家族响应使用 `active_skill_version_id`，解析响应使用 `is_active`。
列表、历史和轨迹的 `limit` 默认 `None` 且无上界；历史支持 offset 或 opaque cursor，不能混用。

图谱读取使用 `list_graphs()`、`graph_snapshots(GraphSnapshotsRequest(...))`、
`graph_snapshot(GraphSnapshotRequest(...))`；删除使用 `delete_graph(GraphResolveRequest(...))`。
项目身份由服务端从凭据解析，HTTP 请求不得自报项目或提供服务器文件路径。
训练、演化和图组织的算法执行能力取决于服务端配置及算法调用方迁移状态，不能仅凭 SDK 方法存在判断可用。
训练请求使用必填的 `algo`、`env`、`init_skill_version_id`、`work_dir`、`tasks`、`split`，
以及默认 `True` 的 `auto_accept`。`algo` 是服务中已配置的训练流水线名称；`env` 是本次训练的环境配置，例如
`{"class_name":"searchqa","config":{"agent":{"model":"YOUR_MODEL"}}}`，由服务端环境工厂校验和创建。
`train()` 返回后台任务；任务成功后的结果包含版本身份、`content`、Base64 `resources` 和双哈希。
旧训练字段 `pipeline`、`environment`、`base_version_id` 不再接受。`work_dir` 是受信任训练服务
的运行目录，不能用作注册或提交 Skill 的源路径。

演化请求使用 `EvolveRequest(skill_version_id="skv-...", auto_accept=False)`，调用
`client.skills.evolve(...)` 返回任务快照。`auto_accept` 默认 `True`；设为 `False` 时保存候选版本，
保留已有 active 版本。通过 `TaskShowRequest(task_id=...)` 查询任务，`status == "succeeded"` 后
从 `result` 读取版本身份、原文 `content`、Base64 `resources`、`content_hash` 和 `skill_hash`。

异步客户端使用同一组请求类型；可用 `AsyncHttpTransport` 单独连接：

```python
import asyncio

from mindmemos_sdk.skills import AsyncSkillClient, ListRequest
from mindmemos_sdk.transport import AsyncHttpTransport


async def main():
    transport = AsyncHttpTransport(base_url="http://127.0.0.1:8001", api_key="local-key", max_retries=0)
    try:
        page = await AsyncSkillClient(transport).list(ListRequest(limit=20))
        print(page["items"])
    finally:
        await transport.aclose()


asyncio.run(main())
```

Skill 客户端直接返回服务响应中的 `data` 字典。服务端未启用的算法接口会返回相应 API 错误；
`MindMemOSClient.skills` 默认不自动重试 Skill 请求，避免网络中断时重复提交写操作。

## CLI

```bash
mindmemos memory add --content "I prefer iced Americano" --user-id alice
mindmemos memory search "coffee preference" --top-k 5 --user-id alice --token-budget 2000
```
