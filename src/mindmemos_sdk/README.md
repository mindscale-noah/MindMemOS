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
与默认用户。官方 Skill 使用独立入口，可用已获授权的项目 Key，也可通过
`--skill-api-key` 单独配置。自部署时，Skill 地址留空可只配置 Memory。已有独立 Skill
地址时，回车保留该地址，`--no-skill` 清除该连接。独立自部署地址需要自己的 Key。
`config show` 显示部署类型及两个服务的生效地址，默认遮住 Key。

也可以先指定部署类型和地址，再交互输入 Key，避免把 Key 放进终端历史：

```bash
mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:8000 \
  --skill-base-url http://127.0.0.1:8001/v1/skills
mindmemos config show
```

新版将连接设置保存到 `settings.v2.json`。官方模式省略两个 `base_url`。
Memory 仍通过 `https://mindmemos.cn/v1/memory/*` 使用原项目 Key。
新版 Skill 的默认 API 根地址是 `https://mindmemos.cn/v2/skills`，由网关转发到独立服务。
公开入口保持稳定时，网关内部路由的变化无需修改本地配置。

- `memory_service.base_url`（`--memory-base-url`）填 Memory 服务地址。SDK 拼接 `/v1/memory/...`。
- `skill_service.base_url`（`--skill-base-url`）填完整的 Skill API 根地址。直连独立服务时以 `/v1/skills` 结尾。自建网关可以填写其公开的版本化 Skill 根路径，例如 `https://gateway.example.com/v2/skills`。SDK 只追加操作名。

已有直连配置若只填了服务地址，SDK 会在解析连接时补上 `/v1/skills`。自部署配置示例如下。

```json
{
  "version": 2,
  "service_type": "custom",
  "memory_service": {
    "base_url": "https://memory.example.com",
    "auth": {"api_key": "memory-key"}
  },
  "skill_service": {
    "base_url": "https://skill.example.com/v1/skills",
    "auth": {"api_key": "skill-key"}
  }
}
```

如果新版配置尚未保存，SDK 会读取旧 `settings.json` 并在内存中解析为 v2。旧 CLI 继续读取原文件。`MindMemOSClient.memory`
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
切换默认目录后也需要重新启动已有进程。原目录中的连接配置会留在原处。
该变量只选择本地配置，远端数据的项目归属由服务端和 API 凭据决定。

You can also pass `base_url`, `api_key`, and `user_id` directly when creating a client.
The constructor's `base_url` is the Memory service address before `/v1/memory`.
Set a custom Skill root separately via `skill_service.base_url` in the SDK configuration.
The default `user_id`
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

Skill 接口按配置的 API 根地址发送请求，官方走 `/v2/skills/*`，直连独立服务走 `/v1/skills/*`。请求类型位于
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

### Skill 评分

同步和异步 Skill 客户端提供相同的评分接口：

```python
from mindmemos_sdk.skills import (
    EvaluationQueryRequest,
    RecommendRequest,
    SkillAnalyzeRequest,
    TrajectoryAnalyzeRequest,
)

# client 是已配置的 MindMemOSClient。
assessment = client.skills.analyze_skill(SkillAnalyzeRequest(skill_version_id="skv-..."))
trajectory = client.skills.analyze_trajectory(TrajectoryAnalyzeRequest(trajectory_id="trajectory-..."))
stored = client.skills.query_evaluations(EvaluationQueryRequest(skill_version_ids=["skv-..."]))
recommended = client.skills.recommend(
    RecommendRequest(graph_ref="my-collection", query="整理 CSV 数据", include_evaluation=True)
)
```

`analyze_skill` 和 `analyze_trajectory` 会复用已完成的评估；`force_refresh=True` 强制重新计算。
轨迹评估可传 `attribute` 覆盖失败归因开关，或传有限数值 `verifier_score` 覆盖已存评分。
`query_evaluations` 只读已存结果，不触发模型调用；可按 `judge_types` 筛选，`limit` 为 1–1000。
返回值沿用 SDK 的 `data` 字典；评分记录可能为 `completed` 或 `failed`，调用方应检查 `status`。

推荐的 `include_evaluation=True` 只附带已有评估，不重新评分、不改变 Graph 排序。
Graph 相关性分数与质量评估分别返回，不能当成同一种分数。
服务端必须配置对应分析器；未启用时返回 API 错误，SDK 不伪造评分或隐式降级。

### Offline 演化

```python
from mindmemos_sdk.skills import EvolveRequest, TaskShowRequest

accepted = client.skills.evolve(
    EvolveRequest(
        pipeline="offline",
        skill_version_ids=["skv-a", "skv-b"],
        editable_skills=["sk-a"],
        allow_create_skills=False,
        auto_accept=False,
    )
)
task = client.skills.show_task(TaskShowRequest(task_id=accepted["task_id"]))
```

`pipeline` 是服务端配置并允许调用的流水线名称。单版本也可使用 `skill_version_id`，
它与 `skill_version_ids` 互斥；集合中每个家族最多选择一个版本。
`editable_skills` 使用输入家族 ID：省略允许修改全部输入，空列表保持全部输入不变。
`allow_create_skills=False` 禁止创建独立的新 Skill。
SDK 的 `auto_accept` 默认 `True`，上述示例显式禁用；CLI 默认禁用。
禁用时仍会保存候选版本，但不会切换已有家族的 active 版本；允许新建时仍会注册新家族。

提交返回后台任务快照。查询到 `status == "succeeded"` 后，从 `result.members` 读取
`updated`、`created`、`unchanged` 成员和对应 `version_id`。失败或取消时检查任务错误。
演化不会自动更新 Graph；采纳候选后仍需发布相应图快照。

异步客户端使用同一组请求类型。直接构造 `AsyncHttpTransport` 时，可以传入完整的 Skill API 根地址，并在 `AsyncSkillClient` 中设置 `path_prefix=""`。已有代码如果传入服务地址并省略 `path_prefix`，客户端仍会追加 `/v1/skills`。

```python
import asyncio

from mindmemos_sdk.skills import AsyncSkillClient, ListRequest
from mindmemos_sdk.transport import AsyncHttpTransport


async def main():
    transport = AsyncHttpTransport(base_url="http://127.0.0.1:8001/v1/skills", api_key="local-key", max_retries=0)
    try:
        page = await AsyncSkillClient(transport, path_prefix="").list(ListRequest(limit=20))
        print(page["items"])
    finally:
        await transport.aclose()


asyncio.run(main())
```

Skill 客户端直接返回服务响应中的 `data` 字典。服务端未启用的算法接口会返回相应 API 错误；
`MindMemOSClient.skills` 默认不自动重试 Skill 请求，避免网络中断时重复提交写操作。

### 统一上报执行轨迹

```python
from mindmemos_sdk import MindMemOSClient, TrajectoryEvent, TrajectoryMessage

event = TrajectoryEvent(
    event_id="harness:session-1:turn-7",
    source="my-harness",
    user_id="alice",
    agent_id="assistant",
    session_id="session-1",
    skill_version_ids=["skv-a", "skv-b"],
    messages=[
        TrajectoryMessage(role="user", content="Summarize the report"),
        TrajectoryMessage(role="assistant", content="Summary complete"),
    ],
)
with MindMemOSClient() as client:
    receipts = client.report_trajectory(event)
    print(receipts.model_dump_json(exclude_none=True))
```

CLI 可用 `mindmemos trajectory report --file event.json --json` 提交同一结构，`--file -` 从标准输入读取。Memory 和 Skill 各有一份回执。Memory 的 `queued` 表示已接收，Skill 的 `stored` 返回一条关联所有精确版本的轨迹。没有已知版本时，Skill 仍接收该轮轨迹。

同一项目内，Skill 用相同 `event_id` 重放同一份输入时返回原轨迹 ID。用 `--skill-only` 或 `--memory-only` 仅投递指定服务。Memory 写入结果为 `unknown` 时，先核实服务端是否接收，再决定是否补报。任一路失败或结果未知时，CLI 仍输出已有回执并以状态码 1 退出。

## CLI

```bash
mindmemos memory add --content "I prefer iced Americano" --user-id alice
mindmemos memory search "coffee preference" --top-k 5 --user-id alice --token-budget 2000
```


Skill 评分与查询：

```bash
mindmemos skill analyze skill my-skill
mindmemos skill analyze skill my-skill --force-refresh
mindmemos skill analyze trajectory TRAJECTORY_ID --no-attribute --verifier-score 0
mindmemos skill evaluations query my-skill another-skill --judge-type static_skill --limit 20
mindmemos skill graph recommend "整理 CSV 数据" --by my-collection --include-evaluation
```

Skill 引用可使用别名、家族 ID 或精确版本 ID；CLI 先解析，再对选定版本评分或查询。
`--request-id` 可为两个 analyze 命令提供追踪身份。
不传 `--attribute` / `--no-attribute` 时沿用服务端归因设置。

Offline 演化支持单个或多个 Skill：

```bash
mindmemos skill evolve my-skill --pipeline offline
mindmemos skill evolve my-skill another-skill --pipeline offline --editable-skill my-skill --no-allow-create-skills
mindmemos skill evolve my-skill another-skill --pipeline offline --read-only --allow-create-skills
mindmemos skill task show TASK_ID
mindmemos skill task progress TASK_ID --follow
```

`--editable-skill` 可重复传入，CLI 将引用解析为输入家族 ID；它与 `--read-only` 互斥。
`--read-only` 保持输入不变，仍可配合 `--allow-create-skills` 从证据创建独立 Skill。
`--auto-accept` 才会激活已有家族的新版本；省略时仅保存候选。
不传 `--pipeline` 时保留原有行为，使用服务端默认流水线；要指定 offline，应显式传入。
命令立即返回任务快照，不等待模型演化完成。所有 Skill 命令成功时输出 JSON；
参数或服务错误输出到 stderr，并返回非零退出码。
