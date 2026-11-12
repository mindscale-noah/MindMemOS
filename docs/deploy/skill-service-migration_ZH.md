# Skill 服务迁移说明

旧 Memory 服务继续提供 Memory 和旧 Skill API。新版 Skill 在独立服务中部署，使用自己的 Skill 仓库和轨迹库。历史数据从旧服务迁入独立服务。**新版 Skill、版本和轨迹不会同步回旧服务。**

## 插件和 SDK 版本对应关系

插件调用 Python SDK 安装的 `mindmemos` CLI。表里的 SDK 版本指插件实际调用的 CLI 所属版本。机器上有多个 Python 环境时，用插件的 `cli` 配置指定正确的可执行文件。

| 使用方式 | `mindmemos-sdk` | OpenClaw 插件 | DeepSeek Harness 插件 | Skill 入口 |
| --- | --- | --- | --- | --- |
| 继续使用旧服务 | `<= 0.1.4` | `@mindmemos/openclaw-plugin` `<= 0.1.3` | `@mindmemos/deepseek-harness-plugin` `<= 0.1.1` | 旧 `/v1/skills/*` |
| 切换到独立 Skill 服务 | `>= 1.0.0` | `@mindmemos/openclaw-plugin` `>= 0.2.0` | `@mindmemos/deepseek-harness-plugin` `>= 0.2.0` | 官方 `/v2/skills/*` |

`<=` 表示小于等于旧版上限，`>=` 表示大于等于新版下限。两款插件按宿主择一安装。表中范围用于区分两代客户端。本文的行为以边界版本为准，范围内其他版本以各自的发行说明为准。落在两代范围之间的版本不在本文配对范围内。

旧版上限对应的发行包已公开发布。截至 2026-11-10，新版下限对应的 SDK 和两款插件尚未公开发布。正式发布前，用源码构建进行联调。 [SDK 发行记录](https://pypi.org/project/mindmemos-sdk/#history) · [OpenClaw 插件发行记录](https://www.npmjs.com/package/@mindmemos/openclaw-plugin?activeTab=versions) · [DeepSeek Harness 插件发行记录](https://www.npmjs.com/package/@mindmemos/deepseek-harness-plugin?activeTab=versions)

### 混装的兼容范围

| 组合 | 当前行为 | 处理方法 |
| --- | --- | --- |
| OpenClaw 插件 `<= 0.1.3` 或 DeepSeek Harness 插件 `<= 0.1.1`，配当前新版 SDK 源码 | `memory add --skill-context-json` 已恢复，显式引用交给原 Memory 接口，未配置独立 Skill 服务也能上报 Memory。旧 SDK 的本地注册和自动上传流程没有恢复。 | 继续使用 Memory 可保留旧插件；要使用新版 Skill，升级插件并配置独立入口。正式发行包的完整宿主链路仍需验收。 |
| 任一插件 `>= 0.2.0`，配 SDK `<= 0.1.4` | 旧 CLI 没有 `trajectory report`，无法执行新版插件的轨迹上报。默认启用多模态召回时，旧 CLI 也会拒绝 `--multimodal`。 | 把插件实际调用的 CLI 换成 SDK `>= 1.0.0`。 |

当前没有自动配套升级。Memory 的显式旧引用参数兼容不代表旧 Skill 管理流程完全兼容。新版 `skill evolve` 接受 `--sync` 和 `--async`，但仍按新版引用解析和任务协议运行；默认保持异步。

## 用户迁移场景

| 场景 | 用户要做什么 | 结果 |
| --- | --- | --- |
| 继续使用旧插件或旧 SDK | 保持旧版配对和现有连接。 | Memory 和旧 Skill 仍走原入口，使用原项目和原 Key。 |
| 官方服务用户升级插件 | 等项目回填和 Key 权限就绪后，同时升级插件和 SDK。确认宿主实际调用的是新版 CLI。 | Memory 仍走 `/v1/memory/*`。新版 Skill 走 `/v2/skills/*`。符合新版权限要求的原 Key 可以继续用。 |
| 直接使用 Python SDK 或 CLI | 常规 Memory 及显式 `memory.add(skill_context=...)` 可继续使用。旧 Skill 管理流程需要适配新版引用和接口。 | 新版轨迹上报接收 `skill_version_ids`。已成功迁入的 Skill 无需重新注册。 |
| 自部署或使用自建网关 | 配置独立 Skill API 地址和凭据。直连独立服务用完整的 `/v1/skills` 根路径。通过网关时使用该网关公开的 Skill 根路径。 | 自定义 Memory 地址不会自动成为新版 Skill 入口。连接配置见 [CLI 使用指南](../cli/instruction_ZH.md#mindmemos-auth)。 |
| 升级后需要退回旧版 | 把插件和 SDK 一起退回旧版配对。保留旧客户端读取的 `settings.json`。 | 旧版仍能访问旧端数据。升级后只写进独立 Skill 服务的数据，要通过新版入口访问。 |

**不支持交叉使用 Skill 服务。** 仅部署旧 Skill 时，新版客户端不能使用旧接口代替独立 Skill 服务。仅部署独立 Skill 时，旧版客户端也不能通过它继续调用旧 Skill API。旧版和独立服务的内部路径都可能是 `/v1/skills`，但请求协议不同。

新版 CLI 可以读取旧 `settings.json` 的连接配置。新版配置保存到 `settings.v2.json`。旧文件仍供旧 CLI 使用。

使用新版插件时，如果本轮只加载了本地 `SKILL.md`，且插件无法识别云端版本 ID，日志会显示 `Skill skipped (no_skill_versions)`。这次对话仍可独立上报到 Memory。

### 原 Key 和新版 Skill 权限

旧 Skill 写权限 `skills:write` 和新版 `skill:write` 是两个权限。新版读取需要 `skill:read` 或 `skill:write`，写入需要 `skill:write`。只有旧权限的 Key 调用新版 Skill 会返回 `403`。

平台代码已支持编辑现有 Key 的权限。网关提供 `skill_scope_backfill` 批量补权工具，默认只预演，运营方核对后显式执行 `--apply`；服务启动时不会自动补权。项目所有者也可以在平台部署权限编辑功能后给原 Key 补齐所需权限。批量补权前应核对旧端实际生效的权限和用户已设置的权限，保留原 Key、项目、有效期和撤销状态。补权作为一次发布步骤执行；用户之后主动撤销新版权限时，不应无判断地重新执行补权。

## 我们会为你迁移哪些数据

官方服务切换时，我们会为原项目做一次历史迁移。你可以继续使用原项目，无需自行导出数据或重新注册已迁入的 Skill。

- **已注册的 Skill。** 我们会复制旧服务已保存的 Skill 正文和版本历史。迁入后使用新版 ID，我们会保留与旧 ID 的对应关系。
- **已关联 Skill 版本的历史轨迹。** 我们会把可以确认版本的轨迹复制到独立 Skill 服务，并关联到迁入后的版本。缺少可用版本关联的轨迹可能仍只在旧端可用。

你的普通 Memory 继续通过原入口访问。尚未注册的 Skill、等待绑定的轨迹和旧版演化摘要留在旧端。

启用转发后，旧端写接口按下表发送增量。请求继续使用原协议；转发在旧端成功保存后异步执行，不等待新端响应。

| 旧端 FastAPI 接口 | 转发策略 |
| --- | --- |
| `POST /v1/skills/register` | 首次注册、基于父版本注册和返回已有版本时，均发送该版本；不依赖轨迹或 evolve。 |
| `POST /v1/skills/evolve` | 发送生成的演进版本，沿用既有行为；新端缺少祖先时补发祖先。 |
| `POST /v1/skills/{cloud_skill_id}/delete` | 发送旧家族 ID，新端软删除对应迁入家族，保留版本、ID 映射和轨迹。该家族在新端追加过的版本也随家族隐藏；不会按同名删除其他家族。 |
| `POST /v1/memory/add`、`POST /v1/memory/add/stream` | 发送保存的 Skill 绑定轨迹。后补绑定、开始处理、完成、分段输出、失败和取消后，再发送同一条记录。失败或取消原因保存为来源元信息，不解释为 Agent 任务成败。 |
| `GET /v1/skills`、`POST /v1/skills/{id}/get`、版本列表与正文查询 | 继续查询旧端，不产生转发写入。 |
| `POST /v1/skills/sync` | 继续比较旧端 published head；不拉取新端独有版本。 |

新端收到后续旧版本时追加版本历史，不自动切换 active。旧端没有独立的公开发布或激活接口，因此本次不新增发布状态通道。新端独有的提交、激活、元信息更新及演进仍不回流旧端。

### 部署与已有数据

1. 先部署独立 Skill 服务的版本、轨迹和删除接收入口。
2. 再部署网关的三个内部转发入口，并保持原有服务签名及项目验证。
3. 最后部署旧 Memory 的接口触发逻辑，并启用已有转发配置。
4. 按原历史迁移流程核对已有版本和轨迹；新触发逻辑不重写旧端表、集合、字段或已有 ID。

新增内部删除入口为 `POST /internal/v1/legacy-skill/deletions`。它使用独立的 `skill:legacy-delete` 权限；Memory 到网关的令牌同时绑定项目和 `cloud_skill_id`。用户 Key 无权调用。

普通注册和 Add 仍在旧库保存成功后返回。旧端保存失败不发送新端写入，已有父子关系和历史数据沿用原格式。回退 Memory 或关闭转发配置后，旧 API 继续工作；已经同步到新端的数据保留。升级前已从旧库删除的家族没有删除事件，不能只靠重扫现存数据推断其删除事实。

本次仅补齐接口触发与接收策略，未调整幂等和并发处理，也未新增持久队列。实时转发仍受现有内存队列和有限重试约束，不能承诺重启或长时间故障后自动恢复。版本和轨迹可用运维回填补齐；失败的删除必须根据已知家族 ID 重新投递。新端软删除后的快照仍保留，但恢复可见性需要运维处理。

### 本地验证

旧端接口与轨迹回写测试：

```sh
uv run pytest src/mindmemos/tests/api/test_skill_routes.py \
  src/mindmemos/tests/pipelines/skill/test_version_store.py \
  src/mindmemos/tests/pipelines/memory_db/test_operation_records.py
```

新旧服务联调与删除鉴权测试：

```sh
uv run pytest tests/integration/test_legacy_skill_relay.py \
  src/mindmemos_skill/tests/test_fastapi.py -k 'legacy or relay'
```

跨模块测试位于根目录 `tests/integration/`，由根 pytest 配置自动发现。测试使用隔离的 Qdrant、SQLite 和进程内 HTTP，不修改云端数据。
