# MindMemOS 与网关特性更新说明

日期：2026-11-12。

本次更新把新版 Skill 管理、版本、轨迹、评分、图谱、训练和演进放到独立 Skill 服务，并在网关接入项目权限、模型池和 Skill 空间。旧 Memory 与旧 Skill API 保留。历史 Skill 数据通过网关侧工具迁入，新旧两端通过单向增量转发衔接。

本文描述源码状态及本地验证结果。代码推送不代表服务已部署，也不代表 SDK 或插件已发布到软件包仓库。

## 对照版本

| 项目 | 本次分支 | 对照基线 |
| --- | --- | --- |
| [MindMemOS](https://github.com/mindscale-noah/MindMemOS) | `feat/mindmemos_skill` | 官方 `main`，`29c92af18c5ad0b670aca01e241db2db9533494c` |
| [网关](https://gitee.com/sxkai2/mindmemos_platform) | `cyq/skill-platform-update-20261112` | `memops-v2`，`0b51a5bcfb562a5db4acd898f8981b3946af14bd` |

MindMemOS 的业务代码提交为 `b3e2cdd`，随后通过 `41f936a` 合并官方同名分支的已有历史；合并没有改变业务文件。网关业务代码提交为 `3b9fb5d`。后续文档提交以这两个业务版本为说明对象。

这里的 v2 指实际远端分支 `memops-v2`。本文比较两个提交的完整文件状态，不把所有提交记录或文件移动都当作新功能。

## MindMemOS 相对 main 的变化

| 范围 | 官方 main | 本次分支 | 对用户的影响 |
| --- | --- | --- | --- |
| Skill 服务 | Skill 依附旧 Memory 服务，使用旧注册、同步与演进协议 | 新增 `src/mindmemos_skill`，提供独立 Python Service、HTTP API 和持久化 | 新 Skill 可独立部署和扩容，使用单独的连接与权限 |
| Skill 身份与版本 | 旧 `cloud_skill_id`、内容哈希及 published head 协议 | Skill 家族、精确版本、不可变版本标签、active 版本、父子谱系和资源快照 | 可按确切版本查看、比较、导出和激活；版本历史与当前激活版本分开 |
| 轨迹 | Memory Add Record 附带旧 Skill 引用和分数 | 独立轨迹、版本关联、执行尝试及评分记录；SDK 提供统一上报入口 | 一次采集可分别投递到 Memory 和 Skill，并读取各自回执 |
| Skill 图谱 | main 不含独立 Skill 图谱服务 | 图谱构建、成员增删、快照、动态 Skill 预览和任务推荐 | 可管理 Skill 组合，并追溯组合使用的具体版本 |
| 评分 | 旧 Skill 演进链路中的评测 | 静态 Skill Judge、轨迹 Judge、历史评分查询和各维度原因 | 可查看安全性、完整性、可执行性、可维护性及评分依据 |
| 训练与演进 | 旧 Skill 演进接口 | TreeSkill、Skill-Forest、基于已保存轨迹的 offline evolution，以及任务进度查询 | 新训练与演进按独立服务协议调用；旧 simple evolution 被替换 |
| Agent 与环境 | main 不含这套独立 Skill 执行框架 | CLI Agent、OpenAI Chat Agent、任务环境、数据划分及训练/评测入口 | 支持 SpreadsheetBench、AgentSkillOS 和其他已接入环境的可复用执行流程 |
| 项目任务并发 | 无新版 Skill 的统一项目准入 | Graph 构建、Judge、Evolve 共享项目任务上限，默认每项目 1 个活动任务 | 冲突返回任务信息，限制可通过 `service.max_active_tasks_per_project` 调整 |
| 本机工作台 | main 不含 `src/mindmemos_ui` | 新增基于 SDK 的 Memory/Skill 工作台、版本、图谱、评测与连接设置 | 本机可同时连接两个服务；与网关网站中的 Skill 空间是两个界面 |
| 开发与部署 | 主要围绕 Memory 的 Makefile 和部署方式 | `scripts/dev/` 统一组织 Memory、Skill、UI 和基础设施 | `make dev`、`make api` 可通过 `SERVICE`、`FEATURE`、`UI` 选择启动范围 |

主要实现位置为 `src/mindmemos_skill/mindmemos_skill/{api,persistence,pipelines,algos,envs}`、`src/mindmemos_ui/` 和 `scripts/dev/`。

### Memory 中保留的能力

最新官方 `main` 已包含以下能力。本次分支保留了这些实现，不把它们列为本次独有特性：

- Vanilla 多模态 Memory：`off`、`basic`、`enhanced`，以及 enhanced 下的 agentic 检索。
- 图片写入、图片检索、S3/MinIO 对象存储、图片下载和签名 URL。
- MemEye、Mem-Gallery 评测适配。
- 控制台内部 Memory 清单、生命周期、来源追踪、关系图和 insights 缓存。

其中图片加载器、对象存储和内部 Memory 路由与本次对照的 `main` 一致。多模态相关文件在本次本地提交中数量较多，是因为它们此前尚未提交到当前分支；这不表示它们都晚于官方 `main`。

Memory 的实际增量集中在旧 Skill 版本、删除及绑定轨迹的转发，另有 `/v1/auth/identity`、部分类型与内部边界整理。公开 Memory 读写入口仍使用 `/v1/memory/*`。

### SDK 与插件的变化

以下版本是两份源码声明的版本，不代表本次执行了公开发包。

| 组件 | main 源码 | 本次源码 |
| --- | --- | --- |
| `mindmemos-sdk` | `0.1.5` | `1.0.0` |
| OpenClaw 插件 | `0.1.5` | `0.2.0` |
| DeepSeek Harness 插件 | `0.1.3` | `0.2.0` |

新版 SDK 分别保存 Memory 和 Skill 连接。官方网关 Skill 根路径为 `/v2/skills`；直连独立服务时使用完整的 `/v1/skills` 根路径。自定义 Memory 地址不会自动成为独立 Skill 地址。

新版配置写入 `settings.v2.json`。旧 `settings.json` 可作为读取来源并继续保留，便于旧客户端使用。SDK 增加同步和异步 Skill 客户端、版本与图谱查询、评分、演进及轨迹上报接口。

新版不再沿用旧 SDK 的本地 Skill 注册、待上传队列与自动上传管理流程。`memory.add(skill_context=...)` 的显式旧引用兼容仍保留；旧 `score`、`task_id` 参数不再属于新版 `memory.add` 签名。执行轨迹应通过新的轨迹协议上报。

插件根据实际加载证据解析精确 Skill 版本。无法确认版本时跳过 Skill 轨迹投递，Memory 可独立写入。统一轨迹回执区分成功、失败、跳过与结果未知；出现超时不能直接认定上游没有写入。

新插件使用新版 CLI 的 `trajectory report`，因此需要配套新版 SDK。源码构建成功与完整宿主验收是两项不同检查，当前插件联调测试仍有失败，见验证记录。

## 网关相对 memops-v2 的变化

| 范围 | memops-v2 | 本次分支 | 使用条件 |
| --- | --- | --- | --- |
| 新旧 Skill 路由 | 旧 Skill 经 Memory 访问 | 旧 `/v1/skills/*` 继续到 Memory；新 `/v2/skills/*` 到独立 Skill | 配置 Skill 内网地址和专用签名密钥 |
| 新 Skill 权限 | 主要使用旧 `skills:write` | 新增 `skill:read`、`skill:write`，现有 Key 可编辑权限 | 旧权限不会自动等价为新版权限 |
| 历史 Key 补权 | 无新版补权工具 | `skill_scope_backfill` 默认预演，显式 `--apply` 后补齐缺失权限 | 保留 Key 值、项目、原权限、有效期及启停状态 |
| 历史 Skill 迁移 | 无独立服务迁入流程 | 网关侧 `skill_migration` 迁移已注册家族、版本与关联轨迹 | 使用旧库只读快照和已初始化的目标 Skill 数据库 |
| 增量转发 | 无新版 Skill 接收链路 | 内部版本、轨迹、删除入口验证 Memory 服务令牌后转发 | 令牌绑定项目与具体记录；公网 ingress 拦截旧 Skill 内部转发路由 |
| Skill 空间 | 导航中的 Skill 路由回到概览 | 提供 Skill 空间、Skill 图谱、Skill 评分、Skill 演进四个入口 | 使用登录用户当前选中项目；按项目切换数据 |
| 版本浏览 | 无独立 Skill 版本界面 | 查看正文、资源、历史、谱系、active 版本及导入导出 | 来自旧服务的记录带来源信息 |
| 评分交互 | 无新版 Skill 静态评分入口 | 可对选定版本开始或刷新评分，展示历史、总分、维度原因和发现 | 需要开启算法转发和可用模型池 |
| Skill 模型调用 | 模型池未对独立 Skill 开放 | 新增 Skill 私有 Chat、Embedding 和原生 Messages 通道 | 独立模型签名密钥、项目所有者模型归属及配额校验 |
| 图谱 Embedding | 无独立 Skill 向量空间约束 | 登记 `space_id`、模型、维度及端点，构建时固定空间快照 | 检索需找到兼容空间；不可用时返回明确冲突 |

网关公开新版 API 使用操作白名单。基础读写与算法转发分开控制。`MINDMEMOS_SKILL_GATEWAY_ENABLE_ALGORITHMS` 默认关闭；开启后可转发组织、评分、推荐与演进。训练仍由私有 Skill 服务提供。

网关登录态的 Skill 空间使用服务端转发接口，浏览器不持有服务签名密钥。评分前端会阻止同次重复点击，服务端按项目与版本阻止并发评分；完成后允许再次刷新。

模型访问使用 `MINDMEMOS_SKILL_MODEL_GATEWAY_SECRET`，区别于 Memory 转发密钥和 Skill API 转发密钥。Claude 可通过平台的 Anthropic Messages 兼容通道运行；平台记录普通与流式请求用量。

主要实现位置为网关的 `backend/src/mindmemos_platform/{main.py,skill_client.py,skill_model_auth.py,skill_messages.py,skill_migration,skill_scope_backfill.py}`、`frontend/src/SkillWorkspace*.tsx` 和 `docker/README.md`。

## 新旧服务如何协作

| 数据或操作 | 入口及流向 | 边界 |
| --- | --- | --- |
| 普通 Memory | 客户端 → 网关 `/v1/memory/*` → Memory | 普通记忆保留在 Memory |
| 旧 Skill 操作 | 旧客户端 → 网关 `/v1/skills/*` → Memory | 继续使用旧协议与旧数据 |
| 新 Skill 操作 | 新客户端 → 网关 `/v2/skills/*` → 独立 Skill | 使用新版本、图谱、轨迹和任务协议 |
| 历史迁移 | 旧 Qdrant 只读快照 → 网关迁移工具 → 独立 Skill 数据库 | 保存旧 ID 到新 ID 的映射；不迁移普通 Memory |
| 旧端新增版本 | 旧端保存成功 → 网关内部接口 → 独立 Skill | 新版本进入历史，不自动切换 active |
| 旧端新增或更新轨迹 | 已绑定 Skill 的 Add Record → 网关内部接口 → 独立 Skill | 后补绑定、处理进度与完成状态可更新同一条迁入记录 |
| 旧端删除家族 | 旧家族 ID → 网关内部接口 → 独立 Skill 软删除 | 不按同名删除；新端同家族追加版本一并隐藏 |
| 新端独有版本与状态 | 新客户端 → 独立 Skill | 不回流旧端；旧客户端无法读取这些新端独有记录 |

实时转发使用内存队列和有限重试。进程重启或持续故障可能丢失待投递事件。已有迁移与回填工具可以补齐版本和轨迹；失败的删除需要根据已知旧家族 ID 重投。本次没有实现持久消息队列，也没有重新设计转发幂等和并发语义。

历史 Key 补权按一次发布步骤执行。若用户之后主动撤销新版权限，不能无判断地反复执行补权，否则仍符合旧权限条件的 Key 会再次获得新版权限。

## 部署与升级顺序

1. 备份旧服务与目标 Skill 数据库，记录旧服务、网关和 Skill 的版本与配置。
2. 部署独立 Skill 接收入口及持久化，验证内部鉴权和项目隔离。
3. 部署网关新路由、内部转发和模型通道，保持原 Memory 与旧 Skill 入口可用。
4. 对固定的旧库只读快照运行迁移预演。核对项目、家族、版本、轨迹和失败清单后，再显式执行写入。
5. 对同一快照重跑并再次预演，确认报告为 `completed`，且 `would_migrate`、`failed`、`unprocessed` 均为零。
6. 预演历史 API Key 补权，再执行一次写入。核对原 Key 的新旧接口权限。
7. 部署旧 Memory 的增量转发触发逻辑并启用转发配置，核对快照之后的版本、轨迹与已知删除。
8. 验证后发布并配套升级 SDK、插件；需要模型操作时再开启算法转发。

迁移按 Skill 家族及其版本/映射、或单条轨迹及其版本关联提交事务。处理中断后可对同一快照重新运行。部分记录失败时，报告可能为 `completed_with_failures` 且退出码仍为 0，因此不能只检查进程退出码。

回退旧客户端和服务后，旧端原有数据仍可使用。新端独有版本与轨迹保留在独立 Skill 数据库，不会自动出现在旧端。不得清空或重建生产 Memory 数据库。

## 当前 main 尚需单独核对的差异

当前分支没有合并本次对照的完整 `main` 历史。多模态核心实现已经带入，但以下差异仍存在：

- `main` README 中的最新多模态 News、评测表与引用说明，在当前 README 中没有完整保留；部分内容转入 `docs/evaluation/`，不能把它们当作已经逐项同步。
- `main` 的 `make db`、`MINIO=0` 及旧清理命令，已被当前开发命令组织方式替代。升级部署脚本前需按当前 Makefile 适配。
- Vanilla 同步去重执行器的默认运行方式不同：`main` 使用独立线程池，当前分支使用带并发准入的默认异步线程调用。
- 测试从根目录迁到各模块目录，评测命令和配置入口也有调整。文件统计包含这些移动及格式整理，不等同于功能数量。

这次发布分支不是官方 `main` 的完整替代版本。合入主线时仍应逐项审阅上述行为和文档差异。

## 本地验证记录

本次验证未调用线上模型，也未执行生产迁移或生产权限写入。下表保留成功、失败和环境限制，不能解读为完整发布验收通过。

| 检查 | 结果 |
| --- | --- |
| MindMemOS Python 回归，排除 `llm` 标记 | 3642 通过、19 失败、108 跳过、8 未选中 |
| 补齐临时 S3 测试依赖后的重点复测 | 76 通过、2 跳过；覆盖对象存储、评测清库、旧 Skill 转发、版本识别和任务管理 |
| MindMemOS 仍失败的用例复测 | 10 项仍失败，见下表；未把多次测试数字相加为一个全套通过数 |
| 网关前端测试 | 167 通过；Node 使用 `--no-experimental-webstorage`，并限制为 2 个测试 worker |
| 网关生产前端构建 | 通过；仍有较大 JavaScript 分包提示 |
| 两款插件构建 | 通过 |
| 插件轨迹集成测试 | 4 项失败，Memory 召回注入结果缺失；构建后复测仍失败 |
| 网关后端回归 | 371 通过、6 失败；使用现有 Python 环境加临时依赖，尚未完成锁定依赖环境验收 |
| 网关本次新增能力与启动配置重点验证 | 186 通过、1 失败；覆盖鉴权、Key 权限、Skill 路由、迁移重跑/中断恢复、旧端转发与补权；环境变量模板断言未同步新增字段 |
| 配置及提交检查 | 两个服务的 `dev.yaml` 与 `dev.example.yaml` 字段结构一致；差异空白检查通过 |

MindMemOS 当前剩余失败分为四组：

| 数量 | 检查位置 | 当前现象 |
| --- | --- | --- |
| 4 | `src/mindmemos/tests/architecture/` | 数据库客户端、原始属性或 Cypher 使用位置不满足架构约束 |
| 3 | `src/mindmemos_skill/tests/test_agent_config.py` | Agent preset 名称、算法归属和示例配置预期不一致 |
| 1 | `src/mindmemos_skill/tests/test_native_async.py` | 检查源码时遇到 UTF-8 BOM，尚未得出异步边界检查通过的结论 |
| 2 | `tests/scripts/test_local_project.py` | 实际 Skill URL 已包含 `/v1/skills`，断言仍期待裸服务地址 |

网关后端首轮失败涉及启动脚本的运行目录/解释器、缺失的 pinned MindMemOS 源码目录、模型端点解析和容器内 Redis 地址 `session:6379`。网关锁定的 `litellm==1.93.0` 在当前安装尝试中进入本地编译并受构建环境限制；本地复测使用已有 `litellm==1.97.0`，不等同于锁定版本验收。

仓库发布测试清单当前包含 213 条，均仍标记为未执行；本次自动测试不能替代真实部署、真实宿主与故障恢复验收。

## 操作参考

MindMemOS 的详细入口：`docs/cli/instruction_ZH.md`、`docs/deploy/instruction_ZH.md`、`docs/deploy/skill-service-migration_ZH.md`、`docs/evaluation/skill_evaluation_ZH.md`。

网关的详细配置、迁移、补权和模型空间设置集中在 `docker/README.md`。其中默认预演与显式写入的边界，应以相应命令和报告字段为准。
