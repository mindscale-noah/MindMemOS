# Memory 与 Skill 均通过 FastAPI 访问时的配置设计

状态：2026-09-21 更新。连接表单、完整 YAML 模板、导入校验和独立 Skill 算法 HTTP 客户端已实现；远程 API Key 与项目映射、UI 全业务远程化及文件上传接口仍须由服务接入层补齐。

## 设计结论

采用 Memory 与 Skill 均通过 FastAPI 访问的方式。普通用户只配置服务连接；Skill 服务的部署者配置数据库、模型和资源；算法开发者维护算法默认参数与协议。

这解决的是客户端需要理解算法参数的问题，并不会消除服务端配置。即使客户端和 Skill 服务运行在同一台机器，也保持相同边界，客户端通过本机 HTTP 地址访问。Python 中的 `create_service()` 保留为服务端组装入口和直接调用入口，算法本身不依赖 FastAPI。

配置页保留 Memory、Skill 两张卡，每张卡管理一个连接配置。Skill 卡不再编辑数据库或训练算法 YAML。服务端配置通过部署文件及环境变量管理，不混进普通客户端页面。

本文仅讨论 Memory 的客户端连接，不涉及 MindMemOS FastAPI 服务端的算法配置。

## 表一：Memory 配置（客户端）

Memory 配置仅包含 Memory 服务连接字段。字段名参考当前 SDK；“建议”描述目标配置职责，不自动改变已有文件结构。

| 配置项 / 字段 | 谁负责 | 普通用户是否配置 | 建议默认与行为 | 保存位置 / 代码职责 |
| --- | --- | --- | --- | --- |
| 服务地址 `base_url` | 用户 / 部署分发者 | 必填，或由部署预填 | 指向实际 Memory FastAPI；现有 example.com 地址不能视为可用服务 | 客户端 `settings.json`；表单使用扁平字段，保存时写回 SDK 嵌套结构 |
| 访问凭据 `auth.api_key` | 用户 / 认证流程 | 服务要求鉴权时必填 | 无通用默认值；掩码显示，支持显式清除 | 客户端凭据；是 Memory 服务凭据，不是大模型 API Key |
| 默认用户 `defaults.user_id` | 用户 / 服务身份体系 | 按 Memory API 契约选填 | 无默认身份时保持为空；不得把客户端填写的 ID 当作鉴权依据 | SDK 默认请求上下文；服务端执行身份校验 |
| 超时 `network.timeout_seconds` | SDK 默认，用户高级覆盖 | 否，放高级设置 | 新配置默认 21600 秒（6 小时）；已有显式超时值原样读取 | SDK HTTP transport |
| Memory 服务端算法、数据库、模型参数 | Memory 服务端维护者 | 客户端不配置 | 不经配置页透传，也不在本文设计范围 | 远端服务内部 |

## 表二：Skill 配置（客户端）

以下为独立客户端连接模型。当前 `HOME/service.yaml` 是服务端组装配置，不能直接改名或改写为此模型。

| 配置项 / 建议字段 | 谁负责 | 普通用户是否配置 | 建议默认与行为 | 保存位置 / 代码职责 |
| --- | --- | --- | --- | --- |
| 服务地址 `base_url` | 用户 / 部署分发者 | 必填，或由部署预填 | 指向 Skill FastAPI；独立于 Memory 地址，允许两个服务分开部署 | 客户端 `HOME/skill-client.yaml`；独立 Skill transport |
| 访问凭据 `api_key` | 用户 / 认证流程 | 服务要求鉴权时必填 | 可留空；本地模式不需要凭据。远程模式按服务要求填写；当前独立 Skill HTTP 入口仍需接入凭据到项目的认证映射 | 客户端凭据存储；服务端认证层验证 |
| 项目身份 | 服务端 | 不配置，也不出现在客户端 YAML | 鉴权服务根据 API Key 确定项目；本地服务启动时使用持久化的默认项目，启动参数为空时自动分配 | 服务端身份解析；不得把 API Key 本身当项目 ID |
| 超时 `timeout_seconds` | 客户端默认，用户高级覆盖 | 否，放高级设置 | 默认 21600 秒（6 小时）；允许用户修改。超时不代表远端任务已停止，禁止自动重试 | HTTP transport；任务超时与网络超时分离 |
| 实验目录 `experiment_dir` | 用户配置，客户端执行记录 | 可使用默认值 `experiments` | 相对路径基于客户端 HOME；记录每次实验的输出信息与日志，不复用缓存或备份目录语义 | `SkillServiceClient` 每次调用创建独立目录，保存 `output.json`、`experiment.log`；失败保留日志 |
| 数据库类型 / URL、模型地址 / API Key | Skill 服务部署者 | 客户端不配置 | 客户端不持有远端数据库和模型凭据 | 见表三，服务端持有 |
| 算法参数、模型角色、内部组件类名 | 算法 / 服务端组装管理 | 客户端连接页不配置 | 不向普通用户暴露完整组装 YAML | 服务端配置类和 Pipeline |
| 本次操作的 Skill、版本、任务、数据划分、Pipeline 选择 | 业务调用者 | 在操作页面或 API 请求中填写 | 是业务输入，不是永久连接设置；可选 Pipeline 应以服务端实际能力为准 | 请求模型；现有 `TrainingRequest` 已包含相关训练输入 |
| 单次预算、并发、算法覆盖项 | 受限的高级调用者 | 首期不开放任意参数覆盖 | 若后续增加，需白名单、类型校验和服务端上限；不能覆盖凭据或协议约束 | 建议后续请求模型；当前训练请求没有算法覆盖字段 |

## 表三：Skill 服务端与算法配置

本表区分“部署者必须决定”“算法默认管理”“系统生成”。它们可以在同一份服务 YAML 中组装，但不应因此全部暴露给客户端。

| 配置项 / 当前字段 | 责任方 | 推荐配置方式与默认 | 代码归属 / 约束 |
| --- | --- | --- | --- |
| HTTP 监听地址、端口 | 部署者 | 启动参数；当前默认 `127.0.0.1:8000` | FastAPI 启动入口；不是算法参数 |
| 服务认证、授权、项目隔离、调用限额 | 服务部署者与服务代码 | 部署级配置；按认证身份检查项目权限 | 需要补齐的 HTTP 边界能力；当前入口没有认证，凭据到项目的映射需由服务认证层完成 |
| 数据库类型 `persistence.database.class_name` | 部署者 | 本地默认 SQLite；共享部署可显式选择 PostgreSQL | `ServiceConfig` 与数据库配置类 |
| 数据库 URL、凭据 | 部署者 | SQLite 本地 HOME 模式可沿用 `workspace.json` 的数据库路径；PostgreSQL 显式配置连接 | 数据库 Config；不由客户端传入。显式相对 URL 的当前语义需保留 |
| SQLite `timeout`，PostgreSQL `pool_size`、`max_overflow`、`pool_timeout` | 部署者高级配置 | 当前默认分别 5 秒、5、10、30 秒 | 数据库连接管理，不属于算法调优 |
| `persistence.config.include_remote` | 部署模板管理 | 按是否需要远端绑定表配置；现有类默认 true，本地工厂默认 false | 只控制相关表注册，不代表自动同步或自动连接远端 |
| `repository_pipeline`、`training_pipelines`、`evolution_pipelines` | 服务端开发者 / 部署者 | 由可运行的服务模板声明；客户端只使用暴露的能力 | 当前默认训练和演进为空，不能因为接入 HTTP 就声称算法已可运行 |
| Chat 模型 `model`、`base_url`、`api_key` | 部署者 | 启用相关功能才要求配置；凭据走服务进程环境或部署密钥 | `OpenAIChatConfig` / `AgentConnectionConfig`；Memory 凭据不能混用 |
| Embedding 模型、地址、凭据、`dimensions`、`send_dimensions` | 部署者 / 模型适配层 | 启用图检索时配置；维度必须与模型和现有向量一致 | `OpenAIEmbeddingConfig`；切换模型或维度不能直接沿用不兼容索引，应另行设计重建与切换 |
| 模型角色 `optimizer`、`router`、`failure_optimizer`，管理算法的 `embedding`、`chat` | 服务端组装管理 | 用部署连接构造各角色；允许显式使用不同模型，不能暗中改变角色 | 当前算法配置就地嵌套模型对象；建议组装函数负责填充，不要求新增全局资源引用框架 |
| 模型网络超时、重试、退避、Embedding `batch_size` | 连接适配层默认，部署者高级覆盖 | 默认来自各模型配置类；供应商限制不同可覆盖 | 传输参数，与算法生成策略分开；避免多层重试相乘 |
| 生成策略：温度、token 限制、推理预算、`top_p`、seed、stop 等 | 算法 / Agent 适配层 | 使用相应算法协议默认值；实验 YAML 可显式覆盖 | `OpenAIChatConfig` 及算法角色配置；模型能力不支持时明确校验，不静默替换 |
| 响应解析、工具参数策略、结构化输出、解析重试、供应商扩展 | Agent 适配层 | 与调用协议匹配，不放普通用户配置 | `response_validation`、`response_format`、`parse_*`、`invalid_tool_arguments`、`extra_body` 等 |
| 图管理阶段 `build_scene`、`llm_linking`、`query_rewrite`、`mode` | 算法开发者 | 当前前三者 true，mode 为 `unified_ppr`；高级部署可选择经验证的组合 | `SkillManageConfig`；启用 LLM linking/rewrite 必须有 chat 配置 |
| 图检索 `dependency_match_threshold`、`seed_top_k`、`score_threshold` | 算法开发者 | 当前默认 0.5、20、-1；仅高级调优 | `SkillManageConfig`，不放连接页 |
| 图传播 `damping`、`max_iter`、`tolerance` | 算法开发者 | 当前默认 0.2、50、1e-6；其中 damping 在当前实现表示重启概率 | `SkillManageConfig`；语义与默认值由算法维护 |
| 图输入限制 `max_input_chars`、`llm_batch_size` | 算法默认，部署者设资源边界 | 当前 1,000,000、32；允许部署级覆盖 | `SkillManageConfig`；客户端不得突破服务端限制 |
| TreeSkill `samples_per_task`、`min_trajectories`、`max_trajectories` | 算法默认，实验者高级覆盖 | 当前 1、1、1000；最小值不能超过最大值 | `TreeSkillTrainerConfig` |
| TreeSkill `rollout_concurrency`、`analysis_concurrency`、`localization_concurrency` | 算法提供默认，部署者限定资源 | 当前 8、16、16；部署应按容量配置，不能直接把算法默认当全服务安全容量 | `TreeSkillTrainerConfig`；全服务队列/并发上限需独立实现 |
| TreeSkill `analysis_max_turns`、`shell_timeout_seconds`、`transcript_max_chars` | 算法 / 执行环境管理 | 当前 20、120 秒、20000；实验可显式覆盖 | 训练配置；服务端执行限制仍为上界 |
| TreeSkill 三阶段 temperature、max_tokens | 算法开发者 | 分析/定位/融合温度 1/0/0；token 限制 4096/2048/4096 | `TreeSkillTrainerConfig`；不从普通客户端连接配置读取 |
| `analysis_adapter`、`annotation_mode`、`require_skill_match`、`success_score_threshold` | 算法 / 环境协议管理 | 当前 generic、required、true、1.0；阈值应匹配环境评分定义 | 协议变更需显式实验配置和记录，不能让普通请求随意关闭匹配检查 |
| `algorithm_version`、`prompt_version`、各阶段 `*_task` | 算法开发者 | 随代码/协议版本管理，作为追踪信息 | 当前配置字段仍可被 YAML 覆盖；建议产品组装层限制任意修改，不能仅改版本名冒充新算法 |
| `reference_routing`、`tree_metadata`、树结构及优化得到的指令 | 算法生成 / 版本管理 | 随 Skill 产物保存与恢复，不作为用户连接配置 | `TreeSkillConfig` 与 Skill 快照；结构和指令不放全局 ENV |
| 执行器、Shell、环境变量、命令超时 | 部署者 / 环境适配层 | 当前使用运行 Python 和平台 Shell；按服务运行环境指定 | `ExecutionConfig`；远端执行使用服务端路径和依赖 |
| 参考实验专用选项、数据加载器、环境配置 | 实验者 / 环境维护者 | 通过专用实验 YAML 显式配置；不塞进普通服务连接文件 | `TreeSkillExperimentConfig`、环境 Config；保留训练/验证/测试分离 |
| 工作目录、运行 ID、最终生效参数、审计快照 | 服务端自动管理 | 每次运行独立目录，记录完整有效配置与版本，凭据脱敏 | Pipeline 负责运行资源和持久化；算法不直接访问业务数据库 |

## 文件组织建议

客户端 HOME 和服务端 HOME 是不同概念，即使在同机也不要隐式绑定。

- 客户端 `HOME/settings.json`：目标职责仅为 Memory 连接配置，不包含 Skill 缓存、备份、注册信息等字段。
- 客户端 `HOME/skill-client.yaml`：Skill HTTP 连接及实验目录配置，每个 HOME 一份。注册记录属于 Skill 运行数据，不出现在连接表单或模板。
- Skill 服务端 `SERVER_HOME/service.yaml`：保留服务端数据库和完整 Pipeline 组装契约，只需填写必填项及默认值覆盖项。
- Skill 服务端 `SERVER_HOME/service.env`：本地部署可选的环境变量来源；线上由部署环境注入密钥。
- 算法源码中的 `Config` 类：参数类型、默认值和校验规则的唯一来源。
- 独立实验 YAML：记录具体实验模型、参数覆盖和数据划分，不要求每位普通用户创建。

客户端凭据和服务端模型凭据分开保存。新增 `skill-client.yaml` 的 ENV 绑定需明确定义加载规则，不能假设现有代码已支持。当前服务 YAML 不展开 `${VAR}`；已有 Chat 连接类可以读取 `MINDMEMOS_AGENT_API_KEY` 和 `MINDMEMOS_AGENT_BASE_URL`，数据库与 Embedding 的密钥注入应由明确的服务端组装逻辑完成，不能只写占位符就认为会生效。

## 代码调用与参数生效流程

1. 客户端分别读取 Memory、Skill 连接设置，分别创建 transport，不再默认共用一个地址和凭据。
2. Skill 服务启动时读取部署 YAML 和已定义的环境变量，创建数据库及所启用能力的模型配置。
3. `create_service(config)` 校验和组装服务。算法配置类补齐未填写参数；现有嵌套组件结构继续使用，FastAPI 不接管算法默认值。
4. 初始化存储，再调用 `create_app(service)`。现有 FastAPI 命令入口已按这一顺序执行；直接使用 `create_app` 的嵌入者需负责初始化。
5. 客户端通过 HTTP 发送业务请求。服务端认证并校验项目权限，然后路由到服务和 Pipeline；这一步的认证授权是待补齐能力。
6. Pipeline 管理数据、版本与运行目录，调用算法，再持久化结果。连接配置不参与每次算法参数调优。
7. 算法调用只发送一次，失败或超时记录状态并交由用户决定，不提供 `max_retries` 配置。需要单次覆盖时，后续新增受限请求字段；在每次运行中构造独立有效配置，不修改共享算法对象。保留配置快照，并受服务端资源上限约束。

普通运行参数建议遵循“算法默认值 < 服务部署覆盖值 < 经允许的单次运行覆盖值”。凭据、鉴权、资源硬上限、项目权限和协议不属于这一自由覆盖链。

## 当前实现与落地顺序

当前已有 Skill FastAPI 适配器，具备注册、列表、版本、导出及训练/演进等路由，但路由存在不代表对应 Pipeline 已启用。`/health` 也只代表应用存活，不代表数据库、模型或训练能力已就绪。

当前 UI 会直接创建本地 Skill 服务；当前 `MindMemOSClient` 为 Memory 与现有云端 Skill 共用 transport。现有 `SkillCloudClient` 面向已部署云接口，不能未经接口核对就拿来调用新 Skill FastAPI。

建议按以下顺序落地：

1. 新增独立 Skill HTTP 客户端及配置，将旧 SDK 中的 Skill 设置和注册数据迁入 Skill 侧，完成后 Memory 配置只包含 Memory 字段。迁移需保留已有数据和已部署接口契约；旧数据兼容不作为目标 Memory 配置结构。
2. 为新服务补齐认证授权与能力查询。配置测试分别验证网络、认证、项目可访问性和能力，不能只请求 `/health`。
3. 核对 UI 操作与 HTTP 路由的覆盖情况，尤其是图管理和动态 Skill 组装；缺少的业务能力先补接口，再切换 UI 调用。
4. 明确文件跨机器传输：客户端本地路径不能作为服务端可读路径。注册需要上传或受支持的远端来源，导出需要下载响应；现有本地路径接口不能原样假设可用于远程部署。
5. 明确长任务协议：推荐训练/演进提交任务后返回任务 ID，再查询状态与结果；该任务接口是后续能力，当前 HTTP 训练路由是直接调用。不自动重试算法或其他写操作；将来如支持人工重试，需要幂等或状态查询来避免重复执行。
6. 把配置页的 Skill 卡换成连接表单。当前 `service.yaml` 保留在部署端，不覆盖、不删除、不当作客户端连接配置迁移。
7. 验证连接配置更新后的客户端重建、旧注册数据保留、本地/远端项目隔离、凭据不泄露、文件上传下载与失败恢复。

本方案不要求修改已商用 `src/mindmemos/` 的数据库。若后续实现触及该服务的数据契约或迁移其 Skill 管理，必须另行提供保留生产数据、可重复且可中断恢复的无感迁移，不能通过换客户端地址跳过兼容工作。

## 源码依据

以下链接均相对本文件，可用于后续实现核对：

- [SDK 配置模型](../../src/mindmemos_sdk/mindmemos_sdk/config/models.py)、[SDK 客户端与共享 transport](../../src/mindmemos_sdk/mindmemos_sdk/client.py)。
- [UI SDK 可编辑字段](../../src/mindmemos_ui/mindmemos_ui/config.py)、[UI 本地服务组装](../../src/mindmemos_ui/mindmemos_ui/app.py)。
- [服务配置契约](../../src/mindmemos_skill/mindmemos_skill/api/config.py)、[工厂](../../src/mindmemos_skill/mindmemos_skill/api/factory.py)、[FastAPI 入口](../../src/mindmemos_skill/mindmemos_skill/api/entries/fastapi.py)。
- [数据库配置](../../src/mindmemos_skill/mindmemos_skill/infra/database/config.py)、[Chat 配置](../../src/mindmemos_skill/mindmemos_skill/agents/buitin_agents/openai_chat/config.py)、[模型连接与 ENV](../../src/mindmemos_skill/mindmemos_skill/agents/config.py)、[Embedding 配置](../../src/mindmemos_skill/mindmemos_skill/infra/embedding/config.py)。
- [图管理算法配置](../../src/mindmemos_skill/mindmemos_skill/algos/manage/skill_manage/config.py)、[TreeSkill 配置](../../src/mindmemos_skill/mindmemos_skill/algos/train/treeskill/config.py)、[执行环境配置](../../src/mindmemos_skill/mindmemos_skill/infra/execution/config.py)。
- [训练请求](../../src/mindmemos_skill/mindmemos_skill/pipelines/typing.py)、[训练 Pipeline 边界](../../src/mindmemos_skill/mindmemos_skill/pipelines/train.py)。


## 已实现的连接页面与完整模板

配置弹窗默认显示逐项表单，可切换 YAML 编辑，也可导入 YAML 后回填表单。两种方式使用同一后端配置模型，缺省字段补齐默认值，未知字段、重复键、非法值均报错。界面和导出不包含 Skill 服务端算法 YAML；保存成功后关闭弹窗。原有 `service.yaml` 不覆盖、不删除。

两张卡以及编辑弹窗都提供完整默认模板下载，模板不包含现有凭据。导出当前配置则会包含用户保存的凭据。客户端配置不使用 ENV 行或 `${VAR}` 插值；API Key 可为空，空字符串规范化为 null。

Memory 全集：

```yaml
base_url: http://127.0.0.1:8000
api_key: null
user_id: null
timeout_seconds: 21600
```

Skill 全集：

```yaml
mode: local
base_url: http://127.0.0.1:8001
api_key: null
timeout_seconds: 21600
experiment_dir: experiments
```

`mode` 可选 local/remote。本地模式只允许回环服务地址；远程模式允许凭据为空以适配服务的认证方式，但不会绕过远端服务的认证要求。两种模式都没有客户端 project_id。

本地 Skill FastAPI 可通过 `--home /path/to/server-home --port 8001` 启动。省略 `--project-id` 或传空字符串时，复用/生成该服务 HOME 的稳定默认项目。显式服务 YAML 与本地 HOME 组装是两个入口，不能混用。此配置页不会自动启动服务，也不会把本地工作台的既有图管理/仓库调用自动切换成远程调用。

`SkillServiceClient(home)` 读取该 HOME 的 Skill 连接文件；其 `train()` / `evolve()` 使用 6 小时默认 HTTP 超时、零自动重试，并在实验目录留下单次调用日志和响应输出。HTTP 失败或超时的日志状态为 failed_or_unknown，不能据此宣称远端算法未执行或已停止。服务端内部的详细算法日志仍由服务端运行目录负责，客户端日志不冒充完整服务端日志。
