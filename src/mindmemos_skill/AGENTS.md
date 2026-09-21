# 项目开发约定

## 改动范围

- 本仓库包含多个 Python 包。新增 Skill 能力默认放在 `src/mindmemos_skill/`，仅在任务需要时修改其他包或根目录配置。
- 修改前检查当前源码、配置和工作区差异，保留用户已有改动。
- 优先编写最少必要、易读的代码，复用标准库、已有公共模块和成熟依赖；不提前实现未确定的业务逻辑。
- 本文件适用于 `src/mindmemos_skill/`。下文涉及的 `config/`、`scripts/`、`docs/` 和 `tests/` 路径均相对于仓库根目录。
- 当前已实现服务组装、Agent 后端、环境 rollout、数据加载与划分、Skill 内容与任务组装、数据库及部分业务持久化能力；中心仓注册、查询、提交、导出与版本管理 API 已实现，自动训练和演化仍为占位。以当前源码和测试区分已实现能力与设计方案，不把局部实现描述为完整业务流程。

## Skill 包模块职责

以下路径相对于 `src/mindmemos_skill/mindmemos_skill/`。

| 模块 | 职责 |
| --- | --- |
| `api/` | 对外统一入口。`MindMemOSSkill` 抽象基类声明业务接口；`create_service` 从配置对象或 YAML 组装 `SkillService`，接入层放在 `entries/`。 |
| `common/` | 跨模块共用的基础操作，例如时间处理，不承载具体业务流程。 |
| `infra/` | 按能力分组的基础设施；`database/` 管理数据库、通用 CRUD 与事务，`execution/` 管理进程执行与清理，不定义业务表或业务查询。 |
| `algos/` | 按 `train/`、`evolve/`、`manage/` 划分算法职责；算法专属 Skill 实现与算法放在一起。 |
| `pipelines/` | 业务操作层；RepositoryPipeline 已实现仓库操作，训练与演进仍为抽象契约。Service 统一委托此层，由其调用 persistence、算法和环境。 |
| `agents/` | Agent 的调用、适配及交互逻辑。 |
| `logging/` | 日志配置及公共日志能力；避免在模块导入时配置全局日志。 |
| `persistence/` | 业务数据表、持久化数据结构、查询和写入操作；复用 `infra/` 的连接能力。 |
| `skill/` | Skill、bundle、辅助资源、快照、恢复与导出的公共协议，以数据为中心，不参与后端注册。 |
| `envs/` | rollout 公共流程与环境实现；`data/` 管理公开任务、私有评分输入、数据加载与划分。 |
| `registry/` | 全包统一的注册与按名称构建机制；需要按名称构造的业务后端复用此处实现。 |

## API 与服务组装

- `api/base.py` 定义全局抽象基类，中心仓接口覆盖注册、列表、详情、提交、导出、历史、差异、切换、元信息及删除；请求与返回类型放在 `pipelines/typing.py`。
- 抽象基类只定义契约，不实现业务逻辑，不创建后端资源。
- `api/factory.py` 的 `create_service(config)` 接受 `ServiceConfig` 或 YAML 路径。全局只有一个 `persistence`，数据库直接定义于其 `database` 字段；`repository_pipeline` 为完整定义，`training_pipelines`、`evolution_pipelines` 为名称到完整定义的字典。工厂预检后构造唯一存储，以 `persistence` 关键字注入各 pipeline。每个 pipeline 用自身嵌套配置独立构造算法、环境和 Agent，允许重复配置，不使用全局后端表、资源名引用、依赖图或隐式仓库选择。请求通过 `pipeline` 选择流程，只有一个时可省略。
- 服务不参与接入模式注册，也不持有 HTTP 应用。`api/entries/fastapi.py` 的 `create_app(service)` 只包装已有 `MindMemOSSkill` 实例，所有路由共享其后端，不重复构造服务。
- 未实现的入口使用 `NotImplementedError`，不得返回伪造的成功结果。
- 导入包不得启动服务、连接数据库或加载模型。Python 调用方通过 `initialize_storage()` 显式建表；HTTP 启动入口在启动前初始化。该操作会幂等迁移既有版本标签，但不承担其他旧表结构迁移。
- 启动统一通过根目录 Makefile 的 `skill-local` 和 `skill-fastapi` 目标；具体入口分别放在 `api/entries/local.py` 和 `api/entries/fastapi.py`。`run_skill.py` 仅供个人调试，不作为正式服务启动入口。
- 服务配置放在 `config/mindmemos_skill/`，默认配置为 `default.yaml`；不要继续使用旧的 `config/skill/` 路径。`skill-local` 当前仅组装、展示并关闭服务，不会执行训练。
- 明确资源所有权：调用方关闭服务；`SkillService.close()` 先逆构造顺序调用 pipeline 的 `close()`，最后关闭数据库。Pipeline 负责自己的组件及构造失败时的部分资源清理，不关闭全局存储。

## 统一注册

- 注册逻辑只在 `registry/` 实现一次，通过共享 `Registrable` 提供注册与构建能力；各业务基类复用它，不维护重复实现。
- 注册使用显式名称与装饰器，各业务类型的注册项相互隔离；空名称、重复名称、未知名称及不符合基类要求的实现应明确报错。
- 注册阶段只记录实现类，不创建实例或启动外部资源。
- 内置实现通过显式模块清单加载，由对应包入口触发；用户扩展必须显式导入。不扫描目录或使用元类自动注册。
- 业务模块可以维护内置实现加载清单，但不得另写注册、查找或构建逻辑。
- `Registrable.register(name)` 统一注册，`resolve(name)` 查找类，`build(name, options, **dependencies)` 校验配置并构造实例。配置模型由实现类的 `config_type` 声明。
- YAML 的 `class_name` 必须与装饰器注册名称一致；pipeline 的 `config` 由具体实现的 `config_type` 校验，并就地定义其全部组件。无 `module` 或字符串 `dependencies` 装配字段；字段归属决定注册类别。
- 注册机制用于业务后端。工厂在构造前校验存储及 pipeline 的类型、配置与构造参数；内部组件校验由具体 pipeline 的嵌套模型负责。未配置训练/演进时调用抛出 `NotImplementedError`，HTTP 返回 501；中心仓接口已接入持久化。
- `Data` 复用统一注册机制，但不是服务配置中的顶层后端类别。普通函数能力（例如进程执行）不为形式统一而额外包装为注册后端。

## 环境与数据边界

- 公共执行入口为同步的 `Environment.rollout(...) -> RolloutResult`；具体环境实现 `_run(*, task, reward_data, skills, workspace)`，负责准备公开材料、调用 Agent、收集轨迹及评分。
- `DataSample` 绑定公开 `Task` 与私有 `RewardData`；`Reward` 是评分输出。答案、私有测试和参考文件不得进入 Agent 提示词或公开任务材料。
- `Data.load()` 校验稳定且唯一的 Task ID；划分使用显式 seed 或已有 split 文件。split 文件保存名称与有序 ID 列表，不保存答案。数据下载由外部脚本负责，不放入基类加载流程。
- 每次独立 rollout 使用独立 `work_dir`，不得并发复用。`audit/` 保存请求与结果，`workspace/` 存放 Agent 可用文件；不得修改进程全局工作目录。目录分离和 cwd 设置不等于文件访问隔离。
- resume 是环境可选能力，由 `supports_resume` 声明，默认 `True` 以兼容已有环境。声明 `False` 的环境收到 `resume=True`（包括参数默认值）时，在组装或文件操作前抛出 `NotImplementedError`；调用方必须传 `resume=False`。
- 只有正式 `audit/result.json` 表示 rollout 完成。支持 resume 的环境在输入一致且结果完整时直接复用；已有组装记录而无结果时复用组装内容后重新执行 `_run`，不恢复轨迹中间步骤。输入变化或存档损坏应明确报错。
- `resume=False` 对匹配输入强制重跑并先移除旧结果，不自动清理工作文件。基类不管理 attempt 或自动重试；外层负责重试、采样及目录分配。
- 执行或评分异常直接抛出，不伪装成零分结果；正常评分失败可以返回零分。具体环境汇总执行及评分的 LLM usage，基类记录时间；resume 保留原始消耗，默认零 usage 不代表已核实无消耗。
- 将影响执行的模型、Agent、评分器和数据版本记录到可审计配置或输入中。配置会明文落盘，不保存 API key 等凭据；源码及外部文件内容变化不会自动纳入恢复核对。

## Skill 与算法边界

- Skill 文件导出的具体操作归属 `skill/`。完整导出统一调用 `Skill.export()`，`SkillSnapshot` 仅负责序列化数据与校验；中心仓先通过 `Skill.from_snapshot()` 恢复数据对象，由 `skill/io.py` 处理目录替换与恢复；RepositoryPipeline 负责版本选择及仓库保护，不实现文件导出流程，不为导出快照实例化动态 Skill 依赖。算法所需的默认执行依赖延迟到实际组装时创建。

- `Skill` 是数据对象，不继承 `Registrable`，不要求空配置，也不作为 YAML 后端。可优化 `bundle` 和辅助 `resources` 分开保存。`assemble(task)` 返回完整的任务专属 Skill 副本，保留具体类、依赖、辅助资源和导出能力，不原地修改源 Skill。
- rollout 公开入口统一接收 `skill_set: SkillSet | None`，不接受旧的 `skills=`。SkillSet 默认全选，动态集合通过 `select(task)` 选择成员，并用 `config_snapshot()` 记录非敏感选择配置；集合按 `max_concurrency` 并行调用成员的 `assemble(task)` 并保持选择顺序。具体环境 `_run` 直接使用已组装的 Skill 列表，不按静态或动态类型分支，也不重复组装。恢复复用已保存的选择和组装内容，输入或选择配置变化要求新目录；旧静态列表记录仅以静态 SkillSet 恢复。
- `snapshot()` / `from_snapshot()` 负责内容快照与数据恢复，保留已有实现元数据而不加载运行时代码。`content_hash` 覆盖源 bundle，`local_snapshot_hash` 同时覆盖 bundle 和 resources；源内容与某次任务组装结果不得混用。
- 算法专属实现由调用方显式构造；需要动态行为时使用该算法自己的恢复入口。`algos/train/treeskill/` 提供任务时路由的 `TreeSkill` 与单轮主动采样的 `TreeSkillTrainer`；独立实验入口位于 `scripts/treeskill/`，尚未接入训练 pipeline。
- SpreadsheetBench 算法适配通过继承公共环境并覆盖 `_system_prompt`、`_user_prompt`、`_get_observation` 实现；不复制 `_run()`、交互循环、工作区与任务评分。环境重算运行时和评分实现归入 `envs/builtin_envs/spreadsheetbench/`。
- 所有算法的主流程保持通用；环境特殊提示词、分析或执行流程放在算法自身的 `env_specific/` 中，主入口文件名与环境注册名对应（例如 `spreadsheetbench.py`）。较长的同环境辅助实现可以按 `spreadsheetbench_*.py` 拆分，配置和数据类型仍分别归入该目录的 `config.py`、`typing.py`。
- TreeSkill 的 SpreadsheetBench reference 训练失败分析允许访问对应训练样本的标准答案，以复现原始 PASS 门控分析流程；这是仅限该训练分析 Agent 的例外。任务执行 Agent、通用分析以及 val/test 不得借此访问训练分析之外的私有答案。进入 Skill 的经验继续执行原始私有答案内容过滤。
- TreeSkill 单条执行异常由环境抛出，训练/实验层记录并跳过；不得将异常伪装成零分或纳入训练证据，不因一条数据失败终止其他独立数据。分析、定位和融合的单项失败也须保留记录。配置、数据契约与产物写入错误仍应明确失败。
- TreeSkill 的 `config.json` 仅保存算法配置，不保存文件 SHA、提示词哈希、源码哈希或数据/Skill 快照。总实验目录还须记录分数分母、分阶段及总耗时、训练指标、模型调用和每次 patch 的提案、结果、修改前后全文与 diff；test 只在训练完成后执行一次用于汇报，不参与选优。
- `SkillTrainer.train()` 接收已准备好的 Skill、Environment、训练集和验证集，返回选中的 Skill；不得修改输入 Skill，也不得直接或通过依赖读写数据库。调用方管理环境生命周期并保存返回结果。

## 持久化与基础设施

- 业务表名统一定义在 `persistence/constants.py`，使用 `Final[str]` 模块级常量；建表、表注册、外键引用、CRUD 和相关测试禁止直接硬编码表名。独立 SQL 迁移文件保留物理名称，不因常量化改写历史迁移。

- 业务数据服务按 `persistence/db_services/` 下的目录划分；业务类型、表和读写逻辑放在对应服务，复用 `infra/database/` 的连接、表注册及事务能力。SQLAlchemy Column 使用英文 `comment` 说明含义。
- `create_persistence(database)` 只组装服务并登记表，数据库由调用方持有并关闭；`ensure_collection()` 显式创建缺失表，不承担已有数据库迁移。字段变更需提供明确迁移方案，不自动重建旧库。
- `version_id` 只用于内部关联和审计；`version_label` 是同项目、同 Skill 内唯一且创建后不变的展示别名。用户标签格式为 `v主.次.修订`；算法发布由 pipeline 传固定 `algorithm` 标识，持久层分配递增修订号与算法尾缀，禁止算法自行拼接标签或用展示顺序代替父子谱系。
- Skill 版本身份、源内容和多父谱系与 rollout 物理执行记录分开。轨迹以 `trajectory_id` 标识，以 `(rollout_id, attempt_no)` 区分重试；新增记录不覆盖历史。
- 本地身份、哈希与远端身份、哈希分别保存，remote 映射带部署及项目范围；已有映射表和操作队列不等于已实现网络同步、worker 或自动重试。
- `algos/manage/skill_manage/` 提供基于现有图表的组织与推荐；算法专属 SkillSet 不解释 Skill 版本。外围调度通过 `SkillGraphStore` 提供同项目版本绑定，发布服务核验完整快照并原子提交图和向量，保留历史。Scene 来自调用方显式记录，不由 LLM 生成。新增模型连边和查询改写使用显式配置及可注入客户端；模型失败明确抛出。图管理范围见 `docs/mindmemos_skill/skill_manage.md`，不得将 Python 算法入口描述为已接通 UI 或商用云服务。

## 代码与验证

- `envs/builtin_envs/` 下所有 README 必须使用英文，且仅按以下顺序包含三项内容：① 官方数据下载链接；② split 文件格式要求；③ 根据官方数据位置初始化 env 的示例代码。禁止加入实现原理、安装说明、运行结果、审计机制、评分规则或其他内容。

- 代码、注释和 docstring 使用英文，独立说明文档优先使用中文。
- 默认使用仓库根 `pyproject.toml` 中的 Ruff 配置统一 Python 基础格式：修改后对本次涉及的 Python 文件运行 `ruff check --fix` 和 `ruff format`，再运行 `ruff check` 与 `ruff format --check` 验证；不对无关文件做批量格式化。
- 长提示词不得写成单个超长源码行；按语义拆成括号内相邻字符串或合适的多行字符串，遵循当前 120 字符行宽。纯格式调整必须保持运行时提示词文本不变；Ruff 不自动拆分的长字符串需手动换行。
- 公有类和函数提供清晰的 docstring；有参数时说明其含义，必要时说明返回值和异常。
- 字段或类变量声明与其紧随的三引号说明保持相邻；每组“声明 + 说明”与下一字段或类变量之间至少空一行。类 docstring 与首个字段或类变量之间也至少空一行，`model_config` 同样遵守此规则。
- 配置定义放在对应模块的 `config.py`，其他数据结构与类型定义放在 `typing.py`；按需创建文件。
- 已知字段和方法使用直接访问，不用字符串反射替代。
- 子模块内部优先使用相对导入，特别是本包的 `logging`，避免与标准库混淆。
- 本包使用 Python >=3.11,<3.14，依赖与可选依赖维护在本包 `pyproject.toml`。目前未加入根 uv workspace，不假定根环境同步会自动安装本包及所有可选依赖。
- 修改后运行与改动相关的检查。主要测试位于根目录 `tests/mindmemos_skill/`，日志测试位于本包 `tests/test_logging.py`；按改动选择测试，不为纯文档更新启动完整业务测试或模型调用。
- 接口和打包改动验证导入、抽象契约及打包；环境、Skill 与持久化改动优先验证恢复一致性、私有数据边界、源对象不变性及历史记录保留等行为。
- 不在未获要求时启动模型调用或昂贵实验。

## Pipeline 调用边界

- Service 不创建 Skill 数据服务或公开原始 database；所有业务操作委托给 pipeline。UI 图查询通过 `service.graph_queries()` 获取项目范围内的查询 pipeline。
- `SkillPersistence.skills` 是统一的 Skill 身份、版本及快照访问入口；`resolve(project_id, reference)` 解析 ID、别名和唯一前缀，`version_by_id(project_id, version_id)` 按精确版本加载并校验项目。事务及版本冲突检测留在 persistence。
- TrainingRequest 传入 tasks、split 和必填 base_version_id，不接受 skill_ref。train 必填，val/test 可缺省；test 只评估已选定候选，不进入 trainer。具体训练和演进 pipeline 未实现时仍返回明确的未实现错误。

- Pipeline 的业务接口、仓库与图查询接口、`close()` 均为异步；Service 与 `create_service()` 沿调用链使用 `await`。`SkillTrainer.train()` 和 `SkillEvolver.evolve()` 同样异步。仅进程入口创建事件循环，不在 pipeline 或训练主流程内使用 `asyncio.run()`；现有同步数据库事务与环境采样通过工作线程执行。
