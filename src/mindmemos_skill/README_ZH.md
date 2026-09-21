# MindMemOS Skill

[English](README.md)

`mindmemos-skill` 用于承载 MindMemOS 中与 Skill 相关的 Python 功能。
当前提供全局抽象接口 `MindMemOSSkill`、后端统一注册机制，以及从配置对象或 YAML
组装服务的 `create_service(config)` 工厂。FastAPI 通过入口层包装已有服务。业务操作仍待实现。

## 环境要求

- Python >=3.11,<3.14。
- Pydantic 2 和 PyYAML 6。

## 本地安装

在仓库根目录执行，将包以可编辑模式安装到当前开发环境：

```bash
uv pip install -e ./src/mindmemos_skill
```

验证导入：

```python
import mindmemos_skill
```

安装包名称为 `mindmemos-skill`，Python 导入名称为 `mindmemos_skill`。
该包目前尚未注册到仓库的 uv workspace。

## 包结构

```text
src/mindmemos_skill/
├── pyproject.toml
├── README.md
├── README_ZH.md
└── mindmemos_skill/
    ├── __init__.py
    ├── api/          # 全局抽象接口与服务工厂
    ├── common/       # 公用 JSON、IO 等操作
    ├── infra/        # 与业务无关的基础设施
    ├── algos/        # Skill 业务算法
    ├── agents/       # Agent 调用
    ├── logging/      # 日志能力
    ├── persistence/  # 业务表、查询与写入
    ├── skill/        # Skill 定义与使用操作
    ├── envs/         # 算法环境及数据结构
    └── registry/     # 统一注册机制
```

包元信息与版本号统一维护在 `pyproject.toml` 中。

中心仓 API 已实现注册、列表、详情、提交、导出、历史、差异、版本切换、元信息修改与引用检查删除。
请求类型、Python 示例和完整 HTTP 路由见[中心仓 API 说明](../../docs/mindmemos_skill/central_repository_api.md)。
自动训练和演化仍未实现。

## 最小环境接口

`Environment` 为算法提供同步的单条轨迹入口：

```python
from mindmemos_skill.envs import Task
from mindmemos_skill.envs.data import RewardData
from mindmemos_skill.skill import Skill, SkillBundle, SkillSet

skill = Skill(
    bundle=SkillBundle(files={"SKILL.md": "Check quantifiers and equality cases."}),
)
result = env.rollout(
    task=Task(id="task-001", instruction="Complete the task and check the answer."),
    work_dir="output/sample-001",
    reward_data=RewardData(),
    skill_set=SkillSet({"skill": skill}, max_concurrency=4),
    resume=True,
)
print(result.output, result.reward.score, result.reward.success)
```

`SkillSet` 默认选择全部成员；动态集合覆写 `select(task)` 返回有序成员 ID，
并通过 `config_snapshot()` 记录不含密钥的选择配置。集合只调用成员的 `assemble(task)`，
不区分静态或动态 Skill。`max_concurrency` 限制单次集合组装的并发数，输出保持选择顺序。
公开入口已移除 `skills=`，统一使用 `skill_set=`。

恢复继续复用已保存的选择和组装结果，不重复推荐或组装；候选、顺序或集合配置变化要求新目录。
源输入一致的旧列表记录可用静态 SkillSet 恢复；`resume=False` 清除旧完成标记并重新选择、组装、执行。

具体环境继承 `Environment[YourConfig]` 并实现
`_run(*, task: Task, reward_data: RewardData, skills: list[Skill], workspace: Path) -> RolloutResult`。
在该方法中准备任务文件、调用 agent 并使用 skills、收集轨迹、计算 reward。
基类只传入已创建的 `work_dir/workspace/` 绝对路径；具体后端必须将它用于执行，不应修改进程的全局工作目录。
agent 可通过具体环境的构造函数注入，沿用现有 YAML dependencies 机制。
`Task` 包含必填的稳定 `id` 和 `instruction`；子类可添加公开的任务专属字段。
私有答案、测试用例和参考文件通过 `RewardData` 子类传入，只用于评分，不交给 agent。
不需要额外评分输入的环境可以显式传入 `RewardData()`。
`rollout` 接收 `SkillSet`，集合选择成员并并行组装；`_run` 接收组装后的 `list[Skill]`。
Skill 自己生成任务副本，只替换 bundle，保留具体类、references 等辅助文件及导出方法。
具体环境不判断静态或动态类型，也不重复组装。
不传 skills 表示无 skill 基线。基类不包含真实模型后端或特定任务评分器。
已接入的 [LiveMath 环境](mindmemos_skill/envs/builtin_envs/livemath/README.md) 提供本地原始数据加载、
单次 OpenAI-compatible 聊天执行及本地选择题评分；统一支持静态与动态 Skill。

`RolloutResult` 包含输出、轨迹、测评结果和审计信息：

- `output`：JSON 可序列化的最终输出。
- `trajectory`：按执行顺序排列的 JSON 字典列表。
- `reward: Reward`：必填 `score`（有限数值）、`success`（bool）、`detail`（str，保存详细测评数据，可使用 JSON 文本）。允许额外 JSON 字段，也可继承扩展。
- `usage: Usage`：默认包含值为 0 的 `input_tokens`、`output_tokens`、`total_tokens`，支持额外字段，例如缓存 token、reasoning token 和按模型统计的数据。具体环境汇总所有 LLM 调用（含测评调用）；总 token 按后端报告填写，不自动推算。默认 0 表示未填入用量，不代表已核实没有消耗。
- `started_at`、`finished_at`：基类自动记录的 UTC 时间。
- `duration_s`：基类使用单调时钟测得的秒数，覆盖 Skill 组装、`_run` 中的准备、agent 执行和评分，不包含缓存读取或结果落盘。

执行或评分异常应直接抛出；正常完成但未通过任务验证可返回 `Reward(score=0, success=False, detail="...")`，仍会保存并复用。
`_run` 返回时不必填写时间，基类会统一补齐。resume 保留首次执行的时间和 usage，不记录为新的执行消耗。

```python
from mindmemos_skill.envs import Reward, RolloutResult, Usage

result = RolloutResult(
    output="42",
    trajectory=[],
    reward=Reward(score=1.0, success=True, detail='{"passed": 3, "total": 3}', passed=3),
    usage=Usage(input_tokens=100, output_tokens=20, total_tokens=120, cached_tokens=40),
)
```

Reward 和 Usage 子类的额外字段会保留到 JSON 和恢复后的结果中。
如果恢复时还需要具体子类的 Python 类型和校验规则，定义 `RolloutResult` 子类、
将 `reward` / `usage` 字段声明为对应子类，并在具体环境设置 `result_type = YourResult`。
默认恢复为通用 Reward / Usage 对象，保留扩展数据。旧版缺少 task ID 或 reward_data 的请求存档不自动迁移，请使用新的工作目录。

`work_dir` 是一次独立 rollout 的根目录，由基类管理两个子目录：

```text
work_dir/
├── audit/
│   ├── request.json
│   └── result.json
└── workspace/
    └── ...
```

`audit/request.json` 保存任务、私有 reward_data、skills、环境类和配置；`audit/result.json` 保存完整结果。
结果通过临时文件替换方式写入，只有正式结果文件表示完成。
`workspace/` 存放公开任务材料、提供给 agent 的 skills 和 agent 产物。
基类不向 `_run` 传入根目录或审计目录，不管理 attempt，也不自动重试或清理工作目录。
`audit/` 和 `workspace/` 不允许使用符号链接。

目录分离不等于文件访问隔离：Agent 后端还需将文件访问范围限制在 `workspace/`，
仅设置 cwd 无法阻止读取 `../audit`。具体环境向 agent 提供 Task 中的公开内容，
不得直接发送可能包含标准答案等私有信息的完整 Task 快照。

- resume 是环境可选能力，`supports_resume` 默认 `True`。不支持的环境声明 `supports_resume = False`，收到 `resume=True`（包括省略参数时的默认值）会在 Skill 组装或文件操作前抛出 `NotImplementedError`；调用时必须显式传 `resume=False`，仍保存审计记录并校验输入。
- 支持 resume 的环境默认启用：输入一致且存在完整结果时直接读取，不调用 agent 或评分器。
- 没有完整结果：本次调用执行一次 `_run`，不恢复到轨迹中间步骤；原有工作文件保留，重试和多次采样由外层管理。
- 任务或 reward_data 的字段/类型、skill 类型或配置、环境类或配置变化：报错，使用新的工作目录。
- `resume=False`：对相同输入强制重跑；先移除旧完成结果，避免失败后错误复用旧结果。
- 损坏的存档明确报错，不静默启动执行。

每条独立采样使用独立目录；同一目录不支持并发调用。将影响执行或评分的模型、参数、
数据版本、agent 和评分器版本等标识放入环境配置，才能参与 resume 核对；
任务子类字段也会保存；执行时将原始 Task 和 Skill 对象传给 `_run`。
Skill 当前按类型和配置保存快照，配置之外的临时状态以及代码变化不会自动检测。配置会明文落盘，不应包含 API key 等凭据。

## 环境数据加载与划分

`envs/data/` 定义 `Data`、`DataSample` 和 `RewardData`，不新增顶层后端类别。
`Data` 复用 `Registrable`，支持 `@Data.register("name")` 和 `Data.build(name, options)`。
具体实现的配置放在对应 `config.py`，实现 `_load() -> list[DataSample]` 读取数据文件。
`load()` 统一验证样本结构、非空且唯一的 Task ID；一次加载全部样本到内存。
基类不自动下载数据，已下载的 Hugging Face 数据、JSONL、Parquet 等格式由具体子类适配。

每条样本由 `DataSample(task=..., reward_data=...)` 绑定：
Task 只包含公开信息，RewardData 是私有评分输入，Reward 才是评分后的输出。
RewardData 默认不定义业务字段且拒绝未知字段，具体数据集应继承并显式声明答案等字段。
例如 `class MathRewardData(RewardData): answer: str`。
路径指向的参考内容变化不会自动被发现，具体实现需在 RewardData 中记录数据版本或内容哈希，参与 resume 核对。

```python
samples = data.load()
splits = data.split(samples, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
data.save_splits(splits, "output/splits.json")

# Restore exact membership from the saved IDs.
splits = data.split(samples, split_file="output/splits.json")
sample = splits["train"][0]
result = env.rollout(
    task=sample.task,
    reward_data=sample.reward_data,
    work_dir="output/rollout-001",
    skill_set=None,
)
```

随机划分要求非负比例之和为 1（容许 1e-9 的误差）及显式整数 seed。
先按 Task ID 排序，再用局部随机数生成器打乱，不修改全局随机状态或原样本列表。
按比例向下取整后，剩余名额按小数余数从大到小分配；相同余数按 split 名称排序。
split 名称顺序不影响随机划分结果。每条样本恰好分配一次；允许空 split 和空数据集。
可复现依赖相同的样本 ID 集合、比例和 seed；数据集增删样本后不保证原归属。

划分文件统一为 JSON，保存 split 名称与有序 ID 列表：

```json
{
  "train": ["task-001", "task-007"],
  "val": ["task-003"],
  "test": []
}
```

文件模式不能同时传 ratios 或 seed。未知 ID、重复 ID、跨 split 重叠、空白名称均会报错。
允许只选择数据的一部分；未列出的样本被排除，按文件给定顺序返回。
`save_splits` 原子保存划分，仅写名称与 ID，不包含任务文本或私有答案。
样本 ID 应来自数据源中的稳定标识，不要在每次加载时随机生成或依赖易变的行号。

## 日志

在应用启动时显式配置一次，再在各模块中获取 logger：

```python
from mindmemos_skill.logging import configure_logging, get_logger

configure_logging(level="INFO", log_file="logs/skill.log")
logger = get_logger("algos")
logger.info("Skill training started")
```

日志同时输出到控制台（stderr）和文件，包含时间、级别、logger 名称和消息。
文件以 UTF-8 追加写入，自动创建父目录；省略 `log_file` 时仅输出到控制台。
`logger.exception(...)` 可记录异常堆栈。包内子模块可使用相对导入
`from ..logging import get_logger`，并通过 `get_logger(__name__)` 获取 logger。

导入模块不会配置日志。配置仅作用于 `mindmemos_skill` 命名空间，不修改根 logger。
重复配置会关闭并替换本模块创建的处理器，保留调用方自行添加的处理器。
请在启动工作线程前完成配置；当前不支持并发重配置、日志轮转或多进程共享文件管理。

## 服务组装

最小配置见仓库中的 `config/mindmemos_skill/default.yaml`。在仓库根目录安装本包后：

服务组装、所有 pipeline 业务操作（含仓库及图查询）、pipeline 与服务关闭均为异步接口。以下示例中的 `await` 应放在异步函数内；可用 `async with await create_service(...) as service:` 自动关闭资源，或显式 `await service.close()`。仅进程入口使用 `asyncio.run(...)`；构造函数和配置校验仍为同步。

```python
from mindmemos_skill import create_service
service = await create_service("config/mindmemos_skill/default.yaml")
```

也可以直接传入配置类，无需生成 YAML：

```python
from mindmemos_skill import ServiceConfig, create_service

service = await create_service(ServiceConfig())
```

两种输入均返回实现 `MindMemOSSkill` 的 `SkillService`。显式空配置创建无存储、无 pipeline 的服务；省略 `create_service()` 的配置参数则使用本地工作区默认仓库。

配置只包含一个 `persistence`（数据库直接定义在 `persistence.database`）、一个可选 `repository_pipeline`，以及 `training_pipelines`、`evolution_pipelines` 两个名称到完整定义的字典。参见 [默认配置](../../config/mindmemos_skill/default.yaml) 与 [接口及组装说明](../../docs/mindmemos_skill/central_repository_api.md)。

每个 pipeline 的 `config` 就地包含自己的 algorithm、environment、agent 等定义；允许重复配置，不使用全局资源名引用。工厂构造一个 `SkillPersistence`，以固定关键字 `persistence` 注入每个 pipeline。具体 pipeline 的构造函数接收 `config` 和 `persistence`，其 `config_type` 使用明确的嵌套模型校验组件设置，再通过各组件已有注册机制构造自己的实例。

`class_name` 与注册名称一致；扩展模块需先显式导入。工厂校验 pipeline 类型、配置及构造参数后再创建数据库；不会扫描候选或补装仓库。Pipeline 必须自行校验其内部实现选择，禁止把存储传给 trainer/evolver。

`service.backends` 仅用于诊断，固定键为 `database`、`persistence`、`repository_pipeline`、`training_pipelines.<名称>` 和 `evolution_pipelines.<名称>`，不是 YAML 引用表。关闭服务先逆序调用各 pipeline 的 `close()`，最后关闭数据库。Pipeline 负责释放自己的组件及构造失败时的部分资源，不关闭全局 persistence。

验证：在仓库根目录执行 `.venv/bin/python -m pytest tests/mindmemos_skill -q`。

## FastAPI 接入

安装 HTTP 接入的可选依赖，再通过 Makefile 启动：

```bash
uv pip install -e './src/mindmemos_skill[fastapi]'
make skill-fastapi
```

Python 直接调用与 HTTP 接入使用同一份 `config/mindmemos_skill/default.yaml`，默认配置 SQLite 中心仓。

HTTP 包装函数位于 `api/entries/fastapi.py`，接收已构建的服务，不重复组装后端：

```python
from mindmemos_skill import ServiceConfig, create_service
from mindmemos_skill.api.entries.fastapi import create_app

service = await create_service(ServiceConfig())
app = create_app(service)
assert app.state.service is service
```

`api/service.py` 承担服务组装与业务接口实现，`entries/local.py` 负责直接组装，
`entries/fastapi.py` 负责添加路由及启动 HTTP 服务。

启动后访问 `http://127.0.0.1:8000/docs` 查看接口，或使用：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/skills/list -H 'Content-Type: application/json' -d '{"project_id":"demo"}'
```

| HTTP 路由 | 对应本地方法 |
| --- | --- |
| `POST /skills/register` | `register_skill()` |
| `POST /skills/delete` | `delete_skill()` |
| `POST /skills/auto-evolve` | `auto_evolve_skill()` |
| `POST /skills/auto-train` | `auto_train_skill()` |

`GET /health` 返回 `{"status":"ok"}`，仅表示应用可访问。中心仓业务接口接收 JSON 请求体，
完整路由与参数见[中心仓 API 说明](../../docs/mindmemos_skill/central_repository_api.md)。
只有自动演化与训练仍返回 HTTP 501。HTTP 启动入口显式初始化表，create_app 本身不初始化。

统一通过仓库根目录的 Makefile 启动，可覆盖配置文件、监听地址和端口：

```bash
make skill-local
make skill-fastapi SKILL_CONFIG=config/mindmemos_skill/default.yaml SKILL_HOST=127.0.0.1 SKILL_PORT=8001
```

`make skill-local` 组装并打印本地服务后退出；`make skill-fastapi` 持续运行 HTTP 服务，
按 Ctrl-C 停止。默认使用 `.venv/bin/python`，可通过 `SKILL_PYTHON` 指定解释器。
Makefile 会加入当前仓库的 Skill 源码路径。

具体启动逻辑分别位于 `mindmemos_skill/api/entries/local.py` 和
`mindmemos_skill/api/entries/fastapi.py`。直接组装服务只需基础依赖，不会导入 FastAPI 或 Uvicorn。
`run_skill.py` 仅保留为个人调试脚本；Python 集成仍可直接调用 `create_service`。

## 数据库基础设施

`infra` 已提供 SQLite + sqlite-vec 默认后端，以及可选的 PostgreSQL + pgvector 后端。
业务在 `persistence` 中定义并集中注册表结构和 Pydantic 校验模型，随后通过表名调用
统一的增删改查、upsert 和向量检索接口。具体后端适配位于 `infra/database/builtin_backends/`。
注册不连接数据库，建表需要显式调用，服务退出时关闭数据库连接池。

完整示例、操作契约和测试方法见 [infra 使用说明](mindmemos_skill/infra/database/README.md)。

## Skill 内容与存取

所有可优化内容都属于 `SkillBundle.files`。首版与旧 SDK 一致，仅管理根目录的
`SKILL.md`，统一换行后使用规范 JSON 与 SHA-256。完整 Skill 还保留 `SkillResources`，
包含 references、scripts、assets 等辅助文件的原始字节；它们不参与动态组装或优化。

```python
skill = Skill.from_path("skills/example")  # Also accepts SKILL.md.
skill.export("output/source")  # Exports SKILL.md and unchanged auxiliary files.
prepared = skill.assemble(task)
prepared.export("output/task-skill")
candidate = skill.with_bundle(SkillBundle(files={"SKILL.md": "Revised instructions"}))
```

`Skill` 基类默认对所有 task 返回内容相同的完整 Skill 副本，可直接使用。
`Skill(bundle=..., resources=...)` 是数据对象，不需要空配置，不参与注册或 YAML 后端构造。
动态能力由具体算法直接继承 `Skill` 并覆盖
`assemble(task) -> Skill`。固定参数放在对应 `config.py` 中；模板、片段和
可优化规则放在源 bundle 内。组装不得修改源内容或 task，也不接收私有评分数据。
`Skill.export(directory, overwrite=False)` 保留完整目录布局；非空目标需要显式指定
`overwrite=True`，覆盖时替换整个目录并清除旧文件。`SkillBundle.export` 只导出 bundle。导出的文件不包含
实现配置；恢复动态实现需要显式配置或保存的 `SkillSnapshot`。

`snapshot()` 保存源内容、辅助文件和已有实现元数据，`Skill.from_snapshot(...)` 只恢复数据，
不根据元数据加载代码或创建运行时依赖；
`with_bundle(...)` 保留实现类型和固定配置，创建新候选，不修改原对象。
`persistence.skills.add_skill(skill, record)` 保留 record 的身份、版本关系和其他元数据，
用完整 Skill 提供的 bundle、resources、哈希及实现配置写入新记录；
`get_skill(version_id)` 恢复完整 Skill。

rollout 的 `audit/request.json` 保存源快照、实际组装后的 bundle 及各自哈希。
普通重试复用已保存的组装结果，已完成的恢复不触发组装或执行；`resume=False`
重新选择、组装并执行。具有完整 task、reward_data 和源快照的旧静态列表记录可恢复；
缺少这些字段的更早请求格式仍须使用新的 work_dir。
源 Skill 版本不会被任务组装结果覆盖。辅助文件单独计入本地快照哈希，
`content_hash` 始终仅对应 bundle；辅助文件变化也会使旧 rollout 输入不匹配。

## TreeSkill 的动态组装

实现位于 `algos/train/treeskill/`，由算法持有，不放入通用 Skill 模块。
该模块实现 tree-only 任务路由；尚不包含训练、进化或候选选择循环。
TreeSkill 由调用方显式构造，导入与构造均不会发起模型请求。
需要恢复路由行为时显式调用 `TreeSkill.from_snapshot(...)`；持久化层的 `get_skill()` 只返回源数据。

```python
from mindmemos_skill.agents import OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkill, TreeSkillConfig
from mindmemos_skill.skill import SkillSet

skill = TreeSkill.from_path(
    "skills/xlsx",
    config=TreeSkillConfig(
        router=OpenAIChatConfig(model="your-model", temperature=0, max_completion_tokens=512),
    ),
)
# Passing this Skill to env.rollout triggers one assembly for that task.
result = env.rollout(task, work_dir, reward_data=reward_data, skill_set=SkillSet({"skill": skill}, max_concurrency=4))
```

流程为：解析源 `SKILL.md` 标题树 → 用公开 task 与完整树请求模型选择子树 →
去除父子重复选择 → 保留选中子树全文与祖先引导内容 → 返回携带本次任务 bundle 的完整 Skill。
代码块中的标题不参与树结构；不设置 top-k 截断；辅助文件保持不变。
源 bundle 每次重新解析，因此 `with_bundle` 更换候选后不会沿用旧树。

有效的空选择产生空指令；无法解析或包含未知 ID 的响应会记录警告并返回完整源 bundle。
网络、供应商和模型执行错误直接抛出，避免把调用失败当成有效路由。
没有可路由标题的 Skill 直接使用源 bundle，不调用模型。

路由由私有方法 `_route(task)` 实现，内部结果包含选中节点、祖先、fallback 原因、
模型原始响应与 usage。对外统一调用 `assemble(task)` 获取完整 Skill；当前路由 usage 不自动计入
环境 agent 的 `RolloutResult.usage`，不要将其当作包含路由开销的总费用。

路由语义与提示词参考本地 TreeSkill 原仓库的
`spreadsheet_agent/tree_only_skill_routing.py`，版本
`bc3b0a9d55870d8ad77c38839c4fa5baf111e638`。实现使用当前 `Task` 的全部公开字段，
不读取私有 reward 数据，不从文件路径推断评分材料；不会复刻原实现对供应商异常的静默回退。

## 本地 CLI harness

`agents/` 还提供 `codex`、`claude` 和 `deepseek_harness` 三个注册实现，
分别调用已安装的 `codex`、`claude` 和 `dsh`。构建与导入不会启动 CLI。
三个实现使用同步 `run(system=..., user=..., workspace=...) -> AgentResult`，
`trajectory` 保留原生事件，`usage` 保存可获得的用量；DeepSeek 当前未解析用量，返回空字典。
调用失败、缺少完成事件或空输出会抛出异常。

```python
from pathlib import Path
from mindmemos_skill.agents import Agent

agent = Agent.build("codex", {"timeout_s": 300})
result = agent.run(system="Solve the task using the workspace files.",
                   user="Summarize report.txt.", workspace=Path("/absolute/task/workspace"))
print(result.output)
```

调用方准备已有的独立工作目录，并使用当前 `Skill.export(directory)` 将已组装的 Skill
分别导出到 `.agents/skills/<name>`、`.claude/skills/<name>` 或 `.dsh/skills/<name>`。
harness 不负责组装或评分，也不接受旧版 request、Skill binding 或 callable Tool 协议；
工具执行由对应 CLI 管理。现有环境若需要逐轮工具回调，应继续使用其对应 Agent 接口。

配置使用 `cli_path`、`model`、`timeout_s`、`extra_args`；
`api_key` / `base_url` 直接传值；未提供的字段在构建配置时分别读取
`MINDMEMOS_AGENT_API_KEY` / `MINDMEMOS_AGENT_BASE_URL`。密钥在 JSON 快照中脱敏。
两者都未配置时，CLI 沿用原生认证与配置。
Codex / Claude 支持 `reasoning_effort`，Claude 另支持 `max_turns` 和 `max_completion_tokens`，
DeepSeek 支持 `profile` / `provider`（默认 `deepseek-official`），每次调用保留独立会话目录，
支持 v0（dsh 0.1.0-rc.6）和 v2 session 格式；其他格式会明确报错。
Claude 从 config 或统一环境变量获得凭据或端点时，会禁用 user/project/local settings 来源，
防止其中的 `env` 覆盖本次选择；这些来源中的其他设置也不会用于本次调用。

自定义网关必须支持对应原生协议：Codex 使用 Responses，Claude 使用 Anthropic Messages，
DeepSeek 使用 Chat Completions。对于提供 `/v1/messages` 的网关，Claude 填写网关根 URL；
Codex 和 DeepSeek 通常填写包含 `/v1` 的 base URL。

## Skill Manage 动态集合

`SkillManageSkillSet` 使用调用方提供的 Skill 和管理算法选择成员，再继承集合的并行组装。版本绑定由外围调度交给 `SkillGraphStore`，集合不解析版本。规则连边、显式 Scene、LLM 连边、查询改写、Embedding 和图快照持久化的使用方式见 [Skill Manage 说明](../../docs/mindmemos_skill/skill_manage.md)。
