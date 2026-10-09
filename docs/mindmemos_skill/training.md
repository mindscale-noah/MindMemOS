# 通用 Skill 训练流程

`training` 是一个通用 TrainingPipeline 实现，配置中的 `trainer.class_name` 可选
`treeskill` 或 `skill_forest`，也可选择显式注册的其他 `SkillTrainer`。Pipeline 不按算法名
分支，不读取算法检查点，也不传递 `resume` 参数。

服务启动配置中的 `training_pipelines` 配置 Trainer。环境属于单次训练，在
`TrainingRequest.env` 中提供 `class_name/config`；Service 不维护环境预设。
`algo` 选择训练流水线，`env.class_name` 选择环境实现。环境参数由该实现自己的配置类校验，
Service 在执行训练时通过 `Environment.build()` 创建环境，再传给 Pipeline。
Python 服务入口可等待训练完成；配置任务管理器后，也可提交后台任务。FastAPI 训练路由提交后台任务。

## 调用与数据划分

```python
from mindmemos_skill import create_service
from mindmemos_skill.api.typing import TrainingRequest, TrainingSplit

async with await create_service("/path/to/your/service.yaml") as service:
    result = await service.run_training(
        TrainingRequest(
            algo="skill_forest",  # Change to "treeskill" to use the same business flow.
            env={"class_name": "searchqa", "config": {"agent": {"model": "YOUR_MODEL"}, "max_turns": 1}},
            project_id="my-project",
            init_skill_version_id=existing_version_id,
            work_dir="output/training/run-001",
            tasks=materialized_samples,
            split=TrainingSplit(train=train_ids, val=val_ids, test=test_ids),
        )
    )
```

`tasks` 使用公共 `DataSample`，包含公开任务和独立私有评分输入。所有任务必须且只能分配到
train/val/test 的一个集合。Task、RewardData 和环境配置只提供业务数据与文件路径，
无需传入 `sha256`、`source_sha256`、文件 hash 字典或 `evaluator_sha256`。系统自动计算外部
输入指纹并保存到私有恢复记录；文件内容变化后需使用新的工作目录。请求必须提供工作目录；相对路径以服务进程工作目录为基准。
Python 调用会等待训练及发布完成。调用方须准备完整 `DataSample` 与 train/val/test 划分。
`auto_accept` 默认是 `True`，发布后把训练结果设为当前生效版本；传入
`auto_accept=False` 时仍保存并返回新版本 ID，但当前生效版本保持不变，可稍后显式切换。
同一 `work_dir` 的重试必须保持该选项一致，否则会被视为不同输入。
HTTP 异步训练请求也包含私有评分输入；服务按选定 Environment 的 task 与 reward_data
模型解析，并在后台任务中保存这些输入，因此应只向受信任的本地训练服务提交。
自定义 Environment 若用于序列化的后台任务，需要声明 `task_type` 和 `reward_data_type`。

HTTP、SDK 和 Python 统一使用 `algo`、`env`、`init_skill_version_id`，不接受旧的
`pipeline`、`environment`、`base_version_id` 训练请求字段。后台任务保存请求中的环境配置，Worker 根据该配置恢复环境和样本类型；
任务受理后若服务中的算法配置发生变化，Worker 会拒绝按新算法配置执行该任务。旧格式检查点和旧格式本地任务不提供兼容恢复；
新格式输入相同、目录相同的任务可安全续跑。

成功发布后返回 `skill_id`、`base_version_id`、`version_id`、`run_id`、原始 `content`、
Base64 `resources`、`content_hash` 和 `skill_hash`。返回的父版本是请求固定的
`init_skill_version_id`，可训练历史版本；发布仍检查训练期间 active 指针是否发生变化。

Pipeline 校验输入版本的项目归属，读取完整快照，将 train 与 val 交给 Trainer，
始终将 test 留在 Trainer 外。最终结果冻结后，Pipeline 对非空 val/test 各评估一次并记录指标，
不根据它们的分数再次选优、修改结果或决定版本是否发布。执行异常阻止发布，不伪装为零分。
TreeSkill 当前不使用外部 val 优化；Forest 在 train 池内部为每棵树划分训练和验证，外部 val
也不参与森林选优。切换其他 Trainer 时，其内部 val 语义由该算法定义。

Forest 当前只接受静态源 Skill，避免把动态实现元数据当作已恢复的运行时行为。
TreeSkill 从源正文训练并返回带路由配置的 TreeSkill，最终评估使用该具体对象；快照入库后保留
其 implementation、配置和 resources。训练与 UI 使用 `common.skill_runtime.load_runtime_skill`
恢复动态实现；未知实现明确报错。源快照保持完整原文，任务选中正文可以为空，标记为非标准格式。
环境保存组装后的完整快照及 `skill_hash`，中断重试复用该快照与原运行依赖，不重复路由。评估使用独立工作区和 `evaluation_concurrency` 限制。

## 自动恢复归属 Trainer

再次调用同一算法、相同输入和相同 `work_dir` 即自动复用兼容工作，没有恢复开关。
Pipeline 始终把 `<work_dir>/train` 传给 Trainer；是否恢复、恢复哪些步骤由 Trainer 自行决定。

- 空目录或不存在的目录：新建训练。
- 兼容且未完成：复用已提交的调用结果，继续未完成工作。
- 兼容且已完成：返回已保存的最终 Skill，TreeSkill 恢复具体动态类及路由依赖。
- 输入、算法、配置、源码指纹变化，或检查点损坏：报错并保留目录。
- 只有旧审计文件、缺少恢复协议的历史目录：报错，需使用新目录，不能直接覆盖。

TreeSkill 复用已提交的采样成功/失败结果和模型响应，重新执行确定性的分析/融合编排；
终态失败不会因恢复获得额外随机采样机会。未提交完成记录的物理请求可能重跑，旧 attempt
保留。Forest 还复用候选生成、验证、波次与集成边界。两者均不恢复环境内部的半条交互轨迹，
也不保证远端模型计费请求 exactly-once。运行目录锁阻止同时写入。

`request.json` 绑定业务输入和配置，`selected_skill.json` 保存发布前的完整候选；
`val/`、`test/` 是独立评估产物，`publication.json` 是成功发布的结果。数据库提交失败时
候选保留；再次调用时 Trainer 与评估函数各自复用兼容结果。

## 发布与既有数据库兼容

发布使用请求指定的父版本，训练期间不持有数据库事务。`auto_accept=True` 时，提交在短事务内
核验当前版本；其他写入推进了当前版本时，发布失败并保留候选，不自动覆盖。
`auto_accept=False` 时仅保存新版本，不更新当前指针。版本标签由持久层分配，算法
后缀为 `treeskill` 或 `skill-forest`，注册名中的下划线统一转换成标签允许的连字符。

同一项目与工作目录产生稳定 run_id。发布版本使用确定性内部 ID，并在已有
`local_metadata.training` 字段记录 run_id 和输入指纹；版本插入、标签分配与当前指针更新
处于同一事务。即使提交成功后进程中断、调用方未收到结果，重试也返回原版本，不新增版本。
如果之后用户已切换或提交其他版本，重试旧请求不改变当前指针。相同 run_id 配不同输入明确冲突。

此功能不新增表或列，不修改历史记录、不更改商用云服务数据契约；无需数据回填或数据库重建。
部署继续使用原数据库，按既有方式初始化缺失表和版本标签；旧代码可以继续读取新版本。
回退服务代码不需要回退数据。只需将原本未启用的训练流程配置为 `training`，并在训练请求中
增加 `work_dir`；未配置训练的服务行为保持不变。

## 不调用 LLM 的 CLI 演示

在仓库根目录执行以下命令，可自动启动独立的本地 HTTP 服务，再用真实 CLI 子进程完成验收：

```bash
.venv/bin/python scripts/skill/demo/verify.py
```

默认每个训练样本暂停 1 秒，便于观察进度。快速验收可使用
`--delay-seconds 0.1`。`--output PATH` 指定一个新的空输出目录；省略时创建
`output/skill_pipeline_demo/<UTC时间>-<随机后缀>/`。脚本结束会关闭服务，保留数据库和审计文件。
每次验收使用独立 SDK home，不覆盖个人认证或原有 Skill 数据库。

演示通过显式导入 `mindmemos_skill.demos` 注册，普通服务不会自动注册这些后端。
验收脚本在自己的输出目录中生成 `service.yaml`，不再维护单独的演示部署模板。该配置使用以下演示后端：

| 流程 | 演示后端 | 实际执行的行为 |
| --- | --- | --- |
| train | `training` + `demo_train`，环境 `demo_offline` | 每个训练样本产生合成轨迹及固定评分；追加 `Demo training` 正文；真实保存候选、评估 val/test、发布版本 |
| evolve | `simple_evolution` + `demo_evolve` | 从真实数据库收集和采样历史轨迹；追加 `Demo evolution` 正文、轨迹数量及任务 ID；真实发布版本 |
| graph | `demo_graph` | 将给定 Skill 版本组织成节点链；真实发布不可变快照；用词语重合度提供确定性推荐 |

所有演示后端均不构造 LLM 或 embedding 客户端，模拟评分不能作为算法效果指标。
train 和 evolve 通过原有 pipeline 发布，保留资源、父版本和版本标签分配。
CLI 不传 `--auto-accept` 时保存新版本但保持当前版本；传入时新版本成为当前版本。

一键验收覆盖以下行为：

- train、evolve 分别启用和不启用 `--auto-accept`，核对当前版本、正文和资源。
- 训练 3 个样本，外部 val/test 各评估 1 个样本；相同目录和精确初始版本的重试复用原发布。
- `task show`、单次 `task progress`、`task progress --follow`，以及 `--after` 排除已读事件；事件序号连续、follow 不重复且终态退出。
- 空轨迹和只有 2 条轨迹时，evolve 在 `sample_size=3` 下失败且不增加版本。
- 上报 10 条轨迹：目标 Skill 有 7 条 completed、1 条 failed，另一 Skill 有 2 条 completed；状态和家族过滤正确。
- 演进按选定版本收集，`traj_collect.trajectory_limit=5` 限定收集数量，再以 `traj_collect.seed=42` 无放回采样 3 条；两次采样一致，排除其他版本的轨迹。
- 图发布、列表、解析、历史、精确快照读取、推荐和重复发布；2 个 Skill 节点、1 条边，相同图成员的重复发布复用快照。

`summary.json` 保存验收结果；`commands.jsonl` 保存每次命令的参数、输入、输出和退出码；
`server.log` 保存服务日志。`train-false/`、`train-true/` 保存训练和评估产物。
`evolution/<run_id>/input.json` 保存 collected / eligible / sampled 数量、采样轨迹 ID
和收集参数，`candidate.json` 保存修改后的 Skill。数量含义分别是保留的匹配记录数、
具有可用任务与历史版本绑定的记录数，以及最终交给 evolver 的记录数；collected 不表示扫描行数。

### 手动运行命令

一个终端启动服务：

```bash
.venv/bin/python scripts/skill/demo/service.py --port 18765
```

该入口只监听 `127.0.0.1`，数据库默认保存到
`output/skill_pipeline_demo/manual/central.db`。本地 task 状态和进度在服务进程内保存，
服务重启后不会保留 task ID；版本和轨迹仍保存在 SQLite 中。

另一个终端在仓库根目录设置独立的 SDK home，并连接本地服务。`demo-local` 是本地演示占位值：

```bash
export MINDMEMOS_HOME="$PWD/output/skill_pipeline_demo/manual/sdk-home"
.venv/bin/mindmemos auth --service-type custom \
  --memory-base-url http://127.0.0.1:18765 --memory-api-key demo-local \
  --skill-shares-memory --user-id demo-user

.venv/bin/mindmemos skill register \
  resources/mindmemos_skill/examples/demo-code-review --alias pipeline-demo

.venv/bin/mindmemos skill train pipeline-demo \
  --pipeline demo --environment-json '{"class_name":"demo_offline","config":{"delay_seconds":0.2}}' \
  --work-dir output/skill_pipeline_demo/manual/train-review \
  --sample-json '{"task":{"id":"t1","instruction":"Review the demo output"},"reward_data":{}}' \
  --train-id t1
```

train 响应中的 `task_id` 用于以下命令；终态 `task show` 的 `result.version_id`
用于读取新版本。要测试自动启用，在训练命令末尾添加 `--auto-accept` 并使用新的工作目录。

```bash
.venv/bin/mindmemos skill task progress TASK_ID --follow
.venv/bin/mindmemos skill task progress TASK_ID --after 1
.venv/bin/mindmemos skill task show TASK_ID
.venv/bin/mindmemos skill show NEW_VERSION_ID
.venv/bin/mindmemos skill show pipeline-demo
```

演进前执行 `skill resolve pipeline-demo`，将返回的精确 `skill_version_id` 放入下列
`SKILL_VERSION_ID`，上报至少 3 条 completed 轨迹：

```bash
for index in 1 2 3 4 5 6 7; do
  .venv/bin/mindmemos skill trajectory report --json-data \
    "{\"user_id\":\"demo-user\",\"agent_id\":\"demo-agent\",\"skill_version_id\":\"SKILL_VERSION_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Demo task $index\"},{\"role\":\"assistant\",\"content\":\"Synthetic output\"}],\"status\":\"completed\"}"
done

.venv/bin/mindmemos skill trajectory list pipeline-demo --status completed
.venv/bin/mindmemos skill evolve pipeline-demo
.venv/bin/mindmemos skill evolve pipeline-demo --auto-accept
```

每次 evolve 提交后先跟踪对应 task 到终态，再开始下一次演进。进度 `evidence` 显示收集和合格数量，
`sampled` 显示交给算法的轨迹数量，之后为 `evolved`、`published`。
演进与 CLI 的 `trajectory list SKILL_REF` 均只收集选定版本的轨迹，不混入同一 Skill 其他版本的轨迹。

图流程可直接使用同一个 Skill，或再注册另一个 Skill 后增加第二个 `--skill`：

```bash
.venv/bin/mindmemos skill graph publish --alias pipeline-demo-graph --skill pipeline-demo
.venv/bin/mindmemos skill graph show pipeline-demo-graph
.venv/bin/mindmemos skill graph history pipeline-demo-graph
.venv/bin/mindmemos skill graph recommend "Review output" --by pipeline-demo-graph --top-k 1
```

当前 graph HTTP 入口同步返回快照 ID，没有异步 task ID；`task progress` 用于 train/evolve。
修改 `traj_collect` 中的 `sample_size`、`trajectory_limit`、`max_page_size` 或 `seed` 后，
重启演示服务即可测试其他收集策略。收集按页进行，直到达到 `trajectory_limit` 或没有更多数据库记录。

## 配置字段与可选编辑器提示

服务配置示例位于 `config/mindmemos_skill/dev.example.yaml`。算法参数填写在所选
`trainer.class_name` 同层的 `config` 中，例如：

```yaml
service:
  home: ./state
  persistence:
    database:
      class_name: sqlite
  repository_pipeline:
    class_name: repository
  training_pipelines:
    forest:
      class_name: training
      config:
        trainer:
          class_name: skill_forest
          config:
            optimizer:
              model: YOUR_MODEL
            num_trees: 1
            expansion_budget: 1
            rollout_concurrency: 2
            model_concurrency: 1
fastapi:
  auth:
    mode: none
    default_project_id: personal
```

算法字段只定义在对应算法目录的 `config.py`。Schema 生成器读取注册实现的 `config_type`，引用这些原始模型及其嵌套模型。字段后的英文说明同时用于 Schema，无需重复维护参数说明。服务预检后，`config` 保存具体配置对象，例如 `SkillForestTrainerConfig`，并保留嵌套凭据的脱敏序列化。

仓库不维护生成的服务配置 Schema。需要编辑器提示时，可自行生成并关联到 YAML：

```bash
mindmemos-skill schema --output /path/to/run/service.schema.json
```

Schema 生成只读取模型和注册信息，不需要部署配置，不构造服务资源。生成后可通过 YAML 首行的 `$schema` 注释或编辑器设置关联该文件。

用户扩展仍使用显式导入和装饰器注册。入口导入扩展模块后调用 `skill_config_schema()`，返回的 Schema 会包含已注册扩展及其原始配置模型。生成后不再自动更新，修改或新增配置类时须重新生成。

### 每次训练提供环境配置

`TrainingRequest.env` 是可序列化的环境定义，不再接受环境预设名称。Python、HTTP 和后台任务
使用相同结构；Pipeline 和 Trainer 仍接收已经创建的 `Environment` 对象。

```yaml
env:
  class_name: searchqa
  config:
    agent:
      model: YOUR_MODEL
      api_key: YOUR_API_KEY
      base_url: https://YOUR_ENDPOINT/v1
    max_turns: 1
```

需要编辑器提示时，可在自己的运行目录生成训练请求 Schema，并将请求 YAML 的
`$schema` 指向生成文件。Schema 包含所选环境自己的字段、说明、默认值和限制。
更新环境配置类后重新生成：

```bash
mindmemos-skill schema --kind training-request \
  --output /path/to/run/request.schema.json
```

环境配置文件可以用 `# yaml-language-server: $schema=./environment.schema.json` 关联提示。
用 `mindmemos-skill schema --kind environment --output /path/to/run/environment.schema.json` 生成或更新它。

远程 CLI 使用 `--environment PATH` 读取仅包含 `class_name/config` 的 YAML 或 JSON 文件，
也可以用 `--environment-json '{"class_name":"searchqa","config":{...}}'` 提供同一结构。

后台任务的凭据传递暂未完成。当前带凭据的环境配置使用 Python `run_training()` 直接执行；
后台入口不会把普通 JSON 中的脱敏凭据当成有效凭据执行。

## 异步任务的配置与恢复

模型连接参数由调用方通过 `api_key`、`base_url` 显式提供。API key 使用普通字符串，与其他配置一起原样序列化、持久化和恢复；任务请求、算法配置文件及动态 Skill 快照不删除 key，也不保存掩码或要求恢复时重新注入。模型配置不从进程环境隐式补齐。

训练任务保存完整请求（含 env 配置、样本和 split）及服务端 pipeline 配置指纹。worker 使用匹配的训练 pipeline 配置执行；服务端配置发生变化时明确失败。

演进任务在提交时保存完整 pipeline 配置，并分配稳定的 `run_id`。worker 从任务配置构造独立的演进 pipeline，执行后关闭它。内置 `simple_evolution` 在运行目录保存原始父版本、采样证据和候选结果，重试复用已保存的数据。发布按项目、操作类型和 run ID 分配稳定版本 ID；版本提交成功但任务状态未写回时，接管执行直接返回已有结果，不创建第二个版本，也不改变后续已选择的版本。

Kafka 模式的任务请求保存在服务数据库中；本地模式仍只保存在当前进程内。跨 worker 恢复中间结果需要相同的工作目录配置和共享文件存储。任务管理器只接管 lease 过期的运行任务，普通算法异常仍标记为 failed，不自动重试。

### Local 与 Kafka 的统一调度

两种模式共用进程内 `TaskDispatcher`：`max_concurrency` 限制完整任务执行的并发数，`max_buffered` 限制已接纳但未结束的任务总数（包含排队和正在执行的任务）。默认分别为 2 和 1000；这些限制属于单个 manager 实例，不是整个部署集群的总额度，也不限制算法内部 rollout 或模型调用的并发。

local 提交在缓冲区满时等待容量；调用方取消等待时，该次未接纳的提交会清理。Kafka 消费与过期任务扫描共用同一个 dispatcher，相同 task ID 在排队或运行期间合并通知。Kafka 缓冲区满时不阻塞消费循环：数据库任务仍由后续通知或恢复扫描再次调度。只有获得执行名额后才 claim 数据库任务，名额持续持有至执行、结果写入和清理完成。

关闭 manager 会停止接收、唤醒等待提交的调用方并取消排队和运行中的工作。local 将已接纳但未完成的任务标记为 cancelled；Kafka 未领取的任务保持原状态，已领取而中断的任务等待 lease 过期后接管。local 仍使用内存状态，Kafka 仍使用数据库状态。


## 无监督历史轨迹演进

具体实现、配置、提示词和裁剪逻辑位于 `algos/evolve/simple_evolve/`；`algos/evolve/` 仅保留公共基类、轨迹类型和包入口，其他演进算法使用独立子目录。

内置 `simple_evolution` / `simple_evolve` 先在 pipeline 中将 Skill 家族引用解析为 active 版本；指定 `base_version_id` 时使用该版本。轨迹查询只接受 `skill_version_id`，直接在数据库 filters 中按该版本筛选，不按家族或绑定 JSON 扫描。再从合格池无放回随机采样。采集和采样配置集中于 `TrajCollectConfig`，通过 `traj_collect` 设置。默认 `sample_size=40`；不足 40 条时任务失败，可显式降低采样量。默认候选池最多保留最近 100 条匹配轨迹，`max_page_size=200` 限制每页读取数量；实际每页最多读取剩余所需数量，持续查询直到收集满或没有更多记录；时间范围、状态和来源不在调用时传入，使用 `collect_skill_trajectories` 的默认值，均不限制。采集不设置单条或总字节上限，也不按记录大小跳过轨迹；消息裁剪仅在模型输入阶段执行。随机采样针对该受限池，不是整个数据库。

采样轨迹按 `batch_size=8` 分组，最多 `concurrency=5` 个模型调用并行提取 patch。最后不足 8 条也独立成组。每组直接输出 `reasoning + edits`，不按成功、失败分组，不向模型提供 reward、status 或数值评分。目标优先级为避免重复错误，其次减少无效操作。

消息文本按角色保留头尾、从中间裁剪，并标记省略：user/system 默认 60000 字符，assistant 默认 12000 字符，tool/其他观察默认 6000 字符。结构化消息的文本字段共享该消息预算；JSON 结构和顶层角色、消息标识保留。任务说明和系统提示使用 user 的预算；历史 Skill 原文不裁剪，超过 `max_skill_chars` 时失败。裁剪只影响模型输入，不修改数据库轨迹。没有额外整条轨迹截断，避免删除用户要求或尾部反馈。

各批 patch 两两并行合并，奇数个时最后一个直接进入下一层，直到剩下一个 patch。40 条默认形成 5 个批次，合并层为 5 → 3 → 2 → 1。所有提取与合并均基于同一份原始 Skill；每次模型返回都检查修改数量和精确文本目标，最终仅应用一次。默认每个 patch 最多 5 项修改。

没有证据支持修改时允许空 patch：返回原版本，报告 `unchanged`，不发布新版本、不改变当前选择。采样证据和最终候选继续按 run ID 保存，中断恢复复用原样本；最终候选生成前的中断会重新执行提取和合并。旧 `max_trajectory_chars` 配置已替换为上述三个角色预算。
