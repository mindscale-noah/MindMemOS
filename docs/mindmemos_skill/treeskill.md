# TreeSkill 单轮主动训练与独立实验

本实现参考 `mindscale-noah/MindMemOS` 的 `feat/treeskill-integration`，固定提交为 `e83aae887f08eb9fa1596384404dc7757a6a7f2f`。算法位于 `mindmemos_skill.algos.train.treeskill`，注册名为 `treeskill`。无需 `mindmemos_eval`、Service 或数据库。

## 200/200 baseline 与演进后对比

`scripts.treeskill.compare` 直接调用本包环境与 Trainer，依次执行初始完整 Skill 的 200 题测试、200 题训练和最终 TreeSkill 的同一组 200 题测试。测试轨迹不进入训练。模型连接与算法设置读取 `config/mindmemos_skill/dev.yaml` 的 TreeSkill pipeline；本入口不连接 HTTP 服务或数据库。

```bash
PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python -u -m scripts.treeskill.compare \
  --data-root /path/to/spreadsheetbench_verified_400 \
  --output output/treeskill_comparison --preflight

PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python -u -m scripts.treeskill.compare \
  --data-root /path/to/spreadsheetbench_verified_400 \
  --output output/treeskill_comparison
```

先将 LibreOffice 程序目录加入 PATH，并验证真实公式重算。`--preflight` 检查固定划分、环境和实际模型响应，不执行正式评测。默认输入为 `resources/mindmemos_skill_eval/treeskill-spreadsheetbench/` 下的 split 和初始 Skill。

当前对比配置使用 Qwen3.5-35B-A3B、seed 41、16,384 completion tokens、600 秒请求超时和最多 10 次额外传输重试。执行 Agent 与失败分析使用 instruct 参数，温度分别为 0.7 和 1.0；成功分析、定位、融合及路由使用 thinking 参数，温度分别为 1.0、0、0、0。thinking 通过 `extra_body.chat_template_kwargs.enable_thinking` 设置，`reasoning_effort` 留空。执行 Agent 额外设置 `stop: ["Observation:"]`。执行及失败分析上限为 100 轮，采样、分析、定位和测试并发均为 16，融合顺序执行。16k 预算、并发和传输重试是本次指定的上游参数覆盖。

目录中的 `baseline/summary.json` 与 `evolved/summary.json` 保存同一测试集上的分数；总 `summary.json` 记录阶段状态及成功率差值。执行异常保留为失败项并阻止进入下一阶段，不能伪装成普通评分失败。同一输入与配置可复用完成的采样检查点；恢复前须检查失败原因及训练状态，不能将不完整或受 API 错误影响的训练结果直接当作完整实验。

路由模型调用或 JSON 解析失败时，按上游行为使用完整 Skill，原因保存于每个测试任务的 `routing/selection.json`。最终报告应列出回退数量。输出可能包含本地连接配置，应留在本地实验目录，不上传含凭据的文件。

## 算法与输入边界

`await TreeSkillTrainer.train(...)` 接收初始 `Skill`、调用方提供的 `Environment`、已加载的训练 `DataSample` 和实验目录。算法自己调用 `Environment.rollout()`，不接受外部轨迹作为训练入口，不修改输入 Skill 或辅助资源。

单轮执行顺序保持为：完整初始 Skill 采样 → 独立轨迹分析 → 对初始树定位证据 → 自底向上融合节点 → 返回最终可路由的 TreeSkill。不会在中间 patch 后重新采样，也不新增多轮训练或验证集选优。原始事件采用参考实现的逐事件截断，`transcript_max_chars` 不是整条轨迹的总字符上限。

通用分析器通过评分区分 success/error/unlabeled。所有定位都针对同一棵初始树；定位失败最多尝试两次，第二次 token 预算为 `min(initial * 2, 16384)`（本次 16k 覆盖；上游原上限为 8192），消息保持一致。融合按深度从叶到根、同深度按源文档位置排序，每个节点读取此前修改后的当前树；`update_node` 在 `create_child` 之前执行。保留原始稳定 ID、Markdown 结构校验、重复子节点校验、内容过滤及 reject 规则。

融合响应严格遵循固定远端提交的 `NodeFusionDecision`：整体和每个操作都必须有非空 `rationale`，只接受 `update_node`、`create_child`、`reject`。一个响应最多包含一次更新和一次新增子树，reject 不能与其他操作混用。未知字段、`update` 别名、缺失字段和截断 JSON 均记录为融合失败，不改写为模型主动 reject，也不修复 JSON 或新增阶段重试。有效响应按更新、再新增子树的顺序执行。单个子树的内容或结构错误不撤销已经通过的更新。更新允许清空正文，无内容叶节点随后不再进入路由集合；新建叶节点必须有非空白正文。

`await run_experiment(...)` 在训练结束后，才把最终 Skill 用于 test。每条 test 数据只请求一次 rollout；模型内部的传输及解析重试仍按模型配置执行。test 不进入 trainer、分析器、融合器或选优逻辑。val 和 test 均可省略或为空；当前不定义验证集选择行为，非空 val 会明确报错。

## 代码职责与远端对齐

主要入口只有 `trainer.py`、`treeskill.py` 和 `construction/`：

| 位置 | 职责 |
| --- | --- |
| `trainer.py` | 组织采样、分析、定位、融合、恢复和训练后测试。 |
| `treeskill.py` | `TreeSkill(Skill)` 的构造、快照和恢复；`assemble()` 路由后返回完整副本，测试和真实 rollout 共用。 |
| `construction/markdown.py` | 唯一的 Markdown 树解析、稳定节点身份、编辑、元数据检查和子树渲染。 |
| `construction/analysis.py` | 从完成的采样提取证据，执行通用轨迹分析。 |
| `construction/localization.py` | 对不可变初始树定位原子经验。 |
| `construction/fusion.py` | 严格校验响应，自底向上更新节点，过滤分析专用内容。 |
| `construction/prompts/` | 定位和融合的原始提示词资源。 |
| `artifacts.py` | 写入阶段产物和 patch，计算评分统计，记录及恢复物理模型请求。 |
| `env_adapter.py`、`env_specific/` | 环境提示词与专用分析；不维护另一套树或路由实现。 |
| `config.py`、`typing.py` | 配置与实际使用的数据结构。 |

分析、定位和融合是函数，不再通过独立阶段类转发调用。`construction/` 不依赖动态 TreeSkill。Trainer 将最终内容与元数据构造成动态 TreeSkill。环境通过统一 `SkillSet.assemble()` 调用它，无需 TreeSkill 专用 rollout 分支。

公共算法默认值与固定远端提交一致：

| 参数 | 默认值 |
| --- | --- |
| 最少 / 最多轨迹 | 1 / 1000 |
| 单事件截断上限 | 20000 字符 |
| 评分注释 / 初始 Skill 检查 | required / true |
| 成功评分阈值 | 1.0 |
| 分析 / 定位并发 | 16 / 16 |
| 分析 / 定位 / 融合温度 | 1.0 / 0.0 / 0.0 |
| 分析 / 定位 / 融合完成预算 | 4096 / 2048 / 4096 |
| 路由温度 / 完成预算 | 0.0 / 512 |

`routing_temperature` 和 `routing_max_tokens` 是 Trainer 的阶段参数。它们覆盖最终路由 Agent 的生成参数，并随最终 Skill 快照保存，实际请求和记录配置一致。提供方连接、凭据及其他生成参数仍来自显式 `router` 配置。

远端的 `trace2skill_reference.yaml` 是单独的 200/200 SpreadsheetBench 实验设置，不是算法默认值。它指定分析并发 128、定位并发 16、分析预算 16384、失败分析最多 100 轮，以及采样并发 16。复现实验时应显式设置这些值，不隐式提高所有环境的默认并发。模型名称、连接、thinking 参数、训练和测试数量同样属于实验配置。

以下差异是当前项目接口与运行契约所需的适配：

- 初始 Skill、任务和私有评分输入使用当前公共类型。内容哈希使用当前公共 Skill 的 UTF-8 内容哈希，而不是远端旧文件映射哈希。
- 远端可接收已有轨迹；当前 Trainer 只从输入任务主动采样，不增加另一套训练入口。
- 路由模型异常按远端行为回退全文并记录原因。任务执行的提供方异常向外传播，由采样层记录失败；分析、定位和融合的独立失败保留记录并继续处理其他项。最终报告需区分模型未产生有效输出、算法拒绝的提案和正常工作簿评分失败。
- 通用任务路由保留环境无关提示词；SpreadsheetBench 路由使用远端原始提示词、公开字段和工作簿预览。
- 保留当前自动恢复与逐次物理请求记录。专用分析复用已完成记录；未完成的单项工作目录先归档，再使用原目录路径重放已记录请求。

## 通用与环境特殊流程

所有算法遵循同一组织规则：主流程通用，环境特殊提示词和流程放在算法自己的 `env_specific/`，主入口按环境命名。TreeSkill 中：

- `env_specific/spreadsheetbench_text_action.py`：reference 分析器、路由和专用适配器；严格要求配套的参考执行环境。
- `env_specific/spreadsheetbench_prompts/`：原始成功分析、失败分析和路由提示词。保留原文件字节；路由提示词按原实现去除末尾空白。
- `envs/builtin_envs/spreadsheetbench/text_action_mode/environment.py`：已注册的文本动作环境；负责消息生成、动作解析与 bash 执行，继承公共交互循环、工作区准备、评分及存档。
- `envs/builtin_envs/spreadsheetbench/text_action_mode/policy.py` 与 `text_action_mode/prompts/`：原始 Action/Observation 协议及执行提示词。
- `envs/builtin_envs/spreadsheetbench/recalculation_cli.py`：向任务目录复制独立的 `.treeskill_recalculate.py`，在 user prompt 中提供由环境 Python 执行的命令。Agent 保存公式后主动调用；脚本从系统 PATH 查找 soffice，在临时副本上清除旧缓存并通过 LibreOffice 命令行转换重算，验证后替换 `output.xlsx`；无需 UNO。评分阶段不自动补算。tool-calling 仍使用原有公共重算实现。
- `envs/builtin_envs/spreadsheetbench/text_action_mode/reference_reward.py`：text-action 执行评分与 reference 训练失败分析共用远端单元格比较规则，均读取指定原文件的缓存值。失败分析的 bash 也使用 text-action 的 managed Python；复制的任务辅助脚本可以继续用于修复后重算。

`TreeSkillTrainer.train(env=...)` 每次根据实际环境的注册名自动选择 adapter，训练配置不再接受 `env_adapter`；请求中的环境配置也无需提供 `adapters`。当前 `spreadsheetbench_text_action` 环境自动加载同名专用 adapter，其余环境使用 `generic`；未注册的自定义环境也回退到 `generic`。

每次调用创建独立 adapter，并输出 INFO 日志，例如：

```text
TreeSkill environment=spreadsheetbench_text_action adapter=spreadsheetbench_text_action (automatic selection)
TreeSkill environment=searchqa adapter=generic (automatic selection)
```

本次选择用于轨迹分析，同时自动写入返回的动态 TreeSkill 配置，供任务路由及快照恢复使用。内部 `TreeSkillConfig.env_adapter` 是产物记录，无需用户在训练时指定。实际选择也参与训练恢复清单的一致性检查；同一 trainer 的不同调用不共享可变的 adapter 状态。

底层 `TreeSkillEnvAdapter` 保留消息渲染及分析器构造接口，供独立路由、历史快照和离线验证使用。

动态 Skill 快照版本 3 保存适配器注册名、配置和唯一的稳定 ID 树元数据，不保存分析器、模型实例或运行目录。恢复时用户扩展必须已经导入，缺失时明确报错。版本 2 的历史快照若记录了 `spreadsheetbench_reference`，加载时会转换为 `spreadsheetbench_text_action`，不修改源快照；重新导出使用新名称。版本 1 和版本 2 的快照仍可加载：在恢复入口移除旧 `reference_routing` 字段并保留原适配器消息协议。所有版本都使用同一套 Markdown 树解析、选择和渲染代码，不再保留旧的 `SkillNode` / `SkillTree` 实现。新建的 generic Skill 使用环境无关提示词。恢复不改写旧文件或数据库，重新导出时写入显式配置。

reference 分析先完成失败轨迹分析，再处理成功轨迹。失败分析 Agent 必须在独立工作区中完成 `evaluate_output` 的 PASS 检查，并生成可解析报告，才会产出经验。按用户批准，仅该训练分析 Agent 可以访问相应训练样本的 `gold.xlsx`。任务执行 Agent 不获得 gold；test 也不会进入失败分析。评分和实验配置中的私有数据保持在审计侧。

## 共享环境后的执行协议

TreeSkill 使用 `SpreadsheetBenchTextActionEnvironment`，与原生工具环境互不继承。两者继承 `SpreadsheetExecutionEnvironment` 的 `_run()` 和交互循环。仍保留原始提示词资源、文本动作解析和 Observation 格式；工作簿与 Skill 统一放在 `workspace/<case-number>/`。提示词中的实际路径随工作区变化，system/user 快照也放在对应 case 目录。

任务评分统一使用公共评分器：空评分范围报错、范围解析及整列检查采用公共评分器语义，并在交互结束后将完整评分报告写入 `reward.detail`，不追加到 `trajectory`。训练失败分析仍使用历史比较器和独立 gold 工作区；执行层与分析层的评分用途保持区分。此版本不再宣称执行及反馈与旧 reference 环境逐字节等价。

重复动作阈值使用 `stagnation_limit`（默认 5）；旧的 `reference_stagnation_limit` 输入已移除。旧 rollout 的配置与新协议不同，继续使用原目录会被输入一致性检查拒绝，应使用新目录。历史输出不删除、不改写；旧目录内的运行脚本不能视为新运行时的可恢复入口。

## 已明确批准的协议差异

为适配当前 Environment 契约，执行或模型请求异常直接向上抛出，由训练/实验层记录并跳过相应数据，不将其转换成零分轨迹，不中断其他独立数据，不重新启动整个任务采样。分析、定位、节点融合的单项异常同样保留失败记录并继续其余工作。原参考分支的部分异常会保留执行失败轨迹或终止整个阶段，因此异常情况下的证据数量可能与原分支不同；实验配置记录这项差异。

传输层内部重试和已批准的模型输出解析重试不受上述数据跳过规则影响。配置错误、无效输入划分、已有实验目录、文件写入失败以及人工中断不会伪装为成功完成。正常完成执行但评分不通过的任务仍是有效零分训练样本。

## SDK 训练入口

训练只保留 `scripts/treeskill/run.py` 一个入口；原 `train.py` 和 `local_service.py` 已合并删除。脚本不创建 Service、Trainer 或 Environment。它在本地读取服务 YAML、用现有 Data 后端加载任务、用 SDK 收集初始 Skill 文件；注册、训练、训练后测试、任务状态查询和版本读取均通过现有 HTTP SDK 完成。`report.py` 只保留报告函数，不再提供独立命令入口。

运行前启动 Skill HTTP 服务。使用项目开发启动器时，需要 `FEATURE=async` 或默认的 `all`；`core` 模式会移除任务管理器，不能处理 SDK 的后台训练请求。运行中的服务必须启用 `service.training_pipelines.treeskill` 和任务管理器，鉴权凭据需要 `skill:write`。直接启动 Skill 服务时，可选择 `task_manager.mode: local`；开发启动器的 async 模式使用 Kafka。

```bash
make dev SERVICE=skill FEATURE=async UI=0

PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python -u -m scripts.treeskill.run \
  --service-config config/mindmemos_skill/dev.yaml \
  --sdk-home .mindmemos \
  --data-root /path/to/spreadsheetbench_verified_400 \
  --train-count 200 --test-count 200 --max-turns 100 \
  --output output/treeskill_qwen9b_sdk
```

服务地址和鉴权来自 SDK home。脚本使用公开入口 `MindMemOSClient`，由根 Client 管理连接、鉴权和资源关闭，不直接构造 SkillClient 或 HTTP transport：

```python
sdk = ConfigManager(args.sdk_home).load()
with MindMemOSClient(config=sdk) as client:
    task = client.skills.train(TrainingRequest(...))
    state = client.skills.show_task(TaskShowRequest(task_id=task["task_id"]))
```

请求不填写 `project_id`，项目由服务鉴权确定。`--service-config` 中的 failure_optimizer 用于构造请求的 instruct policy；未配置时使用 optimizer。Policy 固定使用参考实现的温度 0.7 和 `Observation:` stop，默认训练 200 条、测试 200 条、每条最多 100 回合。TreeSkill optimizer、router、failure_optimizer 和各阶段并发由运行中的服务配置决定。客户端读取的 YAML 应与服务实际加载的配置一致，修改服务模型配置后需要重启服务。配置中的模型名只是配置值，不由脚本固定。

当前 TreeSkill 示例配置使用 `qwen3.5-9b`：成功分析、定位和融合使用 thinking 解码，失败分析、policy 和 router 使用 instruct 解码。分析并发为 128，采样、定位和测试并发为 16；thinking 请求超时为 1800 秒，instruct 请求超时为 600 秒，模型重试次数为 3。各阶段 token 预算保留为 16384，这是相对上游 reference 配置的明确覆盖项。默认初始 Skill 仍为仓库内的简版；严格复现实验需通过 `--initial-skill` 指定授权的完整 Human-Written xlsx 包。

初始 Skill 的标准 `SKILL.md` 和全部辅助文件通过 SDK 上传；返回版本的源文件导出到 `evolved_skill/<skill-name>/`，完整版本（包括 TreeSkill implementation、router 配置及辅助文件）另存为 `evolved_skill.json`。仅导出的源目录不包含独立的动态运行配置，恢复动态行为时使用完整版本快照。

脚本按固定划分选择训练和测试任务，直接提交一个 `TrainingRequest(algo="treeskill")`：`tasks` 包含两组数据，`split.train` 与 `split.test` 分别绑定对应 ID，`split.val` 为空。服务通用训练 pipeline 只把 train 传给 Trainer，最终候选冻结后执行 test；test 不参与采样、分析、融合或选优。`--test-count 0` 可只训练不测试，不执行初始 baseline。请求保留事务式公式重算。

测试并发取自服务端 `training_pipelines.treeskill.config.evaluation_concurrency`，已移除客户端 `--concurrency`。测试异常遵循服务端现有契约：任务标记失败，不把执行错误算成零分，也不沿用旧客户端继续其余测试并标为 partial 的行为。服务在测试成功后发布新版本；`training_result.json` 保存任务最终结果。

SDK 不上传工作簿或下载算法审计目录。服务与脚本需共享数据集和输出目录的绝对路径；`work_dir` 是服务端路径。测试成绩和审计文件由服务写入 `training/test/` 与 `training/train/`，脚本的报告函数读取这些共享文件。

脚本保存 `registration.json`、`training_task.json`、`training_status.json`、`training_result.json` 和 `service_progress.jsonl`。重复使用相同输出目录时复用已注册版本与任务，不重复提交；已保存的进度序号避免重复追加事件。`submission.json` 绑定输入、环境、服务地址和工作目录，输入变化时拒绝复用已有任务。旧直接调用入口的产物布局不作为新 SDK 任务恢复入口，应使用新输出目录。任务状态的跨服务重启恢复能力取决于任务管理器：local 队列不持久化，Kafka 模式提供持久任务状态。

报告明确记录 `baseline_skipped=true` 和 baseline 状态 skipped，不计算前后提升。可通过同一个入口重新汇总共享目录中的结果，不连接服务：

```bash
PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python -m scripts.treeskill.run \
  --report-only --output output/treeskill_qwen9b_sdk
```

patch 统计区分实际修改 `applied`、模型主动不修改 `model_declined`、内容或结构校验拒绝 `validation_rejected`、被接受但全文未改变 `accepted_no_change`。融合或 JSON 解析失败也会计入诊断；原始算法审计中保留原因及全文前后对比。

离线测试不访问模型或下载数据：

```bash
PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/pytest \
  tests/scripts/test_treeskill_comparison.py \
  tests/scripts/test_treeskill_training.py \
  tests/integration/test_treeskill_sdk_runner.py -q
```

### 配置与数据划分

历史参考实验使用 Qwen instruct/thinking 参数、seed=41、200/200 划分、采样并发 16、分析并发 128、定位并发 16；失败分析使用独立 instruct 参数，融合顺序执行。仓库已移除对应 YAML 预设，复现实验时需自行提供等价配置。

当前训练入口读取 `resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json` 的固定有序 train/test ID，没有重新抽样。数量参数取对应列表前缀；Data 后端负责加载并划分数据。

脚本通过现有 Data 后端加载任务和私有评分数据，再构建环境。没有新增数据集下载动作；数据下载沿用仓库现有环境脚本。

## 实验目录

```text
experiment/
├── config.json                  # 算法配置（API key 等凭据显示为 [REDACTED]）
├── summary.json                 # 最终状态、分数、分母、训练指标、耗时与模型 usage
├── preflight/                   # reference 重算预检产物（启用时）
├── training/
│   ├── config.json              # 同样仅保存算法配置
│   ├── summary.json             # 训练阶段结果
│   ├── final_skill.json         # 可恢复的最终 TreeSkill 快照
│   ├── final_skill/             # <skill-name>/ 下保存 SKILL.md 与完整辅助资源
│   ├── tree_metadata.json       # 最终树的稳定节点身份
│   ├── rollouts/train/          # 每个任务/采样的 audit、workspace 或 error.json
│   ├── stages/                  # evidence、analysis、localization、fusion 的阶段汇总
│   ├── items/                   # Trainer 管理的 analysis、localization、fusion 单项结果
│   │   └── <stage>/<item>/
│   │       ├── completed.json   # 成功工作单元的恢复检查点
│   │       └── attempt-001/
│   │           ├── outcome.json # AgentResult、失败信息、耗时和算法结果
│   │           ├── reports/     # 返回给 Trainer 保存的分析报告
│   │           └── workspace/   # 失败分析的工具工作区（需要时创建）
│   └── patches/
│       └── 0001-001-update_node/
│           ├── patch.json       # 提案、依据、目标节点、接受/拒绝原因
│           ├── before.md
│           ├── after.md
│           └── change.diff
└── test/                        # 最终 Skill 的独立测试执行记录，test 为空时不创建
```

不复用已有总实验目录，避免将不同配置或失败后的部分产物混在一起。每次实验都使用新目录；输入 Skill 和历史目录不被清理。

TreeSkill 和 Skill-Forest 的 `config.json` 自动将嵌套配置中的非空 API key、token、password、secret 和 authorization 字段替换为 `[REDACTED]`，保留模型、端点和生成参数。运行配置与恢复身份校验仍使用调用方显式提供的真实凭据。`scripts.treeskill.no_skill` 使用 `--service-config`（默认 `config/mindmemos_skill/dev.yaml`）补入本地凭据，不从实验快照读取 API key。

此脱敏仅覆盖上述配置记录；可恢复的动态 Skill 快照和 rollout 审计文件仍沿用已有序列化协议，整个实验目录不能因此视为已完成凭据清理。已有历史文件不会被自动改写。

### 分数与状态解释

`train` 是初始完整 Skill 在采样任务上的分数；`test` 是最终路由 Skill 在 held-out 任务上的分数，二者不是同一数据集上的前后提升。

- `mean_score = score_sum / completed`；没有有效完成样本时为 `null`。
- `success_rate = success_count / completed`。
- `expected`、`completed`、`failed`、`skipped` 同时报告。异常样本不进入上述分母，`complete=false` 明确表示未完成全部请求。
- `status=completed` 表示流程与请求执行完成；存在跳过数据或训练阶段失败记录时为 `partial`；全局异常为 `failed`。
- `duration_s` 是整个入口的墙钟耗时；`stages` 是分阶段墙钟耗时，`rollout_duration_s` 是各执行时长之和，并发时可大于墙钟时间。
- 内部通过公共 `stage()` 记录四类阶段：`treeskill.rollout`（并发训练轨迹采集）、`treeskill.analyze`（轨迹经验分析）、`treeskill.merge`（经验定位及节点融合）和 `treeskill.test`（最终 Skill 的独立测试集执行与评分汇总）。每个实际执行的阶段产生子 span 与包含状态、`duration_ms` 的结束日志，继承当前请求或任务的 trace。
- 阶段时长为该阶段的墙钟时间，包含并发工作等待；不把各轨迹执行时长相加当作 rollout 阶段时长。没有有效分析时跳过合并，没有测试样本时不产生测试阶段。单项失败被算法捕获并继续时，诊断保存在单项失败日志和汇总中；阶段 `succeeded` 表示该批处理返回，仍需结合汇总的失败数量判断数据完整性。

- `model_usage` 从 Trainer 保存的单项结果统计优化与路由的逻辑 Agent 调用，`scope=logical_agent_calls`。一次 `run()` 内部的解析重试不额外增加 `calls`；成功返回的 usage 包含这些解析尝试。底层 HTTP 重试不单独计数。
- 最终抛异常的调用可能无法提供 token 用量；`failed_calls`、`interrupted_items` 和 `usage_complete` 显式表示覆盖情况。中断前尚未返回的模型调用无法准确计数或计费，不当作零消耗。环境执行的模型消耗另列在 train/test 的 `usage`。
- 算法函数直接调用普通 `OpenAIChatAgent`，返回分析、定位、融合结果及模型响应。文件保存、patch diff、统计和检查点由 Trainer 负责，算法不创建记录型 Agent 或专用 `chat()` 协议。工具执行所需的工作簿由 Trainer 请求环境适配器准备，分析器仅操作该工作区。
- 训练产出的 TreeSkill 使用普通 Router。Trainer 评估时先路由一次，将所选内容作为静态 Skill 交给 rollout，并在该执行目录的 `routing/` 下保存返回的选择结果与 AgentResult；独立使用 TreeSkill 时不自动写训练目录。

## OpenAIChatAgent 的结构化输出与解析重试

`OpenAIChatConfig.response_format` 或 `agent.run(response_format=...)` 支持 JSON object/JSON Schema 参数。构造 Agent 时的 `output_parser(text)` 回调返回字符串或具体 Pydantic 模型；回调抛出异常会触发有界解析重试。

`parse_max_attempts` 表示包含首次请求的总次数，默认 3。`parse_retry_mode="direct"` 重发原请求；`"feedback"` 追加被拒绝的 assistant 回复和校正消息。默认校正提示词与参考分支一致，也可用 `parse_feedback(exception)` 自定义。解析重试与传输重试分别计数；传输最终失败直接向上抛出。成功返回时 usage 汇总全部解析尝试，输入 messages 不被原地修改。输出解析回调与既有工具解析 `parse_response` 不能同时配置。

## 验证边界

提供离线端到端脚本，以及提示词哈希、用户消息、树修改、稳定 ID、定位重试、内容过滤、PASS 门控、私有数据边界、异常跳过、patch 审计和结构化模型调用的回归测试。离线通过不代表已经复现真实模型的 benchmark 分数。本次没有启动付费模型实验，也没有把未运行的 LibreOffice/真实模型集成描述为已验证。

训练入口直接等待异步模型流程，不在业务代码内创建事件循环。`Environment.rollout()`、具体环境 `_run()`、`SkillSet.assemble()`、单个 `Skill.assemble()` 和模型调用均为原生异步接口，调用方逐层 `await`，不使用线程包装同步接口。

## 服务训练与自动恢复

TreeSkill 可通过通用 `training` Pipeline 配置使用，与 Skill-Forest 共用业务流程。
详见 [通用训练流程](training.md)。直接调用 `TreeSkillTrainer.train()` 时，同一输入与
同一工作目录自动复用兼容的采样、已完成的单项分析、定位、融合及最终产物；恢复结果保留 TreeSkill 动态路由。
恢复粒度是完整工作单元，不恢复到模型请求或解析重试中间。失败项不写成功检查点；下次调用重新执行。
融合按自底向上顺序保存成功前缀及对应的树内容、稳定节点 ID。出现失败后仍处理其他节点，但不提交依赖该失败位置的后续检查点；重试时重算该后缀。
上游重试改变了下游输入时，Trainer 校验旧检查点后将其归档，再重新计算下游工作。历史尝试保留，恢复读取不会重复计算其 usage。
当前恢复协议为 2，不读取旧的逐模型请求检查点；旧协议目录需要新的工作目录。
旧目录缺少恢复清单时明确拒绝，不自动覆盖。独立 `run_experiment` 包装器仍要求全新实验根目录。

## 实时进度日志

算法使用普通 logger 和 f-string 输出消息，不维护额外的进度对象。控制台、任务进度和 CLI 使用同一消息字符串，
按 `事件 | 身份或结果字段 | 耗时` 组织，字段使用 `key=value`。控制台增加 UTC 时间、级别、logger 和任务关联字段，
不再追加单独的 JSON 指标行。异常堆栈保留为独立的诊断文本。

```text
[2026-10-10T09:53:00.370965Z] [INFO] [mindmemos_skill.algos.train.treeskill.trainer] | project_id=p task_id=t attempt=1 | [treeskill.train.fusion] progress | proposed_patches=14 accepted_patches=13 rejected_patches=1 | node_id=019 reused=False
```

采样、分析、定位和融合在单项完成时记录结果，阶段汇总使用已有结果或报告中的统计。
答错属于有效评分，执行错误通过 `error` 字段或异常日志说明；复用项通过 `reused=True` 标识。
总耗时取训练报告的 `duration_s`，轨迹原始耗时和 usage 保留在实验产物中，不冒充恢复后的新增消耗。
等待期间不输出定时心跳，也不为了日志额外维护 running、pending 或最长执行耗时。

异步任务入口使用 `async with async_task_log_reporter(...)` 绑定项目、任务和执行次数；
算法与子协程直接调用普通 logger，结束或取消时排空已入队日志。原有 OpenTelemetry 阶段和关联字段继续保留。

日志不输出 Skill 正文、完整轨迹或 patch 内容，详情仍保存在阶段产物和 patch diff 文件中。
