# TreeSkill 单轮主动训练与独立实验

本实现参考 `mindscale-noah/MindMemOS` 的 `feat/treeskill-integration`，固定提交为 `e83aae887f08eb9fa1596384404dc7757a6a7f2f`。算法位于 `mindmemos_skill.algos.train.treeskill`，注册名为 `treeskill`。无需 `mindmemos_eval`、Service 或数据库。

## 算法与输入边界

`await TreeSkillTrainer.train(...)` 接收初始 `Skill`、调用方提供的 `Environment`、已加载的训练 `DataSample` 和实验目录。算法自己调用 `Environment.rollout()`，不接受外部轨迹作为训练入口，不修改输入 Skill 或辅助资源。

单轮执行顺序保持为：完整初始 Skill 采样 → 独立轨迹分析 → 对初始树定位证据 → 自底向上融合节点 → 返回最终可路由的 TreeSkill。不会在中间 patch 后重新采样，也不新增多轮训练或验证集选优。原始事件采用参考实现的逐事件截断，`transcript_max_chars` 不是整条轨迹的总字符上限。

通用分析器通过评分区分 success/error/unlabeled。所有定位都针对同一棵初始树；定位失败最多尝试两次，第二次 token 预算为 `min(initial * 2, 8192)`，消息保持一致。融合按深度从叶到根、同深度按源文档位置排序，每个节点读取此前修改后的当前树；`update_node` 在 `create_child` 之前执行。保留原始稳定 ID、Markdown 结构校验、重复子节点校验、内容过滤及 reject 规则。

融合操作与子树检查对照 `TreeSkill-colleague-share` 的 `bc3b0a9` 实现：附加说明字段（例如子节点的 `rationale`）保留在 patch 审计中，不写入 Skill，也不导致整组拒绝；操作别名 `update` 归一为 `update_node`。非法操作组合产生 reject，单个子树的内容或结构错误只拒绝该操作，不撤销已经通过的更新。与原版一致，更新允许清空节点正文，清空后的无内容叶节点不再进入可路由节点集合；新建叶节点仍必须有非空白正文。JSON 截断仍记录为融合失败，不自动补全或新增重试，融合预算仍使用 `fusion_max_tokens`。

`await run_experiment(...)` 在训练结束后，才把最终 Skill 用于 test。每条 test 数据只请求一次 rollout；模型内部的传输及解析重试仍按模型配置执行。test 不进入 trainer、分析器、融合器或选优逻辑。val 和 test 均可省略或为空；当前不定义验证集选择行为，非空 val 会明确报错。

## 通用与环境特殊流程

所有算法遵循同一组织规则：主流程通用，环境特殊提示词和流程放在算法自己的 `env_specific/`，主入口按环境命名。TreeSkill 中：

- `env_specific/spreadsheetbench_text_action.py`：reference 分析器、路由和专用适配器；严格要求配套的参考执行环境。
- `env_specific/spreadsheetbench_prompts/`：原始成功分析、失败分析和路由提示词。保留原文件字节；路由提示词按原实现去除末尾空白。
- `envs/builtin_envs/spreadsheetbench/text_action.py`：已注册的文本动作环境；负责消息生成、动作解析与 bash 执行，继承公共交互循环、工作区准备、评分及存档。
- `envs/builtin_envs/spreadsheetbench/text_action_policy.py` 与 `text_action_prompts/`：原始 Action/Observation 协议及执行提示词。
- `envs/builtin_envs/spreadsheetbench/recalculation.py`：环境侧事务式公式重算；同目录 `runtime/uno_worker.py` 是由 LibreOffice 自己的 Python 启动的独立工作进程，不依赖本包或 Pydantic。
- `envs/builtin_envs/spreadsheetbench/reference_reward.py`：仅供 reference 训练失败分析保留的历史单元格比较规则。任务执行统一使用环境侧 `reward.py`。

`TreeSkillTrainer.train(env=...)` 每次根据实际环境的注册名自动选择 adapter，训练配置不再接受 `env_adapter`；请求中的环境配置也无需提供 `adapters`。当前 `spreadsheetbench_text_action` 环境自动加载同名专用 adapter，其余环境使用 `generic`；未注册的自定义环境也回退到 `generic`。

每次调用创建独立 adapter，并输出 INFO 日志，例如：

```text
TreeSkill environment=spreadsheetbench_text_action adapter=spreadsheetbench_text_action (automatic selection)
TreeSkill environment=searchqa adapter=generic (automatic selection)
```

本次选择用于轨迹分析，同时自动写入返回的动态 TreeSkill 配置，供任务路由及快照恢复使用。内部 `TreeSkillConfig.env_adapter` 是产物记录，无需用户在训练时指定。实际选择也参与训练恢复清单的一致性检查；同一 trainer 的不同调用不共享可变的 adapter 状态。

底层 `TreeSkillEnvAdapter` 保留消息渲染及分析器构造接口，供独立路由、历史快照和离线验证使用。

动态 Skill 快照版本 2 保存适配器注册名和配置，不保存分析器、模型实例或运行目录。恢复时用户扩展必须已经导入，缺失时明确报错。版本 2 的历史快照若记录了 `spreadsheetbench_reference`，加载时会转换为 `spreadsheetbench_text_action`，不修改源快照；重新导出使用新名称。版本 1 的快照仍可加载：旧稳定 ID 路由恢复 SpreadsheetBench 协议，旧普通路由恢复原有消息格式；新建的 generic Skill 使用环境无关提示词。恢复不改写旧文件或数据库，重新导出时写入显式配置。

reference 分析先完成失败轨迹分析，再处理成功轨迹。失败分析 Agent 必须在独立工作区中完成 `evaluate_output` 的 PASS 检查，并生成可解析报告，才会产出经验。按用户批准，仅该训练分析 Agent 可以访问相应训练样本的 `gold.xlsx`。任务执行 Agent 不获得 gold；test 也不会进入失败分析。评分和实验配置中的私有数据保持在审计侧。

## 共享环境后的执行协议

TreeSkill 使用 `SpreadsheetBenchTextActionEnvironment`，与原生工具环境互不继承。两者继承 `SpreadsheetExecutionEnvironment` 的 `_run()` 和交互循环。仍保留原始提示词资源、文本动作解析和 Observation 格式；工作簿、Skill 与重算辅助程序统一放在 `workspace/<case-number>/`。提示词中的实际路径随工作区变化，system/user 快照也放在对应 case 目录。

任务评分统一使用公共评分器：空评分范围报错、范围解析及整列检查采用公共评分器语义，并在交互结束后将完整评分报告写入 `reward.detail`，不追加到 `trajectory`。训练失败分析仍使用历史比较器和独立 gold 工作区；执行层与分析层的评分用途保持区分。此版本不再宣称执行及反馈与旧 reference 环境逐字节等价。

重复动作阈值使用 `stagnation_limit`（默认 5）；旧的 `reference_stagnation_limit` 输入已移除。旧 rollout 的配置与新协议不同，继续使用原目录会被输入一致性检查拒绝，应使用新目录。历史输出不删除、不改写；旧目录内的运行脚本不能视为新运行时的可恢复入口。

## 已明确批准的协议差异

为适配当前 Environment 契约，执行或模型请求异常直接向上抛出，由训练/实验层记录并跳过相应数据，不将其转换成零分轨迹，不中断其他独立数据，不重新启动整个任务采样。分析、定位、节点融合的单项异常同样保留失败记录并继续其余工作。原参考分支的部分异常会保留执行失败轨迹或终止整个阶段，因此异常情况下的证据数量可能与原分支不同；实验配置记录这项差异。

传输层内部重试和已批准的模型输出解析重试不受上述数据跳过规则影响。配置错误、无效输入划分、已有实验目录、文件写入失败以及人工中断不会伪装为成功完成。正常完成执行但评分不通过的任务仍是有效零分训练样本。

## 本地服务对照脚本

在仓库根目录运行。可安装 `mindmemos-skill[spreadsheetbench]` 后运行；下面的 `PYTHONPATH` 用于直接检查当前工作区源码。

对照脚本的离线测试不访问模型或下载数据：

```bash
PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/pytest \
  tests/scripts/test_treeskill_comparison.py -q
```

`scripts/treeskill/run.py` 通过已运行的本地 Skill 服务训练，并在当前进程完成两次测试。默认取训练划分前 30 题，每题采样 1 条轨迹；初始完整 Skill 和演进后的动态 TreeSkill 分别执行同一测试划分的 200 题，各执行一次。阶段顺序为初始 Skill 测试、服务训练、演进后 Skill 测试；两次测试的结果均不进入训练请求，也不用于选优。

```bash
PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python scripts/treeskill/run.py \
  --service-config config/mindmemos_skill/dev.yaml \
  --sdk-home .mindmemos \
  --data-root /path/to/spreadsheetbench_verified_400 \
  --train-count 30 --test-count 200 --concurrency 32 --max-turns 30 \
  --output output/treeskill_30_200
```

服务地址和鉴权取自 SDK home 的设置；执行模型取自服务 YAML 的 TreeSkill `optimizer`。采样、分析、定位并发由运行中的服务配置决定，`--concurrency` 仅设置当前进程的测试并发。服务必须与所传 YAML 使用相同的 TreeSkill 配置，并与当前进程共享输出目录和数据集的绝对路径。服务重载配置后，应使用新的输出目录。

加 `--preflight-only` 只检查本地输入、Python 工具环境和真实公式重算，不调用模型；加 `--probe-only` 还验证服务鉴权和真实模型响应。预检成功后可在相同输入和目录运行正式实验。正常完成的 rollout 会复用；已记录的执行失败不会在同一目录自动重跑。改变输入要求新目录。首题执行异常会阻止该阶段的批量调用，普通评分失败仍计入有效零分样本。

显式添加 `--no-recalculation` 时，训练和两次测试都关闭 `transactional_recalculation`，跳过 LibreOffice/UNO 预检，并且不向任务工作区写入重算助手或追加重算指令。实验协议记录该选项；必须使用新的输出目录。评分仍比较工作簿中保存的单元格缓存值，新增公式没有缓存值时可能被判为错误，因此应将这组结果标为未重算公式的实验。

根目录保存 `events.jsonl`、`service_progress.jsonl`、`comparison.json`、`patch_diagnostics.json`、`patches.csv` 和逐题配对的 `paired_results.csv`。两次测试分别位于 `baseline/` 和 `evolved/`；训练算法的审计位于 `training/train/`。每条测试结果保留源请求、任务实际组装的 Skill、模型轨迹、评分、usage 和耗时。演进后的测试从服务版本快照恢复动态 TreeSkill，保留任务时路由，不仅加载导出的 Markdown。

可在实验期间重新汇总已落盘的中间结果：

```bash
PYTHONPATH=.:src/mindmemos_skill:src/mindmemos_sdk .venv/bin/python scripts/treeskill/report.py \
  --output output/treeskill_30_200
```

patch 统计区分实际修改 `applied`、模型主动不修改 `model_declined`、内容或结构校验拒绝 `validation_rejected`、被接受但全文未改变 `accepted_no_change`。融合模型调用或 JSON 解析异常可能没有 `patch.json`，因此同时统计 analysis/localization/fusion 的失败记录和模型失败尝试。`result.accepted=true` 本身不足以证明 patch 生效；原因、全文前后对比和 diff 均保留在算法原始审计目录。

参考公式重算需要可运行的 LibreOffice 和包含 pyuno 的 Python，可通过 `SPREADSHEETBENCH_SOFFICE_PATH`、`SPREADSHEETBENCH_LIBREOFFICE_PYTHON` 指定。重算运行时的环境变量和工作区辅助文件统一使用 SpreadsheetBench 命名；旧环境变量不再读取。预检会真实验证公式缓存值；缺少运行时会明确失败，不静默禁用公式重算。

### 配置与数据划分

历史参考实验使用 Qwen instruct/thinking 参数、seed=41、200/200 划分、采样并发 16、分析并发 128、定位并发 16；失败分析使用独立 instruct 参数，融合顺序执行。仓库已移除对应 YAML 预设，复现实验时需自行提供等价配置。

当前对照脚本读取 `resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json` 的固定有序 train/test ID，没有重新抽样；val 必须为空。`--train-count`、`--test-count` 取对应列表前缀，并严格检查请求数量和划分互斥。

脚本先通过现有 Data 后端加载任务和私有评分数据，校验划分互斥，再构建环境。没有新增数据集下载动作；数据下载沿用仓库现有环境脚本。

## 实验目录

```text
experiment/
├── config.json                  # 仅算法配置（含算法模型参数，不含 API key 或文件哈希）
├── summary.json                 # 最终状态、分数、分母、训练指标、耗时与模型 usage
├── preflight/                   # reference 重算预检产物（启用时）
├── training/
│   ├── config.json              # 同样仅保存算法配置
│   ├── summary.json             # 训练阶段结果
│   ├── final_skill.json         # 可恢复的最终 TreeSkill 快照
│   ├── final_skill/             # SKILL.md 与完整辅助资源
│   ├── tree_metadata.json       # 最终树的稳定节点身份
│   ├── rollouts/train/          # 每个任务/采样的 audit、workspace 或 error.json
│   ├── stages/                  # evidence、analysis、localization、fusion 与 reference 报告
│   ├── model_calls/             # optimizer、failure、router 的逐请求、逐解析重试记录
│   └── patches/
│       └── 0001-001-update_node/
│           ├── patch.json       # 提案、依据、目标节点、接受/拒绝原因
│           ├── before.md
│           ├── after.md
│           └── change.diff
└── test/                        # 最终 Skill 的独立测试执行记录，test 为空时不创建
```

不复用已有总实验目录，避免将不同配置或失败后的部分产物混在一起。每次实验都使用新目录；输入 Skill 和历史目录不被清理。

### 分数与状态解释

`train` 是初始完整 Skill 在采样任务上的分数；`test` 是最终路由 Skill 在 held-out 任务上的分数，二者不是同一数据集上的前后提升。

- `mean_score = score_sum / completed`；没有有效完成样本时为 `null`。
- `success_rate = success_count / completed`。
- `expected`、`completed`、`failed`、`skipped` 同时报告。异常样本不进入上述分母，`complete=false` 明确表示未完成全部请求。
- `status=completed` 表示流程与请求执行完成；存在跳过数据或训练阶段失败记录时为 `partial`；全局异常为 `failed`。
- `duration_s` 是整个入口的墙钟耗时；`stages` 是分阶段墙钟耗时，`rollout_duration_s` 是各执行时长之和，并发时可大于墙钟时间。
- `model_usage` 统计优化与路由请求，包含解析失败重试；环境执行的模型消耗另列在 train/test 的 `usage`。没有报告 usage 的请求不会被当成已核实的零消耗。

## OpenAIChatAgent 的结构化输出与解析重试

`OpenAIChatConfig.response_format` 或 `agent.run(response_format=...)` 支持 JSON object/JSON Schema 参数。构造 Agent 时的 `output_parser(text)` 回调返回字符串或具体 Pydantic 模型；回调抛出异常会触发有界解析重试。

`parse_max_attempts` 表示包含首次请求的总次数，默认 3。`parse_retry_mode="direct"` 重发原请求；`"feedback"` 追加被拒绝的 assistant 回复和校正消息。默认校正提示词与参考分支一致，也可用 `parse_feedback(exception)` 自定义。解析重试与传输重试分别计数；传输最终失败直接向上抛出。成功返回时 usage 汇总全部解析尝试，输入 messages 不被原地修改。输出解析回调与既有工具解析 `parse_response` 不能同时配置。

## 验证边界

提供离线端到端脚本，以及提示词哈希、用户消息、树修改、稳定 ID、定位重试、内容过滤、PASS 门控、私有数据边界、异常跳过、patch 审计和结构化模型调用的回归测试。离线通过不代表已经复现真实模型的 benchmark 分数。本次没有启动付费模型实验，也没有把未运行的 LibreOffice/真实模型集成描述为已验证。

训练入口直接等待异步模型流程，不在业务代码内创建事件循环。`Environment.rollout()` 和 `SkillSet.assemble()` 为异步接口；具体环境 `_run()` 与单个 `Skill.assemble()` 的同步实现通过工作线程执行。

## 服务训练与自动恢复

TreeSkill 可通过通用 `training` Pipeline 配置使用，与 Skill-Forest 共用业务流程。
详见 [通用训练流程](training.md)。直接调用 `TreeSkillTrainer.train()` 时，同一输入与
同一工作目录自动复用兼容的采样、模型响应及最终产物；恢复结果保留 TreeSkill 动态路由。
旧目录缺少恢复清单时明确拒绝，不自动覆盖。独立 `run_experiment` 包装器仍要求全新实验根目录。
