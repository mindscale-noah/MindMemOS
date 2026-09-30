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

- `env_specific/spreadsheetbench.py`：reference 分析器、SpreadsheetBench 环境子类和原始路由用户消息。子类只覆盖 `_system_prompt`、`_user_prompt`、`_get_observation`，继承公共交互循环、工作区准备、评分及存档。
- `env_specific/spreadsheetbench_prompts/`：原始成功分析、失败分析、执行和路由提示词。保留原文件字节；路由提示词按原实现去除末尾空白。
- `env_specific/spreadsheetbench_policy.py`：原始 Action/Observation 协议。
- `envs/builtin_envs/spreadsheetbench/recalculation.py`：环境侧事务式公式重算；同目录 `runtime/uno_worker.py` 是由 LibreOffice 自己的 Python 启动的独立工作进程，不依赖本包或 Pydantic。
- `envs/builtin_envs/spreadsheetbench/reference_reward.py`：仅供 reference 训练失败分析保留的历史单元格比较规则。任务执行统一使用环境侧 `reward.py`。

通用训练入口接受公共 Environment 协议。`TreeSkillTrainer` 初始化时按 `env_adapter.class_name` 构造注册式适配器，默认 `generic` 使用环境无关的分析与路由；`spreadsheetbench_reference` 保留原始 SpreadsheetBench 分析和路由协议，并校验配套执行环境。训练返回的 Skill 保留同一个适配器选择，路由继续使用融合后的稳定节点元数据。

```yaml
env_adapter:
  class_name: spreadsheetbench_reference
  config: {}
```

用户继承 `TreeSkillEnvAdapter` 并通过 `@TreeSkillEnvAdapter.register("my_env")` 注册，入口显式导入扩展模块后即可在配置中选择。只改变文本时覆盖 `render_trajectory()`、`render_task()` 或 `build_analysis_messages()`、`build_routing_messages()`；需要多轮工具分析时覆盖 `create_analyzer(runtime)`。默认分析器调用 Adapter 的消息方法；特殊分析器可以使用自己的消息协议。`analysis_system` 和 `routing_system` 接受完整提示词文本覆盖，reference 分析拒绝不适用的 `analysis_system` 配置。

动态 Skill 快照版本 2 保存适配器注册名和配置，不保存分析器、模型实例或运行目录。恢复时用户扩展必须已经导入，缺失时明确报错。版本 1 的快照仍可加载：旧稳定 ID 路由恢复 SpreadsheetBench 协议，旧普通路由恢复原有消息格式；新建的 generic Skill 使用环境无关提示词。恢复不改写旧文件或数据库，重新导出时写入显式配置。

reference 分析先完成失败轨迹分析，再处理成功轨迹。失败分析 Agent 必须在独立工作区中完成 `evaluate_output` 的 PASS 检查，并生成可解析报告，才会产出经验。按用户批准，仅该训练分析 Agent 可以访问相应训练样本的 `gold.xlsx`。任务执行 Agent 不获得 gold；test 也不会进入失败分析。评分和实验配置中的私有数据保持在审计侧。

## 共享环境后的执行协议

TreeSkill 环境现在是 `SpreadsheetBenchEnvironment` 的薄子类，不再维护独立 `_run()`。仍保留原始提示词资源、文本动作解析和 Observation 格式；工作簿、Skill 与重算辅助程序统一放在 `workspace/<case-number>/`。提示词中的实际路径随工作区变化，system/user 快照也放在对应 case 目录。

任务评分统一使用公共评分器：空评分范围报错、范围解析及整列检查采用公共评分器语义，并在交互结束后将完整评分报告追加到返回轨迹。这份报告不会送回本次执行 Agent。训练失败分析仍使用历史比较器和独立 gold 工作区；执行层与分析层的评分用途保持区分。此版本不再宣称执行及反馈与旧 reference 环境逐字节等价。

原 `reference_stagnation_limit` 输入仍可读取，序列化统一为 `stagnation_limit`。旧 rollout 的配置与新协议不同，继续使用原目录会被输入一致性检查拒绝，应使用新目录。历史输出不删除、不改写；旧目录内的运行脚本不能视为新运行时的可恢复入口。

## 已明确批准的协议差异

为适配当前 Environment 契约，执行或模型请求异常直接向上抛出，由训练/实验层记录并跳过相应数据，不将其转换成零分轨迹，不中断其他独立数据，不重新启动整个任务采样。分析、定位、节点融合的单项异常同样保留失败记录并继续其余工作。原参考分支的部分异常会保留执行失败轨迹或终止整个阶段，因此异常情况下的证据数量可能与原分支不同；实验配置记录这项差异。

传输层内部重试和已批准的模型输出解析重试不受上述数据跳过规则影响。配置错误、无效输入划分、已有实验目录、文件写入失败以及人工中断不会伪装为成功完成。正常完成执行但评分不通过的任务仍是有效零分训练样本。

## 独立脚本

在仓库根目录运行。可安装 `mindmemos-skill[spreadsheetbench]` 后运行；下面的 `PYTHONPATH` 用于直接检查当前工作区源码。

离线端到端测试不访问模型或下载数据：

```bash
PYTHONPATH=src/mindmemos_skill .venv/bin/python scripts/treeskill/smoke.py \
  --output-dir .local_doc/treeskill/my-smoke-run
```

正式实验脚本接受 YAML 配置和一个全新的总实验目录：

```bash
PYTHONPATH=src/mindmemos_skill .venv/bin/python scripts/treeskill/run.py \
  --config config/mindmemos_skill/treeskill/generic.yaml \
  --output-dir outputs/treeskill/generic-run
```

```bash
PYTHONPATH=src/mindmemos_skill .venv/bin/python scripts/treeskill/run.py \
  --config config/mindmemos_skill/treeskill/spreadsheetbench_reference.yaml \
  --output-dir outputs/treeskill/reference-run
```

后两个命令会调用配置的模型。凭据通过现有的 `MINDMEMOS_AGENT_API_KEY`、`MINDMEMOS_AGENT_BASE_URL` 提供；模型名必须使用对应服务实际接受的名称。配置中的模型字符串保留参考分支，不替换成猜测的部署名。

加 `--preflight-only` 只检查数据、初始 Skill 和本地重算运行时，不调用模型；结果仍写入指定目录，状态为 `preflight`。随后正式实验须使用另一个新目录。reference 配置需要自行提供授权的原始 xlsx Skill，脚本会校验 `SKILL.md`、`recalc.py`、`LICENSE.txt` 三个文件的 SHA-256，不下载或替换它们。

参考公式重算需要可运行的 LibreOffice 和包含 pyuno 的 Python，可通过 `SPREADSHEETBENCH_SOFFICE_PATH`、`SPREADSHEETBENCH_LIBREOFFICE_PYTHON` 指定。重算运行时的环境变量和工作区辅助文件统一使用 SpreadsheetBench 命名；旧环境变量不再读取。预检会真实验证公式缓存值；缺少运行时会明确失败，不静默禁用公式重算。

### 配置与数据划分

`generic.yaml` 是通用算法结合现有 SpreadsheetBench 环境的示例，使用前 8 条训练任务。`spreadsheetbench_reference.yaml` 保留参考脚本的 Qwen instruct/thinking 参数、seed=41、200/200 划分、采样并发 16、分析并发 128、定位并发 16；失败分析使用独立 instruct 配置。融合顺序执行。

`resources/mindmemos_skill/treeskill/reference/splits.json` 保存参考分支有序的 train/test ID，没有重新抽样。参考资源中的 val 是 train 的副本，且原实验脚本不使用 val；迁移文件按已确认协议将 val 置空，这一转换写在 `provenance.json`。`train_limit`、`test_limit` 只取各有序列表的前缀。若不需要 test，可复制划分文件并将 test 改为空列表。

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

训练入口直接等待异步模型流程，不在业务代码内创建事件循环。同步环境采样通过工作线程执行；`Environment.rollout()` 与 `Skill.assemble()` 的现有同步契约不变。

## 服务训练与自动恢复

TreeSkill 可通过通用 `training` Pipeline 配置使用，与 Skill-Forest 共用业务流程。
详见 [通用训练流程](training.md)。直接调用 `TreeSkillTrainer.train()` 时，同一输入与
同一工作目录自动复用兼容的采样、模型响应及最终产物；恢复结果保留 TreeSkill 动态路由。
旧目录缺少恢复清单时明确拒绝，不自动覆盖。独立 `run_experiment` 包装器仍要求全新实验根目录。
