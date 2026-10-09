# Skill-Forest 训练算法

`Algorithm.build("skill_forest", options)` 构造异步 `SkillForestTrainer`，实现当前
`SkillTrainer.train()` 契约。算法不读写数据库，环境由调用方提供和关闭；返回完整的
`Skill`，保留输入对象的辅助资源，不原地修改输入。此入口已通过通用 `training` Pipeline 接入 Service；配置与发布说明见 [通用训练流程](training.md)。UI 训练表单尚未接入。

## 数据与执行流程

1. 只在传入的 `train_samples` 内，为每棵树按 `seed + tree_id` 独立无放回划分。
   训练部分为 `floor(train_ratio * N)`，其余作为该树固定 validation；两部分必须非空。
   不额外 bootstrap，不做重复训练采样。`validation_samples` 保留为外部评估数据，
   不参与反思、树内评分、最终选择或合并；其 ID 必须与训练池分离。test 不进入 trainer。
2. Root 的采集范围由两个开关决定。`root_share_train_trajectory: true` 时，对所有树的
   train 和 validation 任务并集执行一次 rollout，结果同时用于各树根节点评分与训练反思；
   该开关优先于 `root_share_validation`。否则，`root_share_validation: true`（默认）时，
   对所有树 validation 的任务并集执行一次 rollout，根节点训练轨迹仍由各树独立采集。
   两个开关都关闭时，每棵树分别对自己的 validation 执行 rollout。每棵树始终只根据
   自己的 validation 计算分数。论文只规定共享 Root validation。
3. Trainer 分波调度多个父节点。每个父节点在固定树内 train 上采样一次，全部子候选
   复用此批轨迹。每个候选独立重新反思：按执行结果分为成功/失败组，
   用候选 seed 和 seed+1 分别打乱，再按 `reflection_minibatch_size`（默认 8）分批，
   使用面向任务的反思提示词提取补丁。每批最多保留 `reflection_edit_budget`（默认 4）
   个编辑，再分别层级归并成功/失败证据，合成该候选补丁。末尾单条 minibatch 也参与反思。
   默认启用 `ranking_enabled`：合并后的编辑数量超过当前深度预算时，调用优化器按重要性
   排序筛选。`ranking_edit_budgets: [4, 3, 2]` 按父节点深度 0/1/2 取值，更深层复用末项；
   未超预算不调用模型。设置 `ranking_enabled: false` 可复现没有最终 ranking 的流程，
   不影响前面的 minibatch 反思预算。无效补丁或内容无变化不进入校验。
4. 子节点按父节点失败任务优先的顺序增量校验，只有 `child_score - parent_score > min_gain`
   才接受。评分统一使用环境的 `Reward.success`，即 Hard 成功率；不假定 `Reward.score`
   一定在 [0, 1]。提前拒绝使用论文精确乐观上界，不使用统计预测或概率拒绝。
5. 先扩展全部 Root，再从全局 frontier 按父子验证增益降序安排扩展波次。
   `expansion_budget` 是全森林父节点扩展总数，包含 Root；必须至少等于树数。
   `max_depth` 以 Root 深度 0 计数。accepted 子节点各自保留，不合并兄弟或做额外 beam 剪枝。
6. 每棵树冻结一个树内得分最高的正增益非 Root 节点，统一在完整 `train_samples` 上重新评分。
   所有树都没有合格节点时明确报错，不把共享 Root 冒充论文的 ensemble 结果。

## 两种最终策略

- `ensemble_strategy: select`（默认）：按完整训练池 Hard 得分选优；同分比较相对初始
  Skill 的两文档 TF–IDF 余弦距离；再同分保留稳定原顺序。
- `ensemble_strategy: hierarchical_merge`：仅保留完整训练得分不低于最佳分减
  `merge_threshold` 的树赢家（默认差值 0.15），按论文的固定 `42:tree_id:node_id`
  SHA-256 排序后相邻两两合并。每次输入两份完整 Skill、可比分数和 unified diff，
  每个合成 Skill 在完整训练池评分一次；不按合成分数再次筛选、拒绝或回退。
  奇数项原样晋级。保留 m 个赢家时，执行 m−1 次合成及相同次数的完整训练评分。

`tree.py` 的划分、补丁应用、精确拒绝、优先选择、树赢家和新颖度计算均为无状态函数。
`generation.py` 只执行传入证据的候选生成，不拥有树状态；frontier、Root 缓存、扩展预算
及候选校验时序由 `trainer.py` 管理。

## 全局并发与失败语义

`rollout_concurrency` 限制一次 train 调用的整个森林中正在执行的环境调用；
`model_concurrency` 同样限制所有反思、补丁归并、编辑 ranking 和最终 Skill 合并调用。
限制器不会按树、节点或归并层级重新创建。`expansion_concurrency` 控制每波并行父节点数，
不替代上述物理调用上限。不同 train 调用拥有独立限制器。

训练收集、候选验证和最终评分共用全局 rollout 队列与固定 worker。
一次收集或一次候选验证构成一个任务组；worker 按组轮转，每次取一题，空闲即补位。
不区分阶段优先级，也不设置每组并发上限；只有一个组时可以使用全部 `rollout_concurrency` 名额。
验证每返回一题就检查精确上界，拒绝后撤销该组未启动任务，已启动任务继续收尾并纳入结果与调用审计。
因此提高利用率可能增加拒绝前已启动的调用。原 `validation_microbatch_size` 配置已删除，不再按批次等待。
父节点扩展仍保留原来的波次与选择规则。

节点的 `rollouts.jsonl` 保存用于分裂节点的完整训练轨迹，`evaluation.jsonl` 保存验证节点增益的完整轨迹。
每行包含任务 ID、Skill 内容哈希和环境返回的 `RolloutResult`；Root 共享的轨迹会标记为非重复物理调用。
训练采样与完整训练评分可按任务并行。发生异常或取消时排空已启动的同步调用再释放资源。

优化器模型请求遇到连接错误或暂时性状态码时默认最多重试 3 次（初次请求加重试共最多 4 次），
每次请求默认超时 21600 秒；可通过 `optimizer.max_retries` 调整重试次数。
无效 JSON/补丁不重试，记为候选无效；
重试耗尽或不可重试的模型异常，以及环境或落盘异常，会终止训练并留存失败，
不填充零分、不悄悄改变评分分母。
算法不更改调用方环境/Agent 自身的配置。

## 反思与补丁协议

公共提示词定义于 `prompts.py`，请求也由该模块按结构化字段组织。
数据集专用反思指令位于 `env_specific/`。每次 `train(env=...)` 开始时，算法读取
`Environment` 的注册名；名称与内置 adapter 的 `livemath`、`alfworld_bounded_history`、
`searchqa`、`officeqa`、`spreadsheetbench_tool_calling` 或 `spreadsheetbench_text_action` 重合时，自动加载同名 adapter，否则使用 `generic`。
未注册的自定义环境也使用 `generic`。匹配使用注册名，不根据 Python 类名或父类猜测。

用户无需传入 `env_adapter`，训练配置也不再接受该字段；请求中的环境配置无需为 Skill-Forest
提供 `adapters`。环境执行、反思、补丁合并、编辑排序和最终 Skill 合成都使用本次训练的环境及适配。
同一个 trainer 可用于不同环境，每次调用创建独立 adapter，并输出 INFO 日志，例如：

```text
Skill-Forest environment=livemath adapter=livemath (automatic selection)
```

实际选用的 adapter 类型及配置写入训练目录的 `config.json`，参与恢复一致性校验。
底层 `SkillForestEnvAdapter` 保留独立构造及提示词渲染接口，供离线验证使用；训练入口的选择由环境名决定。


每个候选在成功、失败两类训练轨迹上独立反思，提取可泛化的文本编辑；同类编辑分层
去重归并，再优先保留纠错规则，补充兼容的成功经验。每个编辑采用 `append`、
`insert_after`、`replace` 或 `delete`，非追加操作必须提供唯一精确目标。
空补丁表示证据不足以支持修改。支持数量缺省时使用实际批次大小。

共享 `RANKING_PROMPT` 在公共 `SkillForestEnvAdapter` 中以三引号多行常量定义，所有环境继承。
Ranking 保持模型给出的优先顺序，去重并过滤越界索引，最多保留当前预算个编辑；没有有效索引
或响应格式无效时标记候选无效，不回退到直接截断。模型传输错误仍终止训练。
候选 patch 应用成功后，将完整候选 Skill 原子写入节点的 `node.json`；反思、排序和模型调用
不再各自创建恢复文件。LiveMath pilot 复用同一流程，只定制数据、随机种子与最终选优规则。

`generic` 使用与 OfficeQA 相同的 Markdown 任务头、初始提示词区和 `[agent]` / `[action]` / `[obs]` 消息结构，并排除评分事件及私有失败原因；自定义工具保留参数，不套用 OfficeQA 的工具默认值。缺少初始提示词或只有部分工具调用时保留现有证据。专用环境使用原 Skill-Forest
项目的 Markdown 反思与补丁合并协议，通过统一 `RolloutResult` 读取已记录的执行证据：

- LiveMath：实际 system/user 输入、模型回答与选项评分。
- SearchQA：实际上下文、逐轮回答、EM/F1 及训练参考答案反馈。
- OfficeQA：任务类型、原版工具命令/对应返回与原始失败原因；正文从模型回复开始，Steps 按回复与工具交换计数。`read` 默认 80 行、最多 4000 字符且不加行号；`grep` 最多 20 个匹配，`glob` 最多 50 个结果。工具仍限定在任务的公开文档白名单内。
- SpreadsheetBench：按原始原生工具分支格式保留多 case 对话与 case 分界；工具返回显示为 `[tool]`，不展开 assistant 的 `tool_calls`，不追加逐格校验报告。任务头部保留失败原因。文本动作环境共用此渲染规则。
- ALFWorld：实际 simulator 任务描述、任务类型、逐步 reasoning/action/observation 及终止原因。

专用反思可读取训练结果里的评分反馈（可能含正确答案）；不直接读取 `reward_data`、
私有 workbook 或重新构造当前版本的执行提示词。此反馈只用于训练反思，不追加到执行 agent 的输入。
文本不截断；缺少必需执行上下文时明确报错，OfficeQA 还校验工具配对。旧 ALFWorld 结果未保存
simulator 描述时使用数据任务描述；旧 SpreadsheetBench 结果未保存轮数时按 assistant 消息计数。
输入任务与结果数量不一致时直接报错。

离线对照验证的是统一结果到优化器输入的渲染协议，不代表各环境执行后端或真实模型效果
与原项目等价。当前 OfficeQA 使用原生文档工具；SpreadsheetBench 保留当前配置所选的
工具交互模式及其实际输入。

层级 Skill 合并接收两份完整正文、完整训练池 Hard 分数及文本差异；分数用于处理
真正冲突，互补规则保留具体条件、动作和验证要求。离线测试覆盖请求协议、编辑应用、
并发调度与数据隔离；真实模型下的补丁质量和最终效果仍需实验验证。

## 使用

独立调用时传入算法配置，而非 Service YAML。调用方需自行配置真实模型连接和
Environment；下面的模型名称仅作占位，不在导入或构造时调用。

```python
from pathlib import Path

from mindmemos_skill.algos import Algorithm

options = {
    "optimizer": {"model": "replace-with-optimizer-model"},
    "num_trees": 3,
}
trainer = Algorithm.build("skill_forest", options)
selected = await trainer.train(
    skill=initial_skill,
    env=environment,
    train_samples=train_samples,
    validation_samples=validation_samples,
    work_dir=Path("output/skill_forest/run_001"),
)
selected.export("output/skill_forest/selected")
```

首次运行使用全新工作目录；恢复时传入相同输入和相同目录。训练目录为：

```text
<work_dir>/
├── config.json                  # 脱敏训练与环境配置、树内划分、输入指纹
├── result.json                  # 完成后的训练结果、选中 Skill、验证与测试汇总
├── skills/
│   └── skill-tree-<id>.md       # 有正增益 winner 的树才生成
└── trees/
    └── tree-<id>/
        └── node-<id>/
            ├── node.json       # patch 应用后的完整 Skill 与节点状态
            ├── evaluation.jsonl # 节点增益验证的轨迹，一行一条
            ├── rollouts.jsonl   # 节点分裂所用的轨迹，一行一条
            └── workspaces/
                ├── train/      # 每个任务独立的临时工作空间
                └── eval/
```

文件按实际发生的阶段创建，未扩展的节点不一定有 `rollouts.jsonl`。`cleanup_workspaces: true`
是默认值：每次 rollout 完成或失败后删除其任务工作空间；设为 `false` 时保留，便于调试。
当通用训练 Pipeline 完成外部验证和测试时，将两项汇总写入 `train/result.json` 的
`validation` 与 `test` 字段；直接调用 Trainer 时这两项为 `null`。

恢复行为：

- `config.json` 的指纹校验原始 Skill、数据及顺序、划分、配置、环境和实现；运行锁阻止两个训练器同时写入。
- `node.json` 在 patch 应用后原子提交完整候选 Skill。若验证尚未完成，恢复复用此 Skill，整组验证从第一题重新执行；`evaluation.jsonl` 重新写入，不复用单题结果。
- 已完成验证的接受或拒绝状态保存在同一 `node.json`。父节点所有候选处理完毕后，才把父节点标记为已扩展；恢复据此重建 frontier 和剩余扩展预算。
- Root 评分未提交时重新执行；最终选择中断时重新评分赢家。单次模型调用和 rollout 不设恢复检查点，硬中断可能重跑已发送的请求。
- 旧格式目录不能直接恢复到新格式；需为旧的未完成运行选择新目录。完成的 `result.json` 可直接复用。

```python
selected = await trainer.train(
    skill=initial_skill,
    env=env,
    train_samples=train_samples,
    validation_samples=validation_samples,
    work_dir=Path("output/skill_forest/run_001"),
)
```

LiveMath 试点入口支持 `--mode run --resume`，覆盖训练、Select 和三轮测试；仅适用于本版本创建的兼容运行。旧试点只有审计产物，缺少恢复提交记录，必须显式迁移并校验，不能直接传 `--resume`。

Trainer 根据目录自动判断恢复，无需传入恢复开关。硬中断时已发送但尚未提交结果的请求可能重跑；不保证远端计费请求的 exactly-once。环境内部步骤和单题结果均不单独恢复。

离线验证命令：

```shell
.venv/bin/pytest -q src/mindmemos_skill/tests/test_skill_forest*.py
```

离线测试覆盖执行协议与数学条件，不代表真实模型效果或各数据集已完成端到端实验。
