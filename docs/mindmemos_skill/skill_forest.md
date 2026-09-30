# Skill-Forest 训练算法

`Algorithm.build("skill_forest", options)` 构造异步 `SkillForestTrainer`，实现当前
`SkillTrainer.train()` 契约。算法不读写数据库，环境由调用方提供和关闭；返回完整的
`Skill`，保留输入对象的辅助资源，不原地修改输入。此入口已通过通用 `training` Pipeline 接入 Service；配置与发布说明见 [通用训练流程](training.md)。UI 训练表单尚未接入。

## 数据与执行流程

1. 只在传入的 `train_samples` 内，为每棵树按 `seed + tree_id` 独立无放回划分。
   训练部分为 `floor(train_ratio * N)`，其余作为该树固定 validation；两部分必须非空。
   不额外 bootstrap，不做重复训练采样。`validation_samples` 保留为外部评估数据，
   不参与反思、树内评分、最终选择或合并；其 ID 必须与训练池分离。test 不进入 trainer。
2. 对所有树 validation 的任务并集执行一次 Root rollout，按任务分发结果，每棵树仍
   根据自己的 validation 计算分数。`root_share_train_trajectory: true` 时，同一 Root
   已有的 validation 轨迹也复用于树内训练，仅补采缺失任务；默认 false，各树独立采集
   Root 训练轨迹。该开关是用户要求的扩展，论文只规定共享 Root validation。
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

每次训练在输出目录追加 `scheduler.jsonl`，记录组提交/停止/完成、单次 rollout 启动/完成/失败/释放。
日志包含事件序号、相对运行时间、组目录、任务 ID、全局及组内排队/执行数量、排队时间与执行耗时。
`group_stopped` 的 `removed` 是撤销的未启动任务数，`group_active` 是还需收尾的调用数；
`rollout_completed.after_stop` 标记停止后返回的结果。日志不写任务正文和私有评分数据。
结合 `rollout_released` 与下一次 `rollout_started` 可定位有无排队工作时的资源空闲，
具体调用输入输出仍查看各任务目录的 `call.json`。日志写入失败会终止相关任务组，不静默丢弃。
训练采样与完整训练评分可按任务并行。共享线程池大小由上述上限决定，不受默认
`asyncio.to_thread` 线程数限制。发生异常或取消时排空已启动的同步调用再释放资源，
不会在后台留下尚未记录的训练调用。

模型调用禁止自动重试，默认超时 21600 秒。无效 JSON/补丁记为候选无效；
真实模型、环境或落盘异常终止训练并留存失败，不填充零分、不悄悄改变评分分母。
算法不更改调用方环境/Agent 自身的配置。

## 反思与补丁协议

公共提示词定义于 `prompts.py`，请求也由该模块按结构化字段组织。
数据集专用反思指令位于 `env_specific/`，通过 `env_adapter.class_name` 显式选择
`generic`、`livemath`、`alfworld_bounded_history`、`searchqa`、`officeqa` 或 `spreadsheetbench`。
环境执行由调用方传入的 Environment 负责。

`SkillForestTrainer` 在初始化时通过 `SkillForestEnvAdapter.build()` 绑定适配器，省略 `env_adapter` 时使用 `generic`。`train()` 不再接收或切换适配器。配置格式为：

```yaml
env_adapter:
  class_name: livemath
  config: {}
```

用户通过 `@SkillForestEnvAdapter.register("my_env")` 注册子类，并在加载配置前显式导入自己的模块。只需覆盖变化的方法：`render_trajectory()` 调整单条证据文本，`build_reflection_messages()` 调整完整反思请求；`build_patch_merge_messages()`、`build_final_patch_messages()`、`build_ranking_messages()` 和 `build_skill_merge_messages()` 分别负责同类补丁合并、最终补丁合并、编辑排序与完整 Skill 的合并消息。通用类提供所有默认行为。

Adapter 的 `config` 可用完整文本覆盖 `analyst_error`、`analyst_success`、`merge_failure`、`merge_success`、`merge_final`、`ranking`、`hierarchical_merge_skills`，未配置时保留原始资源。Adapter 不持有本次运行的模型、目录或中间任务状态，评分与并发规则仍由算法负责。


每个候选在成功、失败两类训练轨迹上独立反思，提取可泛化的文本编辑；同类编辑分层
去重归并，再优先保留纠错规则，补充兼容的成功经验。每个编辑采用 `append`、
`insert_after`、`replace` 或 `delete`，非追加操作必须提供唯一精确目标。
空补丁表示证据不足以支持修改。支持数量缺省时使用实际批次大小。

共享 `RANKING_PROMPT` 在公共 `SkillForestEnvAdapter` 中以三引号多行常量定义，所有环境继承。
Ranking 保持模型给出的优先顺序，去重并过滤越界索引，最多保留当前预算个编辑；没有有效索引
或响应格式无效时标记候选无效，不回退到直接截断。模型传输错误仍终止训练。
调用前保存 `pre_ranking_patch.json`，完整请求与响应保存到 `ranking/call.json`，筛选结果
写入 `patch.json`。LiveMath pilot 复用同一流程，只定制数据、随机种子与最终选优规则。

`generic` 保持公开轨迹 JSON 请求，并排除评分事件。五个专用环境使用原 Skill-Forest
项目的 Markdown 反思与补丁合并协议，通过统一 `RolloutResult` 读取已记录的执行证据：

- LiveMath：实际 system/user 输入、模型回答与选项评分。
- SearchQA：实际上下文、逐轮回答、EM/F1 及训练参考答案反馈。
- OfficeQA：任务类型、文档工具名称/参数/对应返回与原始失败原因。
- SpreadsheetBench：完整多 case 对话、工具交换、case 分界及一次逐格校验报告。
- ALFWorld：实际 simulator 任务描述、任务类型、逐步 reasoning/action/observation 及终止原因。

专用反思可读取训练结果里的评分反馈（可能含正确答案）；不直接读取 `reward_data`、
私有 workbook 或重新构造当前版本的执行提示词。此反馈只用于训练反思，不追加到执行 agent 的输入。
文本不截断；缺少必需执行上下文或工具配对不完整时明确报错。旧 ALFWorld 结果未保存
simulator 描述时使用数据任务描述；旧 SpreadsheetBench 结果未保存轮数时按 assistant 消息计数。
输入任务与结果数量不一致时直接报错。

离线对照验证的是统一结果到优化器输入的渲染协议，不代表各环境执行后端或真实模型效果
与原项目等价。当前 OfficeQA 使用原生文档工具；SpreadsheetBench 保留当前配置所选的
工具交互模式及其实际输入。

层级 Skill 合并接收两份完整正文、完整训练池 Hard 分数及文本差异；分数用于处理
真正冲突，互补规则保留具体条件、动作和验证要求。离线测试覆盖请求协议、编辑应用、
并发调度与数据隔离；真实模型下的补丁质量和最终效果仍需实验验证。

## 使用

配置示例为 `config/mindmemos_skill/skill_forest/default.yaml`，是算法配置而非 Service YAML。
调用方需自行配置真实模型连接和 Environment，示例模型名称仅作占位，不在导入或构造时调用。

```python
from pathlib import Path

import yaml

from mindmemos_skill.algos import Algorithm

options = yaml.safe_load(Path("config/mindmemos_skill/skill_forest/default.yaml").read_text())
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

首次运行使用全新工作目录；恢复时传入相同输入和相同目录，Trainer 自动复用兼容检查点。输出包括脱敏 `config.json`、树内
`splits.json`、初始/最终 Skill 快照、`forest.json`、逐候选 patch/diff/决策、逐调用
请求/结果、最终赢家/合并谱系和 `summary.json` 的实际调用数与用量。
`status.json` 只有在选定结果写入后才标为 completed。usage 字段存在不代表提供方
确认了用量，默认零值不能解读为无消耗。

恢复行为：

- `generated.json` 在单个候选的反思、合并、排序及 patch 应用全部完成后原子提交；恢复直接加载最终 Skill。
- `validated.json` 提交接受、拒绝、提前拒绝、无效 patch 或无变化的终态。正常拒绝不会在恢复时获得额外生成机会。
- 每条 rollout 和 optimizer 响应完成后保存带输入指纹的 `completed.json`。恢复先加载完整结果，只有缺失任务进入共享队列；验证从已有分数计算提前拒绝，剩余题目保持原编号。
- 每波父节点列表在执行前保存在 `waves/`。恢复按固定波次重建森林，加载候选与决策，不重复扣除扩展预算。最终 Select 和合并复用相同的调用缓存。
- `recovery_manifest.json` 校验原始 Skill、数据及顺序、划分、配置、环境和源码指纹；协议或源码变化时拒绝恢复。运行锁防止两个 trainer 同时写入同一目录。
- `recovery.jsonl` 单独记录恢复和缓存命中；调度日志包含运行 session ID。重做未完成调用前，将旧 `call.json` 移入 `attempts/`，统计仍计入历史物理调用，缓存命中不计作新调用。

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

Trainer 根据目录自动判断恢复，无需传入恢复开关。硬中断时，已发送但尚未提交结果的请求可能重跑；不保证远端计费请求的 exactly-once。环境未完成的内部步骤也不恢复，只恢复完整单题结果。

离线验证命令：

```shell
.venv/bin/pytest -q tests/mindmemos_skill/test_skill_forest*.py
```

离线测试覆盖执行协议与数学条件，不代表真实模型效果或各数据集已完成端到端实验。
