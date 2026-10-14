# MindMemOS Eval + Skill-Forest 五数据集测试交接指南

本文供测试执行者从零准备数据、检查环境、运行评测并交付结果。所有命令均在仓库根目录执行。

**本次已确认：先测 7 棵树的 Select 和 Merge；target / optimizer 均为 `gpt-5.4-mini`，`reasoning_effort=medium`；父节点扩展 / 训练 rollout / optimizer / 测试并发均先用 20。目标是对齐 Skill-Forest 论文协议。**

当前已提供两套可加载配置及完整运行命令，但“严格复现”尚有下文列出的实现核对项。配置能加载、离线测试通过或小样本运行成功，都不代表论文效果已经复现。不要把这些配置称为经过搜索验证的“全局最优”。

## 1. 测什么，以及如何与论文比较

运行入口是 `mindmemos-eval skill run`，算法固定为 `skill_forest`。它在进程内建立独立 SQLite Skill service，无需启动 Memory server、HTTP Skill 服务、UI 或 Kafka，也无需 `.mindmemos` 项目凭据。

| 数据集 / CLI 名称 | train | val | test | 执行协议 |
| --- | ---: | ---: | ---: | --- |
| LiveMath / `livemath` | 35 | 18 | 124 | 单轮选择题，无外部工具 |
| ALFWorld / `alfworld` | 39 | 18 | 134 | bounded-history，最多 50 步，保留最近 2 步 |
| SpreadsheetBench / `spreadsheetbench` | 80 | 40 | 280 | verified 400，tool-calling，最多 30 轮 |
| SearchQA / `searchqa` | 400 | 200 | 1400 | 单轮上下文问答，最多 6000 字符上下文 |
| OfficeQA / `officeqa` | 50 | 24 | 172 | 本地文档工具，最多 24 轮 |

这些数量来自已提交的 `resources/mindmemos_skill_eval/skill_forest-*/data_splits/splits.json`，与本次核对的论文源稿一致。“全量”指这些固定划分，不是上游数据仓库中所有题目。Memory 基准、HiTab、WikiTableQuestions 和 SpreadsheetBench 912 不在本轮范围。

每个数据集比较 `no_skill`、`initial_skill`、`trained_skill`。每份冻结 Skill 测试 3 次，以三次 **Hard 成功率的均值**作为主指标；这是一次训练后的三次测试，不是三次独立训练。不得选择三轮中的最好一轮，也不得用 test 选择树赢家、聚合策略或超参数。

Skill-Forest 只在 released train 内按每树 `seed + tree_id` 划分树内 train/validation，比例为 0.5。外部 val 和 test 不参与森林优化与最终选优。不要将外部 val 合并进 train；不要把树内 validation 与表中的 val 混淆。

## 2. 正式运行前的论文对齐检查

本次依据本机 `skill-forest/paper/sections/05_results.tex`、`04_method.tex`、`99_appendix.tex` 及当前实现核对。论文源稿不在本仓库内；交接时项目负责人应附同一版本的论文和原实验配置/轨迹，不能只给执行者一个本机路径。

| 项目 | 已核对情况 / 处理要求 |
| --- | --- |
| 模型、划分数量、树数、深度、分支数、预算 | 当前方案与论文 7-tree 设置一致；数据 ID、顺序及初始 Skill 全文仍以固定资源和原实验记录为准 |
| 补丁应用 | 通用配置原为 `exact`；本指南专用配置改为 `reference`，复用原 SkillOpt 的行后插入与缺失目标处理规则。它是协议选择，不代表更高分 |
| Select 选优 | 完整训练池 Hard 分数优先；同分按相对初始 Skill 的 TF–IDF 新颖度选优 |
| Select 与 Merge 的共同森林 | **当前 CLI 不支持从一次森林构建同时导出两种聚合结果。两条 `skill run` 会分别重新训练，即使 seed 相同也不保证森林相同。** 严格复现论文的同源聚合比较，需要先提供共享森林的聚合入口；否则结果只能标注为两次独立训练的实现评测 |
| 初始 Skill 注入 | 当前 `skill run` 加载标准 Skill。仓库历史 `env_test/paper_repro` 配置使用 `skill_body_only: true`，而 `skill run` 没有该开关。需核对实际模型输入中的 frontmatter、包装和正文，不能仅比较文件正文 |
| 请求失败行为 | 当前共用模型默认 `retry_policy=transient`、`retry_max_wait_s=60`；历史 paper_repro 配置为 `all` / `30`。SpreadsheetBench 历史配置还有 `invalid_tool_arguments=empty`，当前共用模型默认 `error`。需以原运行记录确认并记录差异；不能把历史测试配置自动当作五数据集全部训练参数 |
| 环境与评分 | 回合上限已核对。仍需按数据集比较实际提示词、工具返回、评分器版本；LiveMath 阶段选项顺序有离线回归，但不能替代其余四项执行后端对照 |
| 重复测试与预算 | 本指南固定 `--test-repeats 3`；总预算和最长时间尚未约定。先用冒烟轨迹估计费用与时间，再安排全量 |

**启动规则：** 数据下载、依赖准备和离线校验可以先做；带模型调用的冒烟用于核实协议和稳定性。上表差异未解决时，可以运行并报告“当前实现的独立 Select/Merge 测评”，但不得宣称完成严格论文复现。执行者应先将差异提交给项目负责人决定；不要自行修改协议追求论文分数。

## 3. 获取代码与安装环境

取得项目负责人交付的仓库版本，在仓库根目录记录：

```bash
git rev-parse HEAD
git status --short
uv --version
```

当前开发工作区存在未提交的算法修改，不能只记录 HEAD 就认为另一台机器拿到相同实现。交接前需由负责人冻结相关改动，提供包含代码、配置示例、初始 Skill、split 和回归 fixture 的版本。运行中不更新代码。

安装 `uv` 后使用项目的 Python 3.13 环境和锁定依赖：

```bash
uv sync --frozen --all-packages --extra alfworld --extra spreadsheetbench
make init
uv run --no-sync mindmemos-eval skill run --help
uv run --no-sync mindmemos-eval skill envs
```

`make init` 复制缺少的本地配置，也可能应用 `.env` 中的模型设置。初始化后重新检查下面两套本地配置。无需执行会删除数据库卷的 `make clean`。

ALFWorld 会启动真实 TextWorld simulator；安装成功不等于 simulator 可运行。SpreadsheetBench 会执行表格操作及 shell 工具，应在独立测试机器或隔离环境中运行。当前协议关闭事务式重算；只有另行开启重算才需要 LibreOffice，并且这会改变协议。

## 4. 下载并校验五个数据集

来源与 revision 固定在 [dataset_download_sources.json](../../config/presets/dataset_download_sources.json)。复用仓库下载脚本，不手工换源、替换文件或重排题目。

```bash
uv run --no-sync python scripts/download_datasets.py list skill

for benchmark in livemath alfworld spreadsheetbench searchqa officeqa; do
  uv run --no-sync python scripts/download_datasets.py download skill \
    --name "$benchmark" || break
done
```

遇到失败先修复该项，再继续。这里不使用 `download skill --all`，因为它还包含本轮不用的三个数据集。

| 数据集 | 默认下载目录下应有的关键内容 |
| --- | --- |
| LiveMath | `data/mindmemos_skill/livemath/data/{202511,202512,202601,202602}/qa_*_final.json` |
| ALFWorld | `data/mindmemos_skill/alfworld/` 下的 `json_2.1.1/`、`configs/base_config.yaml`、`logic/` 及下载归档展开内容 |
| SpreadsheetBench | `data/mindmemos_skill/spreadsheetbench/spreadsheetbench_verified_400/dataset.json` 和配套工作簿 |
| SearchQA | `data/mindmemos_skill/searchqa/data/*.parquet` |
| OfficeQA | `data/mindmemos_skill/officeqa/officeqa_full.csv` 和 `treasury_bulletins_parsed/` 文档 |

每个数据集目录还必须有 `download_receipt.json`。下载完成后逐项验证：

```bash
for benchmark in livemath alfworld spreadsheetbench searchqa officeqa; do
  uv run --no-sync python scripts/download_datasets.py download skill \
    --name "$benchmark" --verify-only || break
done
```

五项都成功后再继续。已有目录只校验，不覆盖；校验失败时保留原目录排查，不能伪造 receipt。下载需要认证时在本地环境配置 `HF_TOKEN`，不要写入公开 YAML。`--offline` 只适用于已有 Hugging Face 缓存或本地数据，不会替代首次联网下载。

自定义位置时，下载和运行都指定相同的父目录：下载使用 `--data-root /path/to/data`，评测使用 `--data-root /path/to/data`；最终数据位于其 `<benchmark>/` 子目录。

SearchQA 的固定来源是 `lucadiliello/searchqa`，revision 为 `c1a979068ba118d85467179b704031d113d689cc`；按原生 `key` 对齐 split，运行时生成 `prepared/searchqa/items.json`，不需要手工转换 Parquet。ALFWorld 的固定游戏清单位于资源目录的 `games.json`。

## 5. 配置模型和森林参数

两套配置已随本指南准备：

| 策略 | 可共享示例 | 本地运行文件 |
| --- | --- | --- |
| Select | `config/mindmemos_eval/forest_select/dev.example.yaml` | `config/mindmemos_eval/forest_select/dev.yaml` |
| Merge | `config/mindmemos_eval/forest_merge/dev.example.yaml` | `config/mindmemos_eval/forest_merge/dev.yaml` |

`make init` 当前不自动创建这两套新增配置。在新机器上执行以下命令，仅复制缺少的文件：

```bash
for variant in forest_select forest_merge; do
  if [ ! -f "config/mindmemos_eval/$variant/dev.yaml" ]; then
    cp "config/mindmemos_eval/$variant/dev.example.yaml" \
      "config/mindmemos_eval/$variant/dev.yaml"
  fi
done
```

只在本地文件填写 `models.target` 和 `models.optimizer` 的 `api_key`、`base_url`。两组连接都必须有效，测试模型还需支持环境要求的工具调用。不要把密钥发到报告或提交到 Git。

本机本地专用配置保留现有 eval 的连接配置；新机器不能依赖这些本机文件。通用 `config/mindmemos_eval/dev.yaml` 保持原样，运行时必须显式使用本指南的 `--config`。

| 参数 | 本次设置 | 含义 |
| --- | --- | --- |
| target / optimizer model | `gpt-5.4-mini` | 用户已确认，与论文模型名称一致 |
| 两者 `reasoning_effort` | `medium` | 用户已确认 |
| 两者 `max_completion_tokens` | 16384 | 当前复现候选值，包含 reasoning；需观察截断 |
| 两者 `timeout_s` / `max_retries` | 600 / 10 | 每次请求超时 / 额外重试次数，不是整题时限 |
| `num_trees` / `train_ratio` / `seed` | 7 / 0.5 / 42 | 树数 / 树内训练比例 / 森林 seed |
| `candidates_per_node` / `max_depth` | 3 / 3 | 每个父节点候选数 / 最大子节点深度 |
| `expansion_budget` | 30 | 整个森林父节点扩展总预算，包含根节点，不是每棵树 30 |
| `ensemble_strategy` | `select` 或 `hierarchical_merge` | 两份配置唯一的非敏感参数差异 |
| `selection_tiebreak` / `patch_application` | `novelty` / `reference` | 同分选优 / 原实现补丁应用语义 |
| `root_share_validation` / `root_share_train_trajectory` | true / false | 共享根验证，不额外共享根训练轨迹 |
| `min_gain` / `early_rejection` | 0.0 / true | 严格正增益；用精确乐观上界提前拒绝 |
| `validation_batch_size` | 8 | 当前实现每批完成后检查提前拒绝 |
| `reflection_minibatch_size` / `reflection_edit_budget` | 8 / 4 | 每批反思轨迹数 / 编辑预算 |
| `merge_batch_size` | 8 | 候选内补丁分层归并批次 |
| `ranking_enabled` / `ranking_edit_budgets` | true / `[4, 3, 2]` | 子节点深度 1/2/3 的编辑预算 |
| `merge_threshold` | 0.15 | Merge 保留完整训练分不低于最佳分减 0.15 的赢家 |
| `cleanup_workspaces` | true | 清理单题工作空间，保留训练审计产物；排查需保留工作簿时应事先关闭 |
| `expansion_concurrency` | 20 | 每波并行父节点上限，可能影响搜索时序；不要在正式运行中变更 |
| `rollout_concurrency` | 20 | 单次训练森林中环境调用的全局上限 |
| `model_concurrency` | 20 | optimizer 调用全局上限 |
| `evaluation_concurrency` | 20 | 外部验证和测试并发上限 |

训练 rollout 的 target 请求和 optimizer 请求可能同时发生，所以“每层 20”不等于整个 API 账户最多 20 请求。多开进程会叠加并发。本轮先串行跑各数据集和策略，确认限流、CPU、内存和磁盘情况后再考虑调整。

`seed=42` 控制森林划分，并不保证模型输出确定。当前模型采样 `temperature`、`top_p` 和 `seed` 未显式指定；不要把森林 seed 当成模型采样 seed。论文未明确的信息应从原运行请求核对，不凭经验补成“最优值”。

改配置时同步维护同目录的 `dev.example.yaml` 和 `dev.yaml` 字段结构及非敏感值；保留本地连接，只在示例中使用公开值或占位值。

## 6. 先做离线检查和冒烟

不调用模型的检查：

```bash
uv run --no-sync pytest -q \
  src/mindmemos_eval/tests/test_skill_eval.py \
  src/mindmemos_skill/tests/test_skill_forest_paper_protocol.py
```

测试文件及其 fixture 必须随交接版本提供。该检查验证接口与部分协议，不验证真实模型效果。

先对五数据集跑两组对照，确认模型、数据、工具和评分链路：

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_select/dev.yaml \
  --benchmark all --smoke --arms no_skill,initial_skill \
  --output-dir output/mindmemos_eval/handoff/smoke_controls_01
```

再验证训练和两种聚合策略：

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_select/dev.yaml \
  --benchmark all --smoke --arms trained_skill \
  --output-dir output/mindmemos_eval/handoff/smoke_select_01

uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_merge/dev.yaml \
  --benchmark all --smoke --arms trained_skill \
  --output-dir output/mindmemos_eval/handoff/smoke_merge_01
```

每条命令结束后检查退出码、`report.md` 和 `summary.json`，再运行下一条。`--smoke` 只取前 4 train / 2 val / 2 test，并把树数、候选数、深度、预算都降为 1；其余模型和环境参数不变。**它不能验证 7 树搜索，也无法充分验证多赢家 Merge。** 若没有正增益子节点，训练会明确失败；小样本中这种失败不等于 API 故障，不要修改成功判定绕过它。

运行会先检查所有选中数据的 receipt、split 和实际依赖；预检失败写 `diagnostics.json`。`skill run` 没有独立 `--check` 参数，不要套用 `skill treeskill --check`。

检查逐题轨迹中的实际输入、模型回答、工具调用和评分：区分真正答错、截断、格式解析失败、超时、限流和工具错误。没有配套轨迹证据时，不根据总分推测原因。

## 7. 全量运行 Select 与 Merge

以下是**现有 CLI 可直接执行的独立训练方案**；第 2 节的同森林聚合要求仍需先解决，才能把它升级为严格同源论文对照。先保留所有结果，不因低分重跑后只汇报较高分。

Select 同时运行三组：

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_select/dev.yaml \
  --benchmark all --test-repeats 3 \
  --arms no_skill,initial_skill,trained_skill \
  --output-dir output/mindmemos_eval/handoff/select7_full_01
```

Merge 运行训练组，对照引用上面同模型、同环境、同 test ID 的结果，避免重复支付对照成本：

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_merge/dev.yaml \
  --benchmark all --test-repeats 3 --arms trained_skill \
  --output-dir output/mindmemos_eval/handoff/merge7_full_01
```

CLI 不会自动跨这两个目录生成增益表。汇总时必须核对两份 `config.json` 中模型、环境、测试 ID 与顺序相同，并注明森林独立训练。若需要 Merge 目录自带三组对照与 gains，把 `--arms` 改为 `no_skill,initial_skill,trained_skill`，这会重新测试两组对照。

`--benchmark all` 顺序为 LiveMath → ALFWorld → SpreadsheetBench → SearchQA → OfficeQA。运行阶段某项失败后继续后续项，最终返回非零；预检阶段任意项失败则整个套件不开始。也可将 `all` 换为单项名称逐个运行，便于失败隔离。每次必须使用新输出目录。

本轮 5 个 test split 合计 2110 题。两组对照加 Select/Merge 各测试三次，共计划 **25,320 次 held-out 任务执行**，还不含训练、外部验证、完整训练池选优和 Merge 评分。多轮环境每题有多次模型请求；该数不能直接当 API 请求数或预算。主结果不包含论文其他方法、5-tree 组、Oracle 或五次完整训练的消融实验。

## 8. 读取结果与失败恢复

| 文件 / 目录 | 用途 |
| --- | --- |
| 根 `config.json` | 脱敏模型配置、实际 forest 配置、环境、split ID 与资源路径 |
| 根 `summary.json` / `summary.csv` / `report.md` | 逐数据集、逐 arm、逐 repeat 的覆盖率、分数、成功率、时间和 token |
| `<benchmark>/results.json` | 原始逐题结果引用及失败原因 |
| `<benchmark>/initial_skill.json` / `trained_skill.json` | 初始与最终 Skill 快照及版本信息（训练成功后具备最终文件） |
| `<benchmark>/training/` | 森林、节点、反思与评测产物，外部 validation/test |
| `diagnostics.json` | 预检失败明细（预检失败时生成） |

训练 pipeline 的第一次测试经核对后用于 repeat 1，repeat 2/3 使用同一发布版本。`training_seconds` 包含训练 pipeline 的外部验证和首次测试，不能当纯优化耗时。

以 `summary.json.rows` 的 `success_rate` 计算三轮 Hard 均值；`repeats` 当前汇总的是 `mean_score`，不得未经检查直接当 Hard。当前内置 `gains` 只比较同一 run 内相同 repeat、ID、seed 的组。报告中的缺失 token 为 `null`，不能补成 0；当前 runner 没有为 `optimizer_usage` 赋完整汇总值，报告出现 null 时应从训练审计另行核算或标明不可得，不能声称已得到完整优化成本。

有效结果应满足全部计划题完成、无执行错误、coverage=1、incomplete=false。普通 reward=0 是完成但失败的任务；运行/评分异常单列。覆盖不全的组不发布完整准确率，也不能只对成功返回的题重新计算高分。

重新生成报告不会调用模型：

```bash
uv run --no-sync mindmemos-eval skill report \
  output/mindmemos_eval/handoff/select7_full_01
uv run --no-sync mindmemos-eval skill report \
  output/mindmemos_eval/handoff/merge7_full_01
```

`skill run` 不提供 `--resume`，非空目录会拒绝运行。中断后保留目录，先用 `report` 读取已有记录，再由负责人决定新目录重跑失败数据集。不能因为底层 trainer 支持恢复就假定本 CLI 自动续跑。`results.json` 的原始结果引用含绝对路径；拷贝目录后应保存现有报告，不能保证换机器后 `report` 可直接重新校验。

## 9. 测试人员应交付什么

交付代码版本与相关未提交改动说明、依赖版本、机器 CPU/内存/系统、模型提供商及实际模型标识、脱敏配置、数据 receipt、split、初始与最终 Skill、完整输出目录和控制台日志。密钥不进入交付包。

每个数据集独立填表，不用题量加权总分掩盖不同任务的变化。若补充论文 Overall，使用五个数据集 Hard 均值的算术平均，并注明这是额外汇总；当前 CLI 不自动计算 Overall。

| 数据集 | No-skill Hard（三轮均值） | Initial Hard | Select-7 Hard | Merge-7 Hard | Select / Merge 相对 Initial 的百分点差 | 覆盖率 / 错误 | 时间与 token（缺失注明） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LiveMath | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| ALFWorld | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| SpreadsheetBench | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| SearchQA | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| OfficeQA | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

附每轮值与标准差、全部失败和重跑记录、协议差异清单。三轮测试波动不能代表重新训练的稳定性；未做统计检验时不写“显著提升”。SpreadsheetBench tool-calling 的结果不要与 TreeSkill text-action 的历史 61% 直接对比。

参考实现：[评测入口](../../src/mindmemos_eval/mindmemos_eval/skills/cli.py)、[执行与预检](../../src/mindmemos_eval/mindmemos_eval/skills/runner.py)、[数据准备](../../src/mindmemos_eval/mindmemos_eval/skills/data.py)、[结果统计](../../src/mindmemos_eval/mindmemos_eval/skills/report.py)、[Skill-Forest 参数](../../src/mindmemos_skill/mindmemos_skill/algos/train/skill_forest/config.py)。
