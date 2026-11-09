# MindMemOS Skill 评测

[English](skill_evaluation.md) · [Memory 评测](memory_evaluation_ZH.md)

以下命令从仓库根目录执行。各节分别列出数据、运行入口和对标指标。空白分数待补充。

```bash
uv sync --all-packages
make init
uv run --no-sync mindmemos-eval skill envs
```

## Skill-Forest

### 数据集与评分

按需下载以下五个基准。下载脚本使用[固定版本的数据源](../../config/presets/dataset_download_sources.json)，默认保存到 `data/mindmemos_skill/`。Hugging Face 受限数据需先开通访问权限，再在环境变量或本地 `.env` 中设置 `HF_TOKEN`。

```bash
for benchmark in livemath alfworld spreadsheetbench searchqa officeqa; do
  uv run --no-sync python scripts/download_datasets.py download skill --name "$benchmark" || break
done
```

| 数据集 | 测评逻辑 |
| --- | --- |
| LiveMath | 提取最终选项，与标准选项匹配。 |
| ALFWorld | 在只保留有限交互历史的环境中运行任务，以模拟器的 `won` 为准。 |
| SpreadsheetBench Verified 400 | 使用原生工具调用。各工作簿用例的指定评分区域均与标准单元格值匹配，整题才通过。默认关闭自动重算。 |
| SearchQA | 使用原生题目 ID 的固定划分，训练 400 题、验证 200 题、测试 1,400 题。归一化后与任一标准答案完全匹配即通过，另记 Token F1 和子串匹配。 |
| OfficeQA | 使用美国财政部文档。固定版本的 OfficeQA 评分器按数值、单位、有序列表和文本规则对最终答案给出二值分数。工具轮数上限由 `max_tool_turns` 设置。 |

固定划分和初始 Skill 在 [resources/mindmemos_skill_eval](../../resources/mindmemos_skill_eval/)。训练前会校验下载记录、划分和运行依赖。

### Skill-Forest Select

将 [Select 示例](../../config/mindmemos_eval/forest_select/dev.example.yaml)复制为 `config/mindmemos_eval/forest_select/dev.yaml`，填写 `models.target` 与 `models.optimizer` 的模型名和 API 连接。评测在进程内创建隔离的本地 Skill 服务。

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_select/dev.yaml \
  --benchmark all --test-repeats 3
```

可用 `--benchmark livemath` 等参数只跑单个基准。Select 用完整训练集给每棵树的优胜 Skill 打分。当 `selection_tiebreak: novelty` 时，最高分并列的候选按相对于初始 Skill 的双文档 TF-IDF 新颖度选优。冻结选中的版本后，在留出的测试集评分。

### Skill-Forest Merge

将 [Merge 示例](../../config/mindmemos_eval/forest_merge/dev.example.yaml)复制为 `config/mindmemos_eval/forest_merge/dev.yaml`，配置相同的模型。

```bash
uv run --no-sync mindmemos-eval skill run \
  --config config/mindmemos_eval/forest_merge/dev.yaml \
  --benchmark all --test-repeats 3
```

Merge 用完整训练集给每棵树的优胜 Skill 打分，保留与最高分相差不超过 `merge_threshold` 的候选，再逐层两两合并。冻结合并产物后，在留出的测试集评分。

### 对标精度

每种配置都在同一测试集运行 `no_skill`、`initial_skill` 和 `trained_skill`。`--test-repeats 3` 使用同一个冻结版本复测三轮，比较平均任务成功率，同时记录执行覆盖率。比较 Select 和 Merge 时固定模型、划分、初始 Skill 和评测设置。复现训练还需核对候选验证批大小与补丁应用方式。

| 数据集 | 无 Skill | 初始 Skill | Select | Merge | 对标方法及精度 |
| --- | ---: | ---: | ---: | ---: | --- |
| LiveMath | | | | | |
| ALFWorld | | | | | |
| SpreadsheetBench Verified 400 | | | | | |
| SearchQA | | | | | |
| OfficeQA | | | | | |

`--smoke` 取前 4 条训练、2 条验证和 2 条测试数据，并缩小搜索预算。`--arms no_skill,initial_skill` 只运行两组对照，可跳过训练。每次运行默认在 `output/mindmemos_eval/skill/` 下生成 `summary.json`、`summary.csv` 和 `report.md`。覆盖不完整时，完整准确率留空。将运行目录传给 `mindmemos-eval skill report` 可用已有结果重生成报告。

## TreeSkill

下载 SpreadsheetBench Verified 400，使用仓库里的 [TreeSkill 划分与初始 Skill](../../resources/mindmemos_skill_eval/treeskill-spreadsheetbench/)。安装 `spreadsheetbench` 扩展，并准备 LibreOffice 用于工作簿重算。

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name spreadsheetbench
```

复现历史协议时，另行取得参考运行目录 `output/treeskill_full_20261012_092954`。在[本地评测配置](../../config/mindmemos_eval/treeskill/dev.example.yaml)中设置 `reference_run`，对齐 `profile` 选中的服务端训练配置（当前示例为 `treeskill`）。参考运行使用 200 题训练、200 题测试、`qwen3.5-35b-a3b`、seed 41、100 回合及 `transactional_recalculation: true`。预检查还会逐项核对参考数据和训练参数。

按[部署指南](../deploy/instruction_ZH.md)配置 SDK 与 Skill HTTP 服务。客户端和服务端需能访问相同绝对路径下的数据和输出目录。

```bash
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action --check
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action
```

评测器通过 SDK 注册初始 Skill、训练动态 TreeSkill，并在留出的测试集运行最终版本。文本动作环境按工作簿单元格值评分。查看 `summary.json` 时一并核对成功率、完成题量和执行错误。也可为其他已注册环境配置相应的数据、划分、初始 Skill 和环境参数。

| 数据集及协议 | TreeSkill | 对标方法及精度 |
| --- | ---: | --- |
| SpreadsheetBench Verified 400，文本动作，留出测试集 | 122/200（61.0%，历史参考） | |

61.0% 来自旧版指南记录的参考实验。比较新结果时使用相同的文本动作与重算协议。

## Offline Evolution

默认使用 SkillsBench v1.0 和各题自带的 Skill。下载数据并将[演进示例](../../config/mindmemos_eval/skill_evolve/dev.example.yaml)复制为 `config/mindmemos_eval/skill_evolve/dev.yaml`。配置执行代理和凭据，启动 Docker 供 Harbor 使用，在 `config/mindmemos_skill/dev.yaml` 中配置演进模型。若切换到其他已注册环境，提供匹配的数据及 `skills` 或 `case_skills`。

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name skillsbench
uv run --no-sync mindmemos-eval skill evolve --check
uv run --no-sync mindmemos-eval skill evolve
```

先用初始 Skill 采集轨迹，将正常完成的轨迹交给 `offline` 演进流程，导出新 Skill 后用相同执行代理和评分器复测**同一批题**。这个协议衡量同题改进。先对每题的多次运行求平均，再对题目求平均。两组默认各运行三次。满分题、无可用变化的题可沿用初始分。执行错误和剔除题目单列。`--check` 只校验数据和 Skill 来源，无模型调用。

| 数据集 | 指标 | 初始 Skill | Offline Evolution | 对标方法及精度 |
| --- | --- | ---: | ---: | --- |
| SkillsBench | 逐题 Avg@N reward 的均值 | | | |

## Skill 管理

### AgentSkillOS 检索

[AgentSkillOS 公开发布](https://huggingface.co/datasets/NPULH/agentskillos-benchmark)提供上游任务。本检索评测还需要配套的候选 Skill 池与相关性标注。使用整理好的检索包（含 `skill_pool_manifest.json`、`splits.json` 和 `queries/{tasks.jsonl,relevance.json}`），或使用原始包（含 `skill_pool_200/`、`ssl_pool_join.json`、`tasks.jsonl`、`relevance.json`）。统一下载器暂未收录该检索包，下载位置待补。

启动 Skill HTTP 服务并配置 SDK。两组使用不同的输出目录。

```bash
for strategy in graph vector; do
  uv run --no-sync python -m scripts.skill_eval.agentskillos.evaluate \
    --dataset-root data/mindmemos_skill/agentskillos_eval \
    --strategy "$strategy" --top-k 50 \
    --output "output/agentskillos_$strategy" || break
done
```

脚本注册完整候选池，建图并固定图版本，逐题生成 Skill 排名。评分跳过 `generic_only` 题目，使用二值核心相关标签。Recall 以完整相关集为分母，再对已评分题目做宏平均。失败请求与覆盖率单独记录。两组对比时固定候选池、标注、建图设置和检索预算。可用 `--reuse-registrations` 传入首次运行生成的 `registrations.json`，复用同一批 Skill 版本。

| 策略 | nDCG@10 | MRR@10 | Recall@50 | FullCoverage@10 |
| --- | ---: | ---: | ---: | ---: |
| Vector | | | | |
| Graph | | | | |

## Skill Judge

评判器分析 Skill 正文与可读资源。长输入按重叠窗口收集发现，再汇总判定。使用仓库内置样例，无需下载。回归检查分析输出、长文档的重叠窗口、覆盖上限，以及敏感或无法读取的资源过滤。标注数据与精度评分协议待补。

```bash
uv run --no-sync pytest -q src/mindmemos_skill/tests/test_analyze_algos.py -k skill_judge
```

| 标注数据集 | 指标 | Skill Judge | 对标方法及精度 |
| --- | --- | ---: | --- |
| | | | |

## Trajectory Judge

评判器结合执行轨迹与可用的 verifier 分数给出判定，并从轨迹提取 Skill 使用情况。使用仓库内置样例，无需下载。回归检查轨迹判定、Skill 使用指标，以及失败时对已检测到的 Skill 进行可选归因。标注数据与精度评分协议待补。

```bash
uv run --no-sync pytest -q src/mindmemos_skill/tests/test_analyze_algos.py -k trajectory_judge
```

| 标注数据集 | 指标 | Trajectory Judge | 对标方法及精度 |
| --- | --- | ---: | --- |
| | | | |

## 环境对照

`skill test` 每次测试一个注册环境。可选无 Skill，或向每道题注入同一组静态 Skill。默认示例使用已下载的 LiveMath 数据和 `config/mindmemos_eval/env_test/dev.yaml` 中配置的模型。

```bash
uv run --no-sync mindmemos-eval skill test --no-skill --split test
uv run --no-sync mindmemos-eval skill test --split test \
  --skills resources/mindmemos_skill_eval/skill_forest-livemath/initial_skill
```

成功率为环境判定成功的题数除以正常完成的题数。对比前一并核对计划题量、实际执行题量、正常完成题量与异常。输出包括 `summary.json`、`report.md` 和 `tasks/trajectories.jsonl`。
