# MindMemOS Skill 评测

[English](skill_evaluation.md) · [Memory 评测](memory_evaluation_ZH.md)

## 准备环境

安装 Git、Make 和 uv。在仓库根目录执行本文命令。Skill 评测在本地进程内运行。

```bash
uv sync --all-packages
make init
```

`make init` 会从 `dev.example.yaml` 创建缺失的本地配置，包括 `config/mindmemos_skill/dev.yaml` 和 `config/mindmemos_eval/skill_train_treeskill/dev.yaml`。它也会从 `.env.example` 创建缺失的 `.env`。本文使用的 Skill-Forest 和 Offline Evolution 配置按各节命令复制。

运行 `make init` 后再编辑本次评测的 `dev.yaml`。如果 `.env` 中已有模型设置，再次运行 `make init` 会把非空设置同步到本地 YAML 的对应字段。

## 命令入口

```bash
uv run --no-sync mindmemos-eval skill show envs
uv run --no-sync mindmemos-eval skill show pipelines
uv run --no-sync mindmemos-eval skill evaluate train --help
```

`train` 支持 `skill-forest` 和 `tree-skill`，`manage` 支持 `skill-graph`，`evolve` 支持 `offline-evolv`。用 `--config FILE` 指定评测 YAML，用 `--env ENV` 选择算法支持的环境。`--output-dir DIR` 优先于 YAML 中的 `output_dir`。

### 评测产物

各 Pipeline 的 evaluator 解析评测参数，再由选中的算法生成自己的结果和报告。以下文件都在本次运行的输出目录中。

| 评测 | 从哪里读结果 | 报告内容 |
| --- | --- | --- |
| Skill-Forest | `report.md`、`summary.json`、`summary.csv` | 对照组结果、重复运行统计、配对增益、训练和失败记录 |
| TreeSkill | `report.md`、`summary.json`、`trajectories.jsonl` | 留出集通过率、执行覆盖、错误和用量 |
| Offline Evolution | `report.md`、`summary.json` | 采集与复测的分数和用量、逐题演进状态 |
| Skill Graph · `retrieval` | `summary.json`、`rankings.json` | 检索指标的宏平均、逐题指标和检索排名 |
| Skill Graph · `accuracy` | `summary.json`、`selected_skills.json`、`tasks/` | 准确率、评分覆盖率、逐题错误与执行产物 |
| 单环境 `direct` | `report.md`、`summary.json`、`tasks/trajectories.jsonl` | 固定 Skill 或无 Skill 下的单组通过率、执行覆盖和用量 |

## Skill-Forest

### 1. 准备数据和运行环境

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
uv run --no-sync python scripts/download_datasets.py download skill --name alfworld
uv run --no-sync python scripts/download_datasets.py download skill --name spreadsheetbench
uv run --no-sync python scripts/download_datasets.py download skill --name searchqa
uv run --no-sync python scripts/download_datasets.py download skill --name officeqa
```

下载目录为 `data/mindmemos_skill/<dataset>/`。下载 Hugging Face 受限数据前，先在数据集页面取得访问权限，再把具有读取权限的令牌填入仓库根目录的 `.env`（`HF_TOKEN=...`）。

若只运行一个基准，仅下载对应数据。运行 ALFWorld 前，按[部署指南](../deploy/instruction_ZH.md#11-安装依赖)准备系统依赖。数据存放在其他位置时，用 `--data-root` 指定包含各数据集子目录的父目录。

#### 数据集划分

划分数量按训练、验证、测试顺序列出。各基准目录也包含对应的初始 Skill。

| 数据集 | 任务环境 | 划分文件 | 划分数量 | 数据集描述 |
| --- | --- | --- | ---: | --- |
| LiveMath | `livemath` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-livemath/data_splits/splits.json) | 35 / 18 / 124 | 数学推理选择题。 |
| ALFWorld | `alfworld_bounded_history` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-alfworld/data_splits/splits.json) | 39 / 18 / 134 | 在文本模拟环境中完成家务任务。 |
| SpreadsheetBench Verified 400 | `spreadsheetbench_tool_calling` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-spreadsheetbench/data_splits/splits.json) | 80 / 40 / 280 | 根据自然语言指令编辑 Excel 工作簿。 |
| SearchQA | `searchqa` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-searchqa/data_splits/splits.json) | 400 / 200 / 1,400 | 根据搜索结果片段回答开放域问题，按原生题目 ID 划分。 |
| OfficeQA | `officeqa` | [splits.json](../../resources/mindmemos_skill_eval/skill_forest-officeqa/data_splits/splits.json) | 50 / 24 / 172 | 根据美国财政部公报回答需要检索和推理的问题。 |

### 2. 配置评测

首次运行时，从仓库内的 [Skill-Forest 评测示例](../../config/mindmemos_eval/skill_train_skill_forest/paper_repro.example.yaml) 创建本地配置。已有 `paper_repro.yaml` 时直接编辑。

```bash
cp -n config/mindmemos_eval/skill_train_skill_forest/paper_repro.example.yaml config/mindmemos_eval/skill_train_skill_forest/paper_repro.yaml
```

#### 模型与运行设置

下表对应评测配置 `paper_repro.yaml`。任务环境由上表中的基准映射决定。

| 用途 | 模型 | 思考模式 | 配置位置 |
| --- | --- | --- | --- |
| 执行任务并生成轨迹 | `gpt-5.4-mini` | `reasoning_effort: medium` | `models.target` |
| 优化、筛选和合并 Skill | `gpt-5.4-mini` | `reasoning_effort: medium` | `models.optimizer` |

两组模型均设 `max_completion_tokens: 16384`。分别填写 `model`、`api_key` 和 OpenAI 兼容 API 的 `base_url`。如果接口不支持 `reasoning_effort`，将该模型的对应字段设为 `null`，交由接口默认行为。

`skill_forest.ensemble_strategy` 默认为 `select`。要评测合并策略，改为 `hierarchical_merge` 后重跑。`evaluation_concurrency` 控制留出集测试的并发数。接口限流时，也可降低 `skill_forest` 下的 `expansion_concurrency`、`rollout_concurrency` 和 `model_concurrency`。

两种策略均用完整训练集给各树的优胜 Skill 打分。`select` 选最高分，平分时按相对初始 Skill 的 TF-IDF 新颖度选优。`hierarchical_merge` 保留分数距最高分不超过 `merge_threshold` 的优胜者，再逐层两两合并。

### 3. 试跑并正式运行

先用已下载的 LiveMath 跑少量题。`--smoke` 会调用模型并缩小搜索规模。

```bash
uv run --no-sync mindmemos-eval skill evaluate train --algo skill-forest \
  --config config/mindmemos_eval/skill_train_skill_forest/paper_repro.yaml \
  --env livemath --smoke --output-dir output/skill-forest-smoke
```

每次运行评测一个数据集。用 `--benchmark` 指定数据集，并为每次运行设置独立的输出目录。

```bash
uv run --no-sync mindmemos-eval skill evaluate train --algo skill-forest \
  --config config/mindmemos_eval/skill_train_skill_forest/paper_repro.yaml \
  --benchmark livemath --test-repeats 3 --output-dir output/skill-forest-livemath
```

### 4. 读取结果

默认结果保存在 `output/mindmemos_eval/skill/<运行目录>/`。命令结束后打印 `report.md` 的路径。查看 `summary.json` 获取结构化结果。如果输入或运行环境检查失败，查看同目录下的 `diagnostics.json`。

从已有结果重生成报告，无需调用模型。

```bash
uv run --no-sync mindmemos-eval skill evaluate rebuild-report output/skill-forest-livemath
```

#### 评分口径

在同一留出测试集比较无 Skill、初始 Skill 和训练后的冻结 Skill。每组运行三轮，取平均任务成功率。

- **LiveMath** 按最终选项与标准选项是否一致评分。
- **ALFWorld** 按模拟器是否判定任务成功评分。
- **SpreadsheetBench** 检查每个工作簿用例的指定单元格值，全部匹配才通过。
- **SearchQA** 检查归一化答案是否与任一参考答案完全匹配。
- **OfficeQA** 按基准的数值、单位、有序列表和文本规则判断答案是否正确。

### 验证结果

Select 和 Merge 各使用 7 棵树。各基准的分数是留出测试集三轮运行的平均 Hard score（%），总体分数对五个基准等权平均。

| 数据集 | 无 Skill | 初始 Skill | Trace2Skill | GEPA | SkillOpt | Select | Merge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LiveMath | 27.42 | 28.23 | 27.69 | 31.45 | 31.72 | 36.02 | 34.68 |
| ALFWorld | 56.97 | 70.65 | 76.62 | 79.85 | 75.37 | 82.34 | 82.84 |
| SpreadsheetBench Verified 400 | 80.24 | 79.40 | 81.79 | 82.26 | 80.71 | 81.67 | 81.07 |
| SearchQA | 76.90 | 76.67 | 77.50 | 79.62 | 80.67 | 80.21 | 80.52 |
| OfficeQA | 28.49 | 43.99 | 54.84 | 53.68 | 52.71 | 53.88 | 55.23 |
| **总体** | 54.00 | 59.79 | 63.69 | 65.37 | 64.24 | 66.82 | 66.87 |

## TreeSkill

### 1. 准备数据和运行环境

下载 SpreadsheetBench Verified 400，并使用仓库内的 [200 题训练 / 200 题测试划分](../../resources/mindmemos_skill_eval/treeskill-spreadsheetbench/data_splits/splits.json)。

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name spreadsheetbench
```

按[部署指南](../deploy/instruction_ZH.md#12-初始化--填写配置文件)安装 LibreOffice，并确认在当前终端运行 `soffice --version` 能显示版本。若 macOS 上的 LibreOffice 安装在 `/Applications`，可用以下命令把 `soffice` 加到当前终端的 PATH。

```bash
export PATH="/Applications/LibreOffice.app/Contents/MacOS:$PATH"
soffice --version
```

### 2. 配置评测输入和输出

`make init` 已创建 TreeSkill 评测配置和 Skill 服务配置。也可以用下列命令从 [TreeSkill 示例](../../config/mindmemos_eval/skill_train_treeskill/dev.example.yaml)和 [Skill 服务示例](../../config/mindmemos_skill/dev.example.yaml)单独复制缺失文件。

```bash
cp -n config/mindmemos_eval/skill_train_treeskill/dev.example.yaml config/mindmemos_eval/skill_train_treeskill/dev.yaml
cp -n config/mindmemos_skill/dev.example.yaml config/mindmemos_skill/dev.yaml
```

打开 `config/mindmemos_eval/skill_train_treeskill/dev.yaml`，检查以下字段。

| 配置项 | 填写内容 |
| --- | --- |
| `data_root` | 使用默认下载路径时保留示例值。如果数据保存在其他位置，填写包含 `dataset.json` 和工作簿文件的 `spreadsheetbench_verified_400` 目录。 |
| `service_config`、`profile` | 保留 `config/mindmemos_skill/dev.yaml` 和 `treeskill`。第 3 步会在这个文件中设置该训练配置。 |
| `split_file`、`initial_skill` | 保留示例值。评测使用仓库自带的 200/200 划分和初始 `xlsx` Skill。 |
| `environment` | 保留示例设置。每题最多 100 轮，并启用工作簿公式重算。 |
| `concurrency` | 示例值为 `16`。如需降低并发，将第 3 步的四个并发字段一起改为相同值。 |
| `output_dir` | 本次评测的结果目录。保留 `null` 时自动创建 `output/mindmemos_eval/treeskill/<时间戳>/`。用 `--output-dir` 指定新目录或空目录时，以命令行为准。 |

相对路径以运行命令时的工作目录为起点。

### 3. 配置 TreeSkill 训练和模型

在 `config/mindmemos_skill/dev.yaml` 中，用下面的内容替换 `service.training_pipelines.treeskill` 这一节，保留同级的其他训练配置。将 `api_key` 和 `base_url` 换成能调用 `qwen3.5-35b-a3b` 的 OpenAI 兼容接口凭据和地址。

#### 模型与运行设置

下表对应本节的 35B 配置。环境为 `spreadsheetbench_text_action`，每题最多 100 轮，启用公式重算。模型项均位于训练配置的 `trainer.config` 下。

| 用途 | 模型 | 思考模式 | 配置位置 |
| --- | --- | --- | --- |
| 分析轨迹并优化 Skill | `qwen3.5-35b-a3b` | `enable_thinking: true` | `optimizer` |
| 失败分析 | `qwen3.5-35b-a3b` | `enable_thinking: false` | `failure_optimizer` |
| 执行训练和测试任务 | `qwen3.5-35b-a3b` | 继承 `failure_optimizer`，关闭 | 评测入口从 `failure_optimizer` 派生，设置 `temperature: 0.7` |
| 选择 Skill 子树 | `qwen3.5-35b-a3b` | `enable_thinking: false` | `router` |

`enable_thinking` 位于各模型的 `extra_body.chat_template_kwargs`。下面的 YAML 让这些角色共用接口连接。

```yaml
    treeskill:
      class_name: training
      config:
        trainer:
          class_name: treeskill
          config:
            optimizer:
              <<: &treeskill_connection
                model: qwen3.5-35b-a3b
                api_key: "<你的 API Key>"
                base_url: "<你的 OpenAI 兼容 API Base URL>"
                max_completion_tokens: 16384
                temperature: 1.0
                seed: 41
              timeout_s: 1800
              top_p: 0.95
              presence_penalty: 1.5
              extra_body:
                top_k: 20
                min_p: 0.0
                repetition_penalty: 1.0
                chat_template_kwargs:
                  enable_thinking: true
            failure_optimizer: &treeskill_execution
              <<: *treeskill_connection
              timeout_s: 600
              top_p: 1.0
              presence_penalty: 2.0
              extra_body:
                top_k: 40
                min_p: 0.0
                repetition_penalty: 1.0
                chat_template_kwargs:
                  enable_thinking: false
            router: *treeskill_execution
            max_trajectories: 200
            analysis_max_turns: 100
            rollout_concurrency: 16
            analysis_concurrency: 16
            localization_concurrency: 16
        evaluation_concurrency: 16
```

### 4. 检查配置并运行

先运行输入检查。数据集、划分和初始 Skill 加载成功后，命令会显示 `200 train / 200 test`。这一步不调用模型，也不运行 LibreOffice。

```bash
uv run --no-sync mindmemos-eval skill evaluate train --algo tree-skill \
  --config config/mindmemos_eval/skill_train_treeskill/dev.yaml --check
```

检查通过后，去掉 `--check` 开始训练和测试。

```bash
uv run --no-sync mindmemos-eval skill evaluate train --algo tree-skill \
  --config config/mindmemos_eval/skill_train_treeskill/dev.yaml --output-dir output/treeskill
```

在本次结果目录查看 `report.md`、`summary.json` 和逐题的 `trajectories.jsonl`。评测按工作簿单元格值判定每题是否通过，报告中的 `success_rate` 是完成题目的通过比例。

### SpreadsheetBench 域内评测结果

下表按执行任务的模型分组。分数单位为百分比，括号内为相对于同模型 Human-Written Skill 的提升（百分点）。上面的 CLI 汇报完成题目的通过率。

| 执行模型 | 方法 | Vrf (%) | Soft (%) | Hard (%) |
| --- | --- | ---: | ---: | ---: |
| Qwen3.5-35B-A3B | No Skill | 19.00 | 18.00 | 4.60 |
| Qwen3.5-35B-A3B | Human-Written Skill | 9.67 | 13.03 | 3.37 |
| Qwen3.5-35B-A3B | Trace2Skill +Combined | 29.67 (+20.00) | 18.80 (+5.77) | 5.73 (+2.36) |
| Qwen3.5-35B-A3B | SkillCAT | 55.50 (+45.83) | 39.02 (+25.99) | 20.61 (+17.24) |
| Qwen3.5-35B-A3B | **TreeSkill (routed)** | **66.00 (+56.33)** | **41.72 (+28.69)** | **25.22 (+21.85)** |
| Qwen3.5-9B | No Skill | 25.50 | 21.00 | 9.21 |
| Qwen3.5-9B | Human-Written Skill | 14.00 | 16.76 | 7.24 |
| Qwen3.5-9B | Trace2Skill +Combined | 16.50 (+2.50) | 18.97 (+2.21) | 8.66 (+1.42) |
| Qwen3.5-9B | **TreeSkill (routed)** | **45.50 (+31.50)** | **30.28 (+13.52)** | **14.58 (+7.34)** |

35B 对照为已报告的基线分数。TreeSkill 两组结果及表中的 9B 对照由本项目实测。

## Offline Evolution

### 1. 准备数据和运行环境

下载 SkillsBench v1.0。评测使用数据集内每题自带的初始 Skill。启动 Docker，并在本机安装 Claude Code。Harbor 0.24.0 需要 Python 3.12 或 3.13，随项目依赖一起安装。

```bash
uv sync --all-packages --python 3.13
uv run --no-sync python scripts/download_datasets.py download skill --name skillsbench
uv run --no-sync harbor --version
docker info
claude --version
```

### 2. 配置执行与演进模型

首次运行时复制[演进配置示例](../../config/mindmemos_eval/skill_evolve_offline/dev.example.yaml)。

```bash
cp -n config/mindmemos_eval/skill_evolve_offline/dev.example.yaml config/mindmemos_eval/skill_evolve_offline/dev.yaml
```

#### 模型与运行设置

任务执行环境为 `skillsbench`（`environment.class_name`），由 Harbor 在 Docker 中运行。Skill 演进由宿主机上的 Claude Code 执行。以下为示例值。

| 用途 | 代理与模型 | 思考模式 | 配置位置 |
| --- | --- | --- | --- |
| 采集初始轨迹和复测 | `claude-code` · `claude-opus-4-6` | 示例未显式设置，由 Harbor 启动的代理决定 | 评测 YAML 的 `environment.config.agent` 和 `environment.config.model` |
| 演进 Skill | `claude` · `deepseek-v4-flash` | `reasoning_effort` 未设置，客户端使用默认行为 | 服务 YAML 中 `service.agents.offline_evolution_evolver_claude.config` |

在评测 YAML 中，保留 `environment.config.agent_env` 列出的变量名。在运行终端导出供应商要求的 `ANTHROPIC_API_KEY`，或网关使用的 `ANTHROPIC_AUTH_TOKEN` 和 `ANTHROPIC_BASE_URL`。本机演进代理的 `model`、`api_key` 和 `base_url` 填在上表对应的服务配置中。`service.evolution_pipelines.offline.config.evolver.config.agent` 选择该代理。

如果数据下载到了其他位置，将评测 YAML 的 `data.config.benchmark_root` 指向包含 `tasks/` 的目录。默认是 `data/mindmemos_skill/skillsbench`。

### 3. 检查输入并运行

`--check` 准备选定任务和初始 Skill。检查通过后，运行第二条命令。若想先实际试跑一题，在评测 YAML 中把 `cases` 设为只含一个 `tasks/` 子目录名的列表，并设 `repeats: 1`。全量时恢复 `cases: null` 和 `repeats: 3`。

```bash
uv run --no-sync mindmemos-eval skill evaluate evolve --algo offline-evolv \
  --config config/mindmemos_eval/skill_evolve_offline/dev.yaml --check
uv run --no-sync mindmemos-eval skill evaluate evolve --algo offline-evolv \
  --config config/mindmemos_eval/skill_evolve_offline/dev.yaml --output-dir output/offline-evolv
```

### 4. 读取结果

默认在 `output/mindmemos_eval/skill_evolve/<时间戳>/` 查看 `report.md` 和 `summary.json`。评测先用初始 Skill 采集轨迹，再用演进后的 Skill 复测同一批题。先对每题的重复执行取平均 reward，再对可计分任务取平均。无可用演进结果的题沿用初始分。正式对比结果待补。

用逐题 `Status` 检查演进结果。`evolved` 表示生成了新 Skill 并安排复测，`unchanged` 表示演进后保留原 Skill，`evolution_failed` 表示演进失败并沿用初始分。写出报告后，命令仍可返回 0，因此验证演进时请同时检查逐题状态和复测结果。

| 数据集 | 指标 | 初始 Skill | Offline Evolution |
| --- | --- | ---: | ---: |
| SkillsBench | 逐题平均 reward 的均值 | — | — |

## Skill Graph 评测

两个模式都从 `--skills-dir` 加载 Skill 并在本地建图。每个 Skill 存放在 `<源 Skill ID>/SKILL.md`，同目录下可包含附带资源。先运行 `make init`，在 `config/mindmemos_skill/dev.yaml` 配置建图和检索所需的聊天模型及向量模型。可用 `--service-config` 指定另一份服务配置。

每次运行使用新的 `--output` 目录。命令对原始任务的公开指令执行检索，把逐题排名写入该目录的 `rankings.json`，再用同一份排名继续评测。金标留在评测侧。

安装 Git 后，下载固定版本的 AgentSkillOS。两种评测共用这份下载。

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name agentskillos
```

如果 `data/mindmemos_skill/agentskillos` 存有之前从 Hugging Face 下载的数据，先移走旧目录，再执行下载命令。

### 检索评分

评测从官方任务 JSON 的 `skills` 字段读取金标。

```bash
uv run --no-sync mindmemos-eval skill evaluate manage --algo skill-graph --env agentskillos --eval-mode retrieval \
  --tasks-dir data/mindmemos_skill/agentskillos/benchmark/AgentSkillOS_bench/tasks \
  --skills-dir path/to/skills \
  --output-dir output/agentskillos_retrieval
```

`summary.json` 的 `per_task` 保存逐题指标，`metrics` 保存宏平均。每题的 Recall@K 用前 K 名命中的金标数除以原始 `skills` 中的金标总数。报告还包含 nDCG、MRR、Hit、Precision 和 FullCoverage。

### 端到端准确率

`accuracy` 使用 Setting 1 的执行模型 `MiniMax-M2.7`。Claude Code 按本次检索排名注入前 K 个 Skill 的完整内容及资源，执行任务并生成产物。AgentSkillOS 官方评估器判定产物是否通过。

运行 `make init`，在 `config/mindmemos_eval/skill_graph_setting_1/dev.yaml` 填写 `agent.base_url` 和 `agent.api_key`。端点需支持 Anthropic Messages API。安装 Claude Code。通过 `execution_top_k` 调整注入深度。 [MiniMax 接入 Claude Code](https://platform.minimax.io/docs/m-plan/claude-code)

```bash
uv run --no-sync mindmemos-eval skill evaluate manage --algo skill-graph --env agentskillos \
  --config config/mindmemos_eval/skill_graph_setting_1/dev.yaml --eval-mode accuracy \
  --skills-dir path/to/skills \
  --output-dir output/agentskillos_accuracy
```

输出目录须为新目录。`summary.json` 中的 `accuracy` 是通过官方评估的任务占成功评分任务的比例。`scoring_coverage` 单独记录评分覆盖率。执行或评分失败会写入 `errors`，并使命令以非零状态退出。`tasks/` 保存逐题结果、产物及轨迹。

### 对标结果

下表记录三项基准的对标结果，`Ours` 指 MindMemOS Skill Graph。上面的命令用于 AgentSkillOS，端到端模式对应 Setting 1。

#### 表 1　Skill 检索

Emb. 表示向量检索，RR 表示重排。每列用粗体标出第一名，用下划线标出第二名，并列同样标记。\* 表示数值引自对应方法的原论文。

| 方法 | ALFWorld Ret@1 | ALFWorld Ret@K | ALFWorld MRR | AgentSkillOS Ret@1 | AgentSkillOS Ret@K | AgentSkillOS MRR | SkillsBench Ret@1 | SkillsBench Ret@K | SkillsBench MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **向量检索** |  |  |  |  |  |  |  |  |  |
| Vector Skills | 0.7 | 4.5 | 0.4 | 56.7 | 83.3 | 66.2 | 3.6\* | 10.8\* | 5.8\* |
| SkillRouter (Emb.) | 13.6 | 48.6 | 26.2 | 50.0 | <ins>86.7</ins> | 64.3 | <ins>77.6</ins> | **93.4** | <ins>83.1</ins> |
| SkillRouter (Emb.+RR) | 12.1 | **74.3** | <ins>36.2</ins> | 56.7 | 70.0 | 61.9 | **81.6** | <ins>88.2</ins> | **84.5** |
| **树结构** |  |  |  |  |  |  |  |  |  |
| AgentSkillOS | 0.7 | 50.7 | 18.5 | 70.0 | 83.3 | 76.7 | 38.2 | 44.7 | 41.0 |
| SkillRAE | 9.3 | 35.6 | 17.5 | 40.0 | 80.0 | 55.2 | 69.7 | 86.8 | 76.8 |
| **图结构** |  |  |  |  |  |  |  |  |  |
| GoS | 8.6 | 9.3 | 8.9 | <ins>76.7</ins> | <ins>86.7</ins> | <ins>80.3</ins> | 50.6\* | 65.5\* | 57.3\* |
| SkillDAG | <ins>20.0</ins> | 44.8 | 29.2 | 43.3 | 56.1 | 60.8 | 66.7\* | 78.2\* | 71.3\* |
| Ours | **49.3** | <ins>71.4</ins> | **57.9** | **80.0** | **93.3** | **83.3** | 69.2 | 86.8 | 76.8 |

#### 表 2　端到端 R 分数

Setting 1 使用 MiniMax-M2.7。每个设置内按基准排名，标记规则同表 1。

| 设置 | 方法 | ALFWorld | AgentSkillOS | SkillsBench |
| --- | --- | ---: | ---: | ---: |
| Setting 1 | Vanilla Skills | 47.1\* | 70.8 | 17.2\* |
|  | Vector Skills | 50.7\* | <ins>72.4</ins> | 10.4\* |
|  | GoS | 54.3\* | 68.0 | 18.7\* |
|  | SkillDAG | <ins>67.9</ins> | 54.9 | <ins>27.3</ins>\* |
|  | Ours | **76.4** | **80.9** | **28.0** |
| Setting 2 | Vector Skills | 60.0 | 74.9\* | 27.3 |
|  | GoS | 63.6 | <ins>96.2</ins> | <ins>35.4</ins> |
|  | SkillDAG | <ins>82.0</ins> | 94.2 | 31.4 |
|  | AgentSkillOS | 13.6 | 83.5\* | 17.2\* |
|  | SkillRouter | 29.3 | 82.3\* | 22.0\* |
|  | SkillRAE | 27.1 | 84.6\* | 29.3\* |
|  | Ours | **86.0** | **97.4** | **43.2** |
| Setting 3 | GoS | 79.3 | <ins>96.2</ins> | 40.1 |
|  | SkillDAG | <ins>85.0</ins> | 92.3 | <ins>49.0</ins> |
|  | Ours | **85.7** | **96.5** | **51.0** |

## 单环境 Skill 测试

### 1. 准备数据

下面用 LiveMath 对比无 Skill 和固定的初始 Skill。若已按 Skill-Forest 一节下载 LiveMath，可直接使用现有数据。

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
```

### 2. 配置测试

`make init` 从 [单环境测试示例](../../config/mindmemos_eval/skill_test_direct/dev.example.yaml) 创建 `config/mindmemos_eval/skill_test_direct/dev.yaml`。默认从下载目录加载 LiveMath，并读取仓库固定的 `test` 划分。

#### 模型与运行设置

| 用途 | 环境 | 模型 | 思考模式 | 配置位置 |
| --- | --- | --- | --- | --- |
| 无 Skill 与固定初始 Skill 两组作答 | `livemath` | `gpt-5.4-mini` | `reasoning_effort: medium` | `environment.class_name`、`environment.config.agent` |

示例的 `max_completion_tokens` 为 `16384`。在 `environment.config.agent` 中填写 `model`、`api_key` 和 `base_url`。两组使用同一模型设置。

### 3. 试跑并运行

先加 `--limit 2` 跑两题。移除该参数后运行完整测试划分。两条命令各自写入新的结果目录。

```bash
uv run --no-sync mindmemos-eval skill evaluate direct \
  --config config/mindmemos_eval/skill_test_direct/dev.yaml --no-skill --limit 2
uv run --no-sync mindmemos-eval skill evaluate direct \
  --config config/mindmemos_eval/skill_test_direct/dev.yaml --limit 2 \
  --skills resources/mindmemos_skill_eval/skill_forest-livemath/initial_skill
```

### 4. 读取结果

在 `output/mindmemos_eval/test/<运行目录>/` 查看 `report.md` 和 `summary.json`。逐题轨迹位于 `tasks/trajectories.jsonl`。LiveMath 按最终选项与标准选项是否一致评分。这个入口使用固定 Skill，可用于验证环境和比较两组通过率。
