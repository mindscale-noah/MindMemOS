# MindMemOS 评测指南

本文集中说明 Memory 基准和 Skill-Forest 五基准的复现入口，以及如何用 `metrics.py` 统计耗时与 token 用量。

---

## TreeSkill：SpreadsheetBench text_action

`mindmemos-eval skill treeskill` 通过公共 HTTP SDK 注册 Skill、提交原生 TreeSkill 训练，
并由同一服务测试最终动态 Skill。使用与 UI 相同的 `.mindmemos` SDK 配置和项目凭据，
注册名为 `xlsx-mindmemos-eval-YYYYMMDD-HHMMSS-ffffff`。不增加 baseline 或自动对比。

默认配置：`config/mindmemos_eval/treeskill/dev.yaml`，示例为同目录的 `dev.example.yaml`。
服务端使用 `config/mindmemos_skill/dev.yaml` 中独立的 `treeskill_mindmemos_eval` profile。
首次接入后，需重启 Skill 服务和 UI 后端加载新代码及 profile；仅修改配置文件不会更新已运行的服务。
CLI 与服务须共享相同绝对路径下的数据和输出目录。

```bash
# 不调用模型、不注册 Skill：核对参考记录、400 条数据、文件内容和配置。
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action --check

# 正式执行原生训练和测试。
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action
```

`--env` 可选择系统内任意已注册环境，不因缺少历史效果保证而拒绝运行。
省略时使用配置中的 `environment.class_name`（默认 `spreadsheetbench_text_action`）。
其他环境使用其原生 `environment.config`，并通过 `data` 或 `samples_file` 提供匹配的数据，
配合 `split_file` 的 train/val/test 划分及 `initial_skill`；无需历史 `reference_run` 或 `data_root`。
例如 `mindmemos-eval skill treeskill --env livemath --config /path/to/math.yaml`。
只有 `spreadsheetbench_text_action` 执行下面的历史协议对齐；其他环境照常训练、测试和发布 UI 结果，效果不保证。

参考实验是 `output/treeskill_full_20261012_092954`，其 held-out 测试为 122/200（61%）。
检查要求固定划分和顺序、完整初始 Skill 及资源、任务和评分元数据、工作簿内容、
原生环境和 trainer 参数一致。历史 optimizer 模型在存档中被脱敏，
`reference_optimizer_model: qwen3.5-35b-a3b` 来自实验所有者确认；
执行模型和 router 模型分别从历史执行请求和最终动态 Skill 核对。
名称按要求加后缀，初始 Skill 正文及资源保持一致。

| 参数 | 本协议 |
| --- | --- |
| 环境 | `spreadsheetbench_text_action` |
| 数据 | verified 400，固定 200 train / 200 test，validation 为空 |
| 模型 | 执行、optimizer、failure optimizer、router 均为 `qwen3.5-35b-a3b` |
| 最大交互轮数 | 100 |
| 工作簿重新计算 | `transactional_recalculation=true` |
| rollout / analysis / localization / test 并发 | 32 / 32 / 32 / 32 |
| 每题采样数 / seed | 1 / 41 |
| 采样、超时和阶段设置 | 从参考 trainer 配置逐项校验 |

结果写入 `output/mindmemos_eval/treeskill/<时间>/`（可用 `--output-dir` 指定空目录）：

- `registration.json`、`task.json`：注册身份、异步任务状态和最终版本。
- `summary.json`、`report.md`：测试题量、测试耗时、token、执行错误和成功率。
- `trajectories.jsonl`：每条测试题的原生结果或明确的失败/未开始记录。
- `runtime.json`：整个注册、训练和测试流程的耗时，与测试耗时分开。
- `training/train/`：原生训练阶段统计、轨迹和最终动态 Skill。
- `training/test/`：服务端原始测试记录、JSONL 和 `report.json`。

正常 reward=0 不计入执行错误。成功率分母是正常完成的题量；未开始题目单列。
无法取得的耗时或 token 为 `null`，已知的部分 token 另列，不补零。
训练或测试失败时命令返回失败，保留任务状态和已生成的测试报告。
完整结束后，UI → Skills → 对应注册名 → 最终版本 → Content 显示 Test evaluation。
旧版本不会自动补写摘要；61% 是参考结果，并非新运行的保证分数。


## 直接测试已注册环境

`mindmemos-eval skill test` 每次只运行一个模式：no-skill 或一组静态 Skill。
多个 Skill 会一起注入每个任务，不进行训练或自动对比。以下命令在仓库根目录执行。

先运行 `make init` 生成本地配置，再编辑 `config/mindmemos_eval/env_test/dev.yaml`。
默认示例使用 LiveMath；`environment.class_name/config` 是环境注册名及其原生配置，
`data.class_name/config` 是已注册数据加载器及其原生配置。模型连接填写在环境的 `agent` 中。
可用 `skill envs` 查看环境注册名；不同环境需要各自适用的数据和配置。

```bash
uv run --no-sync mindmemos-eval skill envs

# No-skill：取配置所选 split 的前 10 题。
uv run --no-sync mindmemos-eval skill test --no-skill --limit 10 --concurrency 2

# 静态 Skill：这两个目录作为一组同时提供给每个任务。
uv run --no-sync mindmemos-eval skill test \
  --skills /path/to/skill_a /path/to/skill_b --limit 10 --concurrency 2
```

`--no-skill` 与 `--skills` 互斥。未指定这两个参数时使用配置中的 `skills`，空列表表示 no-skill。
Skill 路径可以是包含标准 frontmatter 的 `SKILL.md`，也可以是其所在目录。
`--config` 指定其他配置，`--env` 覆盖环境注册名，`--split-file/--split` 选择固定划分，
`--limit` 取有序前缀，`--output-dir` 指定空的输出目录。所有相对路径均相对于执行命令的目录。

没有现成 Data 加载器时，可在配置中令 `data: null` 并设置 `samples_file`，或使用
`--samples /path/to/samples.json` 覆盖数据源。JSON 文件是数组，每项只有 `task` 和
`reward_data` 两个字段，内容必须符合所选环境的原生任务与评分数据模型。
`split_file: null` 表示评测全部输入；提供 split 文件时，其 ID 必须存在于输入数据中。

报告固定包含六项信息：

| 项目 | 口径 |
| --- | --- |
| 数据量 | `planned` 选中数量、`attempted` 实际执行数量、`completed` 正常完成数量 |
| 耗时 | `wall_seconds` 为准备和执行的墙钟耗时；每题耗时另存轨迹记录 |
| Token 消耗 | `input_tokens/output_tokens/total_tokens`；有缺失时总量为 null，`reported_usage` 保存已知部分，`usage_missing` 标记缺失任务数 |
| 执行失败 | `errors` 仅统计 API 超时等执行或评分异常；正常 reward=0 不计入 |
| 成功率 | 环境 `reward.success=true` 的任务数 / 正常完成数；全部异常时为 null |
| 逐题轨迹 | `tasks/trajectories.jsonl` 每个已结束任务一行，包含任务、结果或错误和原生轨迹 |

每个任务结束后立即追加 JSONL，行顺序为完成顺序，`position` 保留原始任务顺序。
`result.trajectory` 保存环境返回的交互记录，`outcome` 包含得分、成功标记、耗时、token 和错误。
执行异常时 `result` 为 null，保留错误及原始任务目录位置；环境没有返回的轨迹或 token 不会虚构补齐。

默认输出位于 `output/mindmemos_eval/test/<run>/`：`summary.json` 是结构化汇总，
`report.md` 是可读报告，`tasks/trajectories.jsonl` 是逐题轨迹；`tasks/task-*/`
保留原生工作目录和审计文件。出现执行异常时命令返回非零；只有正常零分时仍返回零。

---

## Memory 评测前置条件

### 1. 启动依赖服务

```bash
make docker SERVICE=memory          # 启动 Qdrant / Neo4j / Kafka / ClickHouse / Grafana 等
```

> **注意**：`make clean` 会在确认后停止受管理的应用并删除 Compose 数据卷。评估前请重新启动服务；手动启动的服务也需要重启，避免复用失效的数据库连接。

### 2. 启动 MindMemOS server

```bash
uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### 3. 配置 LLM API Key

评测时，`add` 阶段的 LLM 调用由 **server** 自己完成，使用 `config/mindmemos/dev.yaml` 里的 key；`answer` 和 `judge` 阶段由 **eval 进程**直接调用，使用 `config/mindmemos_eval/memory_evaluation/locomo.example.yaml`（或 `config/mindmemos_eval/memory_evaluation/personamem.example.yaml`） 里的 key。两套 key 相互独立，需要分别配置。

```yaml
# config/mindmemos_eval/memory_evaluation/locomo.yaml
runner:
  llm:
    model: gpt-4.1-mini
    api_key: <YOUR_KEY>
    base_url: https://your-llm-provider/v1
  answer_llm:
    api_key: <YOUR_KEY>          # 可以和 llm 相同
    base_url: https://your-llm-provider/v1
  judge_llm:
    api_key: <YOUR_KEY>
    base_url: https://your-llm-provider/v1
```

验证 server 和 LLM 是否都可用：

```bash
# server 正常返回 JSON（非 connection refused / 500）
curl -s -X POST http://127.0.0.1:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <任意已有 key>" \
  -d '{"query":"test","user_id":"test"}' | head -c 100

# LLM API 正常返回（非 Invalid token）
curl -s -X POST https://your-llm-provider/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

---

## Memory 数据集下载

使用 Python 3.12+，从仓库根目录执行以下命令。数据本体不随仓库分发；统一下载脚本从官方来源
获取数据，版本固定在 [`config/presets/dataset_download_sources.json`](../../config/presets/dataset_download_sources.json)。

```bash
python -m pip install huggingface_hub
python scripts/download_datasets.py list memory
python scripts/download_datasets.py download memory --name locomo
python scripts/download_datasets.py download memory --name longmemeval
python scripts/download_datasets.py download memory --name personamem
python scripts/download_datasets.py download memory --name memoryagentbench
```

默认下载到 `data/mindmemos_memory/<dataset>/`，保留官方文件格式：

| 数据集 | 官方来源 | 相对于该数据集下载目录的文件 |
| --- | --- | --- |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) | `locomo10.json` |
| LongMemEval | [longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `longmemeval_s_cleaned.json`、`longmemeval_m_cleaned.json`、`longmemeval_oracle.json` |
| PersonaMem v1 | [PersonaMem-v1](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) | `questions_{32k,128k,1M}.csv` 与对应的 `shared_contexts_{32k,128k,1M}.jsonl` |
| MemoryAgentBench | [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) | 官方四个数据分区的 `data/*.parquet`，以及 `entity2id.json` |

复制评测配置，然后在本地副本中更新数据路径和 LLM API key：

```bash
cp config/mindmemos_eval/memory_evaluation/locomo.example.yaml config/mindmemos_eval/memory_evaluation/locomo.yaml
```

```yaml
benchmarks:
  locomo:
    dataset: data/mindmemos_memory/locomo/locomo10.json
  longmemeval:
    dataset: data/mindmemos_memory/longmemeval/longmemeval_s_cleaned.json
  personamem:
    dataset: data/mindmemos_memory/personamem/questions_32k.csv
    context_dataset: data/mindmemos_memory/personamem/shared_contexts_32k.jsonl
```

PersonaMem 的问题文件和上下文文件需使用相同的上下文长度；使用 `config/mindmemos_eval/memory_evaluation/personamem.yaml`
时也需更新这两个字段。LongMemEval 下载的是官方 cleaned 版本，不是示例配置原先引用的本地 smoke 样本。
MemoryAgentBench 保留官方 Parquet 格式，当前评测器的本地 `dataset` 入口接受 JSON/JSONL，
需要先转换所需分区再配置该路径。下载脚本不执行格式转换，也不筛选 FactConsolidation 子集。

使用 `--data-root` 指定父目录、`--verify-only` 校验已有下载，或使用 `--offline` 从已有
Hugging Face 缓存准备数据：

```bash
python scripts/download_datasets.py download memory --name locomo --data-root /path/to/memory
python scripts/download_datasets.py download memory --name locomo --verify-only
python scripts/download_datasets.py download memory --name personamem --offline
```

自定义目录示例会写入 `/path/to/memory/locomo/`，校验该目录时也需传入相同的 `--data-root`。
LoCoMo 首次通过直接链接下载需要联网。已有目录只校验，不覆盖；每个下载目录中的
`download_receipt.json` 记录来源与本地文件哈希。实验 split 随代码提交到 GitHub，脚本不下载或生成 split。

---

## 各数据集复现

除非特别说明，下面的命令都从仓库根目录执行。LoCoMo、LongMemEval、PersonaMem 和 MemoryAgentBench 的数据集不随仓库分发，需要先从对应 benchmark 的官方渠道获取。

> **API key 文件**：`--api-key-output` 会完整写入一个新的 `api_keys.yaml`，仅包含本次评测生成的身份。请使用独立路径，并让服务端的 `auth.api_key_file` 指向该文件；不要覆盖日常开发环境正在使用的 key 文件。

### LoCoMo

1. 按上节下载 LoCoMo，并将复制后的配置中 `benchmarks.locomo.dataset` 设置为：

   ```text
   data/mindmemos_memory/locomo/locomo10.json
   ```

2. 复制配置并填入 eval 进程使用的 LLM API key：

   ```bash
   cp config/mindmemos_eval/memory_evaluation/locomo.example.yaml \
     config/mindmemos_eval/memory_evaluation/locomo.yaml
   ```

3. 启动服务后运行：

   ```bash
   uv run python -m mindmemos_eval.cli memory \
     --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
     --benchmark-list locomo \
     --algorithm schema \
     --manifest-output reports/locomo_schema.jsonl \
     --api-key-output config/mindmemos/eval_api_keys.yaml
   ```

### LongMemEval

LongMemEval 与 LoCoMo 共用 `config/mindmemos_eval/memory_evaluation/locomo.example.yaml`。将数据集路径写入复制后的配置，例如：

```yaml
benchmarks:
  longmemeval:
    dataset: data/mindmemos_memory/longmemeval/longmemeval_s_cleaned.json
```

运行评测：

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list longmemeval \
  --algorithm vanilla \
  --manifest-output reports/longmemeval_vanilla.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

先用 `--limit 2 --session-limit 2` 做全链路冒烟测试，再移除限制运行完整数据集。

### PersonaMem

1. 按上节下载 PersonaMem，并在复制后的配置中分别设置 `dataset` 和 `context_dataset`：

   ```text
   data/mindmemos_memory/personamem/questions_32k.csv
   data/mindmemos_memory/personamem/shared_contexts_32k.jsonl
   ```

2. 复制配置并填入 eval 进程使用的 LLM API key：

   ```bash
   cp config/mindmemos_eval/memory_evaluation/personamem.example.yaml \
     config/mindmemos_eval/memory_evaluation/personamem.yaml
   ```

3. 启动服务后运行：

   ```bash
   uv run python -m mindmemos_eval.cli memory \
     --benchmark-config config/mindmemos_eval/memory_evaluation/personamem.yaml \
     --benchmark-list personamem \
     --algorithm schema \
     --manifest-output reports/personamem_schema.jsonl \
     --api-key-output config/mindmemos/eval_api_keys.yaml
   ```

### MemoryAgentBench

1. 复制 MemoryAgentBench 配置：

   ```bash
   cp config/mindmemos_eval/dreaming_evaluation/memoryagentbench.example.yaml \
     config/mindmemos_eval/dreaming_evaluation/memoryagentbench.yaml
   ```

2. 按配置中的 `benchmarks.memoryagentbench.dataset` 准备数据文件，并确认 `sub_dataset`、`chunk_size` 与目标实验一致。

3. 运行当前分支支持的 Vanilla 基线：

   ```bash
   uv run python -m mindmemos_eval.cli memory \
     --benchmark-config config/mindmemos_eval/dreaming_evaluation/memoryagentbench.yaml \
     --benchmark-list memoryagentbench \
     --algorithm vanilla \
     --manifest-output reports/memoryagentbench_vanilla.jsonl \
     --api-key-output config/mindmemos/eval_api_keys.yaml
   ```

> **当前边界**：上述命令复现 MemoryAgentBench 的 Vanilla 基线。当前 `develop` 分支的 `memory` CLI 尚未暴露 Add → Dreaming → Answer/Search → Score 的完整开关，因此不能仅凭配置文件名复现主 README 表格中的 Vanilla + Dreaming 行。

### Skill-Forest 五基准测评

Skill 测评在进程内创建隔离的本地 Skill service，不需要启动前述 Memory server 或 HTTP 服务。
安装依赖并初始化共享配置：

```bash
uv sync --all-packages --extra alfworld --extra spreadsheetbench
make init
```

在 `config/mindmemos_eval/dev.yaml` 中填写 `models.target` 和 `models.optimizer` 的模型名、
`api_key`、`base_url`；四类并发和原生环境参数也在该文件配置。
OfficeQA 的回合限制字段为 `max_tool_turns`。

```bash
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
uv run --no-sync mindmemos-eval skill run --smoke
uv run --no-sync mindmemos-eval skill run
```

默认运行 LiveMath 的完整固定 split，包含 `no_skill`、`initial_skill`、`trained_skill` 三组。
`--smoke` 取固定 split 的前 4 train / 2 val / 2 test，并使用一棵树、一个候选、depth/budget 1。
只运行对照时使用 `--arms no_skill,initial_skill`，不会创建训练服务，也不要求 optimizer 连接。

```bash
uv run --no-sync python scripts/download_datasets.py download skill --all
uv run --no-sync mindmemos-eval skill run --benchmark all
uv run --no-sync mindmemos-eval skill run --benchmark spreadsheetbench --test-repeats 3
uv run --no-sync mindmemos-eval skill report output/mindmemos_eval/skill/<run-directory>
```

`--benchmark all` 按顺序执行 LiveMath、ALFWorld、SpreadsheetBench、SearchQA、OfficeQA。
运行前统一校验全部选中数据的下载 receipt、固定 split 和真实运行依赖；校验失败不会提交训练。
运行阶段某个基准失败后继续其他基准，最终返回非零退出码。

每次运行使用新目录；`--output-dir` 指向非空目录会报错。结果包括 `summary.json`、`summary.csv`、
`report.md`、精确 Skill 版本及原始 rollout 引用。训练 pipeline 的第一轮测试经核验后直接采用，
后续 repeat 使用同一个发布版本和独立工作目录。执行或评分错误单独记录；覆盖不完整时不输出完整准确率。
`report` 只重算保存结果的统计，不调用模型。Token 缺失保留为 null，optimizer 成本与测试成本分别记录。

SpreadsheetBench 固定使用原生 tool-calling，默认 `transactional_recalculation: false`。
SearchQA 统一使用 `lucadiliello/searchqa`，固定 revision 为
`c1a979068ba118d85467179b704031d113d689cc`。下载脚本将原始 Parquet 保存到
`data/mindmemos_skill/searchqa/data/`。split 目录只保留 `splits.json`，使用该来源的原生
`key`，保持原项目 train 400、val 200、test 1,400 的题目及顺序。
评测直接保留源数据的 `question`、`context`、`answers`，仅将 `key` 作为 task ID；
选中记录写入本次运行的 `prepared/searchqa/items.json`，不修改下载文件，也不需要 ID 映射资源。

---

## 快速冒烟测试（推荐先跑）

用 `--limit 2` 把数据量砍到最小，验证全链路是否跑通：

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/smoke_vanilla.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --limit 2 \
  --session-limit 2 \
  --judge-runs 1
```

> **警告**：``--api-key-output`` 会**完整写入**一个新的 ``api_keys.yaml`` 文件，
> 仅包含本次生成的 benchmark 身份。请勿直接指向服务端正在使用的
> ``config/mindmemos/api_keys.yaml``（除非是隔离环境）。
> 建议使用独立路径（如 ``config/mindmemos/eval_api_keys.yaml``），
> 然后将服务端配置指向该文件，或手动将生成的 key 合并到服务端文件中。

---

## 正式评测

### 跑全量 benchmark

三个 benchmark 放在一条命令里串行执行（保证 `api_keys.yaml` 不冲突）：

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/vanilla_run.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --judge-runs 1
```

> **add 前清库**：add 阶段开始前，eval 进程会按本次 run 的 `project_id` 直接清空 Qdrant 和
> Neo4j 中对应的数据，确保每次 run 从干净状态开始（每次 run 都有独立的 `project_id`）。这要求
> eval 进程能访问 Qdrant 和 Neo4j；用与服务端相同的环境变量配置（`MINDMEMOS_QDRANT_URL`、
> `MINDMEMOS_NEO4J_URI`、`MINDMEMOS_NEO4J_USERNAME`、`MINDMEMOS_NEO4J_PASSWORD`、
> `MINDMEMOS_NEO4J_DATABASE`），或用 `--qdrant-url` / `--neo4j-*` 参数。`--no-add` 跳过清库、
> 复用已有数据；`--skip-clean` 跳过清库但仍执行 add。

### 只跑部分 benchmark

```bash
# 只跑 LoCoMo 和 LongMemEval
--benchmark-list locomo,longmemeval
```

### 复用已入库的数据（跳过 add 阶段）

重新评测**同一 project** 上一次已入库的数据时，组合使用 ``--reuse-api-key``（复用上次的
api_key/project_id）和 ``--no-add``（跳过 add 阶段**以及** add 前的清库，保留下来的数据可直接复用）。
``--reuse-api-key`` 一次只复用一个 project，因此只传一个 benchmark：

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list locomo \
  --manifest-output reports/vanilla_run_retry.jsonl \
  --reuse-api-key config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --no-add \
  --judge-runs 1
```

> 搭配 ``--reuse-api-key`` 时**不需要** ``--api-key-output``（不会生成新身份）。要复用另一个
> benchmark 的数据，把 ``--reuse-api-key`` 指向含该 benchmark key 的文件，并单独传该 benchmark。

### 跳过 judge 阶段（只要答案不打分）

```bash
--no-score
```

### 调整并发

```bash
--max-conv-concurrency 5    # 同时处理的 conversation/sample 数
--max-qa-concurrency 10     # 每个 conversation 内同时处理的问题数
--max-search-concurrency 10
--max-score-concurrency 10
```

---

## 关键参数说明

| 参数 | 说明 |
|------|------|
| `--algorithm` | 算法 profile，当前支持 `vanilla` / `schema` |
| `--limit N` | 每个 benchmark 最多处理 N 条（locomo=N 个对话，longmemeval=N 个 sample，personamem=N 道题） |
| `--session-limit N` | LongMemEval 每个 sample 最多 add 前 N 个 session，用于加速冒烟测试 |
| `--judge-runs N` | 每道题独立跑 N 次 judge，取多数票；默认 1。这 N 次是**串行**跑的，judge 阶段耗时基本随 N 线性增长。PersonaMem 不适用（确定性判分） |
| `--no-add` | 跳过 add 阶段**以及** add 前的清库；复用该 project 已在 Qdrant/Neo4j 中的数据 |
| `--no-score` | 跳过 judge 阶段 |
| `--manifest-output` | 评测结果写入的 JSONL 文件，供后续 metrics 统计使用 |
| `--api-key-output` | **全新写入** API key 文件，仅含本次生成的 key。勿直接覆盖服务端正在使用的文件。fresh run 必填；搭配 `--reuse-api-key` 时可选（不使用） |
| `--reuse-api-key` | 复用已有 API key 文件（即上次的 api_key/project_id）；配合 `--no-add` 重新评测其数据。一次只能复用一个 benchmark |
| `--skip-clean` | 跳过 add 前清空该 run 的 `project_id`（Qdrant/Neo4j）。`--no-add` 从不清库；本参数用于跳过清库但仍执行 add |

---

## 统计耗时与 Token

评测跑完后，执行以下命令生成统计报告：

```bash
uv run python -m mindmemos_eval.memory.metrics \
  --manifest reports/vanilla_run.jsonl \
  --output reports/vanilla_run_metrics.jsonl \
  --xlsx-output reports/vanilla_run_metrics.xlsx \
  --json-output reports/vanilla_run_metrics_sheets.json
```

`--output` 写的是每个 run 一行的原始数据（manifest 内容 + Qdrant/ClickHouse 查询结果），不是下面这四个整理好的 sheet。想要程序化读取下面四个 sheet 的内容，用 `--json-output`——它和 xlsx 的四个 sheet 内容完全一致，只是换成 `{"summary": [...], "eval_metrics": [...], "llm_by_task": [...], "percentiles": [...]}` 这样的 JSON 结构（每个 sheet 一个数组，数组元素是以列名为 key 的字典）。

生成的 xlsx（或 `--json-output`）包含四个 sheet：

### summary sheet

每个 benchmark 一行，包含：

| 列 | 来源 | 说明 |
|----|------|------|
| `add_count` / `add_total_ms` / `add_avg_ms` | Qdrant | add 请求数和耗时 |
| `search_count` / `search_total_ms` / `search_avg_ms` | Qdrant | search 请求数和耗时 |
| `memory_count` | Qdrant | 最终存储的 memory 条数 |
| `llm_calls` / `llm_total_tokens` / `llm_prompt_tokens` | ClickHouse | server 侧 LLM 调用（add 阶段的 chunk/extract） |
| `search_total_tokens` / `search_avg_tokens/query` / `search_avg_tokens/call` | ClickHouse | search 阶段 token 总量，按 `TraceId` 聚合到"每次 search 请求"级别（不是每次 LLM call）——`SearchResult` 不回传 per-call token，这里的数字都是跑完之后从 ClickHouse OTel trace 重新算出来的 |
| `overall_accuracy` | manifest | 评测得分 |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_total_tokens` | manifest | eval 侧 answer 阶段 token |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_total_tokens` | manifest | eval 侧 judge 阶段 token |
| `add_token_avg` / `answer_token_avg` / `judge_token_avg` | ClickHouse / manifest | 平均每次 add 调用 / 每道题 answer / 每道题 judge 的 token 数 |

### eval_metrics sheet

每个 benchmark 一行，包含：

| 列 | 来源 | 说明 |
|----|------|------|
| `overall_accuracy` / `correct` / `total` | manifest | 整体准确率和题数 |
| `search_llm_calls` / `search_prompt_tokens` / `search_completion_tokens` / `search_total_tokens` | manifest，缺失时回退 ClickHouse 逐 query 聚合 | search 阶段 token（`vanilla`/`fast` 两边都是 0） |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_completion_tokens` / `answer_total_tokens` | manifest | answer 阶段 token |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_completion_tokens` / `judge_total_tokens` | manifest | judge 阶段 token |
| `build_elapsed_seconds` / `search_elapsed_seconds` / `answer_elapsed_seconds` / `total_elapsed_seconds` | manifest | 各阶段累计耗时（秒） |
| `by_question_type` / `by_topic` | manifest | 按题型 / 主题拆分的准确率 |

### llm_by_task sheet

按 LLM 任务类型分行，包含：

| 列 | 来源 | 说明 |
|----|------|------|
| `memory.add.extract` | ClickHouse | server 侧 add 时的 LLM 抽取调用 |
| `eval.answer` | manifest | eval 侧 answer 生成调用 |
| `eval.judge` | manifest | eval 侧 judge 打分调用 |

### percentiles sheet

每个操作的 token 和耗时分位数（min / max / p50 / p95，没有均值列——均值只在 summary sheet）：

| 列 | 来源 | 说明 |
|----|------|------|
| `add` token 分位数 | ClickHouse（逐次 add 调用） | |
| `add` time 分位数 | Qdrant（逐请求） | |
| `search` token 分位数 | ClickHouse（逐次 search **请求**，把一次 search 内部所有 `search.*` LLM call 按 `TraceId` 加总后再算分位数） | 反映真实的单次 search 成本，不是单次 LLM call 的成本 |
| `search` time 分位数 | Qdrant（逐请求） | |
| `eval.answer` token 分位数 | manifest（逐题记录） | 真实分位数 |
| `eval.answer` time 分位数 | manifest（仅有总耗时） | `time_p50_ms` = 平均值，非真实分位数 |
| `eval.judge` token 分位数 | manifest（逐题记录） | 真实分位数 |
| `eval.judge` time 分位数 | 无逐题时间 | `time_p50_ms` 为空 |

> PersonaMem 使用确定性评分，无 answer/judge LLM 调用，percentiles sheet 中不产生 `eval.*` 行。
> `vanilla`/`fast` search 完全不调用 LLM，所以这类 run 的 `search` token 分位数是空的；只有 `agentic`/`schema`（一次 query 会触发多次 LLM call）才会填上数据。
