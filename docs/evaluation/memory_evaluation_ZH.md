# MindMemOS Memory 评测

[English](memory_evaluation.md) · [Skill 评测](skill_evaluation_ZH.md)

本指南覆盖 MindVanilla、MindSchema、Dreaming 和 Feedback。历史精度来自仓库 README。空白单元格留待补充，冒烟或回归测试结果单独记录。

## Memory 评测前置条件

在仓库根目录先运行 `uv sync --all-packages` 和 `make init`。使用独立评测服务及存储，并将服务端 `auth.api_key_file` 指向下面命令使用的 `config/mindmemos/eval_api_keys.yaml`。先填写服务端模型配置，再启动服务。评测器会为每次新运行生成项目与 key。

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
uv run --no-sync python scripts/download_datasets.py list memory
uv run --no-sync python scripts/download_datasets.py download memory --name locomo
uv run --no-sync python scripts/download_datasets.py download memory --name longmemeval
uv run --no-sync python scripts/download_datasets.py download memory --name personamem
uv run --no-sync python scripts/download_datasets.py download memory --name memoryagentbench
```

默认下载到 `data/mindmemos_memory/<dataset>/`，保留官方文件格式：

| 数据集 | 官方来源 | 相对于该数据集下载目录的文件 |
| --- | --- | --- |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) | `locomo10.json` |
| LongMemEval | [longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `longmemeval_s_cleaned.json`、`longmemeval_m_cleaned.json`、`longmemeval_oracle.json` |
| PersonaMem v1 | [PersonaMem-v1](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) | `questions_{32k,128k,1M}.csv` 与对应的 `shared_contexts_{32k,128k,1M}.jsonl` |
| MemoryAgentBench | [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) | 官方四个数据分区的 `data/*.parquet`，以及 `entity2id.json` |

更新 `make init` 生成的本地 `config/mindmemos_eval/memory_evaluation/locomo.yaml`，填入评测模型的 API key 和数据路径。

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
uv run --no-sync python scripts/download_datasets.py download memory --name locomo --data-root /path/to/memory
uv run --no-sync python scripts/download_datasets.py download memory --name locomo --verify-only
uv run --no-sync python scripts/download_datasets.py download memory --name personamem --offline
```

自定义目录示例会写入 `/path/to/memory/locomo/`，校验该目录时也需传入相同的 `--data-root`。
LoCoMo 首次通过直接链接下载需要联网。已有目录只校验，不覆盖；每个下载目录中的
`download_receipt.json` 记录来源与本地文件哈希。实验 split 随代码提交到 GitHub，脚本不下载或生成 split。

---

## MindVanilla 与 MindSchema

两种 profile 复用同一批数据与评分器。MindVanilla 走 Vanilla 写入和检索，示例矩阵使用 `fast` 检索。MindSchema 走 Schema 写入和检索，示例矩阵使用 `agentic` 检索。比较完整 profile 时保留各自配置，并记录写入、作答与判分模型、检索参数和数据版本。若要单独比较写入算法，另行固定检索设置。

先按[数据集下载](#memory-数据集下载)准备数据和本地 `locomo.yaml`。以下命令逐个执行。切换 `profile` 后重跑，给两组使用独立的新评测项目。

### MindVanilla

将 `profile` 设为 `vanilla`，运行下方各基准命令。读取对标表中的 MindVanilla 行。

```bash
profile=vanilla
```

### MindSchema

将 `profile` 设为 `schema`，用同一批数据复跑。读取对标表中的 MindSchema 行。

```bash
profile=schema
```

### LoCoMo

下载 `locomo`，将 `benchmarks.locomo.dataset` 指向 `locomo10.json`。按会话写入带时间的对话，再逐题检索、生成答案，由 LLM 按参考答案判对错。默认跳过 category 5。汇总 overall 与各题型正确率。

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list locomo \
  --algorithm "$profile" \
  --manifest-output "reports/locomo_$profile.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

**历史对标精度（%）**

| 方法                       | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
| :------------------------- | :--------: | :-------: | :------: | :---------: | :-----: |
| Mem0                       | 68.97 | 61.70 | 58.26 | 50.00  | 64.20 |
| MemU                       | 74.91 | 72.34 | 43.61 | 54.17  | 66.67 |
| MemOS                      | 85.37 | 79.43 | 75.08 | 64.58  | 80.76 |
| Zep                        | 90.84 | 81.91 | 77.26 | 75.00  | 85.22 |
| EverOS                     | 96.67 | 91.84 | 89.72 | 76.04  | 93.05 |
| **MindMemOS-MindVanilla** | 92.03 | 85.82 | 83.80 | 66.67  | 87.60 |
| **MindMemOS-MindSchema**  | **96.79** | **93.97** | **90.34** | **82.29** | **94.03** |

### LongMemEval

下载 `longmemeval`，将 `benchmarks.longmemeval.dataset` 指向所选版本。下例使用 `longmemeval_s_cleaned.json`。每题写入自己的历史会话，再按题检索并作答。LLM 按题型判分，报告整体、题型宏平均和拒答准确率。

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list longmemeval \
  --algorithm "$profile" \
  --manifest-output "reports/longmemeval_$profile.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

全量前可先加 `--limit 2 --session-limit 2` 验证流程。

| 版本 | 指标 | MindVanilla | MindSchema | 对标方法及精度 |
| --- | --- | ---: | ---: | --- |
| LongMemEval-S | Overall accuracy | | | |

### PersonaMem

下载 `personamem`，在 `locomo.yaml` 中配置同一上下文长度的问答 CSV 和上下文 JSONL。也可在 `make init` 生成的 `personamem.yaml` 中更新相同字段，再替换命令中的配置路径。按题使用对应的对话前缀构建和检索记忆。模型选择答案，评测器与标准选项确定性匹配，并按题型汇总。

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list personamem \
  --algorithm "$profile" \
  --manifest-output "reports/personamem_$profile.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

**历史对标精度（%）**。使用与历史运行相同的上下文长度后再比较。

| Method | Recall | Ack. Lat. | Trk. Evo. | Revisit | Suggest | Recom. | General. | Overall |
| :----- | :----: | :-------: | :-------: | :-----: | :-----: | :-----: | :------: | :-----: |
| Mem0 | 46.51 | 41.18 | 65.47 | 90.91 | 12.90 | 34.55 | 43.86 | 51.61 |
| MemU | 64.34 | 64.71 | 66.20 | 87.88 | 31.18 | 67.27 | 84.21 | 65.70 |
| MemOS | 53.49 | 82.35 | 66.91 | 79.80 | 41.94 | 69.09 | 75.44 | 63.67 |
| EverOS | 74.42 | 64.71 | 64.03 | 85.86 | 35.48 | 65.45 | 84.21 | 67.57 |
| **MindMemOS-MindVanilla** | 76.74 | 88.24 | 65.47 | 87.88 | 17.20 | 80.00 | 82.46 | 67.74 |
| **MindMemOS-MindSchema** | 81.40 | 64.71 | 64.75 | 82.83 | 47.31 | 76.36 | 73.68 | 70.63 |

## Dreaming

### MemoryAgentBench FactConsolidation

下载 `memoryagentbench` 后，将目标 `metadata.source` 的 Parquet 记录转成 JSONL。以下以单跳 6k 子集为例。派生文件放在下载目录之外，以便原始下载仍能通过校验。

```bash
mkdir -p data/mindmemos_memory/prepared
uv run --no-sync python - <<'PYTHON'
import json
from pathlib import Path

import pyarrow.parquet as parquet

root = Path("data/mindmemos_memory")
source = "factconsolidation_sh_6k"
paths = sorted((root / "memoryagentbench/data").glob("*.parquet"))
target = root / "prepared" / f"{source}.jsonl"
matches = 0
with target.open("w", encoding="utf-8") as output:
    for path in paths:
        for batch in parquet.ParquetFile(path).iter_batches():
            for record in batch.to_pylist():
                if (record.get("metadata") or {}).get("source") == source:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    matches += 1
if not matches:
    raise SystemExit(f"No records matched {source}")
PYTHON
```

参考 [MemoryAgentBench 示例](../../config/mindmemos_eval/dreaming_evaluation/memoryagentbench.example.yaml)，在 `make init` 生成的 `memoryagentbench.yaml` 中填写模型连接。把 `benchmarks.memoryagentbench.dataset` 指向 `data/mindmemos_memory/prepared/factconsolidation_sh_6k.jsonl`，并核对 `sub_dataset` 与 `chunk_size`。

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/dreaming_evaluation/memoryagentbench.yaml \
  --benchmark-list memoryagentbench \
  --algorithm vanilla \
  --manifest-output "reports/memoryagentbench_vanilla.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

评测器分块写入上下文，检索记忆后生成答案。FactConsolidation 使用归一化的 Substring Exact Match，命中任一参考答案即得分。上面的命令运行 Vanilla 对照。历史 Dreaming 实验在写入与问答之间执行记忆巩固。当前矩阵 CLI 未串联这一步，完整复现命令待补充。Dreaming 的合并与归档行为可用仓库内置样例回归验证。

```bash
uv run --no-sync pytest -q src/mindmemos/tests/pipelines/dreaming
```

**历史对标精度**。SH / MH 分别是单跳 / 多跳，score 为四种上下文规模下的平均 Substring Exact Match，archived 为归档比例。

| Method                             | SH score | SH archived | MH score | MH archived |
|:-----------------------------------| :------: | :---------: | :------: | :---------: |
| **GPT-4o-mini**                    |  |  |  |  |
| Mem0                               | 0.180 | — | 0.020 | — |
| MemoRAG                            | 0.270 | — | 0.070 | — |
| HippoRAG-v2                        | 0.540 | — | 0.050 | — |
| MindMemOS-MindVanilla              | 0.635 | — | 0.118 | — |
| **MindMemOS-MindVanilla + Dreaming** | 0.738 | 21.4% | 0.180 | 19.4% |
| **GPT-5-mini**                     |  |  |  |  |
| Infini Memory                      | 0.800 | — | 0.220 | — |
| MindMemOS-MindVanilla              | 0.900 | — | 0.190 | — |
| **MindMemOS-MindVanilla + Dreaming** | 0.920 | 23.5% | 0.250 | 21.5% |

## Feedback

**数据**。使用仓库内置的合成对话、反馈与记忆记录，无需下载。显式反馈测试检查根据上下文规划并执行记忆操作，隐式反馈测试检查会话归组、信号检测和已处理记录过滤。

```bash
uv run --no-sync pytest -q src/mindmemos/tests/pipelines/feedback
```

| 数据集 | 指标 | Feedback | 对标方法及精度 |
| --- | --- | ---: | --- |
| | | | |

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

### 运行 LoCoMo、LongMemEval 和 PersonaMem

三个 benchmark 放在一条命令里串行执行（保证 `api_keys.yaml` 不冲突）：

```bash
uv run python -m mindmemos_eval.cli memory \
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
uv run python -m mindmemos_eval.cli memory \
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

反思与合并结果统一保持批次提交顺序，不随请求完成先后改变。
生成协议版本为 5，不能续用旧版本训练检查点。


优化器响应统一按参考协议处理：保留原始 JSON 顺序与元数据，按报告批次大小补支持次数，并保存请求与回答。无效模型输出使用证据回退，传输故障仍报错；不再提供 strict/reference 切换。
