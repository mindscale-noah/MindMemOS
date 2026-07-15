# MindMemOS 评测指南

本文说明如何对 MindMemOS 跑三个标准 benchmark（LoCoMo、LongMemEval、PersonaMem），以及如何用 `metrics.py` 统计耗时与 token 用量。

---

## 前置条件

### 1. 启动依赖服务

```bash
make db          # 启动 Qdrant / Neo4j / Kafka / ClickHouse / Grafana 等
```

> **注意**：若之前执行过 `make db-clean`（会删除所有 volume），务必同时重启 MindMemOS server，否则 server 的旧连接池会导致 search 返回 500。

### 2. 启动 MindMemOS server

```bash
uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### 3. 配置 LLM API Key

评测时，`add` 阶段的 LLM 调用由 **server** 自己完成，使用 `config/mindmemos/dev.yaml` 里的 key；`answer` 和 `judge` 阶段由 **eval 进程**直接调用，使用 `config/mindmemos_eval/memory_evaluation.yaml` 里的 key。两套 key 相互独立，需要分别配置。

```yaml
# config/mindmemos_eval/memory_evaluation.yaml
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

## 数据集下载

示例配置中引用的 benchmark 数据集**不在**本仓库中，需自行从各 benchmark 作者（LoCoMo、LongMemEval、PersonaMem）处获取，并按 ``config/mindmemos_eval/memory_evaluation_locomo.example.yaml`` 中的路径放置。

复制示例配置并填入 LLM key：
```bash
cp config/mindmemos_eval/memory_evaluation_locomo.example.yaml config/mindmemos_eval/memory_evaluation_locomo.yaml
```

---

## 快速冒烟测试（推荐先跑）

用 `--limit 2` 把数据量砍到最小，验证全链路是否跑通：

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
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
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/vanilla_run.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --judge-runs 1
```

### 只跑部分 benchmark

```bash
# 只跑 LoCoMo 和 LongMemEval
--benchmark-list locomo,longmemeval
```

### 跳过 add 阶段（数据已入库，重新评测）

用 ``--reuse-api-key`` 复用上次生成的 key，避免覆盖：

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/vanilla_run_retry.jsonl \
  --reuse-api-key config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --no-add \
  --judge-runs 1
```

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
| `--no-add` | 跳过 add 阶段（数据已在 Qdrant 中时使用） |
| `--no-score` | 跳过 judge 阶段 |
| `--manifest-output` | 评测结果写入的 JSONL 文件，供后续 metrics 统计使用 |
| `--api-key-output` | **全新写入** API key 文件，仅含本次生成的 key。勿直接覆盖服务端正在使用的文件；建议用独立路径（如 `eval_api_keys.yaml`） |
| `--reuse-api-key` | 复用已有 API key 文件，配合 `--no-add` 重新评测时可避免覆盖 key 文件 |

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
