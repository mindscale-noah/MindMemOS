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

## 快速冒烟测试（推荐先跑）

用 `--limit 2` 把数据量砍到最小，验证全链路是否跑通：

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/smoke_vanilla.jsonl \
  --api-key-output config/mindmemos/api_keys.yaml \
  --algorithm vanilla \
  --limit 2 \
  --session-limit 2 \
  --judge-runs 1
```

---

## 正式评测

### 跑全量 benchmark

三个 benchmark 放在一条命令里串行执行（保证 `api_keys.yaml` 不冲突）：

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/vanilla_run.jsonl \
  --api-key-output config/mindmemos/api_keys.yaml \
  --algorithm vanilla \
  --judge-runs 1
```

### 只跑部分 benchmark

```bash
# 只跑 LoCoMo 和 LongMemEval
--benchmark-list locomo,longmemeval
```

### 跳过 add 阶段（数据已入库时）

```bash
--no-add
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
| `--judge-runs N` | 每道题独立跑 N 次 judge，取多数票；默认 1；EverMemOS 论文用 3 |
| `--no-add` | 跳过 add 阶段（数据已在 Qdrant 中时使用） |
| `--no-score` | 跳过 judge 阶段 |
| `--manifest-output` | 评测结果写入的 JSONL 文件，供后续 metrics 统计使用 |
| `--api-key-output` | 本次 run 生成的 API key 写入路径，需与 server 的 `auth.api_key_file` 一致 |

---

## 统计耗时与 Token

评测跑完后，执行以下命令生成统计报告：

```bash
uv run python -m mindmemos_eval.memory.metrics \
  --manifest reports/vanilla_run.jsonl \
  --output reports/vanilla_run_metrics.jsonl \
  --xlsx-output reports/vanilla_run_metrics.xlsx
```

生成的 xlsx 包含四个 sheet：

### summary sheet

每个 benchmark 一行，包含：

| 列 | 来源 | 说明 |
|----|------|------|
| `add_count` / `add_total_ms` / `add_avg_ms` | Qdrant | add 请求数和耗时 |
| `search_count` / `search_total_ms` / `search_avg_ms` | Qdrant | search 请求数和耗时 |
| `memory_count` | Qdrant | 最终存储的 memory 条数 |
| `llm_calls` / `llm_total_tokens` / `llm_prompt_tokens` | ClickHouse | server 侧 LLM 调用（add 阶段的 chunk/extract） |
| `overall_accuracy` | manifest | 评测得分 |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_total_tokens` | manifest | eval 侧 answer 阶段 token |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_total_tokens` | manifest | eval 侧 judge 阶段 token |

### eval_metrics sheet

每个 benchmark 一行，包含完整的评测指标（accuracy、各类别得分、search/answer/judge 的 token 和耗时）。

### llm_by_task sheet

按 LLM 任务类型分行，包含：

| task | 说明 |
|------|------|
| `memory.add.extract` | server 侧 add 时的 LLM 抽取调用（来自 ClickHouse） |
| `eval.answer` | eval 侧 answer 生成调用（来自 manifest） |
| `eval.judge` | eval 侧 judge 打分调用（来自 manifest） |

### percentiles sheet

每个操作的 token 和耗时分位数（min / max / p50 / p95）：

| operation | token 分位数来源 | time 分位数来源 |
|-----------|--------------|--------------|
| `add` | ClickHouse（逐请求） | Qdrant（逐请求） |
| `search` | ClickHouse（逐请求） | Qdrant（逐请求） |
| `eval.answer` | manifest 逐题记录（真实分位数） | manifest 仅有总耗时，`time_p50_ms` = 平均值 |
| `eval.judge` | manifest 逐题记录（真实分位数） | 无逐题时间，`time_p50_ms` 为空 |

> PersonaMem 使用确定性评分，无 answer/judge LLM 调用，percentiles sheet 中不产生 `eval.*` 行。

---

## 数据流说明

```
eval 进程                          MindMemOS Server (:8000)       外部 LLM
─────────────────                  ────────────────────────        ─────────────────
add 阶段：
  mindmemos_sdk.add()  ──HTTP──►  /v1/memory/add
                                   └─ chunk/extract ──────────►  yibuapi (dev.yaml key)
                                   └─ 写 Qdrant + Neo4j
                                   └─ 写 ClickHouse (OTel trace)

search 阶段：
  mindmemos_sdk.search() ──HTTP──► /v1/memory/search
                                   └─ 查 Qdrant 返回 memories

answer 阶段：
  LLMClient.complete()  ─────────────────────────────────────►  yibuapi (memory_evaluation.yaml key)

judge 阶段：
  LLMClient.complete()  ─────────────────────────────────────►  yibuapi (memory_evaluation.yaml key)
```

---

## 常见问题

### search 返回 HTTP 500

`make db-clean` 后重建了容器但未重启 server，server 持有旧连接。重启 server 即可：

```bash
# Ctrl+C 之前的 uvicorn 进程，然后
uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### answer/judge 阶段卡住不动（进度条 0%）

eval 侧 LLM API key 失效。检查并更新 `config/mindmemos_eval/memory_evaluation.yaml` 中 `llm` / `answer_llm` / `judge_llm` 的 `api_key`。

```bash
# 验证 key 是否有效
curl -s -X POST https://your-llm-provider/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
# 正常返回 choices[]，异常返回 {"error": {"message": "Invalid token"...}}
```

### xlsx 中 ClickHouse llm_calls = 0

ClickHouse 的 trace 数据依赖 OTel collector 正常运行。确认 `mindmemos-otel-collector` 容器健康，以及 `config/mindmemos/dev.yaml` 中的 `telemetry.endpoint` 配置正确。

### add 成功但 search 失败

两个阶段使用不同 API key：add 用 `api_keys.yaml`（server 侧），search 同样用 `api_keys.yaml`。如果 `make db-clean` 后重新跑，旧 key 已不在新 server 的内存中，需要重新跑评测命令（会写入新 key）或重启 server 让其热重载 `api_keys.yaml`。
