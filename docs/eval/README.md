# MindMemOS Evaluation Guide

This guide explains how to run the three standard benchmarks (LoCoMo, LongMemEval, PersonaMem) against MindMemOS, and how to use `metrics.py` to collect timing and token statistics.

---

## Prerequisites

### 1. Start dependency services

```bash
make db          # starts Qdrant / Neo4j / Kafka / ClickHouse / Grafana
```

> **Important**: If you previously ran `make db-clean` (which deletes all volumes), you must also restart the MindMemOS server. Otherwise the server's stale connection pool will cause search requests to return HTTP 500.

### 2. Start the MindMemOS server

```bash
uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### 3. Configure LLM API keys

The **add** stage LLM calls are made by the **server** using the keys in `config/mindmemos/dev.yaml`.
The **answer** and **judge** stage LLM calls are made directly by the **eval process** using the keys in `config/mindmemos_eval/memory_evaluation.yaml`.
These two key sets are completely independent and must be configured separately.

```yaml
# config/mindmemos_eval/memory_evaluation.yaml
runner:
  llm:
    model: gpt-4.1-mini
    api_key: <YOUR_KEY>
    base_url: https://your-llm-provider/v1
  answer_llm:
    api_key: <YOUR_KEY>          # can be the same as llm
    base_url: https://your-llm-provider/v1
  judge_llm:
    api_key: <YOUR_KEY>
    base_url: https://your-llm-provider/v1
```

Verify both the server and LLM API are reachable before running:

```bash
# Server should return JSON (not connection refused / 500)
curl -s -X POST http://127.0.0.1:8000/v1/memory/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <any existing key>" \
  -d '{"query":"test","user_id":"test"}' | head -c 100

# LLM API should return choices[] (not "Invalid token")
curl -s -X POST https://your-llm-provider/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

---

## Smoke Test (recommended first)

Use `--limit 2` to minimize data volume and verify the full pipeline end-to-end:

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

## Full Evaluation

### Run all benchmarks

All three benchmarks are executed sequentially in a single command to avoid `api_keys.yaml` conflicts:

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/vanilla_run.jsonl \
  --api-key-output config/mindmemos/api_keys.yaml \
  --algorithm vanilla \
  --judge-runs 1
```

### Run a subset of benchmarks

```bash
--benchmark-list locomo,longmemeval
```

### Skip the add stage (data already ingested)

```bash
--no-add
```

### Skip the judge stage (generate answers only)

```bash
--no-score
```

### Tune concurrency

```bash
--max-conv-concurrency 5    # concurrent conversation/sample build tasks
--max-qa-concurrency 10     # concurrent question-answer tasks per conversation
--max-search-concurrency 10
--max-score-concurrency 10
```

---

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--algorithm` | Algorithm profile: `vanilla` or `schema` |
| `--limit N` | Max items per benchmark (conversations for LoCoMo, samples for LongMemEval, questions for PersonaMem) |
| `--session-limit N` | LongMemEval only — add at most N sessions per sample; useful for smoke tests |
| `--judge-runs N` | Run judge N times per question, decide by majority vote; default 1; EverMemOS paper uses 3 |
| `--no-add` | Skip memory ingestion (reuse data already in Qdrant) |
| `--no-score` | Skip judge/scoring stage |
| `--manifest-output` | JSONL file where eval results are written; required for metrics collection |
| `--api-key-output` | Path where generated API keys are written; must match the server's `auth.api_key_file` |

---

## Collecting Timing and Token Statistics

After the evaluation completes, run:

```bash
uv run python -m mindmemos_eval.memory.metrics \
  --manifest reports/vanilla_run.jsonl \
  --output reports/vanilla_run_metrics.jsonl \
  --xlsx-output reports/vanilla_run_metrics.xlsx
```

The output xlsx contains four sheets:

### `summary` sheet

One row per benchmark. Key columns:

| Column | Source | Description |
|--------|--------|-------------|
| `add_count` / `add_total_ms` / `add_avg_ms` | Qdrant | Number of add requests and latency |
| `search_count` / `search_total_ms` / `search_avg_ms` | Qdrant | Number of search requests and latency |
| `memory_count` | Qdrant | Final number of stored memory items |
| `llm_calls` / `llm_total_tokens` / `llm_prompt_tokens` | ClickHouse | Server-side LLM calls during add (chunk/extract) |
| `overall_accuracy` | manifest | Benchmark score |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_total_tokens` | manifest | Eval-side answer stage tokens |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_total_tokens` | manifest | Eval-side judge stage tokens |

### `eval_metrics` sheet

One row per benchmark with the complete set of evaluation metrics: accuracy, per-category scores, and search / answer / judge token counts and elapsed times.

### `llm_by_task` sheet

One row per LLM task type per run:

| Task | Description |
|------|-------------|
| `memory.add.extract` | Server-side LLM extraction calls during add (from ClickHouse) |
| `eval.answer` | Eval-side answer generation calls (from manifest) |
| `eval.judge` | Eval-side judge scoring calls (from manifest) |

### `percentiles` sheet

Token and timing percentiles (min / max / p50 / p95) per operation:

| Operation | Token percentiles | Time percentiles |
|-----------|------------------|-----------------|
| `add` | Per-request data from ClickHouse | Per-request data from Qdrant |
| `search` | Per-request data from ClickHouse | Per-request data from Qdrant |
| `eval.answer` | Per-question data from manifest (true percentiles) | Only total elapsed available; `time_p50_ms` = average |
| `eval.judge` | Per-question data from manifest (true percentiles) | No per-question timing; `time_p50_ms` is empty |

> PersonaMem uses deterministic (multiple-choice) scoring — no answer/judge LLM calls are made, so no `eval.*` rows appear in percentiles.

---

## Data Flow

```
Eval process                       MindMemOS Server (:8000)        External LLM
────────────────                   ────────────────────────         ─────────────────
Add stage:
  sdk.add()          ──HTTP──►    /v1/memory/add
                                   └─ chunk/extract ──────────►   LLM  (dev.yaml key)
                                   └─ write Qdrant + Neo4j
                                   └─ write ClickHouse (OTel)

Search stage:
  sdk.search()       ──HTTP──►    /v1/memory/search
                                   └─ query Qdrant → return memories

Answer stage:
  LLMClient.complete() ──────────────────────────────────────►    LLM  (memory_evaluation.yaml key)

Judge stage:
  LLMClient.complete() ──────────────────────────────────────►    LLM  (memory_evaluation.yaml key)
```

---

## Troubleshooting

### `search` returns HTTP 500

The server was not restarted after `make db-clean`. The old connection pool is stale. Restart the server:

```bash
# Ctrl+C the running uvicorn process, then:
uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### Answer/judge stage hangs at 0% progress

The eval-side LLM API key is invalid or expired. Update `api_key` under `llm`, `answer_llm`, and `judge_llm` in `config/mindmemos_eval/memory_evaluation.yaml`, then re-run with `--no-add` to skip re-ingesting data.

```bash
# Verify the key
curl -s -X POST https://your-llm-provider/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

### `llm_calls = 0` in the xlsx for ClickHouse columns

ClickHouse trace data depends on the OTel collector running correctly. Confirm the `mindmemos-otel-collector` container is healthy and that `telemetry.endpoint` in `config/mindmemos/dev.yaml` points to the correct address.

### Add succeeds but the run starts from scratch on re-run

Each run calls `new_identity()` to generate a fresh `project_id` / `api_key`, so data from a previous run is not reused automatically. If the add stage already completed and you only need to re-run answer/judge, pass `--no-add`.
