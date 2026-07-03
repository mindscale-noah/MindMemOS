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
| `--judge-runs N` | Run judge N times per question, decide by majority vote; default 1; EverMemOS paper uses 3. The N runs execute sequentially per question, so judge-stage latency scales roughly linearly with N |
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
| `search_total_tokens` / `search_avg_tokens/query` / `search_avg_tokens/call` | ClickHouse | Search-stage token totals, aggregated per search request (grouped by `TraceId`, not per individual LLM call) |
| `overall_accuracy` | manifest | Benchmark score |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_total_tokens` | manifest | Eval-side answer stage tokens |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_total_tokens` | manifest | Eval-side judge stage tokens |
| `add_token_avg` / `answer_token_avg` / `judge_token_avg` | ClickHouse / manifest | Average tokens per add call / per question (answer) / per question (judge) |

> Search-stage tokens are *not* returned by the SDK response — `SearchResult` only carries `request_id` and `memories`. All search token numbers are reconstructed after the run from ClickHouse OTel trace data (see [Data Flow](#data-flow)).

### `eval_metrics` sheet

One row per benchmark with the complete set of evaluation metrics: accuracy, per-category scores, and search / answer / judge token counts and elapsed times. `search_llm_calls` / `search_prompt_tokens` / `search_completion_tokens` / `search_total_tokens` fall back to the ClickHouse per-query aggregate whenever the manifest itself doesn't carry them (e.g. `vanilla`/`fast` search makes no LLM calls, so these are 0 either way).

### `llm_by_task` sheet

One row per LLM task type per run:

| Task | Description |
|------|-------------|
| `memory.add.extract` | Server-side LLM extraction calls during add (from ClickHouse) |
| `eval.answer` | Eval-side answer generation calls (from manifest) |
| `eval.judge` | Eval-side judge scoring calls (from manifest) |

### `percentiles` sheet

Token and timing percentiles (min / max / p50 / p95, no average — averages live in the `summary` sheet) per operation:

| Operation | Token percentiles | Time percentiles |
|-----------|------------------|-----------------|
| `add` | Per-add-call data from ClickHouse | Per-request data from Qdrant |
| `search` | Per-search-*request* data from ClickHouse — all `search.*` LLM calls within one search (e.g. `multi_query`, `sufficiency_check` under agentic search) are summed by `TraceId` before computing percentiles, so this reflects real per-query cost | Per-request data from Qdrant |
| `eval.answer` | Per-question data from manifest (true percentiles) | Only total elapsed available; `time_p50_ms` = average |
| `eval.judge` | Per-question data from manifest (true percentiles) | No per-question timing; `time_p50_ms` is empty |

> PersonaMem uses deterministic (multiple-choice) scoring — no answer/judge LLM calls are made, so no `eval.*` rows appear in percentiles.
> `vanilla`/`fast` search makes no LLM calls at all, so `search` token percentiles are empty for those runs; only `agentic`/`schema` search (which fans out to multiple LLM calls per query) populates them.

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
                                   └─ (agentic/schema only) LLM calls ►  LLM  (dev.yaml key)
                                   └─ write ClickHouse (OTel)

Answer stage:
  LLMClient.complete() ──────────────────────────────────────►    LLM  (memory_evaluation.yaml key)

Judge stage:
  LLMClient.complete() ──────────────────────────────────────►    LLM  (memory_evaluation.yaml key)
```

`SearchResult` (the SDK's return value from `sdk.search()`) only carries `request_id` and `memories` — it does **not** return per-call token usage, even though the server may have made several LLM calls internally (agentic/schema search). Search-stage token statistics are therefore never read from the SDK response; `metrics.py` reconstructs them after the run by querying ClickHouse directly, grouping the underlying `search.*` LLM call spans by `TraceId` to get true per-search-request costs (see the `percentiles` sheet above).

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

### A run is much slower than expected

A few independent factors compound:

- **`--algorithm schema` defaults to `agentic` search**, which fans out to several sequential LLM calls per query (`multi_query`, `sufficiency_check`, etc.). If you only need to validate that add/search plumbing works — not to benchmark search quality — override with `--search-strategy fast` to skip the extra LLM round-trips.
- **`--limit N` only trims top-level items** (conversations for LoCoMo, samples for LongMemEval, questions for PersonaMem), not the work inside each item. A single LoCoMo conversation can contain 15+ sessions and ~200 QA pairs, so `--limit 1` alone does not guarantee a fast run — LongMemEval's `--session-limit N` caps sessions per sample, but LoCoMo has no equivalent flag. For a fast smoke test on LoCoMo, trim a copy of the dataset JSON directly (fewer `session_*` keys and a shorter `qa` list per conversation) and point a scratch `benchmarks.locomo.dataset` at it.
- **`--judge-runs N > 1` multiplies judge latency serially** (see the Key Parameters table above).
- Benchmarks in `--benchmark-list` run **sequentially**, not in parallel — split into separate invocations if you want to run more than one at once.
