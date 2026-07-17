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
The **answer** and **judge** stage LLM calls are made directly by the **eval process** using the keys in `config/mindmemos_eval/memory_evaluation_locomo.example.yaml` (or `memory_evaluation_personamem.example.yaml`).
These two key sets are completely independent and must be configured separately.

```yaml
# config/mindmemos_eval/memory_evaluation_locomo.yaml
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

## Dataset Download

The benchmark datasets referenced in the example config are **not** included in this repository.
Download them from the respective benchmark authors (LoCoMo, LongMemEval, PersonaMem) and place
them at the paths shown in ``config/mindmemos_eval/memory_evaluation_locomo.example.yaml``.

Copy the example config and adjust LLM keys before running:
```bash
cp config/mindmemos_eval/memory_evaluation_locomo.example.yaml config/mindmemos_eval/memory_evaluation_locomo.yaml
```

---

## Smoke Test (recommended first)

Use `--limit 2` to minimize data volume and verify the full pipeline end-to-end:

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

> **Warning**: ``--api-key-output`` writes a **fresh** ``api_keys.yaml`` with only the generated
> benchmark identities. Do NOT point it at the server's live key file
> (``config/mindmemos/api_keys.yaml``) unless you are running in an isolated environment.
> Use a separate path (e.g. ``config/mindmemos/eval_api_keys.yaml``) and point the server at
> that file for the duration of the evaluation, or merge the generated keys into the server's
> key file manually.

---

## Full Evaluation

### Run all benchmarks

All three benchmarks are executed sequentially in a single command to avoid `api_keys.yaml` conflicts:

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo,longmemeval,personamem \
  --manifest-output reports/vanilla_run.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --judge-runs 1
```

> **Pre-add cleanup**: before the add stage the eval process clears the run's
> `project_id` directly from Qdrant and Neo4j, so each run starts from a clean project
> (each run gets its own `project_id`). This requires Qdrant and Neo4j reachable from the
> eval process; configure them via the same env vars as the server (`MINDMEMOS_QDRANT_URL`,
> `MINDMEMOS_NEO4J_URI`, `MINDMEMOS_NEO4J_USERNAME`, `MINDMEMOS_NEO4J_PASSWORD`,
> `MINDMEMOS_NEO4J_DATABASE`) or the `--qdrant-url` / `--neo4j-*` flags. `--no-add` skips
> cleanup and reuses existing data; `--skip-clean` skips cleanup while still adding.

### Reuse previously added memories (skip the add stage)

When re-running evaluation against memories already ingested by a prior run of the **same
project**, combine ``--reuse-api-key`` (reuse that run's api_key/project_id) with
``--no-add`` (skip the add stage *and* the pre-add cleanup, so the prior memories stay in
place). ``--reuse-api-key`` reuses exactly one project, so pass a single benchmark:

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_locomo.yaml \
  --benchmark-list locomo \
  --manifest-output reports/vanilla_run_retry.jsonl \
  --reuse-api-key config/mindmemos/eval_api_keys.yaml \
  --algorithm vanilla \
  --no-add \
  --judge-runs 1
```

> ``--api-key-output`` is **not** needed with ``--reuse-api-key`` (no fresh identities are
> generated). To rerun a different benchmark's memories, point ``--reuse-api-key`` at the
> file containing that benchmark's key and pass that benchmark alone.

### Run a subset of benchmarks

```bash
--benchmark-list locomo,longmemeval
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
| `--judge-runs N` | Run judge N times per question, decide by majority vote; default 1. The N runs execute sequentially per question, so judge-stage latency scales roughly linearly with N. Not applicable to PersonaMem (deterministic scoring) |
| `--no-add` | Skip memory ingestion **and** the pre-add cleanup; reuse memories already in Qdrant/Neo4j for this project |
| `--no-score` | Skip judge/scoring stage |
| `--manifest-output` | JSONL file where eval results are written; required for metrics collection |
| `--api-key-output` | Path where **fresh** API keys are generated. Writes a complete file — do NOT point at the server's live key file. Required for fresh runs; optional (unused) with `--reuse-api-key` |
| `--reuse-api-key` | Path to an existing API key file from a prior run. Reuses that run's api_key/project_id; use with `--no-add` to rerun against its memories. Only one benchmark per invocation |
| `--skip-clean` | Skip clearing the run's `project_id` from Qdrant/Neo4j before add. `--no-add` never clears; this flag skips cleanup while still adding |

---

## Collecting Timing and Token Statistics

After the evaluation completes, run:

```bash
uv run python -m mindmemos_eval.memory.metrics \
  --manifest reports/vanilla_run.jsonl \
  --output reports/vanilla_run_metrics.jsonl \
  --xlsx-output reports/vanilla_run_metrics.xlsx \
  --json-output reports/vanilla_run_metrics_sheets.json
```

`--output` writes one row per run of raw data (manifest content plus the Qdrant/ClickHouse query results) — it is *not* the same as the four curated sheets below. To consume those four sheets programmatically, use `--json-output` instead: it mirrors the xlsx sheets exactly, just as `{"summary": [...], "eval_metrics": [...], "llm_by_task": [...], "percentiles": [...]}` (one array per sheet, each entry a dict keyed by column name).

The output xlsx (or `--json-output`) contains four sheets:

### `summary` sheet

One row per benchmark. Key columns:

| Column | Source | Description |
|--------|--------|-------------|
| `add_count` / `add_total_ms` / `add_avg_ms` | Qdrant | Number of add requests and latency |
| `search_count` / `search_total_ms` / `search_avg_ms` | Qdrant | Number of search requests and latency |
| `memory_count` | Qdrant | Final number of stored memory items |
| `llm_calls` / `llm_total_tokens` / `llm_prompt_tokens` | ClickHouse | Server-side LLM calls during add (chunk/extract) |
| `search_total_tokens` / `search_avg_tokens/query` / `search_avg_tokens/call` | ClickHouse | Search-stage token totals, aggregated per search request (grouped by `TraceId`, not per individual LLM call) — `SearchResult` never returns per-call token usage, so these numbers are always reconstructed after the run from ClickHouse OTel trace data |
| `overall_accuracy` | manifest | Benchmark score |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_total_tokens` | manifest | Eval-side answer stage tokens |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_total_tokens` | manifest | Eval-side judge stage tokens |
| `add_token_avg` / `answer_token_avg` / `judge_token_avg` | ClickHouse / manifest | Average tokens per add call / per question (answer) / per question (judge) |

### `eval_metrics` sheet

One row per benchmark. Key columns:

| Column | Source | Description |
|--------|--------|-------------|
| `overall_accuracy` / `correct` / `total` | manifest | Overall accuracy and question count |
| `search_llm_calls` / `search_prompt_tokens` / `search_completion_tokens` / `search_total_tokens` | manifest, falls back to the ClickHouse per-query aggregate when missing | Search-stage tokens (`vanilla`/`fast` search is 0 either way) |
| `answer_llm_calls` / `answer_prompt_tokens` / `answer_completion_tokens` / `answer_total_tokens` | manifest | Answer-stage tokens |
| `judge_llm_calls` / `judge_prompt_tokens` / `judge_completion_tokens` / `judge_total_tokens` | manifest | Judge-stage tokens |
| `build_elapsed_seconds` / `search_elapsed_seconds` / `answer_elapsed_seconds` / `total_elapsed_seconds` | manifest | Cumulative elapsed time per stage (seconds) |
| `by_question_type` / `by_topic` | manifest | Accuracy broken down by question type / topic |

### `llm_by_task` sheet

One row per LLM task type per run:

| Column | Source | Description |
|--------|--------|-------------|
| `memory.add.extract` | ClickHouse | Server-side LLM extraction calls during add |
| `eval.answer` | manifest | Eval-side answer generation calls |
| `eval.judge` | manifest | Eval-side judge scoring calls |

### `percentiles` sheet

Token and timing percentiles (min / max / p50 / p95, no average — averages live in the `summary` sheet) per operation:

| Column | Source | Description |
|--------|--------|-------------|
| `add` token percentiles | ClickHouse (per add call) | |
| `add` time percentiles | Qdrant (per request) | |
| `search` token percentiles | ClickHouse (per search *request* — all `search.*` LLM calls within one search, e.g. `multi_query`, `sufficiency_check` under agentic search, summed by `TraceId` before computing percentiles) | Reflects real per-query cost, not per-LLM-call cost |
| `search` time percentiles | Qdrant (per request) | |
| `eval.answer` token percentiles | manifest (per question) | True percentiles |
| `eval.answer` time percentiles | manifest (total elapsed only) | `time_p50_ms` = average, not a true percentile |
| `eval.judge` token percentiles | manifest (per question) | True percentiles |
| `eval.judge` time percentiles | none | `time_p50_ms` is empty |

> PersonaMem uses deterministic (multiple-choice) scoring — no answer/judge LLM calls are made, so no `eval.*` rows appear in percentiles.
> `vanilla`/`fast` search makes no LLM calls at all, so `search` token percentiles are empty for those runs; only `agentic`/`schema` search (which fans out to multiple LLM calls per query) populates them.
