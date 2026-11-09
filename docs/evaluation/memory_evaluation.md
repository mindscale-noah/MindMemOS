# MindMemOS Memory Evaluation

[简体中文](memory_evaluation_ZH.md) · [Skill evaluation](skill_evaluation.md)

This guide covers MindVanilla, MindSchema, Dreaming, and Feedback. Historical scores come from the repository README. Blank cells are reserved for results. Record smoke and regression results separately.

## Memory prerequisites

From the repository root, run `uv sync --all-packages` and `make init`. Use a dedicated evaluation service and stores. Set the server's `auth.api_key_file` to `config/mindmemos/eval_api_keys.yaml`, as used in the commands below. Configure the server-side models before starting it. The evaluator generates a project and key for each fresh run.

### 1. Start dependency services

```bash
make docker SERVICE=memory          # starts Qdrant / Neo4j / Kafka / ClickHouse / Grafana
```

> **Important**: `make clean` stops managed applications and deletes Compose volumes after confirmation. Start the services again before evaluation. Restart any manually launched server too, so it does not reuse stale database connections.

### 2. Start the MindMemOS server

```bash
uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
```

### 3. Configure LLM API keys

The **add** stage LLM calls are made by the **server** using the keys in `config/mindmemos/dev.yaml`.
The **answer** and **judge** stage LLM calls are made directly by the **eval process** using the keys in `config/mindmemos_eval/memory_evaluation/locomo.example.yaml` (or `config/mindmemos_eval/memory_evaluation/personamem.example.yaml`).
These two key sets are completely independent and must be configured separately.

```yaml
# config/mindmemos_eval/memory_evaluation/locomo.yaml
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

## Memory Dataset Download

Run the following commands from the repository root with Python 3.12+. Dataset payloads are not
included in this repository. The shared downloader retrieves official releases at the revisions
recorded in [`config/presets/dataset_download_sources.json`](../../config/presets/dataset_download_sources.json).

```bash
uv run --no-sync python scripts/download_datasets.py list memory
uv run --no-sync python scripts/download_datasets.py download memory --name locomo
uv run --no-sync python scripts/download_datasets.py download memory --name longmemeval
uv run --no-sync python scripts/download_datasets.py download memory --name personamem
uv run --no-sync python scripts/download_datasets.py download memory --name memoryagentbench
```

Downloads default to `data/mindmemos_memory/<dataset>/` and preserve the official file formats:

| Dataset | Official source | Files relative to its download directory |
| --- | --- | --- |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) | `locomo10.json` |
| LongMemEval | [longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json`, `longmemeval_oracle.json` |
| PersonaMem v1 | [PersonaMem-v1](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) | `questions_{32k,128k,1M}.csv` and matching `shared_contexts_{32k,128k,1M}.jsonl` |
| MemoryAgentBench | [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) | `data/*.parquet` for all four official partitions, plus `entity2id.json` |

In the local `config/mindmemos_eval/memory_evaluation/locomo.yaml` created by `make init`, configure model credentials and dataset paths.

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

For PersonaMem, use a question file and context file with the same context size; update the same
fields when using `config/mindmemos_eval/memory_evaluation/personamem.yaml`. LongMemEval downloads the official cleaned
release, not the local smoke fixture previously referenced by the example config.
MemoryAgentBench remains in official Parquet format: the current evaluator's local `dataset` input
accepts JSON/JSONL, so convert the required partition before setting that path. The downloader does
not perform this conversion or select a FactConsolidation subset.

Use `--data-root` to change the parent directory, `--verify-only` to check an existing download,
and `--offline` to materialize Hugging Face datasets from an existing cache:

```bash
uv run --no-sync python scripts/download_datasets.py download memory --name locomo --data-root /path/to/memory
uv run --no-sync python scripts/download_datasets.py download memory --name locomo --verify-only
uv run --no-sync python scripts/download_datasets.py download memory --name personamem --offline
```

The custom-root example writes to `/path/to/memory/locomo/`; use the same `--data-root` when verifying
that download. LoCoMo's initial direct-URL download requires network access. Existing directories are
verified rather than overwritten. Each download records its source and local hashes in
`download_receipt.json`. Experiment split manifests are maintained with the code for GitHub; the
script does not download or generate them.

---

## MindVanilla and MindSchema

Both profiles share the datasets and scorers. MindVanilla uses Vanilla ingestion and retrieval with `fast` search in the example matrix. MindSchema uses Schema ingestion and retrieval with `agentic` search in that matrix. For a profile comparison, keep each profile's settings and record the ingestion, answer, and judge models, retrieval settings, and dataset release. To isolate the ingestion algorithm, fix the retrieval settings in a separate comparison.

Follow [dataset download](#memory-dataset-download) to prepare the data and local `locomo.yaml`. Run the commands sequentially. Change `profile` and repeat with a fresh evaluation project for each arm.

### MindVanilla

Set `profile` to `vanilla` and run the benchmark commands below. Use the MindVanilla rows in the comparison tables.

```bash
profile=vanilla
```

### MindSchema

Set `profile` to `schema` and rerun the same datasets. Use the MindSchema rows in the comparison tables.

```bash
profile=schema
```

### LoCoMo

Download `locomo` and set `benchmarks.locomo.dataset` to `locomo10.json`. Ingest timestamped dialogue by session, then retrieve memories and generate an answer for each question. An LLM judge compares the answer with the reference. The runner skips category 5 by default and reports overall and per-category accuracy.

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list locomo \
  --algorithm "$profile" \
  --manifest-output "reports/locomo_$profile.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

**Historical comparison (%).**

| Method                       | Single-hop | Multi-hop | Temporal | Open-domain | Overall |
| :--------------------------- | :--------: | :-------: | :------: | :---------: | :-----: |
| Mem0                         | 68.97 | 61.70 | 58.26 | 50.00  | 64.20 |
| MemU                         | 74.91 | 72.34 | 43.61 | 54.17  | 66.67 |
| MemOS                        | 85.37 | 79.43 | 75.08 | 64.58  | 80.76 |
| Zep                          | 90.84 | 81.91 | 77.26 | 75.00  | 85.22 |
| EverOS                       | 96.67 | 91.84 | 89.72 | 76.04  | 93.05 |
| **MindMemOS-MindVanilla**    | 92.03 | 85.82 | 83.80 | 66.67  | 87.60 |
| **MindMemOS-MindSchema**     | **96.79** | **93.97** | **90.34** | **82.29** | **94.03** |

### LongMemEval

Download `longmemeval` and set `benchmarks.longmemeval.dataset` to the chosen release. This example uses `longmemeval_s_cleaned.json`. Each question has its own ingested history. After retrieval and answer generation, an LLM applies question-type-specific judging. The report includes overall, task-averaged, and abstention accuracy.

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list longmemeval \
  --algorithm "$profile" \
  --manifest-output "reports/longmemeval_$profile.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

Add `--limit 2 --session-limit 2` to smoke-test the workflow before running the full release.

| Release | Metric | MindVanilla | MindSchema | Reference method and score |
| --- | --- | ---: | ---: | --- |
| LongMemEval-S | Overall accuracy | | | |

### PersonaMem

Download `personamem` and configure the question CSV and context JSONL for the same context length in `locomo.yaml`. You can also update the same fields in the local `personamem.yaml` created by `make init` and use it in the command. The evaluator builds and retrieves memories from the conversation prefix for each question. It matches the model's choice against the gold option deterministically and aggregates accuracy by question type.

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
  --benchmark-list personamem \
  --algorithm "$profile" \
  --manifest-output "reports/personamem_$profile.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

**Historical comparison (%).** Match the historical context length before comparing a new run.

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

Download `memoryagentbench` and convert the Parquet records with the target `metadata.source` to JSONL. This example selects the single-hop 6k subset. Keep the derived file outside the download directory so that the original download still passes verification.

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

Configure model access in the local `memoryagentbench.yaml` created by `make init`. See the [MemoryAgentBench example](../../config/mindmemos_eval/dreaming_evaluation/memoryagentbench.example.yaml). Set `benchmarks.memoryagentbench.dataset` to `data/mindmemos_memory/prepared/factconsolidation_sh_6k.jsonl` and check `sub_dataset` and `chunk_size`.

```bash
uv run --no-sync mindmemos-eval memory \
  --benchmark-config config/mindmemos_eval/dreaming_evaluation/memoryagentbench.yaml \
  --benchmark-list memoryagentbench \
  --algorithm vanilla \
  --manifest-output "reports/memoryagentbench_vanilla.jsonl" \
  --api-key-output config/mindmemos/eval_api_keys.yaml
```

The evaluator ingests context in chunks, retrieves memories, and generates answers. FactConsolidation uses normalized Substring Exact Match against any accepted reference. The command runs the Vanilla control. The historical Dreaming experiment consolidates memories between ingestion and question answering. The current matrix CLI has no consolidation stage. The repository fixtures exercise merge and archive behavior.

```bash
uv run --no-sync pytest -q src/mindmemos/tests/pipelines/dreaming
```

**Historical comparison.** SH and MH denote single-hop and multi-hop. Scores average Substring Exact Match across four context sizes. Archived columns show the archive proportion.

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

**Data.** Repository fixtures supply synthetic conversations, feedback, and memory records. Explicit feedback tests cover planning and applying memory changes from context. Implicit feedback tests cover session grouping, signal detection, and filtering previously processed records.

```bash
uv run --no-sync pytest -q src/mindmemos/tests/pipelines/feedback
```

| Dataset | Metric | Feedback | Reference method and score |
| --- | --- | ---: | --- |
| | | | |

---

## Smoke Test (recommended first)

Use `--limit 2` to minimize data volume and verify the full pipeline end-to-end:

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

> **Warning**: ``--api-key-output`` writes a **fresh** ``api_keys.yaml`` with only the generated
> benchmark identities. Do NOT point it at the server's live key file
> (``config/mindmemos/api_keys.yaml``) unless you are running in an isolated environment.
> Use a separate path (e.g. ``config/mindmemos/eval_api_keys.yaml``) and point the server at
> that file for the duration of the evaluation, or merge the generated keys into the server's
> key file manually.

---

## Full Evaluation

### Run LoCoMo, LongMemEval, and PersonaMem

All three benchmarks are executed sequentially in a single command to avoid `api_keys.yaml` conflicts:

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
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
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation/locomo.yaml \
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
