# MindMemOS Evaluation Guide

This guide explains how to run the three standard benchmarks (LoCoMo, LongMemEval, PersonaMem) against MindMemOS, and how to use `metrics.py` to collect timing and token statistics.

---

## Skill evolution

Run `mindmemos-eval skill evolve --config config/mindmemos_eval/skill_evolve/dev.yaml --check`
to validate inputs, then omit `--check` to collect initial-Skill runs, evolve, export complete
published members, and retest the same cases. See `dev.example.yaml` and the Chinese guide
for configuration. The default connection uses isolated SQLite; SDK connections are optional.
`concurrency` bounds concurrent cases during collection and retesting; repeats within each
case stay sequential. All collection finishes before sequential evolution/export, and all
evolution finishes before retesting. Evolution itself is never parallelized.
Evolution retries cover registration, reporting, and evolution, resuming after saved steps.
Execution retries apply only to network/API/authentication faults; retryable HTTP statuses
are 401/403/408/429/5xx, so other 4xx request errors are not retried.
Only collection and test execution tokens are counted. This matched-task protocol measures
feedback-driven improvement, not held-out generalization. Use a user systemd unit for long runs.

---

## TreeSkill: SpreadsheetBench text_action

`mindmemos-eval skill treeskill` registers a Skill through the public HTTP SDK and submits
native TreeSkill training plus held-out testing to the Skill service. It shares the UI's
`.mindmemos` SDK configuration and project credentials. Each registration is named
`xlsx-mindmemos-eval-YYYYMMDD-HHMMSS-ffffff`. No baseline or automatic comparison is added.

Use `config/mindmemos_eval/treeskill/dev.yaml` (with its matching example). The service
profile is `treeskill_mindmemos_eval` in `config/mindmemos_skill/dev.yaml`. Restart the
Skill service and UI backend after installing this change so the running processes load
the new profile and reporting code. Client and service must share the same absolute data
and output paths.

```bash
# Validate without model calls or registration.
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action --check

# Run native training and testing.
uv run --no-sync mindmemos-eval skill treeskill --env spreadsheetbench_text_action
```

`--env` accepts any registered environment; lacking a reference score does not block execution.
If omitted, it uses `environment.class_name` in the configuration (default: `spreadsheetbench_text_action`).
For other environments, provide native `environment.config`, either `data` or `samples_file`,
a train/val/test `split_file`, and `initial_skill`. They do not require `reference_run` or `data_root`.
For example: `mindmemos-eval skill treeskill --env livemath --config /path/to/math.yaml`.
Only `spreadsheetbench_text_action` uses the historical alignment below. Other environments
still train, test, and publish results to the UI, without a score guarantee.

The reference is `output/treeskill_full_20261012_092954`: 122/200 held-out successes (61%).
Validation checks the exact 200/200 split and order, full initial Skill and resources,
task/scoring metadata, workbook content, environment settings, and trainer parameters.
The historical optimizer model was redacted in the saved configuration;
`reference_optimizer_model: qwen3.5-35b-a3b` records the experiment owner's confirmation.
Execution and router models are verified against recorded requests and the final Skill.
The initial name receives the requested suffix; the body and resources stay intact.

The protocol uses `spreadsheetbench_text_action`, the verified 400 dataset, no validation
set, `qwen3.5-35b-a3b` for all model roles, 100 turns, transactional recalculation enabled,
rollout/analysis/localization/test concurrency 32, one sample per task, and seed 41.
Other sampling and timeout settings are checked against the reference trainer.

Outputs are under `output/mindmemos_eval/treeskill/<timestamp>/`, or an empty `--output-dir`:

- `registration.json` and `task.json`: registration, asynchronous task state, and publication.
- `summary.json` and `report.md`: test counts, test time, tokens, execution errors, success rate.
- `trajectories.jsonl`: native outcomes or explicit failed/unstarted records for test tasks.
- `runtime.json`: full pipeline wall time, separately from test timing.
- `training/train/`: native training statistics, trajectories, and the final dynamic Skill.
- `training/test/`: original test artifacts, service JSONL, and `report.json`.

Zero reward is a completed outcome, not an execution error. Success rate divides successes
by completed tasks; unstarted tasks are separate. Missing timing or token totals remain
null, with known partial usage reported separately. Failed service tasks return a failing
exit code and preserve available reports. After publication, UI → Skills → registration
name → final version → Content displays Test evaluation. Existing versions are not
backfilled, and the historical 61% is not a guaranteed score for a new run.


## Test a registered environment directly

`mindmemos-eval skill test` runs exactly one mode: no-skill or a fixed static Skill collection.
All supplied Skills are injected together into each task. Run commands from the repository root.
Run `make init`, then edit `config/mindmemos_eval/env_test/dev.yaml` with the native environment,
model connection, and registered Data loader settings. The example selects LiveMath.

```bash
uv run --no-sync mindmemos-eval skill envs
uv run --no-sync mindmemos-eval skill test --no-skill --limit 10 --concurrency 2
uv run --no-sync mindmemos-eval skill test --skills /path/to/skill_a /path/to/skill_b --limit 10
```

`--no-skill` and `--skills` are mutually exclusive. Without either flag, the configured `skills`
list is used; an empty list means no-skill. Skill paths accept directories or `SKILL.md` files.
`--config`, `--env`, `--split-file`, `--split`, `--limit`, `--concurrency`, and `--output-dir`
override their respective settings. Relative paths are resolved from the working directory.
Each output directory must be empty.

Alternatively, `--samples samples.json` replaces the Data loader with an array of objects containing
`task` and `reward_data`, validated against the selected environment's native models.
Set `split_file: null` in the configuration to evaluate all input records.

Reports contain these six required categories:

| Category | Meaning |
| --- | --- |
| Data count | Planned, attempted, and normally completed tasks |
| Elapsed time | Wall time including setup and execution; individual durations remain in the task records |
| Token usage | Input, output, total; null when incomplete, with known sums in `reported_usage` and missing task count in `usage_missing` |
| Execution failures | API timeouts and execution/scoring exceptions; normal zero rewards are not execution failures |
| Success rate | Native `reward.success=true` count divided by normally completed tasks; null when none completed |
| Trajectories | One JSONL record per finished task, including failures |

Results are written under `output/mindmemos_eval/test/<run>/`: `summary.json`, `report.md`, and
`tasks/trajectories.jsonl`. JSONL is appended as tasks finish; `position` retains input order.
`result.trajectory` contains native interaction records, and `outcome` contains timing, usage,
score, success, and errors. Failed executions have `result: null`; unavailable partial trajectories
or token usage are not invented. Native task artifacts remain under `tasks/task-*/`.
Execution errors produce a nonzero exit code; ordinary zero rewards do not.

---

## Memory prerequisites

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
python -m pip install huggingface_hub
python scripts/download_datasets.py list memory
python scripts/download_datasets.py download memory --name locomo
python scripts/download_datasets.py download memory --name longmemeval
python scripts/download_datasets.py download memory --name personamem
python scripts/download_datasets.py download memory --name memoryagentbench
```

Downloads default to `data/mindmemos_memory/<dataset>/` and preserve the official file formats:

| Dataset | Official source | Files relative to its download directory |
| --- | --- | --- |
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) | `locomo10.json` |
| LongMemEval | [longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | `longmemeval_s_cleaned.json`, `longmemeval_m_cleaned.json`, `longmemeval_oracle.json` |
| PersonaMem v1 | [PersonaMem-v1](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v1) | `questions_{32k,128k,1M}.csv` and matching `shared_contexts_{32k,128k,1M}.jsonl` |
| MemoryAgentBench | [ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) | `data/*.parquet` for all four official partitions, plus `entity2id.json` |

Copy the evaluation config, then update the paths and LLM keys in your local copy:

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

For PersonaMem, use a question file and context file with the same context size; update the same
fields when using `config/mindmemos_eval/memory_evaluation/personamem.yaml`. LongMemEval downloads the official cleaned
release, not the local smoke fixture previously referenced by the example config.
MemoryAgentBench remains in official Parquet format: the current evaluator's local `dataset` input
accepts JSON/JSONL, so convert the required partition before setting that path. The downloader does
not perform this conversion or select a FactConsolidation subset.

Use `--data-root` to change the parent directory, `--verify-only` to check an existing download,
and `--offline` to materialize Hugging Face datasets from an existing cache:

```bash
python scripts/download_datasets.py download memory --name locomo --data-root /path/to/memory
python scripts/download_datasets.py download memory --name locomo --verify-only
python scripts/download_datasets.py download memory --name personamem --offline
```

The custom-root example writes to `/path/to/memory/locomo/`; use the same `--data-root` when verifying
that download. LoCoMo's initial direct-URL download requires network access. Existing directories are
verified rather than overwritten. Each download records its source and local hashes in
`download_receipt.json`. Experiment split manifests are maintained with the code for GitHub; the
script does not download or generate them.

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

### Run all benchmarks

All three benchmarks are executed sequentially in a single command to avoid `api_keys.yaml` conflicts:

```bash
caffeinate -si uv run python -m mindmemos_eval.cli memory \
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
caffeinate -si uv run python -m mindmemos_eval.cli memory \
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


## Skill-Forest evaluations

Skill evaluations create an isolated in-process service and use the shared settings in
`config/mindmemos_eval/dev.yaml`. Configure both `models.target` and `models.optimizer`
with explicit model, API key, and base URL values. No HTTP or Memory server is required.

```bash
uv sync --all-packages --extra alfworld --extra spreadsheetbench
make init
uv run --no-sync python scripts/download_datasets.py download skill --name livemath
uv run --no-sync mindmemos-eval skill run --smoke
uv run --no-sync mindmemos-eval skill run
```

The default evaluates LiveMath's full committed split with no Skill, the initial Skill,
and one frozen trained Skill. Smoke mode uses ordered 4/2/2 prefixes and one tree,
one candidate, and depth/budget one. Use `--arms no_skill,initial_skill` to skip training.

For a paper comparison, use `--test-repeats 3` and report the mean of all three tests
of the frozen Skill. The default single test is a different reporting protocol.
`skill_forest.validation_batch_size` defaults to 8; early rejection is checked after
each complete batch. `selection_tiebreak: novelty` follows both the paper and its standalone selector:
score each tree winner on the full training set, then break top-score ties by maximum
pairwise TF-IDF novelty relative to the initial Skill. `validation` retains the earlier
training entry point's validation-score, depth, and text-length ordering; it is not
the paper-alignment setting. `patch_application: exact` requires unique targets and an entirely valid patch;
`reference` uses the original SkillOpt line insertion and missing-target behavior.
Also verify the model, initial Skill text, and task inputs when reproducing a run.

Skill-Forest algorithm version 3 aligns tree partition and reflection seeds and uses
the archived phase-specific option order for LiveMath. Shared root validation and
tree-local candidate validation retain their respective reference seeds. Use a new
training directory; earlier algorithm checkpoints are not compatible with version 3.

```bash
uv run --no-sync python scripts/download_datasets.py download skill --all
uv run --no-sync mindmemos-eval skill run --benchmark all
uv run --no-sync mindmemos-eval skill run --benchmark spreadsheetbench --test-repeats 3
uv run --no-sync mindmemos-eval skill report output/mindmemos_eval/skill/<run-directory>
```

The five benchmarks are LiveMath, ALFWorld, SpreadsheetBench, SearchQA, and OfficeQA.
All selected inputs, receipts, and runtimes are checked before any training starts.
Runtime failures are recorded and other benchmarks continue; incomplete runs exit nonzero.
Each run requires an empty output directory and produces JSON, CSV, and Markdown reports.
The pipeline's verified first test is adopted; subsequent repeats use the exact same
published version in fresh workspaces. Missing usage is null, and infrastructure errors
are distinct from normal wrong answers. The report command makes no model calls.

SpreadsheetBench uses native tool calling with recalculation disabled by default.
OfficeQA uses the native `max_tool_turns` setting. SearchQA uses `lucadiliello/searchqa`
at revision `c1a979068ba118d85467179b704031d113d689cc`. The download script stores its
Parquet files under `data/mindmemos_skill/searchqa/data/`. The single `data_splits/splits.json`
uses native HF `key` values and preserves the original 400 / 200 / 1,400 split and order.
Evaluation copies selected `question`, `context`, and `answers` fields unchanged into the
run's `prepared/searchqa/items.json`, using `key` as task ID. No ID mapping resource is needed.

Reflection and merge results retain batch submission order, regardless of request completion order.
Generation protocol version 5 rejects checkpoints from earlier versions.


Optimizer responses always follow the reference protocol: preserve original JSON order and metadata, use reported batch sizes for support counts, and save requests and responses. Invalid model output uses the documented evidence fallback; transport errors propagate.
