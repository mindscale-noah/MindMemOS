# MindMemOS Evaluation Guide

This guide explains how to run the three standard benchmarks (LoCoMo, LongMemEval, PersonaMem) against MindMemOS, and how to use `metrics.py` to collect timing and token statistics.

---

## Prerequisites

### 1. Start dependency services

```bash
make db          # starts Qdrant / Neo4j / Kafka / ClickHouse / Grafana / MinIO
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

## Multimodal: MemEye / Mem-Gallery (`vanilla`, multimodal mode off / basic / enhanced)

Both benchmarks are long-term memory tests with images interleaved in multi-session dialogues,
evaluating the multimodal side of `vanilla`. The flow:

1. One `memory.add` call per session.
2. One `memory.search` call per question; a question's own images are passed as `query_images`.
3. The text memories and images returned by search, plus the question images, go to one shared
   vision-capable answer model (`answer_llm`); the dataset's own captions are never sent to the
   memory system.
4. A `judge_llm` scores the answers with the benchmark's original rubric.

| | MemEye | Mem-Gallery |
|---|---|---|
| Size | 8 tasks, MCQ + OPEN, 742 questions | 20 tasks, 1711 OPEN questions |
| Data | Hugging Face `MemEyeBench/MemEye` (~260 MB) | Hugging Face `Ethan-Bei/Mem-Gallery` (~530 MB) |
| Metrics | EM (MCQ, option-rotation debiased), B-1, Judge (MemEye rubric), macro mean over tasks | F1, B-1, Judge (original rubric), micro mean over questions |

Each benchmark runs four configurations. **The mode is set per evaluation project through
`project_override_config`; the server's global `mode` stays at its default `"off"`** (quote `"off"`
in YAML — a bare `off` parses as a boolean):

| Config | `algo_config.multimodal.mode` | Search strategy | Notes |
|---|---|---|---|
| `b0` | `off` | fast | Text-only vanilla, no images sent |
| `b1` | `basic` | fast | Store originals, retrieve by image similarity |
| `enhanced` | `enhanced` | fast | VLM extraction + visual facts + evidence images |
| `enhanced_agentic` | `enhanced` | agentic | Multimodal agent retrieval |

The multimodal modes themselves are documented in the
[`algo_config.multimodal` section](../config/README.md) of the config reference.

### 1. Prerequisites

- **Machine:** 16 GB+ RAM, 10 GB free disk (datasets, stored originals, models). GPU optional —
  the local SigLIP2 embedding server runs on CUDA, Apple Silicon (MPS) or CPU (slower).
- **Software:** Docker (Qdrant, Neo4j, Kafka, …), [uv](https://docs.astral.sh/uv/), Python 3.13;
  see the [main README](../../README.md) for installation.
- **An OpenAI-compatible LLM service providing:**
  - a vision-capable chat model, e.g. `gpt-4.1-mini` — used by the server's enhanced extraction
    and agent, and by the evaluation's answerer and judge;
  - a text embedding model (vanilla needs it anyway);
  - a reranker (recommended; without it visual facts are not reranked).
- **A multimodal embedding service** mapping text and images into one vector space. Use the
  bundled local SigLIP2 server (step 3) or your own OpenAI-protocol-compatible service.

### 2. Download the datasets

```bash
uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download as get
get('MemEyeBench/MemEye', repo_type='dataset', local_dir='datasets/memeye')
get('Ethan-Bei/Mem-Gallery', repo_type='dataset', local_dir='datasets/memgallery')
"
```

The layout after the download — each evaluation config's `dataset` field points at its `data/`:

```
datasets/
├── memeye/data/        dialog/*.json   image/<task>/...
└── memgallery/data/    dialog/*.json   image/<task>/...
```

- `datasets/` is gitignored.
- If the data lives elsewhere, symlink `datasets/memeye` / `datasets/memgallery` to the real
  location.
- Behind a restricted network, set `HF_ENDPOINT` to a Hugging Face mirror.

### 3. Start the local multimodal embedding server (SigLIP2)

```bash
uv run --no-project --with torch --with transformers --with pillow \
  python scripts/mm_embed_server.py --port 8090
```

- The first start downloads `google/siglip2-base-patch16-384` (~1.5 GB) from Hugging Face.
- Device auto-selection: CUDA → MPS → CPU; or pass `--device cpu`.
- Check the service:

  ```bash
  curl localhost:8090/health
  # {"status": "ok", "model": "siglip2-base-patch16-384", "dimensions": 768}
  ```

This script is for development and evaluation only — torch is not a project dependency. Bring
your own multimodal embedding API in production.

### 4. Server configuration

```bash
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml
```

In `config/mindmemos/dev.yaml`:

1. Fill in `chat_model_router`, `embed_model_router` and `rerank_model_router` as in the
   [main README](../../README.md). The chat model must accept image input.
2. Configure the multimodal embedding:

   ```yaml
   mm_embed_model_router:
     endpoints:
       - model: siglip2-base-patch16-384
         api_base: http://127.0.0.1:8090/v1
         dimensions: 768
   ```

3. Allow the server to read dataset images by path. The evaluation sends images as file paths, so
   this must be the **absolute** path after resolving symlinks:

   ```yaml
   blob_store:
     local_root: data/blobs                 # where originals are stored
     allowed_path_roots:
       - /absolute/path/to/MindMemOS/datasets
   ```

4. Point the server at the evaluation's key file (written by the runner, auto-reloaded by the
   server):

   ```yaml
   auth:
     api_key_file: eval_api_keys.yaml
   ```

5. Leave `algo_config.multimodal.mode` at `"off"` (evaluations enable it per project via
   `project_override_config`).

Start the server:

```bash
make dev   # Docker dependencies + API at http://127.0.0.1:8000
```

The server does not hot-reload config or code — restart it after editing `dev.yaml`.

### 5. Evaluation configuration

```bash
cp config/mindmemos_eval/memory_evaluation_multimodal.example.yaml config/mindmemos_eval/memory_evaluation_multimodal.yaml
cp config/mindmemos_eval/memory_evaluation_memgallery.example.yaml config/mindmemos_eval/memory_evaluation_memgallery.yaml
```

Fill `runner.llm`, `runner.answer_llm` and `runner.judge_llm` (`model`, `base_url`, `api_key`) in
both files. The answer model must be vision-capable; all three can be the same model. Keep the
other fields at their defaults — the runner script generates the mode, search strategy and
image-sending flags.

Optional (`execution_params.<benchmark>`): `max_answer_images` (images attached to the answerer,
default 8) and `image_max_side` (longest side before attaching, default 1024).

### 6. One-command runs (recommended)

Smoke test first: one task per benchmark, 2 questions per stage, all four configurations — about
45 minutes in total (each configuration still ingests one full task's dialogues):

```bash
uv run python scripts/run_multimodal_bench.py --smoke
```

Then the real runs:

```bash
uv run python scripts/run_multimodal_bench.py                 # both benchmarks × four configs, 1 run each
uv run python scripts/run_multimodal_bench.py --runs 2        # 2 runs each (recommended for comparisons)
uv run python scripts/run_multimodal_bench.py --benchmarks memgallery --configs enhanced enhanced_agentic
```

What the script does:

- **Pre-flight checks:** server reachable, dataset directories present, `allowed_path_roots`
  covering the datasets, multimodal embedding configured, `auth.api_key_file` matching the
  evaluation key file. It exits with a message on any problem.
- **Strictly serial.** The moment a run has a failed answer or judge call (invalid result) it
  stops, preserving the remaining budget.
- **Resumable.** Re-running the same command skips runs that already have a valid result.
- **Output:** per-run manifests (`<bench>_<config>_r<k>.jsonl`), logs and generated configs plus
  a `summary.md` under `reports/multimodal/<tag>/` (`<tag>` defaults to the date, `smoke` for
  smoke tests); per-question details under `runs/mindmemos_eval/<run_id>/<benchmark>/`.

**Time:** one pass of both benchmarks × four configurations is about 7 hours (single run,
gpt-4.1-mini; wall time depends heavily on LLM latency, `enhanced_agentic` being the slowest).

### 7. Running one configuration by hand

MemEye (8 tasks, MCQ + OPEN, images in the dialogue) runs with one shared answerer: the
`answer_llm` (a vision model) sees the text memories and images returned by search, plus the
question's own images. The algorithm is always `vanilla`; the profile's `project_override_config`
sets `algo_config.multimodal.mode`, and `search_params.multimodal: true` makes every search use the
stored images. With `execution_params.memeye.send_images: true` the dialogue and question images
are sent by path, so the server's `blob_store.allowed_path_roots` must cover the dataset; mode
`off` rejects images, so its baseline sets `send_images: false`. Dataset captions are never sent.
The config loader rejects unknown `search_params` / `execution_params` keys at load time (the
known keys per benchmark live in `src/mindmemos_eval/mindmemos_eval/memory/config.py`), and
`send_images: true` against an off-mode project fails the same load-time consistency check —
the conflict surfaces before any request is made, not as a mid-run `add.images_unsupported`.

1. Download `MemEyeBench/MemEye` from Hugging Face and link it so `datasets/memeye/data` holds
   `dialog/` and `image/`.
2. For modes `basic` and `enhanced`, configure `mm_embed_model_router` on the server (see
   `docs/config/README.md`).
3. Copy `config/mindmemos_eval/memory_evaluation_multimodal.example.yaml` to
   `memory_evaluation_multimodal.yaml`, fill in the LLM keys and set the mode (the example uses
   `enhanced`; for B0 use `"off"` with `send_images: false`, for B1 `basic`; add
   `search_strategy: agentic` for the multimodal agent), then run:

```bash
uv run python -m mindmemos_eval.cli memory \
  --benchmark-config config/mindmemos_eval/memory_evaluation_multimodal.yaml \
  --benchmark-list memeye --algorithm vanilla \
  --manifest-output reports/memeye_enhanced.jsonl \
  --api-key-output config/mindmemos/eval_api_keys.yaml \
  --server-config config/mindmemos/dev.yaml
```

`--server-config` lets the project reset find the server's collections and blob root (local blob
backend only), so stored images are deleted with the project; with `blob_store.backend: s3` the
reset still clears the collections but leaves the bucket objects untouched.

`eval_result.metrics` reports, in percent: `EM` (MCQ exact match averaged over the four option
rotations, macro mean over tasks), `B-1` (OPEN BLEU-1) and `Judge` (OPEN, MemEye 0–1 judge rubric),
plus per-task values. `eval_result.cost` holds client-side add/search call counts and latencies and
answerer/judge token usage; `eval_result.valid` is false when any answer or judge call failed.
Per-question rows (retrieved items, predictions, scores) are written under
`output_root/<run_id>/memeye/`. For a smoke test set `execution_params.memeye.max_questions: 2` and
`benchmarks.memeye.tasks: [Brand_Memory_Test]`.

Multimodal runs follow the same runner protocol as the other benchmarks: search defaults (`top_k`,
`rerank`, `search_strategy`) come from the runner config / CLI (`--top-k`, `--rerank`,
`--search-strategy`), with benchmark/profile `search_params` still winning over them;
`--max-search-concurrency` caps the search calls. `--judge-runs N` runs the OPEN judge N times per
question and averages the scores. `metrics.overall_accuracy` follows the locomo convention (0–1)
using the headline metric (`EM` for MemEye, `F1` for Mem-Gallery), and the answer/judge token
fields use locomo's names, so the metrics collection below also works on multimodal manifests.

### 8. Reading the results

The `eval_result` in each manifest holds:

- `metrics`: scores in percent, per-task (`per_task`) scores, and per-question-type
  (`per_point`) scores for Mem-Gallery;
- `cost`: client-side add/search call counts and latencies, answerer and judge tokens;
- `valid`: false as soon as any answer or judge call failed — such a result is unusable.

### 9. Reference results

Setup: `vanilla` algorithm; every row answers with `gpt-4.1-mini` (the answerer must accept
images); image embeddings local SigLIP2 (`siglip2-base-patch16-384`). The MindMemOS rows are this
repository's implementation (answerer & judge both `gpt-4.1-mini`, mean of 2 runs, measured
2026-10); the baseline rows are the paper's reproductions under identical conditions with the same
answerer (mean of 3 runs; source: a paper under double-blind review, not linked yet). Judge scores
are not shared across the two groups — the judge model differs (`gpt-4.1-mini` here vs
`GPT-5.4-mini` there); MindMemOS Judge, b0 → enhanced_agentic: 26.26 / 47.80 / 50.80 / 53.30 on
MemEye and 70.68 / 85.06 / 87.06 / 89.14 on Mem-Gallery.

| Method | MemEye EM | MemEye B-1 | Mem-Gallery F1 | Mem-Gallery B-1 |
|---|---|---|---|---|
| MemoryOS | 45.75 | 13.12 | 43.09 | 37.76 |
| A-MEM | 49.73 | 21.11 | 44.08 | 38.37 |
| SimpleMem | 50.34 | 9.42 | 34.73 | 28.90 |
| HyperMem | 47.82 | 19.64 | 44.57 | 33.42 |
| M2A | 52.31 | 6.31 | 15.04 | 10.28 |
| MMA | 50.96 | 16.32 | 26.99 | 20.60 |
| SRAG (multimodal semantic RAG) | 58.36 | 31.48 | 41.07 | 35.03 |
| UniversalRAG | 60.49 | 11.23 | 48.36 | 41.90 |
| MindMemOS b0 (mode off, text-only) | 43.03 | 20.32 | 38.72 | 32.59 |
| MindMemOS b1 (basic) | 59.77 | 27.02 | 54.73 | 47.75 |
| MindMemOS enhanced | 63.45 | 25.49 | 55.66 | 48.98 |
| MindMemOS enhanced_agentic | **65.89** | 27.09 | **57.06** | **50.45** |

**When comparing results:**

- **MemEye is noisy:** few OPEN questions, so a single-run Judge moves 2–3 points (10–20 on a
  single task), and it drifts with the LLM service over time — the same code rerun two days apart
  dropped enhanced's Judge from 52.8 to 50.2 and agentic's EM from 67.8 to 65.5. Run each
  configuration at least twice and rerun the baseline in the same period.
- **Mem-Gallery is stable:** single-run Judge moves less than 0.6.
- **Never run two evaluations concurrently:** sharing one LLM service under concurrent load costs
  MemEye about 5 Judge points. The numbers above are all serial runs.
- **VR counting questions:** the Mem-Gallery answerer labels image results, facts and visual
  facts with image_ids (the VS type answers with image_ids). In fast retrieval the answerer
  sometimes counts repeated ids, over-counting on VR.

### 10. Approaches that were tried and did not work

| Approach | Result | Why |
|---|---|---|
| Image label in front of each visual fact, agent keeps only 5 unselected facts | Fast Judge −2.8, agentic EM −1.3 | The reranker matches the label, concentrating visual facts on a few images |
| Describe the query images with a VLM first, then append to the query | +0.2–0.4 on image questions only | The answerer already sees the question images, and image-to-image search already finds the stored counterpart — little headroom |
| Image name (what the dialogue calls the image) in front of visual facts | MemEye agentic EM −2.1; Mem-Gallery fast slightly down, agentic +0.4 | Names are often relative ("the same room from the other side"), again concentrating facts |

### 11. FAQ

| Symptom | Cause and fix |
|---|---|
| `add.images_unsupported` / `search.images_unsupported` | The project's `mode` is `off`. b0 never sends images; for other configs it means `project_override_config` did not apply — check that you use the script-generated config |
| `image.path_denied` / `image.path_disabled` | `blob_store.allowed_path_roots` does not cover the dataset's real path (mind symlinks) or is empty |
| Smoke test stuck in add, or slow | enhanced ingestion calls a VLM; tens of seconds per add is normal. Check the embedding and LLM service latency |
| Result `valid: false` | An answer or judge call failed — usually LLM throttling or instability. Rerun the same command once the service recovers; only the failed run repeats |
| Dimension mismatch after switching the multimodal embedding model | The image collection is built per `dimensions`. Delete the `image_item_v1` collection in Qdrant (or use a new `database.qdrant.image_collection` name) and restart |
| Disk usage growth | Every evaluation uses a fresh project id; originals land in `data/blobs/<project>/`, records stay in Qdrant and Neo4j. Delete old projects' blob directories in dev; `make db-clean` wipes all database volumes — careful |
| `multimodal_agentic_failed` warning | One of the agent's LLM calls failed; that search falls back to fast-retrieval results. Run validity is unaffected; occasional occurrences are normal |

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
| `--judge-runs N` | Run judge N times per question, decide by majority vote; default 1. The N runs execute sequentially per question, so judge-stage latency scales roughly linearly with N. Not applicable to PersonaMem (deterministic scoring). MemEye / Mem-Gallery judge scores are continuous — the N scores are averaged instead |
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
