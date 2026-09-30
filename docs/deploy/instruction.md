# MindMemOS Deployment & Configuration Guide

<p align="center">
  <strong><a href="instruction.md">English</a></strong>
  &nbsp;&nbsp;│&nbsp;&nbsp;
  <strong><a href="instruction_ZH.md">简体中文</a></strong>
</p>

## 1. Overview

MindMemOS uses a `uv workspace` to manage three core Python packages, organized into server, client, and evaluation layers:

```text
Applications / Agents ──────────────┐
Agent Plugins ───── CLI ─────────────┼──> mindmemos_sdk ── HTTP ──> mindmemos
mindmemos_eval ──────────────────────┘
```

- `mindmemos` is the core server-side algorithm package. It provides FastAPI endpoints, memory and Skill workflows, model calls, data persistence, and asynchronous tasks.
- `mindmemos_sdk` provides the Python SDK and CLI for applications and plugins. It calls `mindmemos` over HTTP without depending on server internals.
- `mindmemos_eval` is an independent evaluation package. It uses `mindmemos_sdk` to call the service, load datasets, run evaluations, and aggregate results.

The main runtime-related directories are:

```text
.
├── src/
│   ├── mindmemos/          # Core server package
│   ├── mindmemos_sdk/      # Python SDK and mindmemos CLI
│   └── mindmemos_eval/     # Benchmark evaluation tools
├── config/
│   ├── mindmemos/          # Server runtime and authentication configuration
│   ├── mindmemos_eval/     # Evaluation task configuration
│   └── presets/            # Algorithm preset resources
├── dockers/                # Qdrant, Neo4j, Kafka, and observability components
├── plugins/                # Agent plugin integrations
├── Makefile                # Entry points for local services and dependencies
└── pyproject.toml          # uv workspace and development dependencies
```

The primary configuration entry points are:

| Scope | Configuration File | Description |
| --- | --- | --- |
| `mindmemos` | `.env` | Docker dependencies, service ports, and connection addresses. |
| `mindmemos` | `config/mindmemos/dev.yaml` | Server models, databases, pipelines, and runtime configuration. |
| `mindmemos` | `config/mindmemos/api_keys.yaml` | API keys, `project_id`, memory algorithms, and access permissions. |
| `mindmemos` | `config/presets/*.json` | Memory algorithm presets. |
| `mindmemos_sdk` | `~/.mindmemos/settings.json` | SDK and CLI connection information and default user. |
| `mindmemos_eval` | `config/mindmemos_eval/*.yaml` | Evaluation models, datasets, concurrency, and algorithm configuration. |

## 2. Minimal Startup Flow

```bash
cp .env.example .env
cp config/mindmemos/dev.example.yaml config/mindmemos/dev.yaml

# Edit .env and config/mindmemos/dev.yaml, then start the service.
make dev-setup
make dev
```

Default addresses:

- FastAPI: `http://127.0.0.1:8000`
- API Docs: `http://127.0.0.1:8000/docs`
- Qdrant: `http://localhost:6333`
- Neo4j Browser: `http://localhost:7474`

`make dev` starts the full Docker dependency stack before starting FastAPI. To start only core dependencies:

```bash
make dev-core          # Qdrant + Neo4j + Kafka
make db-observability  # Qdrant + Neo4j + Kafka + ClickHouse + OTel + Grafana
```

Stop local dependencies:

```bash
make dev-down
```

## 3. Required Environment Variables

Configuration file selection:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_CONFIG_NAME` | Selects the configuration name; `dev` reads `config/mindmemos/dev.yaml`. | `dev` |
| `MINDMEMOS_CONFIG_PATH` | Specifies a configuration file path directly; takes precedence over `MINDMEMOS_CONFIG_NAME` when set. | Empty |

Qdrant:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_QDRANT_URL` | HTTP address used by FastAPI to access Qdrant. | `http://localhost:6333` |
| `MINDMEMOS_QDRANT_HTTP_PORT` | Qdrant HTTP port exposed by Docker. | `6333` |
| `MINDMEMOS_QDRANT_GRPC_PORT` | Qdrant gRPC port exposed by Docker; also overrides `database.qdrant.grpc_port` in the configuration. | `6334` |
| `MINDMEMOS_QDRANT_PREFER_GRPC` | Whether the Qdrant client prefers gRPC. | `false` |
| `MINDMEMOS_QDRANT_API_KEY` | Qdrant API key; can be left empty for an unauthenticated local setup. | Empty |
| `MINDMEMOS_GRAFANA_QDRANT_URL` | HTTP address used by the Grafana container to access Qdrant. | `http://qdrant:6333` |

Neo4j:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_NEO4J_URI` | Bolt address used by FastAPI to access Neo4j. | `bolt://localhost:7687` |
| `MINDMEMOS_NEO4J_HTTP_PORT` | Neo4j Browser port exposed by Docker. | `7474` |
| `MINDMEMOS_NEO4J_BOLT_PORT` | Neo4j Bolt port exposed by Docker. | `7687` |
| `MINDMEMOS_NEO4J_USERNAME` | Neo4j username; also used as the username in Docker `NEO4J_AUTH`. | `neo4j` |
| `MINDMEMOS_NEO4J_PASSWORD` | Neo4j password; also used as the password in Docker `NEO4J_AUTH`. | `mindmemos_dev_password` |

Optional dependencies:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_KAFKA_BOOTSTRAP_SERVERS` | Kafka address; the service starts consumers/producers only when `kafka.enabled=true` in the configuration. | `localhost:9092` |
| `MINDMEMOS_TELEMETRY_ENDPOINT` | OTel HTTP endpoint; telemetry is reported only when `telemetry.enabled=true` in the configuration. | `http://localhost:4318` |
| `MINDMEMOS_CLICKHOUSE_USER` / `MINDMEMOS_CLICKHOUSE_PASSWORD` / `MINDMEMOS_CLICKHOUSE_DB` | ClickHouse/Grafana observability data configuration. | See `.env.example` |

API bind address:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MINDMEMOS_API_HOST` | Host used by `make dev` / `make api` to start FastAPI. | `127.0.0.1` |
| `MINDMEMOS_API_PORT` | Port used by `make dev` / `make api` to start FastAPI. | `8000` |

## 4. Docker

Start local dependencies with:

```bash
docker compose --env-file .env -f dockers/docker-compose.memory.yml up -d --wait qdrant neo4j kafka kafka-ui kafka-exporter
```

`make dev-core` starts Qdrant, Neo4j, Kafka, Kafka UI, and kafka-exporter. `make dev` starts the full Docker dependency stack before starting FastAPI. `make db` remains available as a compatibility entry point for the full dependency tier and is equivalent to `make db-observability`.

Core services in Docker Compose:

- `qdrant`: stores memory/entity/source vectors and payloads.
- `neo4j`: stores graph relationships.
- `kafka`: asynchronous task queue; it can run even when it is disabled in the default configuration.
- `clickhouse` + `otel-collector` + `grafana`: observability stack; disable `telemetry.enabled` in the configuration when observability is not needed.

For local deployment, the port variables in `.env` must align with the connection addresses in `config/mindmemos/dev.yaml`. At startup, environment variables also override the following configuration fields:

- `database.qdrant.url`
- `database.qdrant.api_key`
- `database.qdrant.grpc_port`
- `database.qdrant.prefer_grpc`
- `database.neo4j.uri`
- `database.neo4j.username`
- `database.neo4j.password`
- `kafka.bootstrap_servers`
- `telemetry.telemetry_endpoint`

## 5. LLM Configuration

LLMs are used for memory extraction, schema processing, dreaming, and other generation tasks. Configure `chat_model_router`:

```yaml
chat_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/gpt-4.1-mini
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 1200
      temperature: 0.0
      num_retries: 3
      extra_body: {}
```

Notes:

- `model` uses LiteLLM-style model names. OpenAI-compatible endpoints usually use `openai/<model-name>`.
- `api_base` should include `/v1`, unless your provider explicitly documents a different format.
- Do not commit `api_key`; keep it in the uncommitted local `config/mindmemos/dev.yaml`.
- Multiple endpoints can be configured, and the router dispatches according to `routing_strategy`.

## 6. Embedding Configuration

Embedding must be configured. At startup, the service validates that the embedding output dimension matches the Qdrant vector dimension.

```yaml
embed_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/qwen3-embedding-4b
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 600
      num_retries: 3
      dimensions: 2560
      extra_body: {}

database:
  qdrant:
    vector_size: 2560
    semantic_vector_name: semantic
    bm25_vector_name: bm25
```

Key points:

- `database.qdrant.vector_size` must equal the actual output dimension of the embedding model.
- If the embedding model supports custom dimensions, `dimensions` and `vector_size` must also match.
- If a Qdrant collection has already been created with an old dimension, changing `vector_size` alone will not migrate it. For local development, run `make db-clean` to clear the volume and rebuild.

## 7. Rerank Configuration (Optional)

Rerank improves search precision by reranking retrieval candidates, but it is not required for service startup. Without an external rerank endpoint, basic add/search still works; the code uses existing recall results or fallback logic.

Configure an external reranker when needed:

```yaml
rerank_model_router:
  routing_strategy: simple-shuffle
  endpoints:
    - model: openai/qwen3-reranker-4b
      api_key: your-api-key
      api_base: https://your-base-url/v1
      timeout: 600
      num_retries: 3

algo_config:
  search:
    rerank:
      enabled: true
      max_query_length: 100
      max_doc_length: 5000
      max_batch_size: 20
      max_concurrent_batches: 1
      request_timeout: 5.0
    vanilla:
      use_reranker: true
    schema_search:
      entity:
        use_reranker: true
```

When not using an external reranker:

```yaml
rerank_model_router:
  routing_strategy: simple-shuffle
  endpoints: []

algo_config:
  search:
    rerank:
      enabled: false
    vanilla:
      use_reranker: false
    schema_search:
      entity:
        use_reranker: false
```

`rerank` is an optional enhancement. For production, stabilize Docker, LLM, and Embedding first, then add rerank.

## 8. Memory Algorithm Version (v1 / v2)

Schema memory extraction has two selectable flows via `algo_config.add.schema.version`:

- `v2` (default): rule-based graph fusion, fewer LLM calls per episode.
- `v1`: develop-compatible LLM-heavy flow, for baseline comparison and rollback.

Set it in the base config for a deployment-wide default (takes effect after a
restart), or in a project's API-key override config to pin one project (project
overrides win and apply to that project's next add request — no restart). Storage
is compatible in both directions: same collections and payload schema, v2 reads
and updates v1 data and vice versa, mixed histories are safe. Output carries a
`mem_extract_version` label (`schema_add` / `schema_add_v1`) so the producing flow
stays distinguishable. v1 remains available for baseline comparison and rollback
until v2 is validated on the LoCoMo/PersonaMem benchmarks.

## 9. Authentication Configuration

Local setup uses API keys by default:

```yaml
auth:
  mode: api_key
  api_key_file: api_keys.yaml
```

`api_key_file` is resolved relative to the configuration file directory, so by default it points to `config/mindmemos/api_keys.yaml`. The local example includes:

- `dev-api-key-001`: vanilla memory
- `dev-api-key-002`: schema memory

Use this header when calling APIs:

```text
Authorization: Bearer <api_key>
```

## 10. Minimal Checklist

Before starting, confirm at least the following:

- Qdrant and Neo4j ports in `.env` do not conflict with services already running locally.
- `config/mindmemos/dev.yaml` exists.
- `chat_model_router.endpoints[0].api_key` / `api_base` / `model` are valid.
- `embed_model_router.endpoints[0].api_key` / `api_base` / `model` are valid.
- `database.qdrant.vector_size` equals the embedding output dimension.
- If rerank is not needed, `rerank_model_router.endpoints` can remain empty and the related `use_reranker` flags should be disabled.
