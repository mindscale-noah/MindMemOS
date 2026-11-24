# MindMemOS Deployment Guide

English | [简体中文](instruction_ZH.md)

Run the Memory API, Skill API, and web workbench from source. The APIs and UI run in the foreground on the host. Docker Compose runs the storage, queue, and monitoring components selected by the launcher.

- [Start local services with the UI](#1-start-local-services-with-the-ui)
- [Deploy the UI](#2-deploy-the-ui) with local APIs or on its own against MindMemOS Cloud or existing APIs
- [Select and manage services](#3-select-and-manage-services) to run a single API, reduce dependencies, or start components separately

## 1. Start local services with the UI

Run these commands from the repository root. A full startup requires Git, Make, uv, Python 3.11–3.13, and a running Docker engine with Docker Compose. Building the UI requires Node.js and npm. The repository defaults to Python 3.13.

### 1.1 Install dependencies

```bash
uv sync --all-packages --extra postgres --extra kafka
export UV_NO_SYNC=1
```

The `postgres` extra supports the Skill example's PostgreSQL database. The `kafka` extra supports Skill's Kafka task queue. Set `UV_NO_SYNC=1` in each terminal that runs Make commands so they use the installed extras. Sync again after dependencies change. Some ALFWorld dependencies require native build tools. On macOS ARM64, allow the TextWorld installer to mount and unmount the Inform7 disk image.

### 1.2 Initialize local configuration

```bash
test -f .env || cp .env.example .env
# Edit .env before initialization to share model settings.
make init
```

`make init` creates missing local files from the examples. It copies nonempty model settings from `.env` into matching fields already present in YAML under `config/`, including service, training, and evaluation configs. Empty variables keep the YAML values. Fill missing fields and provider-specific settings in the relevant YAML. Running `make init` again reapplies the nonempty settings.

| Model | API key | URL | Model name | Output dimensions |
| --- | --- | --- | --- | --- |
| Chat | `MINDMEMOS_CHAT_API_KEY` | `MINDMEMOS_CHAT_BASE_URL` | `MINDMEMOS_CHAT_MODEL` | — |
| Embedding | `MINDMEMOS_EMBEDDING_API_KEY` | `MINDMEMOS_EMBEDDING_BASE_URL` | `MINDMEMOS_EMBEDDING_MODEL` | `MINDMEMOS_EMBEDDING_DIMENSIONS` |
| Rerank | `MINDMEMOS_RERANK_API_KEY` | `MINDMEMOS_RERANK_BASE_URL` | `MINDMEMOS_RERANK_MODEL` | — |

If you set any Embedding variable, supply a positive integer output dimension. Initialization updates matching Embedding dimensions and Qdrant's `vector_size`. Use the full endpoint required by your Rerank provider.

Check the files for the services you will run.

- In `config/mindmemos/dev.yaml`, complete the model connections you use. Match the actual Embedding output dimensions to the model configuration and `database.qdrant.vector_size`.
- In `config/mindmemos_skill/dev.yaml`, check storage and fill in model connections and credentials for the graph management, training, or evolution pipelines you use.
- In `config/presets/local.project.yaml`, replace the placeholders in `projects.<name>.api_key` with your own service API keys. Keep existing project IDs to retain access to their data. To read and write through both APIs in the UI, select a project with `memory:read`, `memory:write`, `skill:read`, and `skill:write`.

The APIs use model provider keys to call models. Clients use project API keys to call MindMemOS. See the [Docker configuration guide](../../dockers/README.md) for database connections and ports, and [task environments and data](#task-environments-and-data) for training and evaluation prerequisites.

### 1.3 Build and start

```bash
uv run --no-sync python scripts/install_nlp_assets.py
uv run --no-sync python scripts/check_nlp_assets.py
make ui-build
make dev PLAN=1
make dev
```

The NLP assets are for Memory. Skip the two NLP commands when running only Skill. `make ui-build` writes the frontend bundle to the UI Python package's `static/` directory. The launcher uses that directory as-is. Rebuild before launching when assets are missing or the frontend has changed.

`PLAN=1` validates configuration and lists application addresses and selected containers. On actual startup, wait for `Development services ready`, then open the [web workbench](http://127.0.0.1:8765).

| Service | Default entry point | Health check |
| --- | --- | --- |
| Memory | [API docs](http://127.0.0.1:8000/docs) | [`/healthz`](http://127.0.0.1:8000/healthz) |
| Skill | [API docs](http://127.0.0.1:8001/docs) | [`/health`](http://127.0.0.1:8001/health) |
| UI | [Workbench](http://127.0.0.1:8765) | Open the page and load **Settings** |
| Kafka UI | [Messages and consumers](http://127.0.0.1:8080) | — |
| Grafana | [Dashboards](http://127.0.0.1:3000) | — |

By default, `make dev` enables both APIs, the UI, Kafka tasks, and OTLP monitoring. Memory uses Qdrant and Neo4j. The Skill example uses a separate PostgreSQL database. The launcher selects local containers from the resolved connection settings.

### 1.4 Multimodal Memory

In `config/mindmemos/dev.yaml`, set `algo_config.multimodal.mode` to `basic` or `enhanced`. The default `"off"` keeps text-only mode. Both image modes require an S3-compatible `blob_store`. Configure `mm_embed_model_router` for image similarity search and a vision-capable chat model for enhanced extraction. See the [multimodal evaluation guide](../evaluation/memory_evaluation.md#multimodal-memory-evaluation) for configuration examples.

`make dev SERVICE=memory UI=0` starts MinIO and initializes the bucket when multimodal mode is enabled globally and the storage endpoint is local. The default port is 9100. Remote storage uses the configured S3 endpoint, bucket, and credentials. To start local MinIO for a project-specific override, run:

```bash
docker compose --env-file .env -f dockers/compose.yaml up -d --wait minio
docker compose --env-file .env -f dockers/compose.yaml run --rm minio-init
```

Install `mindmemos[s3]` for production. Set `MINDMEMOS_S3_ENDPOINT`, `MINDMEMOS_S3_BUCKET`, `MINDMEMOS_S3_ACCESS_KEY`, and `MINDMEMOS_S3_SECRET_KEY` for the connection. Set `MINDMEMOS_S3_PRESIGN_ENDPOINT` if clients need a different address from the internal storage endpoint. For signed proxy URLs, forward `GET /v1/memory/images/{blob_id}` with the original query parameters through the gateway.

For rolling upgrades, keep multimodal mode `"off"` while updating the Memory API and Kafka workers. Keep existing collection names, project IDs, and vector dimensions. With `database.qdrant.auto_create` enabled, startup adds the provenance index to the existing Add Record collection and existing records are immediately queryable. Enable multimodal mode once the upgraded API and workers are ready. To roll back, disable multimodal mode and drain queued image tasks before rolling back the processes. Retain Qdrant and object storage data. The Console's [lifecycle and relationship graph API](internal-memory-api.md) uses project-scoped gateway tokens.

## 2. Deploy the UI

### Start the UI with local APIs

With the default authentication mode, `make dev` and `make api UI=1` write the started API addresses and project credentials to `.mindmemos/settings.v2.json`. Set `UI_HOME` to use a different directory. Set `PROJECT=<name>` to override `default_project` in the project file. Both APIs still accept credentials for projects authorized to use them.

### Connect to MindMemOS Cloud or existing APIs

If you already have the Python dependencies, start with the build. For a fresh UI-only environment, run all three commands.

```bash
uv sync --package mindmemos-ui --inexact
make ui-build
UV_NO_SYNC=1 make ui
```

`make ui` starts the workbench and reads the existing SDK configuration. It defaults to the repository's `.mindmemos/` directory. Set `UI_HOME` or `MINDMEMOS_HOME` to use another directory. Use separate directories to keep connections for different environments apart.

```bash
UV_NO_SYNC=1 make ui UI_HOME="$PWD/.mindmemos/cloud" UI_PORT=8865
```

Open the [workbench](http://127.0.0.1:8765) or use the address printed at startup, then go to **Settings → SDK configuration**.

- **MindMemOS** uses the official service URL. Enter your official API key. Memory and Skill share the credential.
- **Self-hosted** uses an existing Memory API root URL and service API key. For a separate Skill API, enter its root URL and credential. Leave the Skill URL blank for a Memory-only configuration. If you only connect a separate Skill API, you can leave the Memory key blank.

Select **Save configuration** and check **Service availability** in the sidebar. **Available** means the health endpoint responded. Load **Memories** or **Skills** to verify credentials and permissions. To verify model access, run a request that calls the model.

The workbench listens on `127.0.0.1` for use on the machine running it. The UI backend connects to the configured APIs through the SDK.

## 3. Select and manage services

### Select services and features

`SERVICE=all|memory|skill` defaults to both APIs. Skill can run independently. These feature choices apply to `make dev`, `make docker`, and `make api`.

| `FEATURE` | Memory tasks | Skill tasks | OTLP monitoring |
| --- | --- | --- | --- |
| `core` | Synchronous | Local asynchronous | Off |
| `async` | Synchronous and Kafka | Kafka | Off |
| `all` (default) | Synchronous and Kafka | Kafka | On |

```bash
make dev SERVICE=skill FEATURE=core         # Skill API and UI
make dev SERVICE=memory FEATURE=core UI=0   # Memory API with UI disabled
```

Use `UI=0` or `UI=1` to override the UI defaults for `make dev` and `make api`. The launcher applies overrides to temporary runtime configs and preserves the source YAML.

### Start dependencies separately or use existing infrastructure

```bash
make docker FEATURE=core             # Start the required local containers
make api FEATURE=core UI=1           # Start APIs with the same scope and configure the UI
```

`make api` requires ready dependencies and defaults to starting the APIs only. For external databases, Kafka, or OTLP, set the endpoints in the YAML or `.env`. The launcher selects containers for loopback addresses. Set `SERVICE_AUTH=1` to retain authentication from the service YAML and configure the UI connections in **Settings**.

| Option | Purpose |
| --- | --- |
| `API_PORT`, `SKILL_PORT`, `UI_PORT` | Override application ports |
| `MEMORY_CONFIG`, `SKILL_CONFIG` | Select service YAML files |
| `PROJECT_CONFIG` | Select the local project YAML |
| `API_RELOAD=1`, `SKILL_RELOAD=1` | Enable API reload for development |

### Check status and stop

- `make status` shows managed applications, addresses, and containers.
- `Ctrl+C` stops the APIs and UI started in the current terminal and leaves containers running.
- `make stop` stops managed applications and containers in the current Compose project and preserves data. Run it before restarting with a smaller scope.
- `make clean` deletes Docker volumes in the current Compose project after confirmation. Use `make stop` for normal restarts.

## Task environments and data

Prepare datasets, agent programs, and system dependencies for the training or evaluation tasks you run. For gated Hugging Face datasets, obtain access and set a read token as `HF_TOKEN` in `.env`. See the [Skill evaluation guide](../evaluation/skill_evaluation.md) for download and run commands.

SpreadsheetBench requires LibreOffice when formula recalculation is enabled. Install it in `/Applications` on macOS or run `sudo apt install libreoffice-calc` on Ubuntu or Debian. In `.env`, you can set `SPREADSHEETBENCH_SOFFICE_PATH` to the absolute path to `soffice`, such as `/Applications/LibreOffice.app/Contents/MacOS/soffice` or `/usr/bin/soffice`. Run the executable with `--version` to verify the installation.

To use Skill directly from Python, see the [Skill service documentation](../../src/mindmemos_skill/README.md).
