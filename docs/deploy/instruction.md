# MindMemOS deployment guide

English | [简体中文](instruction_ZH.md)

This guide starts local services from the repository root. Python services run on the host. Docker Compose manages databases and monitoring components.
`make dev` manages service processes in the foreground. This guide does not configure public access, a reverse proxy, or a background system service.

## 1. Start all features with `make dev`

The default command starts Memory API, Skill API, and UI. It also enables Kafka tasks and OTLP monitoring.
The launcher selects local containers from the configured connection addresses. Remote connections do not cause the launcher to create those containers.

### 1.1 Prepare software and Python dependencies

Prepare these tools:

- Git, Make, and uv.
- Python 3.11–3.13. The repository selects 3.13 through `.python-version` by default.
- Docker and Docker Compose. Start the Docker engine.
- Node.js and npm for the UI build.
- System build tools. Some ALFWorld dependencies require native compilation.

On macOS ARM64, the TextWorld build mounts and unmounts an Inform7 installer image. Allow the installation process to do these operations.
The project includes a runtime patch for TextWorld 1.7.0 on Python 3.13.

Install dependencies from the repository root:

```bash
uv sync --all-packages --extra postgres --extra kafka
```

The root `uv.lock` manages workspace dependencies. All Python services use the root `.venv`.
Skill includes FastAPI and Python packages for its built-in environments in the default dependencies. Only `postgres` and `kafka` are optional extras.
The Skill example selects PostgreSQL. All features require Kafka, so this command includes both extras.

### 1.2 Initialize local configuration

Run this command:

```bash
make init
```

Initialization copies `.example` templates to missing local files. Existing files are not recopied. It then applies nonempty model settings from `.env` to all actual YAML configurations under `config/`, including existing files, Memory/Skill services, training, and evaluation. `.example.yaml` templates retain public example values.

To share model connections, create `.env`, fill the optional settings below, and initialize:

```bash
cp .env.example .env  # Only when .env does not exist
# Edit .env and fill the optional model settings below
make init
```

| Model type | API key | Base URL | Model name | Output dimensions |
| --- | --- | --- | --- | --- |
| Chat | `MINDMEMOS_CHAT_API_KEY` | `MINDMEMOS_CHAT_BASE_URL` | `MINDMEMOS_CHAT_MODEL` | — |
| Embedding | `MINDMEMOS_EMBEDDING_API_KEY` | `MINDMEMOS_EMBEDDING_BASE_URL` | `MINDMEMOS_EMBEDDING_MODEL` | `MINDMEMOS_EMBEDDING_DIMENSIONS` |
| Rerank | `MINDMEMOS_RERANK_API_KEY` | `MINDMEMOS_RERANK_BASE_URL` | `MINDMEMOS_RERANK_MODEL` | — |

Each nonempty setting replaces its corresponding fields. Empty or undefined settings preserve the YAML values; fill them manually in `config/`. Any Embedding setting requires a positive integer dimension. Initialization updates every embedding `dimensions` field and Qdrant `vector_size`. Base URLs are copied verbatim into `base_url` or `api_base`; provide the complete rerank endpoint required by your provider.

Run `make init` again after editing `.env` to replace all corresponding fields with the new values. Chat settings apply to every chat model block, including training optimizers, routers, and evaluation targets. To use different models, edit the YAML files after initialization and avoid overwriting them again with shared `.env` settings. Initialization retains YAML comments, other settings, and local connections, and does not print model credentials.

Check these deployment files:

| File | Required settings |
| --- | --- |
| `.env` | Database addresses, ports, passwords, and monitoring connections |
| `config/mindmemos/dev.yaml` | Chat, Embedding, and Rerank model names, API keys, and addresses |
| `config/mindmemos_skill/dev.yaml` | Storage connections and model connections for enabled management, training, and evolution pipelines |
| `config/presets/local.project.yaml` | Project IDs, service API keys, permissions, and the default project |

`make init` also creates evaluation configuration files. Service startup does not require an evaluation run.

Check these conditions when you configure models:

1. Match the Memory embedding dimensions to `database.qdrant.vector_size`.
2. Set the connection parameters in each enabled Skill model block.
3. Configure local service API keys separately from model provider API keys.
4. Give the default project these permissions: `memory:read`, `memory:write`, `skill:read`, and `skill:write`.
5. Preserve existing project IDs to retain access to existing data.

The default launcher reads all local projects. It creates temporary authentication files for both APIs.
`default_project` selects the project for UI. `PROJECT=<name>` changes this selection without restricting the projects accepted by the APIs.
The default UI configuration directory is `.mindmemos` in the repository root.

Additional dependencies:

- `HF_TOKEN`: The access token used by the dataset downloader for gated or private Hugging Face datasets. Sign in to Hugging Face, create a token with read access to the target dataset in [Access Tokens](https://huggingface.co/settings/tokens), and put its full value in `.env`, for example `HF_TOKEN=hf_...`. For gated datasets, first accept the terms or request access on the dataset page; a token alone does not grant dataset access. Leave it empty for public datasets that allow anonymous downloads. See the [Hugging Face token documentation](https://huggingface.co/docs/hub/security-tokens).
- `SPREADSHEETBENCH_SOFFICE_PATH`: The absolute path to the LibreOffice `soffice` executable, used to recalculate workbooks and save cached formula results when SpreadsheetBench formula recalculation is enabled. Set an executable file path, not an installation directory. Tasks only need it when they use recalculation.

Install LibreOffice: on macOS, download the installer for your processor architecture from the [LibreOffice website](https://www.libreoffice.org/installation-instructions/) and install the application in `/Applications`. On Ubuntu/Debian, run `sudo apt update && sudo apt install libreoffice-calc`.

Then set the actual executable path in `.env`, for example:

```dotenv
# macOS
SPREADSHEETBENCH_SOFFICE_PATH=/Applications/LibreOffice.app/Contents/MacOS/soffice
# Common Linux path (use this instead of the macOS setting)
# SPREADSHEETBENCH_SOFFICE_PATH=/usr/bin/soffice
```

On Linux, use `command -v soffice` to find the path. Run `"<absolute path to soffice>" --version` and confirm that it prints the LibreOffice version. If the variable is empty, the program searches for a packaged LibreOffice runtime, `soffice` on PATH, and common installation paths. An explicit absolute path does not require adding `soffice` to PATH.

### 1.3 Prepare NLP and UI resources

After you prepare configuration files, install and check NLP models:

```bash
uv run --no-sync python scripts/install_nlp_assets.py
uv run --no-sync python scripts/check_nlp_assets.py
```

The installer reads spaCy model names from Memory configuration.
Build the UI:

```bash
make ui-build
```

`make dev` does not build the UI. The UI page may fail to open if frontend resources are missing.
Before SpreadsheetBench tasks that require formula recalculation, install and configure LibreOffice as described in section 1.2.
Prepare datasets and external evaluators separately. Service startup does not download all benchmark data.

### 1.4 Check the plan and start services

Preserve the selected dependencies in the current terminal:

```bash
export UV_NO_SYNC=1
make dev PLAN=1
```

`UV_NO_SYNC=1` tells the Makefile's `uv run` to use the synchronized environment.
Repeat the synchronization command in section 1.1 after dependency declarations change.
`make setup` synchronizes dependencies, installs Git hooks, and prepares NLP models. After that command, synchronize the required extras again.

`PLAN=1` checks configuration and shows the startup plan. It does not start containers or APIs, or test external connections and model calls.
After you check the plan, start services:

```bash
make dev
```

This command is equivalent to:

```bash
make dev SERVICE=all FEATURE=all UI=1
```

With local example connections, the command starts these components:

| Category | Components |
| --- | --- |
| Python services | Memory API, Skill API, UI |
| Memory storage | Qdrant, Neo4j |
| Skill storage | PostgreSQL, selected by Skill database configuration |
| Asynchronous tasks | Kafka, Kafka UI, and the Skill task worker |
| Monitoring | OpenTelemetry Collector, ClickHouse, Grafana, Kafka Exporter |

The launcher waits for dependency ports and application health checks. Wait for `Development services ready` before you access services.

### 1.5 Check services and stop them

These addresses use default ports. If you change ports, use the addresses in the startup plan.

| Service | Address |
| --- | --- |
| Memory health check | `http://127.0.0.1:8000/healthz` |
| Memory API documentation | `http://127.0.0.1:8000/docs` |
| Skill health check | `http://127.0.0.1:8001/health` |
| Skill API documentation | `http://127.0.0.1:8001/docs` |
| UI | `http://127.0.0.1:8765` |
| Kafka UI | `http://127.0.0.1:8080` |
| Grafana | `http://127.0.0.1:3000` |

Check health from another terminal:

```bash
curl -f http://127.0.0.1:8000/healthz
curl -f http://127.0.0.1:8001/health
UV_NO_SYNC=1 make status
```

A successful health check means the service responds. Send a business request with the configured model to check its connection.
Use these stop commands:

| Action | Result |
| --- | --- |
| Press `Ctrl+C` in the foreground terminal | Stop this session's APIs and UI, but keep containers running |
| `UV_NO_SYNC=1 make stop` | Stop managed workspace applications and containers in the current Compose project, but preserve data |
| `UV_NO_SYNC=1 make clean` | After confirmation, delete Docker volumes in the current Compose project and their data |

Use `make stop` for normal shutdown. `make clean` deletes database data. Do not use it for a normal restart.

## 2. Deploy core features

Complete sections 1.1–1.3 first. Then select the services and features you need.
This section retains `UV_NO_SYNC=1` and uses the service configuration you prepared.

### 2.1 Select services and features

| Parameter | Value | Memory | Skill |
| --- | --- | --- | --- |
| `SERVICE` | `all` | Start Memory API | Start Skill API |
| `SERVICE` | `memory` | Start Memory API | Do not start Skill API |
| `SERVICE` | `skill` | Do not start Memory API | Start Skill API |
| `FEATURE` | `core` | Enable synchronous features; disable Kafka tasks and OTLP export | Enable synchronous features and local asynchronous task management; disable Kafka and OTLP export |
| `FEATURE` | `async` | Enable synchronous features and Kafka tasks; disable OTLP export | Enable synchronous features and Kafka task management; disable OTLP export |
| `FEATURE` | `all` | Enable synchronous features, Kafka tasks, and OTLP export | Enable synchronous features, Kafka task management, and OTLP export |
| `UI` | `0`, `1` | Disable or enable the shared UI; `make dev` enables it by default | Disable or enable the shared UI; `make dev` enables it by default |

Changing `FEATURE` does not modify the original YAML files in `config/`. The launcher generates temporary YAML files for the services to read at startup. These files are located at `mindmemos-dev-<random characters>/memory.yaml` and `mindmemos-dev-<random characters>/skill.yaml` under the operating system's temporary directory. The paths change on each launch, and the directory is cleaned up when the launcher exits.

`FEATURE` makes these changes to the runtime configuration:

- `core`: Disable Kafka and OTLP monitoring data export. Memory supports synchronous interfaces only, while Skill uses Python's native asynchronous task management.
- `async`: Enable Kafka asynchronous tasks and disable OTLP monitoring data export.
- `all`: Enable Kafka asynchronous tasks and OTLP monitoring data export. Use Grafana dashboards to monitor Memory and Skill service logs.

Use one of these commands:

```bash
# Both APIs and UI; synchronous Memory features and local asynchronous Skill tasks.
make dev SERVICE=all FEATURE=core

# Only Memory API, without UI.
make dev SERVICE=memory FEATURE=core UI=0

# Only Skill API, without UI.
make dev SERVICE=skill FEATURE=core UI=0

# Both APIs and asynchronous tasks, without OTLP monitoring.
make dev SERVICE=all FEATURE=async
```

Run one startup command at a time. Run `make stop` before you change the scope.
A smaller scope does not stop extra containers from an earlier startup.

### 2.2 Start containers, APIs, and UI separately

Start containers required for synchronous features:

```bash
make docker SERVICE=all FEATURE=core
```

Then start the APIs:

```bash
make api SERVICE=all FEATURE=core
```

Finally, start the UI in another terminal:

```bash
make ui
```

### 2.3 Use existing infrastructure or change ports

Set existing database, Kafka, and OTLP addresses in service YAML or `.env`.
The launcher creates containers only for localhost or loopback addresses. It does not create remote services.
After external dependencies become available, use `make api` for the selected APIs.

Example startup parameters:

```bash
make dev FEATURE=core API_PORT=8100 SKILL_PORT=8101 UI_PORT=8865
make dev FEATURE=core MEMORY_CONFIG=/absolute/path/memory.yaml SKILL_CONFIG=/absolute/path/skill.yaml
```

These are separate examples. Select one command for each startup.
`UI_HOME` selects the UI's SDK configuration directory. `PROJECT_CONFIG` selects the local project configuration file.
`API_RELOAD=1` enables development reload for Memory API. `SKILL_RELOAD=1` enables it for Skill API.

The default launcher uses local project authentication. Set `SERVICE_AUTH=1` to retain authentication policies from service YAML.
In that mode, configure UI service addresses and credentials separately. The launcher does not configure them automatically.
