# MindMemOS Skill

## Environment data

`mindmemos_skill.envs.data` exports `Data`, `DataSample`, and `RewardData`.
Implement `Data._load()` to adapt local dataset files into samples. `load()`
validates sample structure and unique, nonblank task IDs. `DataSample` pairs
public `Task` inputs with a typed `RewardData` subclass containing private
answers, tests, or references. `RewardData` forbids undeclared fields;
`Reward` remains the evaluated output. The environment requires `reward_data`
as a keyword argument, passes it only to its trusted `_run` implementation,
and records its type and fields in audit/request.json for resume checking.
Reference file versions or hashes must be provided by the dataset adapter.

Data uses the shared registry (`Data.register`, `Data.build`) without adding a
top-level service backend category. It loads all samples into memory and does
not download data automatically. Source formats and configurations belong to
concrete adapters, including adapters for already downloaded datasets.

```python
import asyncio

samples = data.load()
splits = data.split(samples, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
data.save_splits(splits, "output/splits.json")
splits = data.split(samples, split_file="output/splits.json")
sample = splits["train"][0]
result = asyncio.run(env.rollout(sample.task, "output/rollout-001", reward_data=sample.reward_data))
```

Random splitting requires an explicit integer seed and nonnegative ratios
summing to one (absolute tolerance 1e-9). IDs are sorted before a local shuffle;
input order, ratio key order, and global RNG state do not affect membership.
Integer counts use largest remainders, with ties resolved by split name.
Every sample is assigned once; empty splits and empty datasets are allowed.

File mode accepts a JSON object such as `{"train": ["q-1"], "val": [], "test": ["q-2"]}`.
It is exclusive with ratios and seed, preserves file order, and allows subsets.
Unknown IDs, duplicate IDs, overlapping splits, and blank names raise errors.
`save_splits` atomically stores only names and IDs, without private input data.
Stable membership requires stable source IDs; changed datasets can change seeded
membership. Use the exported manifest to freeze a selection.

## Minimal environment contract

`await Environment.rollout(task, work_dir, *, reward_data, skill_set=None, resume=True)` executes one
sample and returns a `RolloutResult` with `output`, ordered `trajectory` records,
and a structured `reward`. The task is a `Task` object with required stable `id` and `instruction` fields; subclasses may add fields.
Supply `skill_set=SkillSet({"skill-id": skill}, max_concurrency=4)`; omitting it runs a baseline.
`SkillSet` selects all members by default. Dynamic collections override `select(task)`
and record explicit selection settings through `config_snapshot()`. The collection calls
each selected member's `assemble(task)` concurrently, bounded by `max_concurrency` per call,
and preserves selection order. `_run` receives the resulting `list[Skill]` without reassembly.
The removed `skills=` rollout argument is not accepted.
Assembly owns creation of the task-local copy and replacement of its content.
Concrete classes, auxiliary resources, and export methods remain available.
Environments never check whether a Skill is static or dynamic or assemble it again.
Agent dependencies can be injected through the existing service factory.
The base class provides persistence, not a model backend or task evaluator.
The built-in [LiveMath environment](mindmemos_skill/envs/builtin_envs/livemath/README.md)
loads official local monthly JSON, uses the shared split API, runs one
OpenAI-compatible chat call with optional static or dynamic Skill instances, and evaluates the
choice locally. Install the `livemath` optional dependencies to run the model.
It exposes only public prompts to the model and keeps private scoring in audit/.

`Reward` requires `score` (finite number), `success` (boolean), and `detail`
(text containing detailed evaluation data). It accepts extra JSON fields and
can be subclassed for environment-specific metrics. `Usage` has nonnegative
`input_tokens`, `output_tokens`, and `total_tokens`, all defaulting to zero,
plus extra JSON fields for provider-specific usage. Implementations aggregate
all LLM calls, including evaluation calls, and supply the provider-reported
total; it is not inferred. Default zeros mean usage was not supplied, not
verified absence of consumption.

The base class fills UTC `started_at`, `finished_at`, and monotonic
`duration_s` around assembly and `_run`, including preparation, agent execution, and scoring.
Cache reads and result persistence are excluded. Resume preserves the original
timing and usage. Saved selection and assembly are reused without selecting or assembling again;
changed candidates or collection settings require a new work directory. Subclass fields on Reward and Usage survive serialization.
To restore their concrete Python types and validators, declare a RolloutResult
subclass with those field types and set the environment's `result_type` to it.
Otherwise, base Reward and Usage objects retain the extension data.
Legacy request records without task IDs or reward_data are not migrated; use a new work directory.

The supplied `work_dir` is the root of one independent rollout. The environment
stores private inputs in `audit/request.json` and atomically publishes the
validated result as `audit/result.json`. Only the absolute `work_dir/workspace/`
path is passed to `_run` for public task files, skills, and agent artifacts.
Resume is an optional environment capability declared by `supports_resume`, which
defaults to `True` for existing environments. An environment declaring
`supports_resume = False` raises `NotImplementedError` when `resume=True`, including
when the argument is omitted. Pass `resume=False` to execute such an environment;
audit persistence and input validation still apply. Unsupported resume is rejected
before skill assembly or filesystem operations.

The two subdirectories cannot be symbolic links. With resume enabled, matching completed runs, including zero-reward runs, are
loaded without execution. Changed task or reward-data fields or types, skill types or config, environment classes, or config
require a new directory. Corrupt records raise errors. Exceptions are propagated
without caching completion. The environment does not manage attempts or retry
automatically; retry and sampling orchestration belong to the caller.
`resume=False` removes the previous completion before rerunning matching inputs.
Existing workspace files remain in place, so implementations must prepare them
appropriately when invoked again. Pass only `workspace` and public task
materials to the agent; do not change the process-wide working directory.
The agent backend must restrict file access to `workspace`: changing cwd alone
does not prevent reading `../audit`. Do not expose the complete Task snapshot
to the agent when it contains private evaluation data.

Use a separate directory per independent sample; concurrent use of the same
directory is unsupported. Include agent/model, data, and evaluator versions and
execution settings in the environment config for resume checking. Code and
dependency state outside config are not automatically fingerprinted.
Task and private reward-data fields (including subclass fields), source Skill snapshots,
and assembled Skill snapshots are serialized for resume checking. `_run` receives the original
task/reward objects and complete task-local Skill copies. Config is persisted in
plain text and preserve explicitly configured model credentials. See [中文说明](README_ZH.md) for usage.

[中文说明](README_ZH.md)

`mindmemos-skill` is the Python package for Skill-related functionality in MindMemOS.
It provides the abstract `MindMemOSSkill` interface, shared registration, and
`create_service(config)` for assembling a service from a typed configuration or YAML.
The FastAPI entry layer wraps an existing service for HTTP access.
Business operations remain unimplemented.

## Requirements

- Python >=3.11,<3.14
- Pydantic 2 and PyYAML 6.

## Local installation

From the repository root, install the package into the active development environment:

```bash
uv pip install -e ./src/mindmemos_skill
```

Verify the import:

```python
import mindmemos_skill
```

The distribution name is `mindmemos-skill`; the Python import name is `mindmemos_skill`.
The package is not yet registered in the repository's uv workspace.

## Package layout

```text
src/mindmemos_skill/
├── pyproject.toml
├── README.md
├── README_ZH.md
└── mindmemos_skill/
    ├── __init__.py
    ├── api/          # Abstract service interface and factory
    ├── common/       # Shared JSON and IO utilities
    ├── infra/        # Business-independent infrastructure
    ├── algos/        # Skill algorithms
    ├── agents/       # Agent invocation
    ├── logging/      # Logging support
    ├── persistence/  # Business tables, queries, and writes
    ├── skill/        # Skill definitions and operations
    ├── envs/         # Algorithm environments and data structures
    └── registry/     # Shared registration mechanism
```

Package metadata and the version are maintained in `pyproject.toml`.

The local Python repository API supports registration, listing, inspection, commit,
export, history, diff, version selection, metadata changes, and reference-safe deletion.
The public HTTP contract is defined in the [target API](../../docs/mindmemos_skill/target_architecture_and_api.md).
The [Python API guide](../../docs/mindmemos_skill/central_repository_api.md) describes
the local service methods. Automatic training and evolution remain unimplemented.

## Logging

Configure logging explicitly at application startup, then obtain module loggers:

```python
from mindmemos_skill.logging import configure_logging, get_logger

configure_logging(level="INFO", log_file="logs/skill.log")
logger = get_logger("algos")
logger.info("Skill training started")
```

Records include the timestamp, level, logger name, and message, and go to both
stderr and the file. Files use UTF-8 append mode; missing parent directories are
created. Omit `log_file` for console-only output. Use `logger.exception(...)` to
include a traceback. Package submodules can use `from ..logging import get_logger`
and `get_logger(__name__)`.

Importing the module does not configure logging. Configuration applies only to
the `mindmemos_skill` namespace and leaves the root logger unchanged. Repeated
configuration closes and replaces handlers created by this module, preserving
caller-added handlers. Configure before starting worker threads; concurrent
reconfiguration, log rotation, and multiprocess shared-file management are not supported.

## Service assembly

Service assembly, pipeline business operations (including repository and graph reads),
and service/pipeline cleanup are asynchronous. Use `await` inside an async function.
Use `async with await create_service(...) as service:` to close owned resources
automatically, or call `await service.close()` explicitly. Only process entry points
should own an event loop with `asyncio.run(...)`. Constructors and configuration
validation remain synchronous.

Install this package, then run from the repository root:

```python
from mindmemos_skill import create_service
service = await create_service("config/mindmemos_skill/dev.yaml")
```

Alternatively, pass a configuration object directly:

```python
from mindmemos_skill import create_service
from mindmemos_skill.config.loader import read_skill_config

config = read_skill_config("config/mindmemos_skill/dev.yaml")
service = await create_service(config)
```

Both inputs produce the same `SkillService` implementing `MindMemOSSkill`. The configuration argument is required. Complete deployments require `service.home` and `fastapi.auth`. None mode requires `default_project_id`; authenticated modes use the project resolved from credentials. A resource-only `ServiceConfig(home=...)` has no implicit project and requires a trusted host authorizer for business access.

Configuration contains one `persistence` with an inline `database`, an optional `repository_pipeline`, and named `training_pipelines` / `evolution_pipelines` mappings containing complete definitions. See the [default configuration](../../config/mindmemos_skill/dev.example.yaml) and [assembly contract](../../docs/mindmemos_skill/central_repository_api.md).

Each pipeline owns its nested algorithm, environment and agent settings. Repeated configuration creates independent components; there are no global resource references. The factory constructs one `SkillPersistence` and passes it to every pipeline through the fixed `persistence` keyword. Pipeline constructors accept `config` and `persistence`; their `config_type` models validate nested settings, and the implementation constructs local components through their existing registries.

Import extensions explicitly before assembly. The factory validates pipeline capabilities, options and constructor signatures before constructing the database. It does not infer or synthesize repository pipelines. Pipelines validate their own internal implementation choices and must never pass persistence into trainers or evolvers.

`service.backends` is diagnostic only. Its keys are `database`, `persistence`, `repository_pipeline`, `training_pipelines.<name>` and `evolution_pipelines.<name>`, not YAML references. Service shutdown closes pipelines in reverse construction order before closing the database. Pipelines release their owned components and clean up partial constructor failures, without closing global persistence.

Run `.venv/bin/python -m pytest src/mindmemos_skill/tests -q` from the repository root.

## FastAPI access

Install the optional HTTP dependencies and start the server from the repository root:

```bash
uv pip install -e './src/mindmemos_skill[fastapi]'
cp config/mindmemos_skill/dev.example.yaml config/mindmemos_skill/dev.yaml
make dev SERVICE=skill
```

Copy `config/mindmemos_skill/dev.example.yaml` to `config/mindmemos_skill/dev.yaml` before deployment. Python and HTTP access share the user-owned `dev.yaml`. Its `service.disabled_services` setting controls business capabilities,
while `fastapi.auth` selects authentication and, in none mode, the default project. Relative home and API key file paths resolve against the YAML directory. Omitted SQLite URLs resolve under home; explicit database URLs retain their existing semantics. For a local Python setup, see the [Skill user guide](../../docs/skill/user-guide.md); the available HTTP settings are shown in [the example configuration](../../config/mindmemos_skill/dev.example.yaml).
The HTTP adapter accepts an existing service and calls its methods without rebuilding backends:

```python
from mindmemos_skill import create_service
from mindmemos_skill.api.entries.fastapi import create_app

service = await create_service("config/mindmemos_skill/dev.yaml")
app = create_app(service)
assert app.state.service is service
```

`config/loader.py` resolves configuration, `api/factory.py` assembles the
service, and `api/service.py` implements business operations. `entries/fastapi.py` contains HTTP routes, request models, and the ASGI application;
`api/auth.py` resolves local keys and verifies Gateway JWTs.
Open `http://127.0.0.1:8001/docs` for API documentation.

The HTTP entry uses `POST /v1/skills/*`. Repository operations available now are
`register`, `list`, `show`, `commit`, `history`, `diff`, `activate`, `update-info`,
`delete`, `version-resolve`, and `trajectories/report`, `show`, `list`. Request bodies do not accept `project_id`; the
deployment binds a local project or resolves a Bearer credential to one project.

`GET /health` returns `{"status":"ok"}` for application availability only.
All target routes and request types are listed in the
[target API](../../docs/mindmemos_skill/target_architecture_and_api.md). Graph,
some algorithm routes currently return explicit unavailable errors.
`Idempotency-Key` is ignored in this phase; write requests do not have a
duplicate-free retry guarantee.
Python callers can use `run_training()` to await a result or `submit_training()` to submit a task.
The HTTP training route submits a background task when a task manager is configured.
The HTTP entry point initializes missing tables; create_app(service) itself does not.

Use the repository Makefile to override the HTTP configuration or address:

```bash
make dev SERVICE=skill SKILL_CONFIG=config/mindmemos_skill/dev.yaml SKILL_HOST=127.0.0.1 SKILL_PORT=8001
```

`make dev SERVICE=skill` runs Uvicorn until Ctrl-C, following the main service's startup
pattern. Set `MINDMEMOS_SKILL_CONFIG_PATH`, `MINDMEMOS_SKILL_API_HOST`, and
`MINDMEMOS_SKILL_API_PORT` in the root `.env` for persistent overrides.

HTTP startup logic lives in `mindmemos_skill/api/entries/fastapi.py`.
Service assembly does not import FastAPI or Uvicorn.
`run_skill.py` remains a personal debugging script. Python integrations can continue
calling `create_service` directly.

## Database infrastructure

`infra` provides SQLite + sqlite-vec by default and optional PostgreSQL + pgvector.
Business-owned SQLAlchemy tables and Pydantic input models are registered centrally,
then accessed by table name through shared CRUD, upsert, and vector-search operations.
Backend-specific adapters live in `infra/database/builtin_backends/`. Registration does not
connect or create tables; initialization is explicit, and service shutdown closes pools.

See the [database guide](mindmemos_skill/infra/database/README.md) for registration examples,
operation semantics, configuration, and integration tests.

## Skill content

`Skill(SkillContent(content=..., resources=...))` represents one source document and
auxiliary files. `content` is the exact SKILL.md text; runtime resources map safe
relative paths to bytes. Standard content requires YAML frontmatter with name and
description. Nonstandard content requires explicit name and description without
rewriting the text. `Skill.from_path()` reads standard local content as UTF-8 bytes
and preserves all auxiliary files.

`Skill.export()` exports the complete directory. A nonempty destination requires
`overwrite=True`, which replaces that directory and removes stale files.
`with_content()` creates a candidate while preserving resources and subclass
dependencies; `with_resources()` explicitly replaces resources. `assemble(task)`
returns a separate task-specific Skill. `SkillSet` selects and assembles members.

`Skill.snapshot()` records content, Base64 resources, source metadata, and runtime
configuration. `content_hash` is SHA-256 of the exact UTF-8 document;
`skill_hash` covers that document and all resource paths and bytes using the shared
canonical encoding. Snapshot restoration revalidates both hashes.
`Skill.from_snapshot()` restores portable data without loading runtime code;
`common.skill_runtime.load_runtime_skill()` restores supported dynamic implementations.

Rollout requests retain source and assembled snapshots. Interrupted retries reuse
the saved selection and assembly without invoking the router again. Incompatible
or corrupt inputs require a new work directory. See the
[content contract](../../docs/mindmemos_skill/central_repository_api.md).

TreeSkill's implementation lives in `algos/train/treeskill/`. Import `TreeSkill`
and `TreeSkillConfig` from that module and construct it explicitly.
Use `TreeSkill.from_snapshot()` explicitly when restoring routing behavior;
`persistence.skills.get_skill()` returns source data without constructing a router.
It parses the source Markdown tree, requests subtree IDs from a router agent,
and renders the selected subtrees with ancestor introductions. References and
other auxiliary files stay unchanged. This implements task-time routing only,
not the training/evolution loop. Provider failures propagate; invalid selections
log an explicit full-source fallback. Private `_route(task)` retains selection
evidence and router usage internally; callers use `assemble(task)`. Router usage
is not yet added to environment-agent usage.
See the Chinese guide for configuration and reference-revision details.

## Local CLI harnesses

The `agents/` catalog includes `codex`, `claude`, and `deepseek_harness`, invoking
separately installed `codex`, `claude`, and `dsh` executables. Importing or building
an agent does not launch its CLI. All three expose synchronous
`run(system=..., user=..., workspace=...) -> AgentResult`. Native events are retained
in `trajectory`; reported usage is preserved in `usage`. DeepSeek usage is currently
unavailable and remains an empty dictionary. Failed, incomplete, or empty runs raise.

```python
from pathlib import Path
from mindmemos_skill.agents import Agent

agent = Agent.build("codex", {"timeout_s": 300})
result = agent.run(system="Solve the task using the workspace files.",
                   user="Summarize report.txt.", workspace=Path("/absolute/task/workspace"))
print(result.output)
```

The caller prepares an existing, independent workspace and exports assembled Skills
with `Skill.export(directory)` to `.agents/skills/<name>`, `.claude/skills/<name>`, or
`.dsh/skills/<name>`, respectively. Harnesses own native tool execution, not Skill
assembly or scoring. They do not accept legacy requests, Skill bindings, or callable
tools. Environments requiring per-turn tool callbacks retain their corresponding
Agent interface.

Shared settings are `cli_path`, `model`, `timeout_s`, and `extra_args`.
`api_key` and `base_url` must be supplied directly in configuration for model calls.
Omitted values remain unset; model calls fail before transport starts. Credentials
are masked in JSON snapshots. Native CLI model variables are filtered from the
inherited process environment before config values are injected into the child.
Codex and Claude support `reasoning_effort`; Claude also supports `max_turns` and
`max_completion_tokens`. DeepSeek supports `profile` and `provider` (default
`deepseek-official`), retains a fresh session directory per run, and validates v0
(dsh 0.1.0-rc.6) and v2 session formats. Unsupported formats raise explicitly.
Claude disables user/project/local settings sources so their `env` values cannot
override the model connection; other settings from those sources are also disabled.

Custom gateways must support the native protocol: Responses for Codex, Anthropic
Messages for Claude, and Chat Completions for DeepSeek. For a gateway exposing
`/v1/messages`, Claude takes the gateway root URL; Codex and DeepSeek typically take
the `/v1` base URL.
