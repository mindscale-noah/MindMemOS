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
samples = data.load()
splits = data.split(samples, ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
data.save_splits(splits, "output/splits.json")
splits = data.split(samples, split_file="output/splits.json")
sample = splits["train"][0]
result = env.rollout(sample.task, "output/rollout-001", reward_data=sample.reward_data)
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

`Environment.rollout(task, work_dir, *, reward_data, skills=None, resume=True)` executes one
sample and returns a `RolloutResult` with `output`, ordered `trajectory` records,
and a structured `reward`. The task is a `Task` object with required stable `id` and `instruction` fields; subclasses may add fields.
Skills are supplied as `list[Skill]`; omitting them runs a baseline.
Both `rollout` and `_run` accept `list[Skill]`. The base class calls
`Skill.assemble(task)` once and passes the returned complete Skill directly to `_run`.
Assembly owns creation of the task-local copy and replacement of its bundle.
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
timing and usage. Subclass fields on Reward and Usage survive serialization.
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
and assembled bundles are serialized for resume checking. `_run` receives the original
task/reward objects and complete task-local Skill copies. Config is persisted in
plain text and must not contain credentials. See [中文说明](README_ZH.md) for usage.

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

The central repository API supports registration, listing, inspection, commit, export,
history, diff, version selection, metadata changes, and reference-safe deletion.
See the [API guide](../../docs/mindmemos_skill/central_repository_api.md) for request types,
configuration, and Python examples. Automatic training and evolution remain unimplemented.

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

Install this package, then run from the repository root:

```python
from mindmemos_skill import create_service
service = create_service("config/mindmemos_skill/default.yaml")
```

Alternatively, pass a configuration object directly:

```python
from mindmemos_skill import ServiceConfig, create_service

service = create_service(ServiceConfig(backends={}))
```

Both inputs produce the same `SkillService` implementing `MindMemOSSkill`.
Configuration describes backends and dependencies, without a `mode` field.
The service exposes business methods and does not own an HTTP application.
The factory revalidates and copies configuration objects and does not start a server.
An empty `backends: {}` still creates an assembly-only service; the default configuration now includes a SQLite repository. For configured backends, each instance name
is constructed once and exposed through `service.backends[name]`.

The following example demonstrates registration and dependency injection only;
it does not implement a database or persistence operations. Save the YAML as
`config/mindmemos_skill/assembly.yaml` before running the Python example.
```python
from pydantic import BaseModel, ConfigDict
from mindmemos_skill import create_service
from mindmemos_skill.infra import Infrastructure
from mindmemos_skill.persistence import Persistence

class ConnectionConfig(BaseModel):
    """Configure an in-memory connection example."""
    model_config = ConfigDict(extra="forbid")
    namespace: str = "default"
    """Namespace used by this example connection."""

class StoreConfig(BaseModel):
    """Configure an in-memory store example."""
    model_config = ConfigDict(extra="forbid")

@Infrastructure.register("MemoryConnection")
class MemoryConnection(Infrastructure[ConnectionConfig]):
    """Hold configuration for the connection example."""
    config_type = ConnectionConfig
    """Configuration model for this implementation."""

@Persistence.register("MemoryStore")
class MemoryStore(Persistence[StoreConfig]):
    """Demonstrate injection of an existing connection instance."""
    config_type = StoreConfig
    """Configuration model for this implementation."""

    def __init__(self, config: StoreConfig, *, connection: MemoryConnection):
        """Store an injected connection.

        Args:
            config: Validated store configuration.
            connection: Shared connection selected by the YAML dependency.
        """
        super().__init__(config)
        self.connection = connection

service = create_service("config/mindmemos_skill/assembly.yaml")
assert service.backends["store"].connection is service.backends["db"]
```

```yaml
backends:
  store:
    module: persistence
    class_name: MemoryStore
    config: {}
    dependencies:
      connection: db
  db:
    module: infra
    class_name: MemoryConnection
    config:
      namespace: demo
```

`class_name` matches the case-sensitive registration name, which may equal the
Python class name. `config` is validated by the implementation's `config_type`;
use `extra="forbid"` to reject unexpected options. Dependency keys name constructor
keyword arguments, and their values reference backend instance names.

Available families are `infra.Infrastructure`, `algos.Algorithm`, `agents.Agent`,
`logging.Logger`, `persistence.Persistence`, and `envs.Environment`.
They share `registry.Registrable` while keeping separate registries. `common`
contains utilities and is not a service backend.

Import extension modules explicitly before calling the factory; YAML does not
import arbitrary Python modules. Only backends use registration; the service is
independent of transport. Package imports do not construct resources.
All registered names, configurations, references, cycles, and constructor signatures
are checked before construction. Dependency order is independent of YAML ordering;
multiple references share the same instance.

Generic resource cleanup and constructor-failure rollback are not yet implemented.
Backend constructors should avoid starting long-lived resources and must clean up
their own failed construction. Business execution and resource lifecycle remain
future work.

Run `.venv/bin/python -m pytest tests/mindmemos_skill -q` from the repository root.

## FastAPI access

Install the optional HTTP dependencies and start the server from the repository root:

```bash
uv pip install -e './src/mindmemos_skill[fastapi]'
make skill-fastapi
```

Python and HTTP access share `config/mindmemos_skill/default.yaml`, without mode selection.
The HTTP adapter accepts an existing service and calls its methods without rebuilding backends:

```python
from mindmemos_skill import ServiceConfig, create_service
from mindmemos_skill.api.entries.fastapi import create_app

service = create_service(ServiceConfig())
app = create_app(service)
assert app.state.service is service
```

`api/service.py` implements service assembly and business operations;
`entries/local.py` assembles the service directly, while `entries/fastapi.py`
wraps it with routes and launches the HTTP server.
Open `http://127.0.0.1:8000/docs` for API documentation.

| HTTP route | Local method |
| --- | --- |
| `POST /skills/register` | `register_skill()` |
| `POST /skills/delete` | `delete_skill()` |
| `POST /skills/auto-evolve` | `auto_evolve_skill()` |
| `POST /skills/auto-train` | `auto_train_skill()` |

`GET /health` returns `{"status":"ok"}` for application availability only.
Central operations accept typed JSON bodies. All routes and request types are listed in the
[API guide](../../docs/mindmemos_skill/central_repository_api.md). Only auto-evolve and auto-train return 501.
The HTTP entry point initializes missing tables; create_app(service) itself does not.

Use the repository Makefile to choose an entry point and override the configuration or address:

```bash
make skill-local
make skill-fastapi SKILL_CONFIG=config/mindmemos_skill/default.yaml SKILL_HOST=127.0.0.1 SKILL_PORT=8001
```

`make skill-local` assembles and prints the local service, then exits.
`make skill-fastapi` runs the HTTP server until Ctrl-C. The default interpreter is
`.venv/bin/python`; override it with `SKILL_PYTHON`. The Makefile adds this checkout's
Skill source directory to the import path.

Startup logic lives in `mindmemos_skill/api/entries/local.py` and
`mindmemos_skill/api/entries/fastapi.py`. Service assembly does not import FastAPI or Uvicorn.
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

`SkillBundle` reads a directory or `SKILL.md`, serializes canonical SDK-compatible
JSON, computes its SHA-256 hash, and exports managed files to a directory. Only
`SKILL.md` belongs to the mutable bundle. `Skill.from_path()` also preserves
auxiliary files such as references and scripts, and `Skill.export()` exports the
complete directory. Nonempty destinations require `overwrite=True`, which replaces
the directory and removes stale files. Resources remain unchanged by assembly and optimization.
`Skill.assemble(task)` defaults to a complete independent Skill copy with unchanged content.
`Skill(bundle=..., resources=...)` is a data object with no configuration or registry.
Algorithms can subclass it and override `assemble(task)` for task-dependent behavior.
All optimizable content belongs in the bundle; implementation config is fixed.

`Skill.snapshot()` captures source content and auxiliary bytes;
`Skill.from_snapshot()` restores data and preserves existing implementation metadata without loading code. `with_bundle()` creates a new candidate.
`persistence.skills.add_skill(skill, record)` preserves record metadata while
replacing bundle/resource/config fields, and `get_skill(version_id)` restores the
complete Skill. Bundle hashes exclude auxiliary files; the local snapshot hash
includes them. Resource changes also invalidate rollout resume.
Rollout requests retain both source and assembled content with hashes. Retries
reuse saved assembly; `resume=False` assembles again. Old rollout request formats
require a new work directory. See [the Chinese guide](README_ZH.md) for examples.

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
`api_key` and `base_url` accept direct values. Missing fields resolve from
`MINDMEMOS_AGENT_API_KEY` and `MINDMEMOS_AGENT_BASE_URL` when config is constructed.
Credentials are masked in JSON snapshots. Without either source, CLIs preserve
their native authentication and configuration. Codex and Claude support
`reasoning_effort`; Claude also supports `max_turns` and `max_completion_tokens`.
DeepSeek supports `profile` and `provider` (default `deepseek-official`), retains a fresh
session directory per run, and validates v0 (dsh 0.1.0-rc.6) and v2 session formats.
Unsupported formats raise explicitly. When Claude resolves a credential or
endpoint from config or the shared environment, it disables user/project/local settings sources so their
`env` values cannot override those choices. Other settings from those sources are
also disabled for that run.

Custom gateways must support the native protocol: Responses for Codex, Anthropic
Messages for Claude, and Chat Completions for DeepSeek. For a gateway exposing
`/v1/messages`, Claude takes the gateway root URL; Codex and DeepSeek typically take
the `/v1` base URL.
