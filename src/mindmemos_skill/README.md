# mindmemos-skill

Skill components for MindMemOS.

This package is the home for reusable skill definitions, runtime helpers, and
related integrations. It owns a backend-neutral storage infra layer and does
not depend on `mindmemos_sdk`.

## Installation

The core package contains the management contracts and SQLite-backed
persistence infrastructure without installing model or PostgreSQL clients:

```bash
pip install mindmemos-skill
```

Install only the runtime capabilities an application uses:

```bash
pip install 'mindmemos-skill[llm]'
pip install 'mindmemos-skill[pgvector]'
pip install 'mindmemos-skill[claude-sdk]'
pip install 'mindmemos-skill[alfworld]'
```

## Storage infra

Storage is split into two independent capabilities:

- `mindmemos_skill.infra.database` stores core structured persistence data.
  It has its own backend registry and ships SQLite by default.
- `mindmemos_skill.infra.vector_store` is an optional algorithm index. It has a
  separate backend registry and ships PostgreSQL + pgvector by default.

SQLite is not a VectorStore, and PGVector is not selected as the core database.
Custom providers are registered independently in the capability they
implement. Infra owns only generic records, schemas, filtering, and adapter
contracts; `mindmemos_skill.persistence` owns the Skill business table catalog.

Bootstrap the core persistence database through `DatabaseConfig`:

```python
from mindmemos_skill.infra.database import (
    DatabaseConfig,
    FieldSpec,
    FieldType,
    TableRegistry,
    TableSpec,
    bootstrap_database,
)

tables = TableRegistry(
    (
        TableSpec(
            name="runtime_logs",
            primary_key="log_id",
            fields=(FieldSpec(name="message", field_type=FieldType.TEXT, nullable=False),),
        ),
    )
)
tables.freeze()

database = await bootstrap_database(
    DatabaseConfig(provider="sqlite", options={"path": ".mindmemos/skill.db"}),
    tables,
)
```

Change `provider` and `options` to use another registered structured database.
Algorithms that need similarity search configure
`infra.vector_store.BackendConfig` separately, so core persistence remains
usable without a vector database.

For Skill persistence, use the business-owned catalog and canonical default
path (`~/.mindmemos/skill/state.db`):

```python
from mindmemos_skill.persistence import bootstrap_skill_database

database = await bootstrap_skill_database()
async with database.transaction() as unit_of_work:
    await unit_of_work.upsert_records("skill_versions", version_records)
    await unit_of_work.upsert_records("skill_sync_state", (sync_state_record,))
```

The current public schema is `mindmemos-skill` version 2, adding Skill Manage
graph tables to the initial version 1 catalog. SQLite records ordered,
forward-only migrations. The `bootstrap_skill_database(...)` helper creates a
consistent backup before an upgrade; `SkillApplication.from_config(...)` uses
the generic bootstrap and requires an operator-provided backup.
Use `get_skill_database_status(...)` to inspect pending versions or
`backup_skill_database(...)` to create a manual backup. Migration failures roll
back both DDL and the version ledger. Use the transaction-bound `unit_of_work`
inside the context; do not call the outer `database` object until the context
exits.

## Skill Manage graph index

The `skill_manage` capability builds and updates a graph over registered,
immutable Skill versions, then retrieves Skills with embedding seeds and graph
ranking. Install `mindmemos-skill[llm,pgvector]` and configure both structured
storage and vector storage on PostgreSQL with the pgvector extension. Plain
SQLite applications remain supported without enabling this capability.

For example, configure an application from environment-provided credentials:

```python
import os

from mindmemos_skill import SkillApplication
from mindmemos_skill.application.skill_manage import (
    SkillManageBuildRequest,
    SkillManageRetrieveRequest,
)
from mindmemos_skill.typing import SkillVersionRef

application = await SkillApplication.from_config({
    "local": {
        "root_dir": "~/.mindmemos/skill",
        "database": {
            "provider": "postgres",
            "options": {"dsn": os.environ["SKILL_DATABASE_DSN"]},
        },
        "vector_store": {
            "provider": "pgvector",
            "options": {"dsn": os.environ["SKILL_DATABASE_DSN"]},
        },
    },
    "runtime": {
        "models": {
            "embed": {
                "kind": "embedding",
                "model": "openai/text-embedding-3-small",
                "dimension": 1536,
                "api_key": os.environ["OPENAI_API_KEY"],
            },
        },
        "algorithms": {
            "graphs": {
                "type": "skill_manage",
                "model_roles": {"embedding": "embed"},
            },
        },
    },
})
try:
    # Use IDs of a version already registered in this application's repository.
    result = await application.build_skill_index(SkillManageBuildRequest(
        project_id="my-project",
        graph_id="my-graph",
        version_refs=(SkillVersionRef(skill_id=skill_id, version_id=version_id),),
        idempotency_key="initial-build-1",
    ))
    hits = await application.retrieve_skills(SkillManageRetrieveRequest(
        project_id="my-project", graph_id="my-graph", query="parse a CSV file",
    ))
finally:
    await application.close()
```

`upsert_skill_index` replaces/adds one version; `remove_skill_from_index` removes
a family or version from the index without deleting the canonical Skill.
Persisted writes accept `expected_revision` for optimistic concurrency and an
`idempotency_key` for replay. A classified transient storage failure before
publication permits the same request/key to open a new attempt; committed
requests replay their result. Unknown failures remain non-retryable. Keep the
request unchanged when retrying; request hashes currently follow the DTO shape,
so changing that shape across releases needs an explicit compatibility review.

Inspect operations with `get_skill_manage_operation(operation_id, project_id=...,
graph_id=...)`. Use `repair_skill_index(project_id=..., graph_id=...)` to reconcile
interrupted writes and `maintain_skill_index(project_id=..., graph_id=...)` to
collect retired generations according to configured retention. The default
version loader assumes a single-tenant local repository; project IDs scope the
index and do not provide repository authorization.

Opening the current persistence catalog applies schema v2 even without a graph
algorithm configured. Back up existing databases before upgrading; PostgreSQL
backups are the operator's responsibility. Older code rejects a v2 migration
ledger, so rolling back requires restoring a compatible backup, not deleting
ledger entries. No database is migrated merely by installing this package.

Run graph integration tests only against a disposable database; they reset test
tables. A dedicated local service is provided:

```bash
docker compose -f dockers/docker-compose.skill-manage-test.yml up -d --wait
MINDMEMOS_SKILL_TEST_PG_DSN=postgresql://mindmemos:mindmemos@127.0.0.1:54329/skill_manage_test \
  make test-skill-manage-pg
```

## Skill application

`SkillApplication` is the public lifecycle root. Its async classmethod compiles
configuration, constructs configured model clients, Agents and algorithms, and
owns their database and lifecycle:

```python
from mindmemos_skill import SkillApplication

application = await SkillApplication.from_config(
    {
        "local": {
            "root_dir": "~/.mindmemos/skill",
            "database": {"provider": "sqlite", "path": "state.db"},
        }
    }
)
try:
    skills = await application.list_skills()
finally:
    await application.close()
```

The classmethod also accepts `CompiledSkillApplicationConfig`, allowing an SDK
config loader to compile once and inject the same normalized configuration.
An embedding SDK or application may also pass an optional transport-neutral
`SkillRemotePort` through `from_config(..., remote=adapter)`. The caller owns
that adapter and its HTTP/Auth connection lifecycle; closing `SkillApplication`
does not close the borrowed remote. Without one, all local capabilities remain
available and the package has no HTTP or SDK dependency.
When a remote is present, `await application.push(skill_ref, version_id=None)`
uses a durable deterministic operation ID, records retry/lease state in the
family outbox and validates the immutable acknowledgement against the same
canonical `SKILL.md` hash used locally. The upload is built only from the persisted
immutable single-file bundle (`SKILL.md`); scripts, local resources and
source-tree files are never read or represented by the remote request.
`await application.pull(skill_ref)` reads every remote metadata page, validates
each immutable content hash, orders missing versions parent-first, and commits
the complete import atomically. The cloud bundle replaces the full executable
blob while private local resources are inherited only on the local machine.
Pull does not rewrite immutable version facts or `last_sync_at`.
`await application.sync(skill_ref)` pushes pending versions parent-first, asks
the remote for missing versions and lifecycle revisions, then commits the
imported versions and `last_sync_at` in one transaction. Edge and cloud never
persist an active or head pointer; omitted-version reads use the shared
`(created_at DESC, version_id DESC)` latest-available selector.
Agent execution appends its trajectory attempt automatically. Product-side
trace2skill and evolve runs use `run_trace2skill(...)` and `run_evolve(...)`:
the algorithm orchestrator resolves persisted inputs, dispatches a configured
algorithm by instance name, and applies `dry_run`, `persist`, or
`persist_and_push` commit policy. Non-dry-run runs persist output trajectories,
normalize changed candidates into immutable evolution versions, and append a
result log; components can add detailed step reports through
`record_algorithm_log(...)`.
The script-side experiment layer additionally evaluates the resulting Skill
through the selected dataset's test split and registered environment. The
generic `skill_evaluation` method accepts either one explicit `SKILL.md`/Skill
directory or true no-Skill execution, and writes a common `test/summary.json`,
`results.jsonl`, and `skill.json` artifact set. This runner-level benchmark
lives under `scripts/mindmemos_skill/`, remains outside this package, and does
not change candidate acceptance.

## Low-level local management

`mindmemos_skill.management` owns the local management rules and can run
without the SDK or a cloud connection. `LocalSkillManager.open()` uses the
canonical SQLite database unless a test or embedding application supplies a
different path:

```python
from mindmemos_skill.management import (
    ExportSkillRequest,
    LocalSkillManager,
    PublishSkillRequest,
    RegisterSkillRequest,
)

manager = await LocalSkillManager.open()
registered = await manager.register(
    RegisterSkillRequest(
        source_path="./my-skill",
        alias="my-skill",
        version_label="1.0.0",
    )
)
candidate = await manager.publish(
    PublishSkillRequest(
        skill_ref=registered.skill_id,
        source_path="./my-skill-next",
        version_label="1.1.0",
    )
)
await manager.export(
    ExportSkillRequest(skill_ref=registered.skill_id, target_path="./exported-skill")
)
await manager.close()
```

Registration and publication persist an immutable version and stable pending
push operation in one transaction. Parent versions must already
belong to the same family, version labels are unique and monotonically ordered
as integer triples. Export restores the complete UTF-8 snapshot and preserves files it does not manage;
if replacement fails partway through, overwritten files are restored.

OpenClaw trace detection now lives with its runtime. `OpenClawSkillRuntime`
interprets OpenClaw `read` / `write` / `edit` calls and emits canonical
`SkillBinding` values directly; management does not parse agent messages.

## Environment registry

Built-in benchmark environments are selected by name. `livemath` and the
bounded-history ALFWorld protocol registered as `alfworld_bounded_history` are currently shipped:

```python
from mindmemos_skill.envs import get_env

env = get_env(name=env_name, config=env_params)
```

Future trainers should pass their configured `env_name` and `env_params`
through this factory rather than importing benchmark classes. Packages outside
MindMemOS can participate in the same selection path:

```python
from mindmemos_skill.envs import BaseEnv
from mindmemos_skill.registry import ComponentType, register

@register(type=ComponentType.ENV, name="my_benchmark")
class MyBenchmarkEnv(BaseEnv):
    ...
```

## Development

From the repository root:

```bash
uv sync
```

The import package is `mindmemos_skill`.
