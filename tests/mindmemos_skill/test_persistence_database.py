from __future__ import annotations

import sqlite3

import pytest
from mindmemos_skill.contracts import SkillBundle
from mindmemos_skill.infra.database import DatabaseConfig, DatabaseScope, bootstrap_database
from mindmemos_skill.persistence import (
    SKILL_TABLE,
    TRAJECTORY_TABLE,
    SkillRecord,
    TrajectoryRecord,
    build_persistence_tables,
    from_database_record,
    to_database_record,
)


@pytest.mark.asyncio
async def test_persistence_catalog_round_trips_business_models_through_sqlite(tmp_path) -> None:
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": str(tmp_path / "state.db")}),
        build_persistence_tables(),
    )
    bundle = SkillBundle.from_files({"SKILL.md": "# Research brief"})
    skill = SkillRecord(
        skill_id="skill-1",
        version_id="version-1",
        name="research-brief",
        bundle=bundle.canonical_json(),
        content_hash=bundle.content_hash,
        local_snapshot_hash=bundle.content_hash,
        version_label="1.0.0",
    )

    await database.upsert_records(SKILL_TABLE, (to_database_record(skill),))
    stored = await database.get_records(SKILL_TABLE, DatabaseScope(), ("version-1",))

    assert from_database_record(stored[0], SkillRecord) == skill
    await database.close()


@pytest.mark.asyncio
async def test_persistence_catalog_enforces_one_attempt_number_per_rollout(tmp_path) -> None:
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": str(tmp_path / "state.db")}),
        build_persistence_tables(),
    )
    first = TrajectoryRecord(
        trajectory_id="trajectory-1",
        trajectory_hash="hash-1",
        task_id="task-1",
        rollout_id="rollout-1",
        attempt_no=0,
        task_instruction="Solve the task",
    )
    duplicate_attempt = first.model_copy(update={"trajectory_id": "trajectory-2"})

    await database.upsert_records(TRAJECTORY_TABLE, (to_database_record(first),))
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        await database.upsert_records(TRAJECTORY_TABLE, (to_database_record(duplicate_attempt),))

    await database.close()
