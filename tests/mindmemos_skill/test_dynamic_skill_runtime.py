from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest
from mindmemos_skill.contracts import SkillRuntimeSpec, SkillVersionCore
from mindmemos_skill.persistence import SKILL_TABLE, bootstrap_skill_database, to_database_record
from mindmemos_skill.skill_runtime import build_default_skill_runtime_coordinator
from mindmemos_skill.typing import Skill, Task, compute_skill_content_hash


def _skill(*, runtime_type: str = "static", runtime_metadata=None) -> Skill:
    blob = {"SKILL.md": "# Demo\n"}
    return Skill(
        skill_id="skill-1",
        version_id="version-1",
        version_label="0.1.0",
        content_hash=compute_skill_content_hash(blob),
        name="demo",
        blob=blob,
        runtime_type=runtime_type,
        runtime_schema_version=1,
        runtime_metadata=runtime_metadata or {},
        created_at=datetime.now(UTC),
    )


def _virtual_skill() -> Skill:
    return _skill(
        runtime_type="virtual_components",
        runtime_metadata={
            "max_initial_components": 1,
            "components": [
                {
                    "component_id": "inspect",
                    "name": "Inspect workbook",
                    "description": "inspect workbook cells",
                    "content": "Inspect the workbook before editing.\n",
                },
                {
                    "component_id": "verify",
                    "name": "Verify output",
                    "description": "verify saved output",
                    "content": "Verify formulas and save the output.\n",
                },
            ],
        },
    )


def test_runtime_fields_round_trip_through_skill_record_and_wire_contract() -> None:
    skill = _virtual_skill()

    restored = Skill.from_record(skill.to_record())
    wire = SkillVersionCore(
        version_id=skill.version_id,
        parent_version_ids=[],
        name=skill.name,
        content_hash=skill.content_hash,
        version_label=skill.version_label,
        origin="local",
        runtime_type=skill.runtime_type,
        runtime_schema_version=skill.runtime_schema_version,
        runtime_metadata=skill.runtime_metadata,
        created_at=skill.created_at,
        updated_at=skill.created_at,
    )

    assert restored.runtime_type == "virtual_components"
    assert restored.runtime_metadata == skill.runtime_metadata
    assert wire.runtime_metadata == skill.runtime_metadata


def test_static_runtime_rejects_nonempty_metadata() -> None:
    with pytest.raises(ValueError, match="static runtime"):
        SkillRuntimeSpec(runtime_metadata={"unexpected": True})


@pytest.mark.asyncio
async def test_virtual_components_assemble_on_task_and_load_progressively() -> None:
    coordinator = build_default_skill_runtime_coordinator()
    skill = _virtual_skill()
    task = Task(task_id="task-1", instruction="Inspect workbook cells")

    async with coordinator.on_task(task=task, skills=[skill]) as scope:
        session = scope.sessions[0]
        assert "Inspect the workbook" in session.initial_content
        assert "Verify formulas" not in session.initial_content
        assert len(session.resources) == 1

        payload = await scope.load(session.resources[0].resource_id)

        assert payload.content == "Verify formulas and save the output.\n"
        assert scope.trace()["skills"][0]["loaded_resource_ids"] == [payload.resource_id]


@pytest.mark.asyncio
async def test_filesystem_projection_materializes_lazy_resources_without_marking_them_loaded() -> None:
    coordinator = build_default_skill_runtime_coordinator()
    task = Task(task_id="task-1", instruction="Inspect workbook cells")

    async with coordinator.on_task(task=task, skills=[_virtual_skill()]) as scope:
        projected = await scope.projected_skills(materialize_resources=True)

        resource_paths = [path for path in projected[0].resources if path.startswith("runtime_resources/")]
        assert resource_paths == ["runtime_resources/001-Verify_output.md"]
        assert projected[0].resources[resource_paths[0]] == "Verify formulas and save the output.\n"
        assert resource_paths[0] in projected[0].content
        assert scope.trace()["skills"][0]["loaded_resource_ids"] == []


@pytest.mark.asyncio
async def test_runtime_rejects_unknown_type_and_schema_version() -> None:
    coordinator = build_default_skill_runtime_coordinator()
    task = Task(task_id="task-1", instruction="demo")
    unknown = _skill().model_copy(update={"runtime_type": "not_installed"})
    unsupported = _virtual_skill().model_copy(update={"runtime_schema_version": 2})

    with pytest.raises(Exception, match="unavailable"):
        async with coordinator.on_task(task=task, skills=[unknown]):
            pass
    with pytest.raises(ValueError, match="does not support schema version 2"):
        async with coordinator.on_task(task=task, skills=[unsupported]):
            pass


@pytest.mark.asyncio
async def test_local_v1_table_persists_runtime_as_first_class_columns(tmp_path) -> None:
    path = tmp_path / "state.db"
    database = await bootstrap_skill_database(path)
    record = _virtual_skill().to_record()
    await database.upsert_records(SKILL_TABLE, (to_database_record(record),))
    await database.close()

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT runtime_type, runtime_schema_version, runtime_metadata FROM skill_versions WHERE version_id = ?",
            (record.version_id,),
        ).fetchone()

    assert row is not None
    assert row[:2] == ("virtual_components", 1)
    assert json.loads(row[2]) == record.runtime_metadata
