"""Version namespaces, database uniqueness, concurrency, and legacy migration."""

import asyncio

import pytest
import pytest_asyncio
from mindmemos_skill.errors import SkillConflictError
from mindmemos_skill.persistence.constants import SKILL_VERSION_LABEL_INDEX, SKILL_VERSIONS_TABLE
from mindmemos_skill.persistence.db_services.skills.migrations import migrate_version_labels
from mindmemos_skill.persistence.db_services.skills.typing import VersionNaming
from mindmemos_skill.pipelines.typing import CommitRequest, HistoryRequest, RegisterRequest, ShowRequest, SwitchRequest
from mindmemos_skill.skill import Skill, SkillBundle
from pydantic import ValidationError
from sqlalchemy import event, inspect, select, update
from sqlalchemy.exc import IntegrityError

from mindmemos_skill import create_service


def snapshot(text="content"):
    return Skill(bundle=SkillBundle(files={"SKILL.md": text})).snapshot()


@pytest_asyncio.fixture
async def service(tmp_path):
    async with await create_service(home=tmp_path / "home") as service:
        yield service


async def register(service, **options):
    return await service.register_skill(RegisterRequest(name="Example", source=snapshot(), **options))


async def commit(service, first, **options):
    return await service.commit_skill(CommitRequest(skill_ref=first.skill.skill_id, source=snapshot("new"), **options))


@pytest.mark.asyncio
async def test_user_and_algorithm_numbering_and_history(service):
    first = await register(service, version_label="v1.2.0")
    algorithm = await commit(service, first, algorithm="treeskill")
    assert algorithm.version.version_label == "v1.2.1-treeskill"
    await service.switch_skill(SwitchRequest(skill_ref=first.skill.skill_id, version_id=first.version.version_id))
    human = await commit(service, first)
    assert human.version.version_label == "v1.2.1"
    again = await commit(service, first, algorithm="treeskill")
    assert again.version.version_label == "v1.2.2-treeskill"
    await service.switch_skill(SwitchRequest(skill_ref=first.skill.skill_id, version_id=first.version.version_id))
    assert (await commit(service, first, algorithm="treeskill")).version.version_label == "v1.2.3-treeskill"
    assert (await commit(service, first, version_bump="minor")).version.version_label == "v1.3.0"
    assert (await commit(service, first, version_bump="major")).version.version_label == "v2.0.0"
    with pytest.raises(SkillConflictError, match="already in use"):
        await commit(service, first, version_label="v1.2.0")
    assert (await register(service, version_label="v1.2.0")).version.version_label == "v1.2.0"
    assert (await register(service, algorithm="grpo")).version.version_label == "v1.0.0-grpo"
    assert (
        await service.show_skill(ShowRequest(skill_ref=first.skill.skill_id, version_id=first.version.version_id))
    ).version == first.version


@pytest.mark.parametrize(
    "label", ["v1", "1.2.0", "v01.2.0", "v1.02.0", "v1.2.00", "v-1.0.0", "v1.2.0-treeskill", "v1.2.0+build", "v1.2.0 "]
)
def test_invalid_manual_labels_are_rejected(label):
    with pytest.raises(ValidationError):
        RegisterRequest(name="Example", source=snapshot(), version_label=label)


@pytest.mark.parametrize(
    "options",
    [
        {"algorithm": "TreeSkill"},
        {"algorithm": "../escape"},
        {"algorithm": "grpo", "version_label": "v1.0.0"},
        {"algorithm": "grpo", "version_bump": "major"},
        {"version_label": "v1.0.0", "version_bump": "minor"},
    ],
)
def test_conflicting_modes_are_rejected(options):
    with pytest.raises(ValidationError):
        VersionNaming(**options)


@pytest.mark.asyncio
async def test_concurrent_commit_does_not_duplicate_or_leave_orphans(service, tmp_path):
    first = await register(service)
    barrier = asyncio.Barrier(2)

    async def publish(algorithm):
        async with await create_service(home=tmp_path / "home") as writer:
            await asyncio.wait_for(barrier.wait(), timeout=10)
            try:
                return (
                    await writer.commit_skill(
                        CommitRequest(
                            skill_ref=first.skill.skill_id,
                            source=snapshot(),
                            base_version_id=first.version.version_id,
                            algorithm=algorithm,
                        )
                    )
                ).version.version_label
            except SkillConflictError:
                return "conflict"

    results = await asyncio.gather(publish(None), publish("treeskill"))
    assert results.count("conflict") == 1
    versions = await service.skill_history(HistoryRequest(skill_ref=first.skill.skill_id))
    assert len(versions) == 2
    assert len({version.version_label for version in versions}) == 2


@pytest.mark.asyncio
async def test_database_uniqueness(service):
    first = await register(service)
    database = service.repository.persistence.database
    duplicate = first.version.model_copy(update={"version_id": "another-id"})
    with pytest.raises(IntegrityError):
        database.insert(SKILL_VERSIONS_TABLE, [duplicate.model_dump()])


@pytest.mark.asyncio
async def test_migration_preserves_data_rolls_back_and_repeats(service):
    first = await register(service)
    for _ in range(3):
        await commit(service, first)
    database = service.repository.persistence.database
    table = database.tables.get(SKILL_VERSIONS_TABLE).table
    with database.engine.begin() as connection:
        index = next(index for index in table.indexes if index.name == SKILL_VERSION_LABEL_INDEX)
        index.drop(connection)
        rows = connection.execute(select(table).order_by(table.c.created_at, table.c.version_id)).mappings().all()
        old_labels = ["old release", "v2.0.0", "v2.0.0", None]
        for row, label in zip(rows, old_labels):
            connection.execute(
                update(table)
                .where(table.c.version_id == row["version_id"])
                .values(version_label=label, local_metadata={"user_note": "keep"})
            )
    before = database.select(SKILL_VERSIONS_TABLE, limit=100)
    pointer = (await service.show_skill(ShowRequest(skill_ref=first.skill.skill_id))).skill.current_version_id
    writes = []

    def interrupt(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE"):
            writes.append(statement)
            if len(writes) == 3:
                raise RuntimeError("simulated interruption")

    event.listen(database.engine, "before_cursor_execute", interrupt)
    try:
        with pytest.raises(RuntimeError, match="simulated interruption"):
            migrate_version_labels(database)
    finally:
        event.remove(database.engine, "before_cursor_execute", interrupt)
    assert database.select(SKILL_VERSIONS_TABLE, limit=100) == before
    assert SKILL_VERSION_LABEL_INDEX not in {
        index["name"] for index in inspect(database.engine).get_indexes(table.name)
    }
    migrate_version_labels(database)
    after = database.select(SKILL_VERSIONS_TABLE, limit=100)
    assert len({row["version_label"] for row in after}) == 4
    assert sum(row["version_label"] == "v2.0.0" for row in after) == 1
    for old, new in zip(before, after):
        assert {k: v for k, v in old.items() if k not in {"version_label", "local_metadata"}} == {
            k: v for k, v in new.items() if k not in {"version_label", "local_metadata"}
        }
        assert new["local_metadata"]["user_note"] == "keep"
        if old["version_label"] != new["version_label"]:
            assert new["local_metadata"]["version_label_migration_v1"]["previous_label"] == old["version_label"]
    migrate_version_labels(database)
    assert database.select(SKILL_VERSIONS_TABLE, limit=100) == after
    assert (await service.show_skill(ShowRequest(skill_ref=first.skill.skill_id))).skill.current_version_id == pointer


@pytest.mark.asyncio
async def test_startup_migrates_late_legacy_writes_and_keeps_existing_aliases(service, tmp_path):
    first = await register(service, version_label="v1.2.0")
    database = service.repository.persistence.database
    legacy = first.version.model_copy(update={"version_id": "late-legacy", "version_label": None})
    database.insert(SKILL_VERSIONS_TABLE, [legacy.model_dump()])
    async with await create_service(home=tmp_path / "home") as reopened:
        rows = await reopened.skill_history(HistoryRequest(skill_ref=first.skill.skill_id))
        labels = {row.version_id: row.version_label for row in rows}
        assert labels[first.version.version_id] == "v1.2.0"
        assert labels["late-legacy"] == "v1.0.0"
    with pytest.raises(ValidationError):
        RegisterRequest(name="Example", source=snapshot(), version_bump="major")
