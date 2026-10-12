"""Version namespaces, database uniqueness, concurrency, and legacy migration."""

import asyncio

import pytest
import pytest_asyncio
from mindmemos_skill.api.typing import ActiveRequest, HistoryRequest, ShowRequest
from mindmemos_skill.errors import SkillConflictError
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.persistence.constants import SKILL_VERSIONS_TABLE
from mindmemos_skill.persistence.db_services.skills.typing import VersionNaming
from mindmemos_skill.skill import Skill, SkillContent
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from mindmemos_skill import create_service

from .service_config import repository_config


def snapshot(text="content"):
    return Skill(
        SkillContent(content=text, name="test-skill", description="Test instructions", standard_format=False)
    ).snapshot()


@pytest_asyncio.fixture
async def service(tmp_path):
    async with await create_service(repository_config(tmp_path / "home", "project")) as service:
        yield service


async def register(service, **options):
    with service.repository.persistence.session(Scope(fields={"project_id": service.default_project_id})) as scoped:
        family = await scoped.skills.register(
            project_id=service.default_project_id,
            name="Example",
            alias=None,
            source=None,
            message=None,
            snapshot=snapshot(),
            naming=VersionNaming(**options),
        )
    return await service.show_skill(ShowRequest(project_id=service.default_project_id, skill_id=family.skill_id))


async def commit(service, first, **options):
    with service.repository.persistence.session(Scope(fields={"project_id": service.default_project_id})) as scoped:
        owner = await scoped.skills.resolve_family(service.default_project_id, first.skill.skill_id)
        (await scoped.skills.commit(owner, snapshot("new"), message="Version test", naming=VersionNaming(**options)))
    return await service.show_skill(ShowRequest(project_id=service.default_project_id, skill_id=first.skill.skill_id))


@pytest.mark.asyncio
async def test_user_and_algorithm_numbering_and_history(service):
    first = await register(service, version_label="v1.2.0")
    algorithm = await commit(service, first, algorithm="treeskill")
    assert algorithm.version.version_label == "v1.2.1-treeskill"
    await service.active_skill(
        ActiveRequest(project_id=service.default_project_id, version_id=first.version.version_id)
    )
    human = await commit(service, first)
    assert human.version.version_label == "v1.2.1"
    again = await commit(service, first, algorithm="treeskill")
    assert again.version.version_label == "v1.2.2-treeskill"
    await service.active_skill(
        ActiveRequest(project_id=service.default_project_id, version_id=first.version.version_id)
    )
    assert (await commit(service, first, algorithm="treeskill")).version.version_label == "v1.2.3-treeskill"
    assert (await commit(service, first, version_bump="minor")).version.version_label == "v1.3.0"
    assert (await commit(service, first, version_bump="major")).version.version_label == "v2.0.0"
    with pytest.raises(SkillConflictError, match="already in use"):
        await commit(service, first, version_label="v1.2.0")
    assert (await register(service, version_label="v1.2.0")).version.version_label == "v1.2.0"
    assert (await register(service, algorithm="grpo")).version.version_label == "v1.0.0-grpo"
    assert (
        await service.show_skill(
            ShowRequest(
                project_id=service.default_project_id,
                skill_id=first.skill.skill_id,
                version_id=first.version.version_id,
            )
        )
    ).version == first.version


@pytest.mark.parametrize(
    "label", ["v1", "1.2.0", "v01.2.0", "v1.02.0", "v1.2.00", "v-1.0.0", "v1.2.0-treeskill", "v1.2.0+build", "v1.2.0 "]
)
def test_invalid_manual_labels_are_rejected(label):
    with pytest.raises(ValidationError):
        VersionNaming(version_label=label)


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
        async with await create_service(repository_config(tmp_path / "home", "project")) as writer:
            with writer.repository.persistence.session(
                Scope(fields={"project_id": writer.default_project_id})
            ) as scoped:
                owner = await scoped.skills.resolve_family(writer.default_project_id, first.skill.skill_id)
            await asyncio.wait_for(barrier.wait(), timeout=10)
            try:
                with writer.repository.persistence.session(
                    Scope(fields={"project_id": writer.default_project_id})
                ) as scoped:
                    family = await scoped.skills.commit(
                        owner, snapshot(), message="Concurrent version", naming=VersionNaming(algorithm=algorithm)
                    )
                    return (await scoped.skills.version(family)).version_label
            except SkillConflictError:
                return "conflict"

    results = await asyncio.gather(publish(None), publish("treeskill"))
    assert results.count("conflict") == 1
    versions = await service.skill_history(
        HistoryRequest(project_id=service.default_project_id, skill_id=first.skill.skill_id)
    )
    assert len(versions) == 2
    assert len({version.version_label for version in versions}) == 2


@pytest.mark.asyncio
async def test_database_uniqueness(service):
    first = await register(service)
    database = service.backends["database"]
    duplicate = first.version.model_copy(update={"version_id": "another-id"})
    with pytest.raises(IntegrityError):
        (await database.insert(SKILL_VERSIONS_TABLE, [duplicate.model_dump()]))
