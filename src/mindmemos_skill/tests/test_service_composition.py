"""Verify business operations shared by direct Python and HTTP callers."""

import pytest
from mindmemos_skill.api.typing import CommitRequest, RegisterRequest, ReportTrajectoryRequest, SkillPageRequest
from mindmemos_skill.common.time_utils import utcnow
from mindmemos_skill.skill import Skill, SkillContent

from mindmemos_skill import create_service

from .service_config import repository_config
from .skill_sources import register_detail, structured


def snapshot(name: str, description: str, **metadata):
    """Build one portable Skill source with validated frontmatter."""
    fields = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    content = f"---\nname: {name}\ndescription: {description}\n{fields}\n---\nBody"
    return Skill(SkillContent(content=content)).snapshot()


@pytest.mark.asyncio
async def test_service_validates_source_metadata_for_repository_calls(tmp_path):
    """Direct callers can request the same source checks used by HTTP."""
    async with await create_service(repository_config(tmp_path / "home", "project")) as service:
        source = snapshot("Alpha", "Initial", category="example")
        with pytest.raises(ValueError, match="name.*frontmatter"):
            await register_detail(
                service, RegisterRequest(skill=structured(source, name="Wrong"), project_id=service.default_project_id)
            )
        registered = await register_detail(
            service, RegisterRequest(skill=structured(source, name="Alpha"), project_id=service.default_project_id)
        )
        assert registered.version.description == "Initial"
        assert registered.version.metadata == {"category": "example"}

        committed = await service.commit_skill(
            CommitRequest(
                skill=structured(snapshot("Alpha", "Updated", category="revised")),
                skill_id=registered.skill.skill_id,
                project_id=service.default_project_id,
            )
        )
        assert committed.version.description == "Updated"
        assert committed.version.metadata == {"category": "revised"}


@pytest.mark.asyncio
async def test_service_filters_skill_pages_with_reported_actor_evidence(tmp_path, monkeypatch):
    """Batch actor evidence preserves same-report and snapshot semantics."""
    async with await create_service(repository_config(tmp_path / "home", "project")) as service:
        alpha = await register_detail(
            service,
            RegisterRequest(
                skill=structured(snapshot("Alpha", "A"), name="Alpha"), project_id=service.default_project_id
            ),
        )
        beta = await register_detail(
            service,
            RegisterRequest(
                skill=structured(snapshot("Beta", "B"), name="Beta"), project_id=service.default_project_id
            ),
        )
        await service.report_trajectory(
            ReportTrajectoryRequest(
                project_id=service.default_project_id,
                user_id="user-a",
                agent_id="agent-a",
                skill_version_id=alpha.version.version_id,
                messages=[{"role": "user", "content": "Used Alpha"}],
            )
        )
        snapshot_at = utcnow()
        await service.report_trajectory(
            ReportTrajectoryRequest(
                project_id=service.default_project_id,
                user_id="user-a",
                agent_id="agent-a",
                skill_version_id=beta.version.version_id,
                messages=[{"role": "user", "content": "Used Beta"}],
            )
        )
        database = service._persistence._database
        select_distinct = database.select_distinct
        queries = []

        def track_distinct(table, field, *, filters=None):
            queries.append((table, field))
            return select_distinct(table, field, filters=filters)

        monkeypatch.setattr(database, "select_distinct", track_distinct)
        matching = await service.list_skill_page(
            SkillPageRequest(
                project_id="project",
                name_pattern="A*",
                user_id="user-a",
                agent_id="agent-a",
                limit=1,
                snapshot_at=snapshot_at,
            )
        )
        assert [item.name for item in matching.items] == ["Alpha"]
        assert matching.has_more is False
        assert len(queries) == 1
        before_late_report = await service.list_skill_page(
            SkillPageRequest(project_id="project", user_id="user-a", agent_id="agent-a", snapshot_at=snapshot_at)
        )
        assert [item.skill_id for item in before_late_report.items] == [alpha.skill.skill_id]
        absent = await service.list_skill_page(
            SkillPageRequest(project_id="project", user_id="user-a", agent_id="agent-b", snapshot_at=snapshot_at)
        )
        assert absent.items == []


@pytest.mark.asyncio
async def test_service_queries_reported_ids_in_bounded_batches(tmp_path, monkeypatch):
    """A large candidate page uses bounded set queries instead of one lookup per Skill."""
    async with await create_service(repository_config(tmp_path / "home", "project")) as service:
        registered = await register_detail(
            service,
            RegisterRequest(
                skill=structured(snapshot("Tracked", "A"), name="Tracked"), project_id=service.default_project_id
            ),
        )
        await service.report_trajectory(
            ReportTrajectoryRequest(
                project_id=service.default_project_id,
                user_id="user-a",
                agent_id="agent-a",
                skill_version_id=registered.version.version_id,
                messages=[{"role": "user", "content": "Used Tracked"}],
            )
        )
        candidates = [registered.skill.model_copy(update={"skill_id": f"absent-{index}"}) for index in range(500)] + [
            registered.skill
        ]

        async def list_candidates(_request):
            return candidates

        monkeypatch.setattr(service, "list_skills", list_candidates)
        database = service._persistence._database
        select_distinct = database.select_distinct
        batch_sizes = []

        def track_distinct(table, field, *, filters=None):
            batch_sizes.append(len(next(condition.value for condition in filters if condition.field == "skill_id")))
            return select_distinct(table, field, filters=filters)

        monkeypatch.setattr(database, "select_distinct", track_distinct)
        page = await service.list_skill_page(
            SkillPageRequest(project_id="project", user_id="user-a", limit=1, snapshot_at=utcnow())
        )
        assert [item.skill_id for item in page.items] == [registered.skill.skill_id]
        assert page.has_more is False
        assert batch_sizes == [500, 1]
