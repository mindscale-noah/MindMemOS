"""Soft deletion preserves storage while hiding active Skill and graph identities."""

from tempfile import TemporaryDirectory

import pytest
from mindmemos_skill.errors import GraphNotFoundError, SkillNotFoundError
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence.constants import SKILL_GRAPHS_TABLE, SKILL_VERSIONS_TABLE, SKILLS_TABLE
from mindmemos_skill.persistence.db_services.graphs.typing import SkillGraphRecord
from mindmemos_skill.persistence.factory import create_persistence
from mindmemos_skill.skill import Skill, SkillContent


async def test_soft_delete_preserves_rows_and_survives_repeat_initialization():
    with TemporaryDirectory() as directory:
        database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{directory}/data.db"))
        persistence = create_persistence(database)
        (await persistence.initialize_storage())
        with persistence.session(Scope(fields={"project_id": "p"})) as session:
            skill = await session.skills.register(
                name="Test",
                alias=None,
                source=None,
                message=None,
                snapshot=Skill(
                    SkillContent(content="hello", name="Test", description="Description", standard_format=False)
                ).snapshot(),
            )
            (
                await session.database.insert(
                    SKILL_GRAPHS_TABLE,
                    [SkillGraphRecord(project_id="p", graph_id="g", name="Graph").model_dump()],
                )
            )
            graphs = session.graphs
            (await session.skills.delete(skill))
            await graphs.delete("g")
            assert (await session.skills.list_skills(limit=10, offset=0)) == []
            assert await graphs.list_graphs() == []
            with pytest.raises(SkillNotFoundError):
                (await session.skills.resolve_family(skill.skill_id))
            with pytest.raises(GraphNotFoundError):
                await graphs.get_graph("g")
            assert (await session.database.count(SKILL_VERSIONS_TABLE, filters={"skill_id": skill.skill_id})) == 1
            assert (await session.database.get(SKILLS_TABLE, {"project_id": "p", "skill_id": skill.skill_id}))[
                "deleted_at"
            ]
            assert (await session.database.get(SKILL_GRAPHS_TABLE, {"project_id": "p", "graph_id": "g"}))["deleted_at"]
        (await persistence.initialize_storage())
        with persistence.session(Scope(fields={"project_id": "p"})) as session:
            assert (await session.skills.list_skills(limit=10, offset=0)) == []
            assert (await session.graphs.list_graphs()) == []

        await database.close()
