"""Soft deletion preserves storage while hiding active Skill and graph identities."""

from tempfile import TemporaryDirectory

import pytest
from mindmemos_skill.errors import GraphNotFoundError, SkillNotFoundError
from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence.constants import SKILL_GRAPHS_TABLE, SKILL_VERSIONS_TABLE, SKILLS_TABLE
from mindmemos_skill.persistence.db_services.skill_management.service import GraphQueryService
from mindmemos_skill.persistence.db_services.skill_management.typing import SkillGraphRecord
from mindmemos_skill.persistence.factory import create_persistence
from mindmemos_skill.skill import Skill, SkillContent


def test_soft_delete_preserves_rows_and_survives_repeat_migration():
    with TemporaryDirectory() as directory:
        persistence = create_persistence(SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{directory}/data.db")))
        persistence.initialize_storage()
        with persistence.session(Scope(fields={"project_id": "p"})) as session:
            skill = session.skills.register(
                project_id="p",
                name="Test",
                alias=None,
                source=None,
                message=None,
                snapshot=Skill(
                    SkillContent(content="hello", name="Test", description="Description", standard_format=False)
                ).snapshot(),
            )
            session.database.insert(
                SKILL_GRAPHS_TABLE,
                [SkillGraphRecord(project_id="p", graph_id="g", name="Graph").model_dump()],
            )
            graphs = GraphQueryService(session.database, "p")
            session.skills.delete(skill)
            graphs.delete("g")
            assert session.skills.list_skills("p", limit=10, offset=0) == []
            assert graphs.list_graphs() == []
            with pytest.raises(SkillNotFoundError):
                session.skills.resolve_family("p", skill.skill_id)
            with pytest.raises(GraphNotFoundError):
                graphs.graph("g")
            assert session.database.count(SKILL_VERSIONS_TABLE, filters={"skill_id": skill.skill_id}) == 1
            assert session.database.get(SKILLS_TABLE, {"project_id": "p", "skill_id": skill.skill_id})["deleted_at"]
            assert session.database.get(SKILL_GRAPHS_TABLE, {"project_id": "p", "graph_id": "g"})["deleted_at"]
        persistence.initialize_storage()
        with persistence.session(Scope(fields={"project_id": "p"})) as session:
            assert session.skills.list_skills("p", limit=10, offset=0) == []
            assert GraphQueryService(session.database, "p").list_graphs() == []
