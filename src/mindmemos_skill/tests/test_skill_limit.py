"""Verify unlimited Skill list and history queries retain their ordering."""

from mindmemos_skill.infra.database import Scope, SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence.factory import create_persistence
from mindmemos_skill.skill import Skill, SkillContent


def test_unlimited_repository_list_and_history(tmp_path):
    """Return every matching row when no query limit is supplied."""
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{tmp_path / 'skills.db'}"))
    persistence = create_persistence(database)
    try:
        persistence.initialize_storage()
        with persistence.session(Scope(fields={"project_id": "project"})) as session:
            registered = []
            for name in ("C", "A", "B"):
                registered.append(
                    session.skills.register(
                        project_id="project",
                        name=name,
                        alias=None,
                        source=None,
                        message=None,
                        snapshot=Skill(
                            SkillContent(content=name, name=name, description="Description", standard_format=False)
                        ).snapshot(),
                    )
                )
            families = session.skills.list_skills("project", limit=None, offset=0)
            assert [family.name for family in families] == ["A", "B", "C"]
            assert len(session.skills.list_skills("project", limit=2, offset=0)) == 2
            assert len(session.skills.history(registered[0], limit=None, offset=0)) == 1
            assert session.skills.history(registered[0], limit=0, offset=0) == []
    finally:
        database.close()
