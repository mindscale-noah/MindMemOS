"""Upgrade an existing database after removal of cloud synchronization."""

from mindmemos_skill.infra.database import SQLiteConfig, SQLiteDatabase
from mindmemos_skill.persistence import create_persistence, ensure_collection
from mindmemos_skill.persistence.constants import SKILL_VERSIONS_TABLE, SKILLS_TABLE


def test_retired_remote_tables_are_removed_without_changing_core_data(tmp_path):
    import sqlite3

    path = tmp_path / "retired.db"
    database = SQLiteDatabase(SQLiteConfig(url=f"sqlite:///{path}"))
    persistence = create_persistence(database)
    try:
        ensure_collection(persistence)
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE remote_skill_bindings_v1 (id TEXT PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE remote_version_bindings_v1 (id TEXT REFERENCES remote_skill_bindings_v1(id))"
            )
            connection.execute("CREATE TABLE remote_trajectory_bindings_v1 (id TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE remote_operations_v1 (id TEXT PRIMARY KEY)")
            connection.execute("INSERT INTO remote_skill_bindings_v1 VALUES ('retired')")
            connection.execute("INSERT INTO remote_version_bindings_v1 VALUES ('retired')")
            connection.execute("CREATE TABLE remote_operations (id TEXT PRIMARY KEY)")
            connection.execute("DROP TABLE remote_trajectory_bindings_v1")
        ensure_collection(persistence)
        ensure_collection(persistence)
        with sqlite3.connect(path) as connection:
            names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert not any(name.startswith("remote_") for name in names)
        assert SKILL_VERSIONS_TABLE in names
        assert SKILLS_TABLE in names
    finally:
        database.close()
