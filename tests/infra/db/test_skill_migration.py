from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.components.skill import serialize_bundle
from mindmemos.infra.db import LegacyCloudSkillMigrator, SkillRelationalRepository, build_cloud_skill_tables
from mindmemos_skill.infra.database import DatabaseConfig, bootstrap_database


class _LegacyRepository:
    def __init__(self) -> None:
        created_at = datetime(2026, 8, 7, tzinfo=UTC).isoformat()
        self._versions = [
            SimpleNamespace(
                payload={
                    "version_id": "legacy-child",
                    "cloud_skill_id": "legacy-family",
                    "parent_version_id": "legacy-root",
                    "skill_name": "legacy",
                    "content_hash": "old-child-hash",
                    "version_label": "1.0.1",
                    "status": "published",
                    "origin": "cloud",
                    "created_at": created_at,
                }
            ),
            SimpleNamespace(
                payload={
                    "version_id": "legacy-root",
                    "cloud_skill_id": "legacy-family",
                    "parent_version_id": None,
                    "skill_name": "legacy",
                    "content_hash": "old-root-hash",
                    "version_label": "1.0.0",
                    "status": "observed",
                    "origin": "edge",
                    "created_at": created_at,
                }
            ),
        ]
        self._blobs = {
            "old-root-hash": SimpleNamespace(payload={"content": serialize_bundle({"SKILL.md": "root"})}),
            "old-child-hash": SimpleNamespace(payload={"content": serialize_bundle({"SKILL.md": "child"})}),
        }

    async def list_versions(self, _project_id, *, limit, cursor):
        assert limit > 0
        return (self._versions, None) if cursor is None else ([], None)

    async def get_blob(self, _project_id, content_hash):
        return self._blobs.get(content_hash)


@pytest.mark.asyncio
async def test_qdrant_v1_backfill_is_topological_restartable_and_ignores_heads():
    database = await bootstrap_database(
        DatabaseConfig(provider="sqlite", options={"path": ":memory:"}),
        build_cloud_skill_tables(),
    )
    repository = SkillRelationalRepository(database)
    migrator = LegacyCloudSkillMigrator(_LegacyRepository(), repository)
    try:
        first = await migrator.migrate_project("project")
        second = await migrator.migrate_project("project")

        assert (first.scanned, first.migrated, first.replayed) == (2, 2, 0)
        assert (second.scanned, second.migrated, second.replayed) == (2, 0, 2)
        child = await repository.get_version("project", "legacy-child")
        assert child.parent_version_ids == ["legacy-root"]
        assert child.status.value == "published"
        assert child.origin.value == "cloud"
        assert child.metadata == {"migration": {"source": "qdrant-skill-v1"}}
        assert "head" not in child.model_dump()
        assert "effective" not in child.model_dump()
    finally:
        await database.close()
