"""Reconcile the current legacy Skill data into the independent service."""

from collections import defaultdict
from datetime import timedelta

from mindmemos.config import QdrantConfig
from mindmemos.infra.db.engine import QdrantEngine
from mindmemos.infra.db.filters import is_empty, match_value
from mindmemos.infra.db.models import QdrantRecord
from mindmemos.typing.skill import SkillTraceSummary
from mindmemos_skill.api.typing import LegacyBinding
from mindmemos_skill.common.time_utils import utcnow
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.infra.database.scoped import ScopedDatabase
from mindmemos_skill.persistence.constants import (
    LEGACY_SKILL_SYNC_ITEMS,
    LEGACY_SKILL_SYNC_PROJECTS,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
)
from mindmemos_skill.persistence.db_services.skills.typing import SkillRecord, SkillVersionRecord
from mindmemos_skill.persistence.session import SkillPersistenceSession
from mindmemos_skill.persistence.store import SkillPersistence
from mindmemos_skill.persistence.typing import LegacySkillSyncItem, LegacySkillSyncProject
from qdrant_client import models as qmodels

from .migrate import (
    SOURCE,
    MigrationError,
    _convert_family,
    _diagnose,
    _display_value,
    _scan,
    _trajectory,
    _verify_version,
)
from .typing import MigrationFailure, ProjectMigration

LEASE_DURATION = timedelta(minutes=5)


async def _item(tx: ScopedDatabase, item: LegacySkillSyncItem) -> None:
    """Commit the ID mapping and latest source state with the corresponding write."""
    await tx.upsert(
        LEGACY_SKILL_SYNC_ITEMS,
        [item.model_dump()],
        update_fields=["target_id", "generation", "pending", "source_deleted", "published_version_id"],
    )


async def _claim(database: ScopedDatabase, project: str) -> LegacySkillSyncProject | None:
    """Fence stale workers with a monotonically increasing project generation."""
    now = utcnow()
    await database.upsert(LEGACY_SKILL_SYNC_PROJECTS, [LegacySkillSyncProject(project_id=project).model_dump()])
    row = LegacySkillSyncProject.model_validate(await database.get(LEGACY_SKILL_SYNC_PROJECTS, project))
    if row.lease_until is not None and row.lease_until > now:
        return None
    generation = row.generation + 1
    expires = now + LEASE_DURATION
    claimed = await database.update(
        LEGACY_SKILL_SYNC_PROJECTS,
        {"generation": generation, "lease_until": expires},
        filters={"project_id": project, "generation": row.generation, "lease_until": row.lease_until},
    )
    return row.model_copy(update={"generation": generation, "lease_until": expires}) if claimed else None


async def _fence(tx: ScopedDatabase, job: LegacySkillSyncProject, *, progress: dict | None = None) -> None:
    """Lock the current lease in the same transaction as the replicated rows."""
    values = {"lease_until": utcnow() + LEASE_DURATION} if progress is None else progress
    updated = await tx.update(
        LEGACY_SKILL_SYNC_PROJECTS, values, filters={"project_id": job.project_id, "generation": job.generation}
    )
    if not updated:
        raise MigrationError("lease_lost", "Another worker took over the project")


async def _save_family(
    database: ScopedDatabase,
    job: LegacySkillSyncProject,
    legacy_id: str,
    family: SkillRecord,
    versions: dict[str, SkillVersionRecord],
) -> tuple[int, int]:
    """Append validated versions and update the selected pointer for mirror families."""
    async with database.transaction() as tx:
        await _fence(tx, job)
        key = {"project_id": family.project_id, "skill_id": family.skill_id}
        previous = await tx.get(SKILLS_TABLE, {"project_id": family.project_id, "skill_id": legacy_id})
        if previous is not None:
            raise MigrationError("old_id_import", "An existing family already uses the legacy ID")
        existing = await tx.get(SKILLS_TABLE, key)
        if existing is not None and existing["registration_source"] != SOURCE:
            raise MigrationError("target_skill_conflict", "Target identity is owned by a native Skill")
        mapping = await tx.get(
            LEGACY_SKILL_SYNC_ITEMS, {"project_id": family.project_id, "kind": "family", "legacy_id": legacy_id}
        )
        if existing is not None and mapping is None and existing["native_owned_at"] is None:
            # The snapshot importer did not record whether a native edit followed it.
            await tx.update(SKILLS_TABLE, {"native_owned_at": utcnow()}, filters=key | {"native_owned_at": None})
        labels = {
            row["version_label"]
            for row in await tx.select(SKILL_VERSIONS_TABLE, filters=key, fields=["version_label"], limit=None)
        }
        new_versions = 0
        for old_id, version in versions.items():
            row = await tx.get(SKILL_VERSIONS_TABLE, version.version_id)
            if row is None:
                if version.version_label in labels:
                    # An unlabeled history entry still retains its original legacy label in provenance.
                    version = version.model_copy(update={"version_label": None})
                await tx.insert(SKILL_VERSIONS_TABLE, [version.model_dump()])
                labels.add(version.version_label)
                new_versions += 1
            else:
                _verify_version(SkillVersionRecord.model_validate(row), version)
                if row["status"] != version.status:
                    await tx.update(
                        SKILL_VERSIONS_TABLE,
                        {"status": version.status},
                        filters=key | {"version_id": version.version_id, "origin": SOURCE},
                    )
            await _item(
                tx,
                LegacySkillSyncItem(
                    project_id=job.project_id,
                    kind="version",
                    legacy_id=old_id,
                    target_id=version.version_id,
                    generation=job.generation,
                ),
            )

        if existing is None:
            await tx.insert(SKILLS_TABLE, [family.model_dump()])
        elif (
            existing["active_version_id"] != family.active_version_id
            or existing["is_deleted"]
            or existing["name"] != family.name
            or existing["metadata"] != family.metadata
        ):
            await tx.update(
                SKILLS_TABLE,
                {
                    "active_version_id": family.active_version_id,
                    "name": family.name,
                    "metadata": family.metadata,
                    "is_deleted": False,
                    "deleted_at": None,
                    "updated_at": max(existing["updated_at"], utcnow()),
                },
                filters=key | {"native_owned_at": None},
            )
        published = next(
            (old for old, version in versions.items() if version.version_id == family.active_version_id), None
        )
        await _item(
            tx,
            LegacySkillSyncItem(
                project_id=job.project_id,
                kind="family",
                legacy_id=legacy_id,
                target_id=family.skill_id,
                generation=job.generation,
                published_version_id=published,
            ),
        )
        return int(existing is None), new_versions


async def _reconcile_deletions(
    source: QdrantEngine, config: QdrantConfig, database: ScopedDatabase, job: LegacySkillSyncProject
) -> None:
    """Confirm missing families against the source before hiding their mirrors."""
    rows = await database.select(
        LEGACY_SKILL_SYNC_ITEMS, filters={"kind": "family", "source_deleted": False}, limit=None
    )
    for row in rows:
        item = LegacySkillSyncItem.model_validate(row)
        if item.generation == job.generation:
            continue
        selection = source.project_filter(job.project_id, conditions=[match_value("cloud_skill_id", item.legacy_id)])
        found, _ = await source.scroll(config.skill_version_collection, scroll_filter=selection, limit=1)
        if found:
            continue
        async with database.transaction() as tx:
            await _fence(tx, job)
            now = utcnow()
            await tx.update(
                SKILLS_TABLE,
                {"is_deleted": True, "deleted_at": now, "updated_at": now},
                filters={"skill_id": item.target_id, "native_owned_at": None, "is_deleted": False},
            )
            await _item(tx, item.model_copy(update={"source_deleted": True, "published_version_id": None}))


async def sync_project(
    source: QdrantEngine, config: QdrantConfig, target: SkillPersistence, project: str, *, page_size: int
) -> ProjectMigration:
    """Perform one resumable reconciliation pass for an authorized namespace."""
    result = ProjectMigration(project_id=project)
    with target.session(Scope(fields={"project_id": project})) as session:
        job = await _claim(session.database, project)
        if job is None:
            return result
        try:
            await _sync_project(source, config, session, job, result, page_size)
        finally:
            # A replacement worker may already own the lease after an interruption.
            await session.database.update(
                LEGACY_SKILL_SYNC_PROJECTS,
                {"lease_until": None},
                filters={"project_id": project, "generation": job.generation},
            )
    return result


async def _sync_project(
    source: QdrantEngine,
    config: QdrantConfig,
    session: SkillPersistenceSession,
    job: LegacySkillSyncProject,
    result: ProjectMigration,
    page_size: int,
) -> None:
    """Read current source data and checkpoint each applied trajectory page."""
    project = job.project_id
    counts = result.counts
    families: dict[str | None, list[QdrantRecord]] = defaultdict(list)
    selection = source.project_filter(project)
    async for point in _scan(source, config.skill_version_collection, selection, page_size):
        families[_display_value(point.payload.get("cloud_skill_id"))].append(point)
    counts.families = len(families)
    counts.versions = sum(map(len, families.values()))
    known = {point.point_id for points in families.values() for point in points}
    registered: dict[str, SkillVersionRecord] = {}
    for legacy_id, points in families.items():
        try:
            if legacy_id is None:
                raise MigrationError("missing_skill_id", "Registered Skill version has no family ID")
            family, versions = await _convert_family(source, config, points, project, published_only=True)
            new_families, new_versions = await _save_family(session.database, job, legacy_id, family, versions)
        except Exception as error:
            reason, detail = _diagnose(error)
            result.failed_skills.append(
                MigrationFailure(project_id=project, skill_id=legacy_id, reason=reason, detail=detail)
            )
        else:
            registered.update(versions)
            counts.verified_families += 1
            counts.new_families += new_families
            counts.new_versions += new_versions
    await _reconcile_deletions(source, config, session.database, job)

    cursor = job.trajectory_cursor
    failures = job.failures if cursor is not None else 0
    failures += len(result.failed_skills)
    selection = source.project_filter(project, filter_=qmodels.Filter(must_not=[is_empty("skill_bindings")]))
    has_summaries = await source.collection_exists(config.skill_trace_summary_collection)
    while True:
        points, cursor = await source.scroll(
            config.add_record_collection, scroll_filter=selection, limit=page_size, offset=cursor
        )
        summaries = {}
        if points and has_summaries:
            summaries = {
                row.point_id: row.payload
                for row in await source.retrieve(
                    config.skill_trace_summary_collection, [point.point_id for point in points]
                )
                if row.payload.get("project_id") == project
            }
        for point in points:
            counts.trajectories += 1
            try:
                record = _trajectory(point, registered, known, counts, project)
                if record is not None and point.point_id in summaries:
                    summary = SkillTraceSummary.model_validate(summaries[point.point_id])
                    if summary.add_record_id != point.point_id:
                        raise MigrationError("invalid_summary_identity", "Legacy summary belongs to another add record")
                    provenance = record.metadata[SOURCE]
                    if isinstance(provenance, dict):
                        provenance["summary"] = summary.model_dump(mode="json")
                pending = any(
                    LegacyBinding.model_validate(binding).version_id not in registered
                    for binding in point.payload.get("skill_bindings") or []
                )
                inserted = 0
                async with session.transaction() as tx:
                    await _fence(tx.database, job)
                    key = {"project_id": project, "kind": "trajectory", "legacy_id": point.point_id}
                    existing = await tx.database.get(LEGACY_SKILL_SYNC_ITEMS, key)
                    target_id = existing["target_id"] if existing is not None else None
                    if record is not None:
                        inserted = await tx.trajectories.replicate_legacy(record)
                        target_id = record.trajectory_id
                    await _item(
                        tx.database,
                        LegacySkillSyncItem(
                            project_id=project,
                            kind="trajectory",
                            legacy_id=point.point_id,
                            target_id=target_id,
                            generation=job.generation,
                            pending=pending,
                        ),
                    )
            except Exception as error:
                reason, detail = _diagnose(error)
                result.failed_trajectories.append(
                    MigrationFailure(project_id=project, trajectory_id=point.point_id, reason=reason, detail=detail)
                )
                failures += 1
            else:
                counts.new_trajectories += inserted
                counts.verified_trajectories += 1
        progress = {"trajectory_cursor": cursor, "failures": failures, "lease_until": utcnow() + LEASE_DURATION}
        if cursor is None:
            progress.update(ready=job.ready or not failures, last_checked_at=utcnow())
        async with session.database.transaction() as tx:
            await _fence(tx, job, progress=progress)
        if cursor is None:
            return
