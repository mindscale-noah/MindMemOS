"""Copy legacy project data without changing the Memory or Skill API contracts."""

import json
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from graphlib import CycleError, TopologicalSorter
from uuid import UUID

from mindmemos.components.skill.bundle import compute_content_hash, deserialize_bundle
from mindmemos.config import QdrantConfig
from mindmemos.errors import SkillBundleError
from mindmemos.infra.db.collections.skill import SkillVersionRepository
from mindmemos.infra.db.engine import QdrantEngine
from mindmemos.infra.db.filters import is_empty
from mindmemos.infra.db.models import QdrantRecord
from mindmemos.mappers.skill import skill_blob_id
from mindmemos.typing.skill import SkillBlob, SkillVersionStatus
from mindmemos_skill.api.legacy_trajectory import build_legacy_binding, build_legacy_trajectory
from mindmemos_skill.api.typing import LegacyBinding, LegacyTrajectory
from mindmemos_skill.common.ids import generate_stable_id
from mindmemos_skill.infra.database import Scope
from mindmemos_skill.infra.database.scoped import ScopedDatabase
from mindmemos_skill.persistence.constants import LEGACY_SKILL_SOURCE, SKILL_VERSIONS_TABLE, SKILLS_TABLE
from mindmemos_skill.persistence.db_services.skills.typing import SkillRecord, SkillVersionRecord
from mindmemos_skill.persistence.db_services.skills.versions import parse_label
from mindmemos_skill.persistence.db_services.trajectories.service import TrajectoryDataService
from mindmemos_skill.persistence.db_services.trajectories.typing import TrajectoryRecord
from mindmemos_skill.persistence.store import SkillPersistence
from mindmemos_skill.skill.content import read_frontmatter
from mindmemos_skill.skill.typing import SkillContent, SkillSnapshot, parse_skill_header
from pydantic import JsonValue, ValidationError
from qdrant_client import models as qmodels
from sqlalchemy.exc import IntegrityError

from .typing import LegacyVersion, MigrationCounts, MigrationFailure, ProjectMigration

SOURCE = LEGACY_SKILL_SOURCE


class MigrationError(ValueError):
    """Data needs operator review before migration can proceed."""

    def __init__(self, reason: str, detail: str):
        """Capture a stable reason category and an operator-safe description."""
        super().__init__(detail)
        self.reason = reason


def _diagnose(error: Exception) -> tuple[str, str]:
    """Keep raw exception text, payloads and credentials out of the report."""
    if isinstance(error, MigrationError):
        return error.reason, str(error)
    if isinstance(error, IntegrityError):
        return "target_integrity_error", "Target database rejected a constraint"
    if isinstance(error, ValidationError):
        return "invalid_data", "Record failed schema validation"
    return "operation_failed", f"Operation failed ({type(error).__name__})"


async def _scan(
    source: QdrantEngine, collection: str, selection: qmodels.Filter | None, page_size: int
) -> AsyncIterator[QdrantRecord]:
    """Scroll by point ID so Qdrant returns a continuation cursor."""
    cursor = None
    while True:
        page, cursor = await source.scroll(collection, scroll_filter=selection, limit=page_size, offset=cursor)
        for point in page:
            yield point
        if cursor is None:
            return


async def discover_projects(
    source: QdrantEngine, config: QdrantConfig, page_size: int, unscoped_skills: list[MigrationFailure]
) -> list[str]:
    """Find all project namespaces with registered Skill versions.

    Args:
        source: Read-only connection to the legacy Qdrant collections.
        config: Names of the deployed Memory collections.
        page_size: Maximum number of versions to read per page.
        unscoped_skills: Report entries for source versions without a valid project.

    Returns:
        Sorted unique project namespaces recorded in the legacy Skill store.
    """
    projects: set[str] = set()
    async for point in _scan(source, config.skill_version_collection, None, page_size):
        project = point.payload.get("project_id")
        if not isinstance(project, str) or not project or project != project.strip():
            unscoped_skills.append(
                MigrationFailure(
                    reason="invalid_project",
                    detail="Skill version has no valid project namespace",
                    skill_id=_display_value(point.payload.get("cloud_skill_id")),
                    version_ids=[point.point_id],
                )
            )
            continue
        projects.add(project)
    return sorted(projects)


def _target_id(project: str, old_id: str, prefix: str) -> str:
    """Derive a replayable central ID from the old project and exact identity."""
    key = json.dumps([SOURCE, prefix, project, old_id], separators=(",", ":"))
    return generate_stable_id(key, prefix)


def _load_family(points: list[QdrantRecord], project: str) -> list[LegacyVersion]:
    """Validate one family and order its versions before any writes."""
    by_id: dict[str, LegacyVersion] = {}
    for point in points:
        try:
            version = LegacyVersion.model_validate(point.payload)
            UUID(version.cloud_skill_id)
        except ValueError as exc:
            raise MigrationError("invalid_version", f"Invalid Skill version {point.point_id}") from exc
        if version.project_id != project or version.version_id != point.point_id:
            raise MigrationError("invalid_version_identity", f"Unexpected owner or identity for {point.point_id}")
        by_id[version.version_id] = version

    lineage = TopologicalSorter()
    for version in sorted(by_id.values(), key=lambda item: (item.created_at, item.version_id)):
        parent = version.parent_version_id
        if parent and parent not in by_id:
            raise MigrationError("missing_parent", f"Missing same-family parent for Skill version {version.version_id}")
        lineage.add(version.version_id, *([parent] if parent else []))
    try:
        return [by_id[version_id] for version_id in lineage.static_order()]
    except CycleError as exc:
        raise MigrationError("cyclic_lineage", "Skill version lineage contains a cycle") from exc


def _display_value(value: JsonValue) -> str | None:
    """Keep only nonempty text for required display metadata."""
    if isinstance(value, str):
        return value.strip() or None
    return None


async def _load_snapshot(
    source: QdrantEngine, cfg: QdrantConfig, version: LegacyVersion
) -> tuple[SkillSnapshot, dict[str, JsonValue]]:
    """Verify the legacy bundle and adapt missing metadata without rewriting SKILL.md."""
    blob_id = skill_blob_id(version.project_id, version.content_hash)
    try:
        points = await source.retrieve(cfg.skill_blob_collection, [blob_id])
    except Exception as exc:
        raise MigrationError("bundle_read_failed", f"Bundle read failed ({type(exc).__name__})") from exc
    if not points:
        raise MigrationError("missing_bundle", f"Bundle missing for Skill version {version.version_id}")
    if points[0].payload.get("project_id") != version.project_id:
        raise MigrationError("bundle_owner_mismatch", f"Bundle belongs to another project for {version.version_id}")
    try:
        if len(points) != 1:
            raise ValueError("Missing bundle")
        blob = SkillBlob.model_validate(points[0].payload)
        files = deserialize_bundle(blob.content)
        if files.keys() != {"SKILL.md"}:
            raise ValueError("Unexpected bundle contents")
        content = files["SKILL.md"]
    except (ValueError, SkillBundleError) as exc:
        raise MigrationError("invalid_bundle", f"Invalid bundle for Skill version {version.version_id}") from exc
    if blob.content_hash != version.content_hash or compute_content_hash(files) != version.content_hash:
        raise MigrationError("bundle_hash_mismatch", f"Bundle digest does not match version {version.version_id}")

    try:
        name, description, metadata = parse_skill_header(content)
        standard = True
    except ValueError:
        try:
            metadata = read_frontmatter(content)
        except ValueError:
            metadata = {}
        name = (
            _display_value(metadata.pop("name", None))
            or _display_value(version.skill_name)
            or f"legacy-skill-{version.cloud_skill_id}"
        )
        description = _display_value(metadata.pop("description", None)) or "Imported from Memory."
        standard = False
    snapshot = SkillSnapshot.from_content(
        SkillContent(content=content, name=name, description=description, standard_format=standard)
    )
    return snapshot, metadata


def _verify_version(existing: SkillVersionRecord, incoming: SkillVersionRecord) -> None:
    """Require the same original snapshot and lineage before reusing an ID."""
    if (
        existing.origin != SOURCE
        or existing.skill_id != incoming.skill_id
        or existing.parent_version_ids != incoming.parent_version_ids
        or existing.snapshot() != incoming.snapshot()
        or existing.local_metadata.get(SOURCE) != incoming.local_metadata[SOURCE]
    ):
        raise MigrationError("target_version_conflict", f"Destination Skill version conflict for {incoming.version_id}")


async def _save_family(
    database: ScopedDatabase,
    versions: list[SkillVersionRecord],
    family: SkillRecord,
    legacy_family_id: str,
    apply: bool,
) -> tuple[int, int]:
    """Validate existing identities and atomically insert a family and its versions."""
    async with database.transaction() as tx:
        previous = await tx.get(SKILLS_TABLE, {"project_id": family.project_id, "skill_id": legacy_family_id})
        if previous is not None and previous["registration_source"] == SOURCE:
            raise MigrationError("old_id_import", f"Skill {legacy_family_id} was imported by the old-ID migration")
        key = {"project_id": family.project_id, "skill_id": family.skill_id}
        existing = await tx.get(SKILLS_TABLE, key)
        if existing is not None:
            owner = SkillRecord.model_validate(existing)
            if owner.registration_source != SOURCE or owner.is_deleted:
                raise MigrationError("target_skill_conflict", f"Destination Skill conflict for {family.skill_id}")

        new_versions = 0
        for version in versions:
            row = await tx.get(SKILL_VERSIONS_TABLE, version.version_id)
            if row is None and apply:
                inserted = await tx.upsert(
                    SKILL_VERSIONS_TABLE, [version.model_dump()], conflict_fields=["project_id", "version_id"]
                )
                if not inserted:
                    row = await tx.get(SKILL_VERSIONS_TABLE, version.version_id)
            if row is None:
                new_versions += 1
            else:
                _verify_version(SkillVersionRecord.model_validate(row), version)

        if existing is not None:
            return 0, new_versions
        if apply:
            inserted = await tx.upsert(SKILLS_TABLE, [family.model_dump()])
            if not inserted:
                row = SkillRecord.model_validate(await tx.get(SKILLS_TABLE, key))
                if row.registration_source != SOURCE or row.is_deleted:
                    raise MigrationError("target_skill_conflict", f"Destination Skill conflict for {family.skill_id}")
                return 0, new_versions
        return 1, new_versions


def _labels(versions: list[LegacyVersion]) -> dict[str, str]:
    """Keep unique canonical labels and give other versions stable fallback labels."""
    counts = Counter(version.version_label for version in versions)
    return {
        version.version_id: (
            version.version_label
            if parse_label(version.version_label) is not None
            and counts[version.version_label] == 1
            and not version.version_label.endswith("-legacy-memory")
            else f"v1.0.{index}-legacy-memory"
        )
        for index, version in enumerate(versions)
    }


def _head(versions: list[LegacyVersion]) -> LegacyVersion:
    """Prefer published and observed heads, then fall back to surviving history."""
    for status in (SkillVersionStatus.PUBLISHED, SkillVersionStatus.OBSERVED, SkillVersionStatus.SUPERSEDED):
        if eligible := [version for version in versions if version.status == status]:
            return max(eligible, key=lambda item: (item.created_at, item.version_id))
    return max(versions, key=lambda item: (item.created_at, item.version_id))


def _trajectory(
    point: QdrantRecord,
    versions: dict[str, SkillVersionRecord],
    known_version_ids: set[str],
    counts: MigrationCounts,
    project: str,
) -> TrajectoryRecord | None:
    """Convert one add record, dropping bindings without an imported exact version."""
    payload = point.payload
    if payload.get("project_id") != project or payload.get("add_record_id", point.point_id) != point.point_id:
        raise MigrationError("invalid_trajectory_identity", "Unexpected owner or identity for add record")
    bindings = []
    failed_versions = []
    try:
        for raw in payload.get("skill_bindings") or []:
            binding = LegacyBinding.model_validate(raw)
            version = versions.get(binding.version_id)
            if version is None:
                if binding.version_id in known_version_ids:
                    failed_versions.append(binding.version_id)
                else:
                    counts.skipped_bindings += 1
                continue
            try:
                bindings.append(build_legacy_binding(binding, version))
            except ValueError as exc:
                raise MigrationError("binding_hash_mismatch", f"Skill binding hash mismatch for {binding.version_id}") from exc
        if failed_versions:
            raise MigrationError(
                "skill_migration_failed", f"Registered versions unavailable: {', '.join(failed_versions)}"
            )
        if not bindings:
            counts.skipped_records += 1
            return None
        legacy = LegacyTrajectory.model_validate(payload)
        try:
            return build_legacy_trajectory(project, point.point_id, legacy, bindings)
        except ValidationError:
            raise
        except ValueError as exc:
            reason = "missing_trajectory_timestamp" if not (
                legacy.request_submitted_at or legacy.event_time or legacy.task_completed_at
            ) else "invalid_dialogue"
            raise MigrationError(reason, str(exc)) from exc
    except ValidationError as exc:
        raise MigrationError("invalid_trajectory", "Add record failed schema validation") from exc


async def _save_trajectory(store: TrajectoryDataService, record: TrajectoryRecord, apply: bool) -> bool:
    """Reuse an identical physical attempt or insert it and its index together."""
    existing = await store.get(record.trajectory_id)
    if existing is None and apply:
        try:
            await store.add(record)
        except IntegrityError:
            existing = await store.get(record.trajectory_id)
            if existing is None:
                raise
    if existing is None:
        return True
    if existing != record:
        raise MigrationError("target_trajectory_conflict", "Destination contains a different trajectory with this ID")
    return False


async def _convert_family(
    source: QdrantEngine, config: QdrantConfig, points: list[QdrantRecord], project: str, *, published_only: bool = False
) -> tuple[SkillRecord, dict[str, SkillVersionRecord]]:
    """Build the complete family before registering any version."""
    versions = _load_family(points, project)
    old_family_id = versions[0].cloud_skill_id
    head = _head(versions)
    published_id = head.version_id
    if published_only:
        published = await SkillVersionRepository(config, engine=source).published_head(project, old_family_id)
        published_id = published.point_id if published is not None else None
        if published_id is not None and published_id not in {version.version_id for version in versions}:
            raise MigrationError("source_changed", "Published version changed during the source scan")
    labels = _labels(versions)
    family_id = _target_id(project, old_family_id, "sk-")
    version_ids = {version.version_id: _target_id(project, version.version_id, "skv-") for version in versions}
    converted = {}
    for version in versions:
        snapshot, metadata = await _load_snapshot(source, config, version)
        parent = version.parent_version_id
        converted[version.version_id] = SkillVersionRecord(
            **snapshot.model_dump(),
            project_id=project,
            skill_id=family_id,
            version_id=version_ids[version.version_id],
            parent_version_ids=[version_ids[parent]] if parent else [],
            version_label=labels[version.version_id],
            origin=SOURCE,
            status=version.status.value,
            metadata=metadata,
            local_metadata={
                SOURCE: {
                    "cloud_skill_id": old_family_id,
                    "version_id": version.version_id,
                    "content_hash": version.content_hash,
                    "name": version.skill_name,
                    "version_label": version.version_label,
                    "origin": version.origin.value,
                }
            },
            created_at=version.created_at,
            updated_at=version.created_at,
        )
    active = converted[published_id or head.version_id]
    family = SkillRecord(
        project_id=project,
        skill_id=family_id,
        name=active.name,
        metadata=active.metadata,
        registration_source=SOURCE,
        active_version_id=active.version_id if published_id is not None else None,
        created_at=min(version.created_at for version in versions),
        updated_at=max(version.created_at for version in versions),
    )
    return family, converted


async def migrate_project(
    source: QdrantEngine,
    config: QdrantConfig,
    target: SkillPersistence,
    project: str,
    *,
    apply: bool,
    page_size: int,
) -> ProjectMigration:
    """Migrate one gateway namespace while collecting independent failures.

    Args:
        source: Read-only connection to the legacy Qdrant collections.
        config: Names of the deployed Memory collections.
        target: Existing Skill storage assembled from the target service schema.
        project: Exact data-plane namespace used by the gateway.
        apply: Commit records when true. Otherwise only validate and count.
        page_size: Maximum number of points in each source scan.

    Returns:
        Observed counts, failed record IDs, and any incomplete scan diagnostics.
    """
    result = ProjectMigration(project_id=project)
    counts = result.counts
    families: dict[tuple[str | None, str | None], list[QdrantRecord]] = defaultdict(list)
    selection = source.project_filter(project)
    try:
        async for point in _scan(source, config.skill_version_collection, selection, page_size):
            skill_id = _display_value(point.payload.get("cloud_skill_id"))
            families[skill_id, None if skill_id else point.point_id].append(point)
            counts.versions += 1
    except Exception as exc:
        counts.families = len(families)
        result.scan_failures.append(
            MigrationFailure(
                project_id=project, reason="skill_scan_failed", detail=f"Skill scan interrupted ({type(exc).__name__})"
            )
        )
        return result

    counts.families = len(families)
    known_version_ids = {point.point_id for points in families.values() for point in points}
    registered: dict[str, SkillVersionRecord] = {}
    with target.session(Scope(fields={"project_id": project})) as session:
        for (skill_id, _), points in families.items():
            try:
                if skill_id is None:
                    raise MigrationError("missing_skill_id", "Registered Skill version has no family ID")
                family, converted = await _convert_family(source, config, points, project)
                new_families, new_versions = await _save_family(
                    session.database, list(converted.values()), family, skill_id, apply
                )
            except Exception as exc:
                reason, detail = _diagnose(exc)
                result.failed_skills.append(
                    MigrationFailure(
                        project_id=project,
                        skill_id=skill_id,
                        version_ids=[point.point_id for point in points],
                        reason=reason,
                        detail=detail,
                    )
                )
                continue
            registered.update(converted)
            counts.verified_families += 1
            counts.fallback_versions += sum(not version.standard_format for version in converted.values())
            counts.new_families += new_families
            counts.new_versions += new_versions

        selection = source.project_filter(project, filter_=qmodels.Filter(must_not=[is_empty("skill_bindings")]))
        try:
            async for point in _scan(source, config.add_record_collection, selection, page_size):
                try:
                    record = _trajectory(point, registered, known_version_ids, counts, project)
                    if record is None:
                        continue
                    counts.new_trajectories += await _save_trajectory(session.trajectories, record, apply)
                except Exception as exc:
                    reason, detail = _diagnose(exc)
                    result.failed_trajectories.append(
                        MigrationFailure(project_id=project, trajectory_id=point.point_id, reason=reason, detail=detail)
                    )
                else:
                    counts.verified_trajectories += 1
                counts.trajectories += 1
        except Exception as exc:
            result.scan_failures.append(
                MigrationFailure(
                    project_id=project,
                    reason="trajectory_scan_failed",
                    detail=f"Trajectory scan interrupted ({type(exc).__name__})",
                )
            )
    return result
