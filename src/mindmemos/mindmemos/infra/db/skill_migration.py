"""Restartable Qdrant-v1 to relational Skill-v2 version backfill."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from mindmemos_skill.contracts import SkillBundle, SkillVersionCore, SkillVersionOrigin, SkillVersionStatus

from ...components.skill import deserialize_bundle
from ...errors import SkillConflictError, SkillVersionNotFoundError
from .skill_relational import SkillRelationalRepository


@dataclass(frozen=True, slots=True)
class CloudSkillMigrationReport:
    scanned: int = 0
    migrated: int = 0
    replayed: int = 0


class LegacyCloudSkillMigrator:
    """Backfill version/blob facts while deliberately ignoring legacy head pointers."""

    def __init__(self, legacy_repository: Any, target: SkillRelationalRepository) -> None:
        self._legacy = legacy_repository
        self._target = target

    async def migrate_project(self, project_id: str, *, page_size: int = 200) -> CloudSkillMigrationReport:
        rows = []
        cursor = None
        while True:
            page, cursor = await self._legacy.list_versions(project_id, limit=page_size, cursor=cursor)
            rows.extend(page)
            if cursor is None:
                break
        ordered = _topological_legacy_versions(rows)
        migrated = 0
        replayed = 0
        family_labels: dict[str, int] = {}
        for row in ordered:
            payload = row.payload
            blob = await self._legacy.get_blob(project_id, payload["content_hash"])
            if blob is None:
                raise SkillConflictError(f"legacy Skill blob is missing: {payload['version_id']}")
            files = deserialize_bundle(blob.payload["content"])
            bundle = SkillBundle.from_files(files)
            cloud_skill_id = str(payload["cloud_skill_id"])
            fallback_index = family_labels.get(cloud_skill_id, 0)
            family_labels[cloud_skill_id] = fallback_index + 1
            created_at = _datetime(payload.get("created_at"))
            version = SkillVersionCore(
                version_id=str(payload["version_id"]),
                cloud_skill_id=cloud_skill_id,
                parent_version_ids=(
                    [str(payload["parent_version_id"])] if payload.get("parent_version_id") else []
                ),
                name=str(payload.get("skill_name") or "migrated-skill"),
                content_hash=bundle.content_hash,
                version_label=str(payload.get("version_label") or f"0.0.{fallback_index}"),
                commit_message=payload.get("commit_message"),
                status=_status(str(payload.get("status") or "draft")),
                version_revision=0,
                origin=_origin(str(payload.get("origin") or "cloud")),
                metadata={"migration": {"source": "qdrant-skill-v1"}},
                created_at=created_at,
                updated_at=created_at,
                received_at=_datetime(payload.get("received_at")) if payload.get("received_at") else created_at,
            )
            operation_id = f"migration:qdrant-v1:{version.version_id}"
            existed = True
            try:
                await self._target.get_version(project_id, version.version_id)
            except SkillVersionNotFoundError:
                existed = False
            try:
                await self._target.create_version(
                    project_id=project_id,
                    operation_id=operation_id,
                    version=version,
                    bundle=bundle,
                )
            except SkillConflictError as exc:
                raise SkillConflictError(f"legacy Skill migration conflict for {version.version_id}: {exc}") from exc
            if existed:
                replayed += 1
            else:
                migrated += 1
        return CloudSkillMigrationReport(scanned=len(rows), migrated=migrated, replayed=replayed)


def _topological_legacy_versions(rows: list[Any]) -> list[Any]:
    remaining = list(rows)
    available: set[str] = set()
    ordered = []
    while remaining:
        ready = [
            row
            for row in remaining
            if not row.payload.get("parent_version_id") or row.payload["parent_version_id"] in available
        ]
        if not ready:
            unresolved = ", ".join(str(row.payload.get("version_id")) for row in remaining)
            raise SkillConflictError(f"legacy Skill graph has missing parents or a cycle: {unresolved}")
        ready.sort(key=lambda row: (str(row.payload.get("created_at")), str(row.payload.get("version_id"))))
        for row in ready:
            ordered.append(row)
            available.add(str(row.payload["version_id"]))
            remaining.remove(row)
    return ordered


def _status(value: str) -> SkillVersionStatus:
    return {
        "published": SkillVersionStatus.PUBLISHED,
        "superseded": SkillVersionStatus.ARCHIVED,
        "rolled_back": SkillVersionStatus.ARCHIVED,
    }.get(value, SkillVersionStatus.DRAFT)


def _origin(value: str) -> SkillVersionOrigin:
    return SkillVersionOrigin.LOCAL if value in {"edge", "local"} else SkillVersionOrigin.CLOUD


def _datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return datetime.now(UTC)


__all__ = ["CloudSkillMigrationReport", "LegacyCloudSkillMigrator"]
