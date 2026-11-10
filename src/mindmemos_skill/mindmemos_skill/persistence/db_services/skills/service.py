"""Transactional central identities, owned snapshots, and reference checks."""

from collections.abc import Sequence
from datetime import datetime

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from mindmemos_skill.common.ids import generate_id, generate_stable_id

from ....common.time_utils import utcnow
from ....errors import (
    SkillAliasConflictError,
    SkillAmbiguousVersionError,
    SkillConflictError,
    SkillNotFoundError,
    SkillVersionNotFoundError,
)
from ....infra.database import Database, Filter
from ....infra.database.scoped import ScopedDatabase
from ....skill import Skill, SkillSnapshot
from ...access import _project_id_from_database
from ...constants import (
    LEGACY_SKILL_SOURCE,
    LEGACY_SKILL_SYNC_ITEMS,
    SKILL_VERSIONS_TABLE,
    SKILLS_TABLE,
)
from .typing import SceneData, SkillRecord, SkillVersionRecord, SkillVersionSummary, SkillVersionWrite, VersionNaming
from .versions import next_label

DEFAULT_HISTORY_FIELDS = (
    "version_id",
    "version_label",
    "commit_message",
    "parent_version_ids",
    "content_hash",
    "skill_hash",
    "implementation",
    "status",
    "origin",
    "created_at",
)

VERSION_VISIBILITY_BATCH_SIZE = 500
"""Maximum version IDs per visibility query, keeping SQL parameter counts bounded."""


class SkillDataService:
    """Implement central repository operations on a borrowed database.

    Args:
        database: Database | ScopedDatabase with the persistence tables already initialized.
        prefixed_ids: Whether new Skill identities use the prefixed format.
    """

    def __init__(self, database: Database | ScopedDatabase, *, prefixed_ids: bool = True):
        self.database = database
        self.prefixed_ids = prefixed_ids

    def new_version_id(self) -> str:
        """Allocate a version ID using the configured rollout format.

        Returns:
            New immutable version identity.
        """
        return generate_id("skv-" if self.prefixed_ids else "v_")

    async def get_scene_data(self, version_id: str) -> SceneData | None:
        """Read a valid exact-version cache; storage and authorization failures propagate.

        Args:
            version_id: Exact source version.

        Returns:
            Valid cache or None for missing or invalid data.
        """
        version = await self.get(version_id)
        value = version.metadata.get("scene_data")
        if value is None:
            return None
        try:
            if not isinstance(value, dict) or "schema_version" not in value:
                return None
            return SceneData.model_validate(value)
        except ValidationError:
            return None

    async def set_scene_data(self, version_id: str, data: SceneData) -> None:
        """Merge only the Scene cache using the version's conditional metadata revision.

        Args:
            version_id: Exact source version.
            data: Newly validated complete cache.

        Raises:
            SkillConflictError: Version metadata changed before the update.
        """
        project_id = _project_id_from_database(self.database)
        data = SceneData.model_validate(data.model_dump())
        version = await self.get(version_id)
        async with self.database.transaction() as tx:
            changed = await tx.update(
                SKILL_VERSIONS_TABLE,
                {
                    "metadata": {**version.metadata, "scene_data": data.model_dump(mode="json")},
                    "version_revision": version.version_revision + 1,
                },
                filters={
                    "project_id": project_id,
                    "version_id": version.version_id,
                    "version_revision": version.version_revision,
                },
            )
            if changed != 1:
                raise SkillConflictError("Version metadata changed concurrently")

    async def add(self, record: SkillVersionWrite) -> None:
        """Insert one version; duplicate identities raise an integrity error.

        Args:
            record: Version input and lineage; the active scope supplies its project.
        """
        (await self.database.insert(SKILL_VERSIONS_TABLE, [record.model_dump(mode="python")]))

    async def get(self, version_id: str) -> SkillVersionRecord:
        """Read an exact visible version within the database's active scope.

        Args:
            version_id: Exact version identity, never an alias or prefix.

        Returns:
            The stored version belonging to a visible Skill family.

        Raises:
            SkillNotFoundError: The version or its visible family is absent.
        """
        _, version = await self.get_with_owner(version_id)
        return version

    async def get_with_owner(self, version_id: str) -> tuple[SkillRecord, SkillVersionRecord]:
        """Read an exact version and its visible owner once within the active scope.

        Args:
            version_id: Exact version identity, never an alias or prefix.

        Returns:
            The owning Skill and its complete stored version.

        Raises:
            SkillNotFoundError: The version or its visible family is absent.
        """
        version_id = await self.legacy_reference("version", version_id)
        row = await self.database.get(SKILL_VERSIONS_TABLE, version_id)
        if row is None:
            raise SkillNotFoundError(f"Version not found: {version_id}")
        family = await self.database.get(SKILLS_TABLE, {"project_id": row["project_id"], "skill_id": row["skill_id"]})
        if family is None or family["is_deleted"]:
            raise SkillNotFoundError(f"Version not found: {version_id}")
        return SkillRecord.model_validate(family), SkillVersionRecord.model_validate(row)

    async def validate_version_visibility(self, version_ids: Sequence[str]) -> None:
        """Require all exact versions and their owners to be visible without loading content.

        Args:
            version_ids: Exact version identities within the active project; duplicates are ignored.

        Raises:
            SkillNotFoundError: Any version or its visible owning Skill is absent.
        """
        identities = list(dict.fromkeys([await self.legacy_reference("version", item) for item in version_ids]))
        if not identities:
            return
        project_id = _project_id_from_database(self.database)
        for start in range(0, len(identities), VERSION_VISIBILITY_BATCH_SIZE):
            batch = identities[start : start + VERSION_VISIBILITY_BATCH_SIZE]
            versions = await self.database.select(
                SKILL_VERSIONS_TABLE,
                filters=[
                    Filter(field="project_id", value=project_id),
                    Filter(field="version_id", op="in", value=batch),
                ],
                fields=["version_id", "skill_id"],
                limit=None,
            )
            owners = {row["version_id"]: row["skill_id"] for row in versions}
            for version_id in batch:
                if version_id not in owners:
                    raise SkillNotFoundError(f"Version not found: {version_id}")
            families = await self.database.select(
                SKILLS_TABLE,
                filters=[
                    Filter(field="project_id", value=project_id),
                    Filter(field="skill_id", op="in", value=list(set(owners.values()))),
                    Filter(field="is_deleted", value=False),
                ],
                fields=["skill_id"],
                limit=None,
            )
            visible = {row["skill_id"] for row in families}
            for version_id in batch:
                if owners[version_id] not in visible:
                    raise SkillNotFoundError(f"Version not found: {version_id}")

    async def add_skill(self, skill: Skill, record: SkillVersionWrite) -> SkillVersionRecord:
        """Insert source content with caller-owned identity and version metadata.

        Args:
            skill: Source Skill; task-specific outputs are never persisted here.
            record: Version identity and metadata. Skill content/config fields
                and resource fields are supplied by the complete Skill. All
                identity, lineage, and other metadata fields are preserved.

        Returns:
            The complete inserted record without changing the input record.
        """
        values = record.model_dump(mode="python")
        values.update(skill.snapshot().model_dump(mode="python"))
        if "project_id" not in values:
            values["project_id"] = _project_id_from_database(self.database)
        stored = SkillVersionRecord.model_validate(values)
        (await self.add(stored))
        return stored

    async def get_skill(self, version_id: str) -> Skill | None:
        """Restore source data without constructing algorithm or backend instances.

        Args:
            version_id: Local version identity.

        Returns:
            A restored Skill, or None when the version does not exist.
        """
        try:
            record = await self.get(version_id)
        except SkillNotFoundError:
            return None
        return Skill.from_snapshot(record.snapshot())

    async def list_versions(
        self, skill_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[SkillVersionRecord]:
        """Read one family's versions in stable creation order.

        Args:
            skill_id: Local family identity.
            limit: Maximum number of versions.
            offset: Number of matching versions to skip.

        Returns:
            Validated local versions.
        """
        rows = await self.database.select(
            SKILL_VERSIONS_TABLE,
            filters={"skill_id": skill_id},
            order_by=["created_at", "version_id"],
            limit=limit,
            offset=offset,
        )
        visible = []
        for row in rows:
            family = await self.database.get(
                SKILLS_TABLE, {"project_id": row["project_id"], "skill_id": row["skill_id"]}
            )
            if family is not None and not family["is_deleted"]:
                visible.append(SkillVersionRecord.model_validate(row))
        return visible

    async def legacy_reference(self, kind: str, reference: str) -> str:
        """Translate an exact legacy ID within the active project."""
        if "-" not in reference or len(reference) != 36:
            return reference
        project_id = _project_id_from_database(self.database)
        row = await self.database.get(
            LEGACY_SKILL_SYNC_ITEMS, {"project_id": project_id, "kind": kind, "legacy_id": reference}
        )
        return row["target_id"] if row and row["target_id"] else reference

    async def resolve_family(self, reference: str) -> SkillRecord:
        """Resolve an exact ID/alias, then a unique ID prefix.

        Args:
            reference: Skill ID, alias, or unique ID prefix.

        Returns:
            The resolved central record.
        """
        reference = await self.legacy_reference("family", reference)
        project_id = _project_id_from_database(self.database)
        candidates = await self._rows(
            SKILLS_TABLE,
            {"project_id": project_id, "is_deleted": False},
            any_filters=[Filter(field="skill_id", value=reference), Filter(field="alias", value=reference)],
        )
        exact = next((row for row in candidates if row["skill_id"] == reference), None)
        if exact is not None:
            if any(row["skill_id"] != exact["skill_id"] for row in candidates):
                raise SkillAliasConflictError("Reference matches both a Skill ID and another alias")
            return SkillRecord.model_validate(exact)
        if candidates:
            if reference.startswith("sk-"):
                prefixed = await self.database.select(
                    SKILLS_TABLE,
                    filters=[
                        Filter(field="project_id", value=project_id),
                        Filter(field="is_deleted", value=False),
                        Filter(field="skill_id", op="starts_with", value=reference),
                    ],
                    limit=2,
                )
                if any(
                    row["skill_id"].startswith(reference) and row["skill_id"] != candidates[0]["skill_id"]
                    for row in prefixed
                ):
                    raise SkillAliasConflictError("Reference matches a Skill alias and another ID prefix")
            return SkillRecord.model_validate(candidates[0])
        matches = [
            row
            for row in (
                await self._rows(
                    SKILLS_TABLE,
                    [
                        Filter(field="project_id", value=project_id),
                        Filter(field="is_deleted", value=False),
                        Filter(field="skill_id", op="starts_with", value=reference),
                    ],
                )
            )
            if row["skill_id"].startswith(reference)
        ]
        if not matches:
            raise SkillNotFoundError(f"Skill not found: {reference}")
        if len(matches) > 1:
            candidates = ", ".join(f"{row['skill_id']} ({row['name']})" for row in matches)
            raise SkillConflictError(f"Ambiguous Skill prefix {reference!r}: {candidates}")
        return SkillRecord.model_validate(matches[0])

    async def resolve(self, reference: str) -> tuple[SkillRecord, SkillVersionRecord]:
        """Resolve a family alias or a unique immutable version prefix.

        Args:
            reference: Alias, Skill ID or prefix, or Skill version ID or prefix.

        Returns:
            Owning family and selected version; a family selects its active version.
        """
        reference = await self.legacy_reference("version", reference)
        project_id = _project_id_from_database(self.database)
        if reference.startswith(("skv-", "v_")):
            if reference in ("skv-", "v_"):
                raise SkillVersionNotFoundError("Skill version prefix is incomplete")
            aliases = await self.database.select(
                SKILLS_TABLE, filters={"project_id": project_id, "alias": reference, "is_deleted": False}, limit=1
            )
            rows = await self._rows(
                SKILL_VERSIONS_TABLE,
                [
                    Filter(field="project_id", value=project_id),
                    Filter(field="version_id", op="starts_with", value=reference),
                ],
            )
            versions = [
                row
                for row in rows
                if row["version_id"].startswith(reference)
                and (
                    owner := (
                        await self.database.get(SKILLS_TABLE, {"project_id": project_id, "skill_id": row["skill_id"]})
                    )
                )
                and not owner["is_deleted"]
            ]
            if aliases:
                if versions:
                    raise SkillAliasConflictError("Reference matches both a Skill alias and a version prefix")
                family = SkillRecord.model_validate(aliases[0])
                return family, (await self.version(family))
            if not versions:
                raise SkillVersionNotFoundError("Skill version not found")
            if len(versions) > 1:
                raise SkillConflictError("Ambiguous Skill version prefix")
            version = SkillVersionRecord.model_validate(versions[0])
            return (await self.resolve_family(version.skill_id)), version
        family = await self.resolve_family(reference)
        return family, (await self.version(family))

    async def list_skills(
        self,
        *,
        limit: int | None,
        offset: int,
        snapshot_at: datetime | None = None,
        after_created_at: datetime | None = None,
        after_skill_id: str | None = None,
    ) -> list[SkillRecord]:
        """List a page of central identities.

        Args:
            limit: Maximum records.
            offset: Number of sorted records to skip.
            snapshot_at: Inclusive creation cutoff for cursor paging.
            after_created_at: Last returned creation time, if continuing a cursor.
            after_skill_id: Last returned Skill ID, if continuing a cursor.
        """
        project_id = _project_id_from_database(self.database)
        if snapshot_at is None:
            rows = await self.database.select(
                SKILLS_TABLE,
                filters={"project_id": project_id, "is_deleted": False},
                order_by=["name", "skill_id"],
                limit=limit,
                offset=offset,
            )
        else:
            base = [
                Filter(field="project_id", value=project_id),
                Filter(field="is_deleted", value=False),
                Filter(field="created_at", op="le", value=snapshot_at),
            ]
            if after_created_at is None:
                rows = await self.database.select(
                    SKILLS_TABLE, filters=base, order_by=["-created_at", "-skill_id"], limit=limit
                )
            else:
                rows = await self.database.select(
                    SKILLS_TABLE,
                    filters=base
                    + [
                        Filter(field="created_at", value=after_created_at),
                        Filter(field="skill_id", op="lt", value=after_skill_id),
                    ],
                    order_by=["-skill_id"],
                    limit=limit,
                )
                if limit is None or len(rows) < limit:
                    rows.extend(
                        (
                            await self.database.select(
                                SKILLS_TABLE,
                                filters=base + [Filter(field="created_at", op="lt", value=after_created_at)],
                                order_by=["-created_at", "-skill_id"],
                                limit=None if limit is None else limit - len(rows),
                            )
                        )
                    )
        return [SkillRecord.model_validate(row) for row in rows]

    async def version(self, skill: SkillRecord, version_id: str | None = None) -> SkillVersionRecord:
        """Read a version only if it belongs to the selected project and Skill.

        Args:
            skill: Resolved owner.
            version_id: Requested identity; None selects the active version.
        """
        if skill.is_deleted:
            raise SkillNotFoundError("Skill not found")
        selected = version_id or skill.active_version_id
        if selected is None:
            raise SkillVersionNotFoundError("Skill has no active version; supply an exact version ID")
        selected = await self.legacy_reference("version", selected)
        row = await self.database.get(SKILL_VERSIONS_TABLE, selected)
        if row is None or row["project_id"] != skill.project_id or row["skill_id"] != skill.skill_id:
            raise SkillNotFoundError(f"Version does not belong to this Skill: {version_id or skill.active_version_id}")
        return SkillVersionRecord.model_validate(row)

    async def version_by_label(self, skill: SkillRecord, label: str) -> SkillVersionRecord:
        """Read a version by its unique family-scoped display label.

        Args:
            skill: Family owning the requested version.
            label: Exact immutable version label.
        """
        rows = await self.database.select(
            SKILL_VERSIONS_TABLE, filters=self._key(skill) | {"version_label": label}, limit=1
        )
        if not rows:
            raise SkillNotFoundError("Version label not found")
        return SkillVersionRecord.model_validate(rows[0])

    async def resolve_version_hash(
        self,
        content_hash: str | None = None,
        *,
        skill_hash: str | None = None,
        skill_id: str | None = None,
        version_id: str | None = None,
    ) -> SkillVersionRecord:
        """Resolve a unique visible version satisfying every supplied constraint.

        Args:
            content_hash: Optional exact-text digest.
            skill_hash: Optional complete-source digest.
            skill_id: Optional exact owning family identity.
            version_id: Optional exact immutable version identity.

        Returns:
            The unique matching version.

        Raises:
            ValueError: No immutable constraint was supplied.
            SkillVersionNotFoundError: No visible version matches all constraints.
            SkillAmbiguousVersionError: Multiple visible versions match.
        """
        project_id = _project_id_from_database(self.database)
        if content_hash is None and skill_hash is None and version_id is None:
            raise ValueError("Supply a hash or version_id")
        if skill_id is not None:
            skill_id = await self.legacy_reference("family", skill_id)
        if version_id is not None:
            version_id = await self.legacy_reference("version", version_id)
        filters = {"project_id": project_id}
        for key, value in (
            ("content_hash", content_hash),
            ("skill_hash", skill_hash),
            ("skill_id", skill_id),
            ("version_id", version_id),
        ):
            if value is not None:
                filters[key] = value
        match = None
        for row in await self._rows(SKILL_VERSIONS_TABLE, filters):
            family = await self.database.get(SKILLS_TABLE, {"project_id": project_id, "skill_id": row["skill_id"]})
            if family is None or family["is_deleted"]:
                continue
            if match is not None:
                raise SkillAmbiguousVersionError("Multiple versions match; supply an exact version ID")
            match = row
        if match is None:
            raise SkillVersionNotFoundError("Skill version not found")
        return SkillVersionRecord.model_validate(match)

    async def history(
        self,
        skill: SkillRecord,
        *,
        limit: int | None,
        offset: int = 0,
        fields: list[str] | None = None,
        snapshot_at: datetime | None = None,
        after_created_at: datetime | None = None,
        after_version_id: str | None = None,
    ) -> list[SkillVersionSummary] | list[dict]:
        """Read visible owned versions in stable ascending creation order.

        Args:
            skill: Resolved owner.
            limit: Maximum rows; None returns every match.
            offset: Rows to skip in offset mode.
            fields: Requested columns; None selects validated history summaries.
            snapshot_at: Inclusive first-page creation cutoff.
            after_created_at: Last returned creation time in cursor mode.
            after_version_id: Last returned identity in cursor mode.

        Returns:
            Summaries or dictionaries matching the requested projection.
        """
        if (after_created_at is None) != (after_version_id is None):
            raise ValueError("History position requires creation time and version ID")
        if after_created_at is not None and snapshot_at is None:
            raise ValueError("Continuation requires snapshot_at")
        if snapshot_at is not None and offset:
            raise ValueError("Snapshot pagination cannot include an offset")
        owner = await self.database.get(SKILLS_TABLE, self._key(skill))
        if owner is None or owner["is_deleted"]:
            raise SkillNotFoundError("Skill not found")
        filters = [Filter(field=key, value=value) for key, value in self._key(skill).items()]
        if snapshot_at is not None:
            filters.append(Filter(field="created_at", op="le", value=snapshot_at))
        projection = list(DEFAULT_HISTORY_FIELDS) if fields is None else fields
        if after_created_at is None:
            rows = await self.database.select(
                SKILL_VERSIONS_TABLE,
                filters=filters,
                fields=projection,
                order_by=["created_at", "version_id"],
                limit=limit,
                offset=offset,
            )
        else:
            rows = await self.database.select(
                SKILL_VERSIONS_TABLE,
                fields=projection,
                filters=filters
                + [
                    Filter(field="created_at", value=after_created_at),
                    Filter(field="version_id", op="gt", value=after_version_id),
                ],
                order_by=["version_id"],
                limit=limit,
            )
            if limit is None or len(rows) < limit:
                rows.extend(
                    (
                        await self.database.select(
                            SKILL_VERSIONS_TABLE,
                            fields=projection,
                            filters=filters + [Filter(field="created_at", op="gt", value=after_created_at)],
                            order_by=["created_at", "version_id"],
                            limit=None if limit is None else limit - len(rows),
                        )
                    )
                )
        return rows if fields is not None else [SkillVersionSummary.model_validate(row) for row in rows]

    async def register(
        self,
        *,
        name: str,
        alias: str | None,
        source: str | None,
        message: str | None,
        snapshot: SkillSnapshot,
        naming: VersionNaming | None = None,
        metadata: dict | None = None,
        file_meta: dict | None = None,
    ) -> SkillRecord:
        """Atomically register a new identity and its initial snapshot.

        Args:
            name: Display name.
            alias: Optional exact shortcut.
            source: Initial provenance reference.
            message: Registration explanation.
            snapshot: Complete validated source content and configuration.
            naming: Semantic label options; omission creates v1.0.0.
            metadata: Initial family metadata.
            file_meta: Metadata captured on the immutable version.

        Returns:
            Newly registered family with its initial version selected.
        """
        project_id = _project_id_from_database(self.database)
        naming = naming or VersionNaming()
        if naming.version_bump != "patch":
            raise ValueError("Initial registration has no version to increment")
        skill_id = generate_id("sk-" if self.prefixed_ids else "sk_")
        version_id = self.new_version_id()
        record = SkillRecord(
            project_id=project_id,
            skill_id=skill_id,
            name=name,
            alias=alias,
            metadata=metadata or {},
            registration_source=source,
            registration_message=message,
            active_version_id=version_id,
        )
        version = self._new_version(record, snapshot, version_id, [], message)
        version = version.model_copy(
            update={
                "version_label": naming.version_label or next_label(None, [], algorithm=naming.algorithm),
                "origin": source or "user",
                "metadata": file_meta or {},
            }
        )
        try:
            async with self.database.transaction() as tx:
                store = SkillDataService(tx)
                (await store._check_alias(record, alias))
                if await tx.count(SKILLS_TABLE, filters={"project_id": project_id, "alias": skill_id}):
                    raise SkillAliasConflictError(
                        "Generated Skill ID conflicts with an existing alias; retry registration"
                    )
                (await tx.insert(SKILL_VERSIONS_TABLE, [version.model_dump()]))
                (await tx.insert(SKILLS_TABLE, [record.model_dump()]))
        except IntegrityError as exc:
            raise SkillAliasConflictError("Registration conflicts with an existing identity or alias") from exc
        return record

    async def commit(
        self,
        skill: SkillRecord,
        snapshot: SkillSnapshot,
        *,
        message: str | None,
        naming: VersionNaming | None = None,
        base_version_id: str | None = None,
        active: bool = True,
        commit_source: str = "user",
        metadata: dict | None = None,
        version_id: str | None = None,
        file_meta: dict | None = None,
    ) -> SkillRecord:
        """Commit a child of an owned version and optionally select it atomically.

        Args:
            skill: Owner and active selection captured before source loading.
            snapshot: Complete new content.
            message: Version explanation.
            naming: Manual label or automatic increment and algorithm namespace.
            base_version_id: Parent version; None selects the captured active version.
            active: Whether to select the new version as active.
            commit_source: User or algorithm responsible for publication.
            metadata: Replacement family metadata; None preserves existing metadata.
            version_id: Optional caller-generated identity for the new version.
            file_meta: Optional version file metadata.

        Returns:
            Updated family; inactive commits preserve its selected version.
        """
        naming = naming or VersionNaming()
        version_id = version_id or self.new_version_id()
        parent_id = base_version_id or skill.active_version_id
        if parent_id is None:
            raise SkillVersionNotFoundError("Skill has no active version; supply an exact base version ID")
        parent_id = await self.legacy_reference("version", parent_id)
        version = self._new_version(skill, snapshot, version_id, [parent_id], message)
        version = version.model_copy(update={"origin": commit_source})
        if file_meta is not None:
            version = version.model_copy(update={"metadata": file_meta or {}})
        try:
            async with self.database.transaction() as tx:
                await self._claim_legacy(tx, skill)
                if active:
                    (await self._advance(tx, skill, skill.active_version_id))
                (await self._insert_child(tx, skill, version, naming, active=active))
                if metadata is not None:
                    (
                        await tx.update(
                            SKILLS_TABLE, {"metadata": metadata, "updated_at": utcnow()}, filters=self._key(skill)
                        )
                    )
                return SkillRecord.model_validate((await tx.get(SKILLS_TABLE, self._key(skill))))
        except IntegrityError as exc:
            raise SkillConflictError("Version identity or label conflicts with an existing version") from exc

    async def algorithm_publication(
        self, run_id: str, input_hash: str, *, operation: str = "training"
    ) -> SkillVersionRecord | None:
        """Find an already published run without changing the active version.

        Args:
            run_id: Stable run identity shared by execution attempts.
            input_hash: Exact request and pipeline identity.
            operation: Publication namespace, shared by retries of the same operation.

        Returns:
            Published version, or None before publication.

        Raises:
            SkillConflictError: The run was previously published with different inputs.
        """
        project_id = _project_id_from_database(self.database)
        suffix = generate_stable_id(f"{operation}:{project_id}:{run_id}")
        identifier = ("skv-" if self.prefixed_ids else "v_") + suffix
        version = None
        for candidate_id in (identifier, ("v_" if self.prefixed_ids else "skv-") + suffix):
            try:
                version = await self.get(candidate_id)
                break
            except SkillNotFoundError:
                continue
        if version is None:
            return None
        if version.project_id != project_id or version.local_metadata.get(operation) != {
            "run_id": run_id,
            "input_hash": input_hash,
        }:
            raise SkillConflictError("Algorithm run was published with different inputs")
        return version

    async def commit_algorithm(
        self,
        skill: SkillRecord,
        snapshot: SkillSnapshot,
        *,
        run_id: str,
        input_hash: str,
        algorithm: str,
        base_version_id: str,
        active: bool = True,
        report: dict | None = None,
        operation: str = "training",
        commit_source: str | None = None,
    ) -> SkillVersionRecord:
        """Publish an algorithm result exactly once using existing version columns.

        Args:
            skill: Owner with the active selection captured before algorithm execution.
            snapshot: Complete selected Skill, including dynamic implementation data.
            run_id: Stable run identity shared by retries.
            input_hash: Validated request and configuration fingerprint.
            algorithm: Registered algorithm name used for automatic version naming.
            base_version_id: Exact version used as the algorithm input and parent.
            active: Whether to select the published version as active.
            report: Compact JSON report attached to the published version metadata.
            operation: Publication namespace, such as training or evolution.
            commit_source: Explicit version origin; None preserves the snapshot origin.

        Returns:
            Newly committed or previously published immutable version.

        Raises:
            SkillConflictError: The active selection changed or the run identity was reused.
        """
        existing = await self.algorithm_publication(run_id, input_hash, operation=operation)
        if existing is not None:
            if not self._matches_algorithm_result(existing, skill, base_version_id, snapshot):
                raise SkillConflictError("Algorithm publication parent or result changed")
            return existing
        identifier = generate_stable_id(
            f"{operation}:{skill.project_id}:{run_id}", "skv-" if self.prefixed_ids else "v_"
        )
        version = self._new_version(
            skill, snapshot, identifier, [base_version_id], f"{operation.capitalize()} run {run_id}"
        )
        if commit_source is not None:
            version = version.model_copy(update={"origin": commit_source})
        version = SkillVersionRecord.model_validate(
            {
                **version.model_dump(mode="python"),
                "metadata": {**version.metadata, "algorithm_report": {**report, "run_id": run_id}}
                if report
                else version.metadata,
                "local_metadata": {operation: {"run_id": run_id, "input_hash": input_hash}},
            }
        )
        try:
            async with self.database.transaction() as tx:
                await self._claim_legacy(tx, skill)
                if active:
                    (await self._advance(tx, skill, skill.active_version_id))
                (await self._insert_child(tx, skill, version, VersionNaming(algorithm=algorithm), active=active))
                return await SkillDataService(tx).get(identifier)
        except (IntegrityError, SkillConflictError):
            # A concurrent identical publication may have committed while acquiring the owner lock.
            existing = await self.algorithm_publication(run_id, input_hash, operation=operation)
            if existing is not None and self._matches_algorithm_result(existing, skill, base_version_id, snapshot):
                return existing
            raise SkillConflictError("Active version or algorithm publication changed") from None

    @staticmethod
    def _matches_algorithm_result(version, skill, base_version_id, snapshot):
        """Compare the complete published output and its captured parent."""
        return (
            version.skill_id == skill.skill_id
            and version.parent_version_ids == [base_version_id]
            and version.content == snapshot.content
            and version.resources == snapshot.resources
            and version.implementation == snapshot.implementation
            and version.config_schema_version == snapshot.config_schema_version
            and version.config == snapshot.config
            and version.content_hash == snapshot.content_hash
            and version.skill_hash == snapshot.skill_hash
        )

    async def _insert_child(self, tx, skill, version, naming, *, active: bool = True):
        """Allocate a unique label and insert a version in the current transaction."""
        store = SkillDataService(tx)
        parent = await store.version(skill, version.parent_version_ids[0])
        labels = [row["version_label"] for row in (await store._rows(SKILL_VERSIONS_TABLE, self._key(skill)))]
        label = naming.version_label or next_label(
            parent.version_label, labels, bump=naming.version_bump, algorithm=naming.algorithm
        )
        if label in labels:
            raise SkillConflictError(f"Version label is already in use: {label}")
        version = version.model_copy(update={"version_label": label})
        (await tx.insert(SKILL_VERSIONS_TABLE, [version.model_dump()]))
        if active:
            (await self._advance(tx, skill, version.version_id))

    async def activate(self, skill: SkillRecord, version_id: str) -> SkillRecord:
        """Select an owned historical version with a compare-and-set update.

        Args:
            skill: Owner and expected active version.
            version_id: Owned version to select.
        """
        async with self.database.transaction() as tx:
            await self._claim_legacy(tx, skill)
            (await self._advance(tx, skill, skill.active_version_id))
            version = await SkillDataService(tx).version(skill, version_id)
            (await self._advance(tx, skill, version.version_id))
            return SkillRecord.model_validate((await tx.get(SKILLS_TABLE, self._key(skill))))

    async def set_meta(
        self,
        skill: SkillRecord,
        *,
        name: str | None,
        alias: str | None,
        clear_alias: bool,
        metadata: dict | None = None,
    ) -> SkillRecord:
        """Modify display metadata while preserving registration provenance.

        Args:
            skill: Resolved identity.
            name: Replacement name, or None to retain it.
            alias: Replacement alias, or None to retain it.
            clear_alias: Explicitly remove the alias.
        """
        values = {"updated_at": utcnow()}
        if name is not None:
            values["name"] = name
        if alias is not None or clear_alias:
            values["alias"] = None if clear_alias else alias
        if metadata is not None:
            values["metadata"] = metadata
        try:
            async with self.database.transaction() as tx:
                await self._claim_legacy(tx, skill)
                await SkillDataService(tx)._check_alias(skill, alias)
                if (await tx.update(SKILLS_TABLE, values, filters=self._key(skill) | {"is_deleted": False})) != 1:
                    raise SkillNotFoundError("Skill was removed during metadata update")
                return SkillRecord.model_validate((await tx.get(SKILLS_TABLE, self._key(skill))))
        except IntegrityError as exc:
            raise SkillAliasConflictError("Alias is already in use") from exc

    async def delete(self, skill: SkillRecord) -> None:
        """Mark a Skill deleted while retaining all versions and references.

        Args:
            skill: Active Skill identity selected by the caller.
        """
        now = utcnow()
        async with self.database.transaction() as tx:
            await self._claim_legacy(tx, skill)
            changed = await tx.update(
                SKILLS_TABLE,
                {"is_deleted": True, "deleted_at": now, "updated_at": now},
                filters=self._key(skill) | {"is_deleted": False},
            )
            if changed != 1:
                raise SkillNotFoundError("Skill not found")

    async def _claim_legacy(self, tx, skill: SkillRecord) -> None:
        """Protect native edits from later legacy publication or deletion."""
        if skill.registration_source != LEGACY_SKILL_SOURCE:
            return
        key = self._key(skill) | {"is_deleted": False}
        changed = await tx.update(SKILLS_TABLE, {"native_owned_at": utcnow()}, filters=key | {"native_owned_at": None})
        if not changed:
            # Also lock an already claimed family before an inactive commit.
            changed = await tx.update(SKILLS_TABLE, {"updated_at": utcnow()}, filters=key)
        if not changed:
            raise SkillNotFoundError("Skill was removed during update")

    async def _rows(self, table, filters, *, any_filters=None):
        """Read all matching records in bounded pages, including histories over 100 rows."""
        rows = []
        while True:
            page = await self.database.select(
                table, filters=filters, any_filters=any_filters, limit=1000, offset=len(rows)
            )
            rows.extend(page)
            if len(page) < 1000:
                return rows

    async def _check_alias(self, skill, alias):
        """Reject aliases that identify another Skill by ID or alias."""
        if alias is None:
            return
        exact = await self.database.get(SKILLS_TABLE, {"project_id": skill.project_id, "skill_id": alias})
        matches = await self.database.select(
            SKILLS_TABLE, filters={"project_id": skill.project_id, "alias": alias}, limit=1
        )
        if (exact and exact["skill_id"] != skill.skill_id) or any(row["skill_id"] != skill.skill_id for row in matches):
            raise SkillAliasConflictError(f"Alias already identifies another Skill: {alias}")

    @staticmethod
    def _key(skill):
        """Return the complete local identity."""
        return {"project_id": skill.project_id, "skill_id": skill.skill_id}

    @staticmethod
    async def _advance(tx, skill, version_id):
        """Fail atomically if another writer already advanced the pointer."""
        changed = await tx.update(
            SKILLS_TABLE,
            {"active_version_id": version_id, "updated_at": utcnow()},
            filters=SkillDataService._key(skill) | {"active_version_id": skill.active_version_id, "is_deleted": False},
        )
        if changed != 1:
            raise SkillConflictError("Active version changed; reload before committing or activating")

    @staticmethod
    def _new_version(skill, snapshot, version_id, parents, message):
        """Build an immutable version from a complete source snapshot."""
        snapshot = SkillSnapshot.model_validate(snapshot.model_dump())
        return SkillVersionRecord(
            project_id=skill.project_id,
            skill_id=skill.skill_id,
            version_id=version_id,
            parent_version_ids=parents,
            commit_message=message,
            **snapshot.model_dump(),
        )


def _references(value, identities):
    """Conservatively detect retained identities in legacy trajectory JSON."""
    if isinstance(value, str):
        return value in identities
    if isinstance(value, dict):
        return any(_references(child, identities) for child in value.values())
    if isinstance(value, list):
        return any(_references(child, identities) for child in value)
    return False
