"""Transactional central identities, owned snapshots, and reference checks."""

from uuid import NAMESPACE_URL, uuid4, uuid5

from sqlalchemy.exc import IntegrityError

from ....common.time_utils import utcnow
from ....errors import SkillConflictError, SkillInUseError, SkillNotFoundError
from ....infra.database import Database
from ....skill import Skill, SkillSnapshot
from ...constants import REMOTE_SKILL_BINDINGS_TABLE, SKILL_VERSIONS_TABLE, SKILLS_TABLE, TRAJECTORIES_TABLE
from .typing import SkillRecord, SkillVersionRecord, VersionNaming
from .versions import next_label


class SkillDataService:
    """Implement central repository operations on a borrowed database.

    Args:
        database: Database with the persistence tables already initialized.
    """

    def __init__(self, database: Database):
        self.database = database

    def add(self, record: SkillVersionRecord) -> None:
        """Insert one version; duplicate identities raise an integrity error.

        Args:
            record: Complete version, including its local lineage.
        """
        self.database.insert(SKILL_VERSIONS_TABLE, [record.model_dump(mode="python")])

    def get(self, version_id: str) -> SkillVersionRecord | None:
        """Read a local version.

        Args:
            version_id: Local version identity.

        Returns:
            The stored version, or None.
        """
        row = self.database.get(SKILL_VERSIONS_TABLE, version_id)
        return SkillVersionRecord.model_validate(row) if row is not None else None

    def add_skill(self, skill: Skill, record: SkillVersionRecord) -> SkillVersionRecord:
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
        stored = SkillVersionRecord.model_validate(values)
        self.add(stored)
        return stored

    def get_skill(self, version_id: str) -> Skill | None:
        """Restore source data without constructing algorithm or backend instances.

        Args:
            version_id: Local version identity.

        Returns:
            A restored Skill, or None when the version does not exist.
        """
        record = self.get(version_id)
        if record is None:
            return None
        snapshot = SkillSnapshot(
            implementation=record.implementation,
            config_schema_version=record.config_schema_version,
            config=record.config,
            bundle=record.bundle,
            content_hash=record.content_hash,
            resources=record.resources,
            local_snapshot_hash=record.local_snapshot_hash,
        )
        return Skill.from_snapshot(snapshot)

    def list_versions(self, skill_id: str, *, limit: int = 100, offset: int = 0) -> list[SkillVersionRecord]:
        """Read one family's versions in stable creation order.

        Args:
            skill_id: Local family identity.
            limit: Maximum number of versions.
            offset: Number of matching versions to skip.

        Returns:
            Validated local versions.
        """
        rows = self.database.select(
            SKILL_VERSIONS_TABLE, filters={"skill_id": skill_id}, order_by=["created_at"], limit=limit, offset=offset
        )
        return [SkillVersionRecord.model_validate(row) for row in rows]

    def resolve(self, project_id: str, reference: str) -> SkillRecord:
        """Resolve an exact ID/alias, then a unique ID prefix.

        Args:
            project_id: Local project scope.
            reference: Skill ID, alias, or unique ID prefix.

        Returns:
            The resolved central record.
        """
        exact = self.database.get(SKILLS_TABLE, {"project_id": project_id, "skill_id": reference})
        aliases = self.database.select(SKILLS_TABLE, filters={"project_id": project_id, "alias": reference}, limit=1)
        if exact is not None:
            if aliases and aliases[0]["skill_id"] != exact["skill_id"]:
                raise SkillConflictError("Reference matches both a Skill ID and another alias")
            return SkillRecord.model_validate(exact)
        if aliases:
            return SkillRecord.model_validate(aliases[0])
        matches = [
            row for row in self._rows(SKILLS_TABLE, {"project_id": project_id}) if row["skill_id"].startswith(reference)
        ]
        if not matches:
            raise SkillNotFoundError(f"Skill not found: {reference}")
        if len(matches) > 1:
            candidates = ", ".join(f"{row['skill_id']} ({row['name']})" for row in matches)
            raise SkillConflictError(f"Ambiguous Skill prefix {reference!r}: {candidates}")
        return SkillRecord.model_validate(matches[0])

    def list_skills(self, project_id: str, *, limit: int, offset: int) -> list[SkillRecord]:
        """List a page of central identities.

        Args:
            project_id: Local project scope.
            limit: Maximum records.
            offset: Number of sorted records to skip.
        """
        return [
            SkillRecord.model_validate(row)
            for row in self.database.select(
                SKILLS_TABLE, filters={"project_id": project_id}, order_by=["name"], limit=limit, offset=offset
            )
        ]

    def version(self, skill: SkillRecord, version_id: str | None = None) -> SkillVersionRecord:
        """Read a version only if it belongs to the selected project and Skill.

        Args:
            skill: Resolved owner.
            version_id: Requested identity; None selects the current version.
        """
        row = self.database.get(SKILL_VERSIONS_TABLE, version_id or skill.current_version_id)
        if row is None or row["project_id"] != skill.project_id or row["skill_id"] != skill.skill_id:
            raise SkillNotFoundError(f"Version does not belong to this Skill: {version_id or skill.current_version_id}")
        return SkillVersionRecord.model_validate(row)

    def version_by_id(self, project_id: str, version_id: str) -> SkillVersionRecord:
        """Load an immutable version within an explicit project.

        Args:
            project_id: Required repository scope.
            version_id: Exact input version identity, never an alias or prefix.

        Raises:
            SkillNotFoundError: The version is absent or belongs to another project.
        """
        record = self.get(version_id)
        if record is None or record.project_id != project_id:
            raise SkillNotFoundError(f"Version not found in project: {version_id}")
        return record

    def history(self, skill: SkillRecord, *, limit: int, offset: int) -> list[SkillVersionRecord]:
        """Read owned versions in stable creation order.

        Args:
            skill: Resolved owner.
            limit: Maximum versions.
            offset: Number of versions to skip.
        """
        return [
            SkillVersionRecord.model_validate(row)
            for row in self.database.select(
                SKILL_VERSIONS_TABLE, filters=self._key(skill), order_by=["created_at"], limit=limit, offset=offset
            )
        ]

    def register(
        self,
        *,
        project_id: str,
        name: str,
        alias: str | None,
        source: str | None,
        message: str | None,
        snapshot: SkillSnapshot,
        naming: VersionNaming | None = None,
    ) -> SkillRecord:
        """Atomically register a new identity and its initial snapshot.

        Args:
            project_id: Local project scope.
            name: Display name.
            alias: Optional exact shortcut.
            source: Initial provenance reference.
            message: Registration explanation.
            snapshot: Complete validated source content and configuration.
            naming: Semantic label options; omission creates v1.0.0.
        """
        naming = naming or VersionNaming()
        if naming.version_bump != "patch":
            raise ValueError("Initial registration has no version to increment")
        skill_id = f"sk_{uuid4().hex}"
        version_id = f"v_{uuid4().hex}"
        record = SkillRecord(
            project_id=project_id,
            skill_id=skill_id,
            name=name,
            alias=alias,
            registration_source=source,
            registration_message=message,
            current_version_id=version_id,
        )
        version = self._new_version(record, snapshot, version_id, [], message)
        version = version.model_copy(
            update={"version_label": naming.version_label or next_label(None, [], algorithm=naming.algorithm)}
        )
        try:
            with self.database.transaction() as tx:
                store = SkillDataService(tx)
                store._check_alias(record, alias)
                if tx.count(SKILLS_TABLE, filters={"project_id": project_id, "alias": skill_id}):
                    raise SkillConflictError("Generated Skill ID conflicts with an existing alias; retry registration")
                tx.insert(SKILL_VERSIONS_TABLE, [version.model_dump()])
                tx.insert(SKILLS_TABLE, [record.model_dump()])
        except IntegrityError as exc:
            raise SkillConflictError("Registration conflicts with an existing identity or alias") from exc
        return record

    def commit(
        self,
        skill: SkillRecord,
        snapshot: SkillSnapshot,
        *,
        message: str | None,
        naming: VersionNaming | None = None,
    ) -> SkillRecord:
        """Allocate a label and atomically commit against the captured current version.

        Args:
            skill: Owner and expected current version captured before source loading.
            snapshot: Complete new content.
            message: Version explanation.
            naming: Manual label or automatic increment and algorithm namespace.
        """
        naming = naming or VersionNaming()
        version_id = f"v_{uuid4().hex}"
        version = self._new_version(skill, snapshot, version_id, [skill.current_version_id], message)
        try:
            with self.database.transaction() as tx:
                # Acquire the owner write lock before reading labels, including on SQLite.
                self._advance(tx, skill, skill.current_version_id)
                self._insert_child(tx, skill, version, naming)
                return SkillRecord.model_validate(tx.get(SKILLS_TABLE, self._key(skill)))
        except IntegrityError as exc:
            raise SkillConflictError("Version identity or label conflicts with an existing version") from exc

    def training_publication(self, project_id: str, run_id: str, input_hash: str) -> SkillVersionRecord | None:
        """Find an already published run without changing the current version.

        Args:
            project_id: Repository owning the run.
            run_id: Stable identity derived from the caller's run directory.
            input_hash: Exact non-secret request and pipeline identity.

        Returns:
            Published version, or None before publication.

        Raises:
            SkillConflictError: The run was previously published with different inputs.
        """
        identifier = "v_" + uuid5(NAMESPACE_URL, f"training:{project_id}:{run_id}").hex
        version = self.get(identifier)
        if version is None:
            return None
        if version.project_id != project_id or version.local_metadata.get("training") != {
            "run_id": run_id,
            "input_hash": input_hash,
        }:
            raise SkillConflictError("Training run was published with different inputs")
        return version

    def commit_training(
        self,
        skill: SkillRecord,
        snapshot: SkillSnapshot,
        *,
        run_id: str,
        input_hash: str,
        algorithm: str,
        report: dict | None = None,
    ) -> SkillVersionRecord:
        """Publish a training result exactly once using existing version columns.

        Args:
            skill: Owner with the original expected current parent.
            snapshot: Complete selected Skill, including dynamic implementation data.
            run_id: Stable run identity shared by retries.
            input_hash: Validated request and configuration fingerprint.
            algorithm: Registered trainer name used for automatic version naming.
            report: Compact JSON report attached to the published version metadata.

        Returns:
            Newly committed or previously published immutable version.

        Raises:
            SkillConflictError: The parent changed or the run identity was reused.
        """
        existing = self.training_publication(skill.project_id, run_id, input_hash)
        if existing is not None:
            if not self._matches_training_result(existing, skill, snapshot):
                raise SkillConflictError("Training publication parent or result changed")
            return existing
        identifier = "v_" + uuid5(NAMESPACE_URL, f"training:{skill.project_id}:{run_id}").hex
        version = self._new_version(skill, snapshot, identifier, [skill.current_version_id], f"Training run {run_id}")
        version = SkillVersionRecord.model_validate(
            {
                **version.model_dump(mode="python"),
                "metadata": {**version.metadata, "algorithm_report": {**report, "run_id": run_id}}
                if report
                else version.metadata,
                "local_metadata": {"training": {"run_id": run_id, "input_hash": input_hash}},
            }
        )
        try:
            with self.database.transaction() as tx:
                self._advance(tx, skill, skill.current_version_id)
                self._insert_child(tx, skill, version, VersionNaming(algorithm=algorithm))
                return SkillDataService(tx).version_by_id(skill.project_id, identifier)
        except (IntegrityError, SkillConflictError):
            # A concurrent identical publication may have committed while acquiring the owner lock.
            existing = self.training_publication(skill.project_id, run_id, input_hash)
            if existing is not None and self._matches_training_result(existing, skill, snapshot):
                return existing
            raise SkillConflictError("Current version or training publication changed") from None

    @staticmethod
    def _matches_training_result(version, skill, snapshot):
        """Compare the complete published output and its captured parent."""
        return (
            version.skill_id == skill.skill_id
            and version.parent_version_ids == [skill.current_version_id]
            and version.bundle == snapshot.bundle
            and version.resources == snapshot.resources
            and version.implementation == snapshot.implementation
            and version.config_schema_version == snapshot.config_schema_version
            and version.config == snapshot.config
            and version.content_hash == snapshot.content_hash
            and version.local_snapshot_hash == snapshot.local_snapshot_hash
        )

    def _insert_child(self, tx, skill, version, naming):
        """Allocate a label and insert a version while holding the owner write lock."""
        store = SkillDataService(tx)
        current = store.version(skill)
        labels = [row["version_label"] for row in store._rows(SKILL_VERSIONS_TABLE, self._key(skill))]
        label = naming.version_label or next_label(
            current.version_label, labels, bump=naming.version_bump, algorithm=naming.algorithm
        )
        if label in labels:
            raise SkillConflictError(f"Version label is already in use: {label}")
        version = version.model_copy(update={"version_label": label})
        tx.insert(SKILL_VERSIONS_TABLE, [version.model_dump()])
        self._advance(tx, skill, version.version_id)

    def switch(self, skill: SkillRecord, version_id: str) -> SkillRecord:
        """Select an owned historical version with a compare-and-set update.

        Args:
            skill: Owner and expected current version.
            version_id: Owned version to select.
        """
        with self.database.transaction() as tx:
            SkillDataService(tx).version(skill, version_id)
            self._advance(tx, skill, version_id)
            return SkillRecord.model_validate(tx.get(SKILLS_TABLE, self._key(skill)))

    def set_meta(self, skill: SkillRecord, *, name: str | None, alias: str | None, clear_alias: bool) -> SkillRecord:
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
        try:
            with self.database.transaction() as tx:
                SkillDataService(tx)._check_alias(skill, alias)
                if tx.update(SKILLS_TABLE, values, filters=self._key(skill)) != 1:
                    raise SkillNotFoundError("Skill was removed during metadata update")
                return SkillRecord.model_validate(tx.get(SKILLS_TABLE, self._key(skill)))
        except IntegrityError as exc:
            raise SkillConflictError("Alias is already in use") from exc

    def delete(self, skill: SkillRecord) -> None:
        """Remove an unreferenced identity and its versions atomically.

        Args:
            skill: Owner and current pointer captured by the caller.

        Raises:
            SkillInUseError: Graphs, trajectories, other versions, or remote records reference the Skill.
        """
        try:
            with self.database.transaction() as tx:
                store = SkillDataService(tx)
                versions = store._rows(SKILL_VERSIONS_TABLE, self._key(skill))
                ids = {row["version_id"] for row in versions}
                for row in store._rows(SKILL_VERSIONS_TABLE, {}):
                    if row["version_id"] not in ids and ids.intersection(row["parent_version_ids"]):
                        raise SkillInUseError("Other versions reference this Skill's history")
                for row in store._rows(TRAJECTORIES_TABLE, {}):
                    if _references(row["skill_bindings"], ids | {skill.skill_id}) or _references(
                        row["injected_skills"], ids | {skill.skill_id}
                    ):
                        raise SkillInUseError("A stored trajectory references this Skill")
                if REMOTE_SKILL_BINDINGS_TABLE in {spec.name for spec in tx.tables.values()}:
                    if tx.count(REMOTE_SKILL_BINDINGS_TABLE, filters={"skill_id": skill.skill_id}):
                        raise SkillInUseError("A remote binding references this Skill")
                if (
                    tx.delete(SKILLS_TABLE, filters=self._key(skill) | {"current_version_id": skill.current_version_id})
                    != 1
                ):
                    raise SkillConflictError("Current version changed before deletion")
                tx.delete(SKILL_VERSIONS_TABLE, filters=self._key(skill))
        except IntegrityError as exc:
            raise SkillInUseError("Stored records still reference this Skill or its versions") from exc

    def _rows(self, table, filters):
        """Read all matching records in bounded pages, including histories over 100 rows."""
        rows = []
        while True:
            page = self.database.select(table, filters=filters, limit=1000, offset=len(rows))
            rows.extend(page)
            if len(page) < 1000:
                return rows

    def _check_alias(self, skill, alias):
        """Reject aliases that identify another Skill by ID or alias."""
        if alias is None:
            return
        exact = self.database.get(SKILLS_TABLE, {"project_id": skill.project_id, "skill_id": alias})
        matches = self.database.select(SKILLS_TABLE, filters={"project_id": skill.project_id, "alias": alias}, limit=1)
        if (exact and exact["skill_id"] != skill.skill_id) or any(row["skill_id"] != skill.skill_id for row in matches):
            raise SkillConflictError(f"Alias already identifies another Skill: {alias}")

    @staticmethod
    def _key(skill):
        """Return the complete local identity."""
        return {"project_id": skill.project_id, "skill_id": skill.skill_id}

    @staticmethod
    def _advance(tx, skill, version_id):
        """Fail atomically if another writer already advanced the pointer."""
        changed = tx.update(
            SKILLS_TABLE,
            {"current_version_id": version_id, "updated_at": utcnow()},
            filters=SkillDataService._key(skill) | {"current_version_id": skill.current_version_id},
        )
        if changed != 1:
            raise SkillConflictError("Current version changed; reload before committing or switching")

    @staticmethod
    def _new_version(skill, snapshot, version_id, parents, message):
        """Build an immutable version from a complete source snapshot."""
        return SkillVersionRecord(
            project_id=skill.project_id,
            skill_id=skill.skill_id,
            version_id=version_id,
            name=skill.name,
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
