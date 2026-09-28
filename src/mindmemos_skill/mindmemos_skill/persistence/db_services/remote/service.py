"""Persist cloud bindings and local requests without calling cloud APIs."""

from ....infra.database import Database
from ...constants import (
    REMOTE_OPERATIONS_TABLE,
    REMOTE_SKILL_BINDINGS_TABLE,
    REMOTE_TRAJECTORY_BINDINGS_TABLE,
    REMOTE_VERSION_BINDINGS_TABLE,
    SKILL_VERSIONS_TABLE,
)
from .typing import (
    RemoteOperationRecord,
    RemoteSkillBindingRecord,
    RemoteTrajectoryBindingRecord,
    RemoteVersionBindingRecord,
)


class RemoteDataService:
    """Keep remote identities scoped to a deployment and cloud project.

    Args:
        database: Borrowed database with core and remote tables registered.
    """

    def __init__(self, database: Database):
        self.database = database

    def has_version(self, local_version_id: str) -> bool:
        """Check for a confirmed cloud mapping on any recorded deployment.

        Args:
            local_version_id: Exact local version identity, already checked by the caller.
        """
        return bool(self.database.count(REMOTE_VERSION_BINDINGS_TABLE, filters={"local_version_id": local_version_id}))

    def bind_skill(self, record: RemoteSkillBindingRecord) -> None:
        """Associate an existing local family with a confirmed cloud lineage.

        Args:
            record: Explicit local-to-cloud family association.
        """
        with self.database.transaction() as database:
            if not database.count(
                SKILL_VERSIONS_TABLE, filters={"project_id": record.project_id, "skill_id": record.skill_id}
            ):
                raise ValueError("Local Skill family does not exist")
            database.insert(REMOTE_SKILL_BINDINGS_TABLE, [record.model_dump(mode="python")])

    def bind_version(self, record: RemoteVersionBindingRecord) -> None:
        """Store a confirmed cloud version without recalculating cloud identity.

        Args:
            record: Cloud metadata/content and its corresponding local version.

        Raises:
            ValueError: Local and remote family associations disagree.
        """
        with self.database.transaction() as database:
            local = database.get(SKILL_VERSIONS_TABLE, record.local_version_id)
            family = database.get(
                REMOTE_SKILL_BINDINGS_TABLE,
                {
                    "project_id": record.project_id,
                    "remote_id": record.remote_id,
                    "cloud_project_id": record.cloud_project_id,
                    "cloud_skill_id": record.cloud_skill_id,
                },
            )
            if (
                local is None
                or family is None
                or local["project_id"] != record.project_id
                or local["skill_id"] != family["skill_id"]
            ):
                raise ValueError("Version must belong to the bound local Skill family")
            database.insert(REMOTE_VERSION_BINDINGS_TABLE, [record.model_dump(mode="python")])

    def bind_trajectory(self, record: RemoteTrajectoryBindingRecord) -> None:
        """Record a confirmed memory/add identity for a local trajectory.

        Args:
            record: Local attempt and confirmed cloud add-record identity.
        """
        self.database.insert(REMOTE_TRAJECTORY_BINDINGS_TABLE, [record.model_dump(mode="python")])

    def enqueue(self, record: RemoteOperationRecord) -> RemoteOperationRecord:
        """Deduplicate requests locally without assuming cloud idempotency.

        Args:
            record: Pending immutable request, excluding credentials.

        Returns:
            The new operation or the previously enqueued matching request.

        Raises:
            ValueError: The deduplication key is reused for a different request.
        """
        if record.status != "pending" or record.attempt_count != 0:
            raise ValueError("New operations must be pending and unattempted")
        with self.database.transaction() as database:
            database.upsert(
                REMOTE_OPERATIONS_TABLE,
                [record.model_dump(mode="python")],
                conflict_fields=["project_id", "remote_id", "cloud_project_id", "idempotency_key"],
            )
            rows = database.select(
                REMOTE_OPERATIONS_TABLE,
                filters={
                    "project_id": record.project_id,
                    "remote_id": record.remote_id,
                    "cloud_project_id": record.cloud_project_id,
                    "idempotency_key": record.idempotency_key,
                },
                limit=1,
            )
            existing = RemoteOperationRecord.model_validate(rows[0])
            if (
                existing.request_hash != record.request_hash
                or existing.request_payload != record.request_payload
                or existing.operation_type != record.operation_type
                or existing.local_version_id != record.local_version_id
                or existing.trajectory_id != record.trajectory_id
                or existing.cloud_skill_id != record.cloud_skill_id
            ):
                raise ValueError("Idempotency key already belongs to a different request")
            return existing
