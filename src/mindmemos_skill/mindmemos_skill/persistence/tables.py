"""Local physical table catalog for unified Skill facts and outbox state."""

from __future__ import annotations

from ..infra.database import FieldSpec, FieldType, IndexSpec, SchemaMigration, TableRegistry, TableSpec

SKILL_TABLE = "skill_versions"
TRAJECTORY_TABLE = "trajectories"
ALGORITHM_LOG_TABLE = "algorithm_logs"
SKILL_SYNC_STATE_TABLE = "skill_sync_state"
SKILL_REMOTE_OPERATION_TABLE = "skill_remote_operations"
SKILL_FAMILY_STATE_TABLE = SKILL_SYNC_STATE_TABLE


def build_persistence_tables() -> TableRegistry:
    specs = (
        TableSpec(
            name=SKILL_TABLE,
            primary_key="version_id",
            fields=(
                _text("skill_id", nullable=False),
                _text("version_id", nullable=False),
                _text("cloud_skill_id"),
                _json("parent_version_ids", nullable=False, default=[]),
                _text("name", nullable=False),
                _text("description"),
                _text("alias"),
                _text("bundle", nullable=False),
                _text("resources", nullable=False, default="{}"),
                _text("content_hash", nullable=False),
                _text("local_snapshot_hash", nullable=False),
                _text("status", nullable=False, default="draft"),
                _integer("version_revision", nullable=False, default=0),
                _text("version_label", nullable=False),
                _text("commit_message"),
                _json("metadata", nullable=False, default={}),
                _json("local_metadata", nullable=False, default={}),
                _text("origin", nullable=False),
                _datetime("created_at", nullable=False),
                _datetime("updated_at", nullable=False),
                _datetime("received_at"),
            ),
            indexes=(
                IndexSpec(name="skill_versions_label_uq", fields=("skill_id", "version_label"), unique=True),
                IndexSpec(name="skill_versions_hash_idx", fields=("content_hash",)),
                IndexSpec(
                    name="skill_versions_latest_idx",
                    fields=("skill_id", "status", "created_at", "version_id"),
                ),
            ),
            scope_scoped=False,
        ),
        TableSpec(
            name=SKILL_SYNC_STATE_TABLE,
            primary_key="skill_id",
            fields=(
                _text("skill_id", nullable=False),
                _datetime("last_version_sync_at"),
                _text("trajectory_pull_cursor"),
                _datetime("last_trajectory_pull_at"),
                _datetime("created_at", nullable=False),
                _datetime("updated_at", nullable=False),
            ),
            scope_scoped=False,
        ),
        TableSpec(
            name=TRAJECTORY_TABLE,
            primary_key="trajectory_id",
            fields=(
                _text("trajectory_id", nullable=False),
                _text("trajectory_hash", nullable=False),
                _text("task_id", nullable=False),
                _text("rollout_id", nullable=False),
                _integer("attempt_no", nullable=False, default=0),
                _text("rollout_type", nullable=False),
                _text("task_instruction", nullable=False),
                _text("task_system_prompt"),
                _json("task_tags", nullable=False, default=[]),
                _json("task_metadata", nullable=False, default={}),
                _text("running_dir"),
                _json("env_metadata", nullable=False, default={}),
                _json("injected_skills", nullable=False, default=[]),
                _text("agent_type", nullable=False),
                _json("agent_profile", nullable=False, default={}),
                _text("status", nullable=False),
                _json("trajectory", nullable=False, default=[]),
                _json("skill_bindings", nullable=False, default=[]),
                _float("reward_score"),
                _text("reward_detail"),
                _json("reward_metadata", nullable=False, default={}),
                _datetime("started_at", nullable=False),
                _datetime("finished_at"),
                _integer("n_turn", nullable=False, default=0),
                _text("error_info"),
                _json("metadata", nullable=False, default={}),
                _integer("metadata_revision", nullable=False, default=0),
                _datetime("metadata_updated_at"),
                _text("source", nullable=False),
                _text("source_add_record_id"),
                _datetime("created_at", nullable=False),
                _datetime("received_at"),
            ),
            indexes=(
                IndexSpec(
                    name="trajectories_rollout_attempt_uq",
                    fields=("rollout_id", "attempt_no"),
                    unique=True,
                ),
                IndexSpec(name="trajectories_task_idx", fields=("task_id",)),
                IndexSpec(name="trajectories_incremental_idx", fields=("started_at", "trajectory_id")),
            ),
            scope_scoped=False,
        ),
        TableSpec(
            name=ALGORITHM_LOG_TABLE,
            primary_key="log_id",
            fields=(
                _text("log_id", nullable=False),
                _text("algorithm_name", nullable=False),
                _text("algorithm_version"),
                _text("component_name", nullable=False),
                _text("step_name", nullable=False),
                _text("status"),
                _json("payload", nullable=False, default={}),
                _datetime("created_at", nullable=False),
            ),
            indexes=(
                IndexSpec(name="algorithm_logs_algorithm_created_idx", fields=("algorithm_name", "created_at")),
            ),
            scope_scoped=False,
        ),
        TableSpec(
            name=SKILL_REMOTE_OPERATION_TABLE,
            primary_key="operation_id",
            fields=(
                _text("operation_id", nullable=False),
                _text("operation_type", nullable=False),
                _text("skill_id"),
                _text("cloud_skill_id"),
                _text("version_id"),
                _text("trajectory_id"),
                _text("request_hash", nullable=False),
                _text("status", nullable=False),
                _integer("attempt_count", nullable=False, default=0),
                _datetime("lease_expires_at"),
                _datetime("next_retry_at"),
                _text("last_error_code"),
                _json("remote_result"),
                _datetime("created_at", nullable=False),
                _datetime("updated_at", nullable=False),
            ),
            indexes=(
                IndexSpec(
                    name="skill_remote_operations_status_idx",
                    fields=("operation_type", "status", "next_retry_at"),
                ),
                IndexSpec(name="skill_remote_operations_skill_idx", fields=("skill_id",)),
            ),
            scope_scoped=False,
        ),
    )
    registry = TableRegistry(
        specs,
        migrations=(
            SchemaMigration(
                namespace="skill-persistence-v2",
                version=1,
                name="create_unified_local_skill_schema",
                tables=(
                    SKILL_TABLE,
                    SKILL_SYNC_STATE_TABLE,
                    TRAJECTORY_TABLE,
                    ALGORITHM_LOG_TABLE,
                    SKILL_REMOTE_OPERATION_TABLE,
                ),
            ),
        ),
    )
    registry.freeze()
    return registry


def _text(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.TEXT, nullable=nullable, default=default)


def _integer(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.INTEGER, nullable=nullable, default=default)


def _float(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.FLOAT, nullable=nullable, default=default)


def _datetime(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.DATETIME, nullable=nullable, default=default)


def _json(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.JSON, nullable=nullable, default=default)


__all__ = [
    "ALGORITHM_LOG_TABLE",
    "SKILL_FAMILY_STATE_TABLE",
    "SKILL_REMOTE_OPERATION_TABLE",
    "SKILL_SYNC_STATE_TABLE",
    "SKILL_TABLE",
    "TRAJECTORY_TABLE",
    "build_persistence_tables",
]
