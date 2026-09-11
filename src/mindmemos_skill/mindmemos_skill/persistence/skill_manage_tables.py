"""Structured table catalog for the ``skill_manage`` capability.

Head/operation/lease records use ``project_id + graph_id`` scope; snapshot
records (generation/membership/node/edge/chunk/mapping) use
``project_id + graph_id + generation_id`` scope. Snapshot tables store the
generation-qualified physical identity in a dedicated ``record_id`` primary-key
field so the same logical node can coexist across generations without being
upserted over.
"""

from __future__ import annotations

from ..infra.database import FieldSpec, FieldType, IndexSpec, TableSpec

SKILL_MANAGE_GRAPHS = "skill_manage_graphs"
SKILL_MANAGE_GENERATIONS = "skill_manage_generations"
SKILL_MANAGE_OPERATIONS = "skill_manage_operations"
SKILL_MANAGE_WRITER_LEASES = "skill_manage_writer_leases"
SKILL_MANAGE_MEMBERSHIPS = "skill_manage_memberships"
SKILL_MANAGE_NODES = "skill_manage_nodes"
SKILL_MANAGE_EDGES = "skill_manage_edges"
SKILL_MANAGE_CHUNKS = "skill_manage_chunks"
SKILL_MANAGE_ID_MAPPINGS = "skill_manage_id_mappings"

SKILL_MANAGE_TABLE_NAMES = (
    SKILL_MANAGE_GRAPHS,
    SKILL_MANAGE_GENERATIONS,
    SKILL_MANAGE_OPERATIONS,
    SKILL_MANAGE_WRITER_LEASES,
    SKILL_MANAGE_MEMBERSHIPS,
    SKILL_MANAGE_NODES,
    SKILL_MANAGE_EDGES,
    SKILL_MANAGE_CHUNKS,
    SKILL_MANAGE_ID_MAPPINGS,
)


def build_skill_manage_tables() -> tuple[TableSpec, ...]:
    return (
        TableSpec(
            name=SKILL_MANAGE_GRAPHS,
            primary_key="graph_id",
            fields=(
                _text("graph_id", nullable=False),
                _integer("active_revision"),
                _text("active_generation_id"),
                _text("algorithm_name"),
                _text("algorithm_version"),
                _json("embedding_profile", nullable=False, default={}),
                _text("status", nullable=False, default="active"),
                _datetime("created_at", nullable=False),
                _datetime("updated_at", nullable=False),
            ),
        ),
        TableSpec(
            name=SKILL_MANAGE_OPERATIONS,
            primary_key="operation_id",
            fields=(
                _text("operation_id", nullable=False),
                _text("idempotency_key", nullable=False),
                _text("request_hash", nullable=False),
                _text("operation_type", nullable=False),
                _text("status", nullable=False),
                _integer("expected_revision"),
                _integer("base_revision"),
                _integer("result_revision"),
                _text("current_generation_id"),
                _integer("attempt_no", nullable=False, default=1),
                _text("algorithm_name"),
                _text("config_hash"),
                _json("request_summary", nullable=False, default={}),
                _json("progress_payload", nullable=False, default={}),
                _integer("result_schema_version"),
                _json("result_payload"),
                _text("error_code"),
                _text("error_message"),
                _boolean("error_retryable"),
                _json("error_details"),
                _text("pending_repair_reason"),
                _integer("writer_fencing_token"),
                _datetime("created_at", nullable=False),
                _datetime("updated_at", nullable=False),
                _datetime("first_started_at"),
                _datetime("attempt_started_at"),
                _datetime("validating_at"),
                _datetime("completed_at"),
                _datetime("repaired_at"),
                _integer("repair_count", nullable=False, default=0),
                _text("last_repair_action"),
            ),
            indexes=(IndexSpec(name="skill_manage_operations_scope_key_uq", fields=("idempotency_key",), unique=True),),
        ),
        TableSpec(
            name=SKILL_MANAGE_WRITER_LEASES,
            primary_key="graph_id",
            fields=(
                _text("graph_id", nullable=False),
                _text("owner_operation_id"),
                _integer("fencing_token", nullable=False, default=0),
                _datetime("acquired_at", nullable=False),
                _datetime("renewed_at", nullable=False),
                _datetime("expires_at", nullable=False),
            ),
        ),
        TableSpec(
            name=SKILL_MANAGE_GENERATIONS,
            primary_key="record_id",
            fields=(
                _text("record_id", nullable=False),
                _text("generation_id", nullable=False),
                _text("operation_id", nullable=False),
                _integer("attempt_no", nullable=False, default=1),
                _integer("base_revision"),
                _integer("target_revision"),
                _text("status", nullable=False),
                _text("content_hash"),
                _text("config_hash"),
                _integer("node_count", nullable=False, default=0),
                _integer("edge_count", nullable=False, default=0),
                _integer("vector_count", nullable=False, default=0),
                _integer("chunk_count", nullable=False, default=0),
                _text("counts_hash"),
                _text("failure_code"),
                _text("failure_message"),
                _boolean("failure_retryable"),
                _datetime("created_at", nullable=False),
                _datetime("ready_at"),
                _datetime("activated_at"),
                _datetime("retired_at"),
                _datetime("failed_at"),
                _datetime("deleted_at"),
            ),
            indexes=(
                IndexSpec(
                    name="skill_manage_generations_op_attempt_uq", fields=("operation_id", "attempt_no"), unique=True
                ),
            ),
        ),
        TableSpec(
            name=SKILL_MANAGE_MEMBERSHIPS,
            primary_key="record_id",
            fields=(
                _text("record_id", nullable=False),
                _text("membership_id", nullable=False),
                _text("skill_id", nullable=False),
                _text("version_id", nullable=False),
                _text("graph_node_id"),
                _text("generation_id", nullable=False),
                _text("content_hash"),
                _text("status", nullable=False, default="active"),
            ),
            indexes=(IndexSpec(name="skill_manage_memberships_skill_idx", fields=("skill_id",)),),
        ),
        TableSpec(
            name=SKILL_MANAGE_NODES,
            primary_key="record_id",
            fields=(
                _text("record_id", nullable=False),
                _text("graph_node_id", nullable=False),
                _text("generation_id", nullable=False),
                _text("node_type", nullable=False),
                _text("skill_id"),
                _text("version_id"),
                _text("parent_node_id"),
                _json("payload", nullable=False, default={}),
                _text("content_hash"),
            ),
        ),
        TableSpec(
            name=SKILL_MANAGE_EDGES,
            primary_key="record_id",
            fields=(
                _text("record_id", nullable=False),
                _text("edge_id", nullable=False),
                _text("generation_id", nullable=False),
                _text("source_node_id", nullable=False),
                _text("target_node_id", nullable=False),
                _text("edge_type", nullable=False),
                _float("weight", nullable=False, default=1.0),
                _float("confidence"),
                _text("provenance"),
                _json("payload", nullable=False, default={}),
            ),
            indexes=(IndexSpec(name="skill_manage_edges_type_idx", fields=("edge_type",)),),
        ),
        TableSpec(
            name=SKILL_MANAGE_CHUNKS,
            primary_key="record_id",
            fields=(
                _text("record_id", nullable=False),
                _text("chunk_id", nullable=False),
                _text("generation_id", nullable=False),
                _text("owner_node_id"),
                _text("skill_id"),
                _text("version_id"),
                _integer("ordinal", nullable=False, default=0),
                _text("content"),
                _text("content_hash"),
                _json("metadata", nullable=False, default={}),
            ),
        ),
        TableSpec(
            name=SKILL_MANAGE_ID_MAPPINGS,
            primary_key="record_id",
            fields=(
                _text("record_id", nullable=False),
                _text("graph_node_id", nullable=False),
                _text("generation_id", nullable=False),
                _text("vector_record_id", nullable=False),
                _text("mapping_type", nullable=False, default="node"),
            ),
            indexes=(IndexSpec(name="skill_manage_id_mappings_node_idx", fields=("graph_node_id",)),),
        ),
    )


def _text(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.TEXT, nullable=nullable, default=default)


def _uuid(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.UUID, nullable=nullable, default=default)


def _integer(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.INTEGER, nullable=nullable, default=default)


def _float(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.FLOAT, nullable=nullable, default=default)


def _boolean(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.BOOLEAN, nullable=nullable, default=default)


def _datetime(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.DATETIME, nullable=nullable, default=default)


def _json(name: str, *, nullable: bool = True, default=None) -> FieldSpec:
    return FieldSpec(name=name, field_type=FieldType.JSON, nullable=nullable, default=default)


__all__ = [
    "SKILL_MANAGE_CHUNKS",
    "SKILL_MANAGE_EDGES",
    "SKILL_MANAGE_GENERATIONS",
    "SKILL_MANAGE_GRAPHS",
    "SKILL_MANAGE_ID_MAPPINGS",
    "SKILL_MANAGE_MEMBERSHIPS",
    "SKILL_MANAGE_NODES",
    "SKILL_MANAGE_OPERATIONS",
    "SKILL_MANAGE_WRITER_LEASES",
    "SKILL_MANAGE_TABLE_NAMES",
    "build_skill_manage_tables",
]
