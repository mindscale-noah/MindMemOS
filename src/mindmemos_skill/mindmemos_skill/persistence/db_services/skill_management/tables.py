"""Project-isolated graph snapshots with one selected version per graph."""

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Double,
    ForeignKeyConstraint,
    Index,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

from ....infra.database import TableSpec, Vector
from ....infra.database.typing import UTCDateTime
from ...constants import (
    SKILL_GRAPH_EDGES_TABLE,
    SKILL_GRAPH_NODES_TABLE,
    SKILL_GRAPH_VERSIONS_TABLE,
    SKILL_GRAPHS_TABLE,
    SKILL_VERSIONS_TABLE,
)
from .typing import (
    SkillGraphEdgeRecord,
    SkillGraphNodeRecord,
    SkillGraphRecord,
    SkillGraphVersionRecord,
)


def _scope():
    """Create fresh composite identity columns for one graph table."""
    return (
        Column("project_id", Text, primary_key=True, comment="Project owning this graph."),
        Column("graph_id", Text, primary_key=True, comment="Logical graph identity within its project."),
    )


def _graph_fk():
    """Constrain a record to an existing project-scoped graph."""
    return ForeignKeyConstraint(
        ["project_id", "graph_id"],
        [f"{SKILL_GRAPHS_TABLE}.project_id", f"{SKILL_GRAPHS_TABLE}.graph_id"],
        ondelete="RESTRICT",
    )


def _version_fk():
    """Constrain a record to an existing snapshot of the same graph."""
    return ForeignKeyConstraint(
        ["project_id", "graph_id", "graph_version_id"],
        [
            f"{SKILL_GRAPH_VERSIONS_TABLE}.project_id",
            f"{SKILL_GRAPH_VERSIONS_TABLE}.graph_id",
            f"{SKILL_GRAPH_VERSIONS_TABLE}.graph_version_id",
        ],
        ondelete="RESTRICT",
    )


def build_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Declare graph tables using the same metadata as local Skill versions.

    Args:
        metadata: Shared catalog containing the local skill_versions definition.

    Returns:
        Four table registrations with insert validation.
    """
    json_object = JSON().with_variant(JSONB(), "postgresql")
    text_list = JSON().with_variant(ARRAY(Text()), "postgresql")
    scope = ("project_id", "graph_id")
    snapshot = (*scope, "graph_version_id")
    graphs = Table(
        SKILL_GRAPHS_TABLE,
        metadata,
        *_scope(),
        Column("public_id", Text, nullable=True, comment="Stable prefixed graph identity."),
        Column("alias", Text, nullable=True, comment="Project-unique graph shortcut."),
        Column("name", Text, nullable=False, comment="Human-readable graph name."),
        Column("description", Text, nullable=True, comment="Optional purpose of the graph."),
        Column(
            "current_version_id", Uuid, nullable=True, comment="Selected ready graph snapshot; null means unpublished."
        ),
        Column(
            "created_at",
            UTCDateTime(),
            nullable=False,
            server_default=func.current_timestamp(),
            comment="Graph creation time.",
        ),
        Column(
            "updated_at",
            UTCDateTime(),
            nullable=False,
            server_default=func.current_timestamp(),
            comment="Last graph metadata update time.",
        ),
        Column(
            "is_deleted",
            Boolean,
            nullable=False,
            server_default=false(),
            comment="Hide this family after soft deletion.",
        ),
        Column("deleted_at", UTCDateTime(), nullable=True, comment="Soft deletion time; None while visible."),
        CheckConstraint("updated_at >= created_at"),
        ForeignKeyConstraint(
            [*scope, "current_version_id"],
            [
                f"{SKILL_GRAPH_VERSIONS_TABLE}.project_id",
                f"{SKILL_GRAPH_VERSIONS_TABLE}.graph_id",
                f"{SKILL_GRAPH_VERSIONS_TABLE}.graph_version_id",
            ],
            name="skill_graphs_current_version_fk",
            use_alter=True,
            ondelete="RESTRICT",
        ),
        Index("skill_graphs_project_updated_idx", "project_id", "updated_at"),
    )
    versions = Table(
        SKILL_GRAPH_VERSIONS_TABLE,
        metadata,
        *_scope(),
        Column("graph_version_id", Uuid, primary_key=True, comment="Immutable graph content identity."),
        Column("public_version_id", Text, nullable=True, comment="Stable prefixed external snapshot identity."),
        Column("status", Text, nullable=False, server_default="staging", comment="Candidate lifecycle stage."),
        Column("node_count", BigInteger, nullable=True, comment="Validated node count."),
        Column("edge_count", BigInteger, nullable=True, comment="Validated directed edge count."),
        Column("vector_count", BigInteger, nullable=True, comment="Validated non-null embedding count."),
        Column("snapshot_hash", Text, nullable=True, comment="Canonical graph content SHA-256."),
        Column("request_hash", Text, nullable=True, comment="Canonical build request SHA-256 for retry deduplication."),
        Column("actor_id", Text, nullable=True, comment="Actor who created this snapshot, when known."),
        Column("message", Text, nullable=True, comment="Explanation supplied when this snapshot was created."),
        Column(
            "metadata",
            json_object,
            nullable=False,
            server_default=text("'{}'"),
            comment="Effective algorithm and embedding configuration.",
        ),
        Column(
            "created_at",
            UTCDateTime(),
            nullable=False,
            server_default=func.current_timestamp(),
            comment="Candidate creation time.",
        ),
        Column("ready_at", UTCDateTime(), nullable=True, comment="Successful validation time."),
        _graph_fk(),
        CheckConstraint("status IN ('staging', 'validating', 'ready', 'failed')"),
        CheckConstraint("node_count IS NULL OR node_count >= 0"),
        CheckConstraint("edge_count IS NULL OR edge_count >= 0"),
        CheckConstraint(
            "vector_count IS NULL OR (node_count IS NOT NULL AND vector_count >= 0 AND vector_count <= node_count)"
        ),
        CheckConstraint("ready_at IS NULL OR ready_at >= created_at"),
        CheckConstraint(
            "status != 'ready' OR (node_count IS NOT NULL AND edge_count IS NOT NULL AND vector_count IS NOT NULL AND snapshot_hash IS NOT NULL AND ready_at IS NOT NULL)"
        ),
        Index("skill_graph_versions_status_idx", *scope, "status", "created_at"),
        Index("skill_graph_versions_request_uq", *scope, "request_hash", unique=True),
    )
    nodes = Table(
        SKILL_GRAPH_NODES_TABLE,
        metadata,
        *_scope(),
        Column("graph_version_id", Uuid, primary_key=True, comment="Snapshot containing this node."),
        Column("node_id", Text, primary_key=True, comment="Stable logical node identity."),
        Column("node_type", Text, nullable=False, server_default="skill", comment="Skill or scene node category."),
        Column("skill_version_id", Text, nullable=True, comment="Local immutable Skill version reference."),
        Column("owner_node_id", Text, nullable=True, comment="Owning Skill node for a scene in this snapshot."),
        Column("name", Text, nullable=False, comment="Display name; not a unique identity."),
        Column("description", Text, nullable=False, server_default="", comment="Parsed capability description."),
        Column("one_line_capability", Text, nullable=False, server_default="", comment="Short capability statement."),
        Column("inputs", text_list, nullable=False, comment="Input descriptions used for linking."),
        Column("outputs", text_list, nullable=False, comment="Output descriptions used for linking."),
        Column("domain_tags", text_list, nullable=False, comment="Domain labels used for retrieval."),
        Column("tooling", text_list, nullable=False, comment="Tool keywords used as retrieval features."),
        Column("example_tasks", text_list, nullable=False, comment="Explicit example task descriptions."),
        Column("rendered_snippet", Text, nullable=False, server_default="", comment="Display and embedding snippet."),
        Column(
            "metadata", json_object, nullable=False, server_default=text("'{}'"), comment="Non-core node attributes."
        ),
        Column("embedding", Vector(), nullable=True, comment="Finite nonzero float32 vector; variable dimensions."),
        Column("embedding_input_hash", Text, nullable=True, comment="SHA-256 of the exact embedding input."),
        _version_fk(),
        ForeignKeyConstraint(
            [*snapshot, "owner_node_id"],
            [
                f"{SKILL_GRAPH_NODES_TABLE}.project_id",
                f"{SKILL_GRAPH_NODES_TABLE}.graph_id",
                f"{SKILL_GRAPH_NODES_TABLE}.graph_version_id",
                f"{SKILL_GRAPH_NODES_TABLE}.node_id",
            ],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["project_id", "skill_version_id"],
            [f"{SKILL_VERSIONS_TABLE}.project_id", f"{SKILL_VERSIONS_TABLE}.version_id"],
            ondelete="RESTRICT",
        ),
        CheckConstraint(
            "(node_type = 'skill' AND skill_version_id IS NOT NULL AND owner_node_id IS NULL) OR (node_type = 'scene' AND skill_version_id IS NULL AND owner_node_id IS NOT NULL AND owner_node_id != node_id)"
        ),
        CheckConstraint(
            "(embedding IS NULL AND embedding_input_hash IS NULL) OR (embedding IS NOT NULL AND embedding_input_hash IS NOT NULL)"
        ),
        Index(
            "skill_graph_nodes_skill_version_uq",
            *snapshot,
            "skill_version_id",
            unique=True,
            postgresql_where=text("node_type = 'skill'"),
            sqlite_where=text("node_type = 'skill'"),
        ),
    )
    edges = Table(
        SKILL_GRAPH_EDGES_TABLE,
        metadata,
        *_scope(),
        Column("graph_version_id", Uuid, primary_key=True, comment="Snapshot containing both endpoints."),
        Column("edge_id", Uuid, primary_key=True, comment="Stable directed relationship identity."),
        Column("source_node_id", Text, nullable=False, comment="Source node in this snapshot."),
        Column("target_node_id", Text, nullable=False, comment="Target node in this snapshot."),
        Column("edge_type", Text, nullable=False, server_default="dependency", comment="Relationship meaning."),
        Column("weight", Double, nullable=False, server_default="1", comment="Finite nonnegative transition weight."),
        Column(
            "metadata",
            json_object,
            nullable=False,
            server_default=text("'{}'"),
            comment="Relationship evidence and extra attributes.",
        ),
        _version_fk(),
        ForeignKeyConstraint(
            [*snapshot, "source_node_id"],
            [
                f"{SKILL_GRAPH_NODES_TABLE}.project_id",
                f"{SKILL_GRAPH_NODES_TABLE}.graph_id",
                f"{SKILL_GRAPH_NODES_TABLE}.graph_version_id",
                f"{SKILL_GRAPH_NODES_TABLE}.node_id",
            ],
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            [*snapshot, "target_node_id"],
            [
                f"{SKILL_GRAPH_NODES_TABLE}.project_id",
                f"{SKILL_GRAPH_NODES_TABLE}.graph_id",
                f"{SKILL_GRAPH_NODES_TABLE}.graph_version_id",
                f"{SKILL_GRAPH_NODES_TABLE}.node_id",
            ],
            ondelete="RESTRICT",
        ),
        UniqueConstraint(*snapshot, "source_node_id", "target_node_id", "edge_type"),
        CheckConstraint("source_node_id != target_node_id"),
        CheckConstraint("edge_type IN ('dependency', 'containment', 'flow', 'semantic', 'workflow', 'alternative')"),
        CheckConstraint("weight >= 0 AND weight <= 1.7976931348623157e308"),
        Index("skill_graph_edges_source_idx", *snapshot, "source_node_id"),
        Index("skill_graph_edges_target_idx", *snapshot, "target_node_id"),
    )
    records = (
        (graphs, SkillGraphRecord),
        (versions, SkillGraphVersionRecord),
        (nodes, SkillGraphNodeRecord),
        (edges, SkillGraphEdgeRecord),
    )
    for table, _model in records:
        for column in table.c:
            if column.name in {
                "project_id",
                "graph_id",
                "node_id",
                "skill_version_id",
                "owner_node_id",
                "source_node_id",
                "target_node_id",
                "name",
                "actor_id",
            }:
                table.append_constraint(CheckConstraint(column.is_(None) | (func.length(column) > 0)))
            if column.name in {"snapshot_hash", "embedding_input_hash", "request_hash"}:
                table.append_constraint(
                    CheckConstraint(
                        f"{column.name} IS NULL OR (length({column.name}) = 64 AND {column.name} NOT GLOB '*[^0-9a-f]*')"
                    ).ddl_if(dialect="sqlite")
                )
                table.append_constraint(
                    CheckConstraint(f"{column.name} IS NULL OR {column.name} ~ '^[0-9a-f]{{64}}$'").ddl_if(
                        dialect="postgresql"
                    )
                )
        for name in ("inputs", "outputs", "domain_tags", "tooling", "example_tasks"):
            if name in table.c:
                table.append_constraint(CheckConstraint(f"json_type({name}) = 'array'").ddl_if(dialect="sqlite"))
                table.append_constraint(
                    CheckConstraint(f"array_position({name}, NULL) IS NULL").ddl_if(dialect="postgresql")
                )
        if "metadata" in table.c:
            table.append_constraint(CheckConstraint("json_type(metadata) = 'object'").ddl_if(dialect="sqlite"))
            table.append_constraint(CheckConstraint("jsonb_typeof(metadata) = 'object'").ddl_if(dialect="postgresql"))
    return tuple(TableSpec(name=table.name, table=table, insert_model=model) for table, model in records)
