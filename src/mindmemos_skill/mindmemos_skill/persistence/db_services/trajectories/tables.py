"""Business tables owned by the trajectories data service."""

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    text,
)

from ....infra.database import TableSpec
from ....infra.database.typing import UTCDateTime
from ...constants import (
    TRAJECTORIES_TABLE,
    TRAJECTORY_CONSUMPTIONS_TABLE,
    TRAJECTORY_REPORT_ACTOR_INDEX,
    TRAJECTORY_REPORT_CURSOR_INDEX,
    TRAJECTORY_VERSIONS_TABLE,
)
from .typing import TrajectoryConsumption, TrajectoryRecord, TrajectoryVersionIndex


def build_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Declare tables without creating resources.

    Args:
        metadata: Shared metadata for cross-service foreign keys.

    Returns:
        Table registrations owned by this service.
    """
    table = Table(
        TRAJECTORIES_TABLE,
        metadata,
        Column("project_id", Text, nullable=False, comment="Project owning this physical rollout attempt."),
        Column("trajectory_id", Text, primary_key=True, comment="Local identity of one physical rollout attempt."),
        Column("trajectory_hash", Text, nullable=False, comment="Hash of local events or reported messages."),
        Column(
            "source", Text, nullable=False, server_default=text("'local'"), comment="Local or Harness-reported source."
        ),
        Column("task_id", Text, nullable=False, comment="Local task identity or generated report task identity."),
        Column("rollout_id", Text, nullable=False, comment="Logical rollout identity shared by retries."),
        Column("attempt_no", Integer, nullable=False, comment="Zero-based physical attempt number."),
        Column("rollout_type", Text, nullable=False, comment="Purpose of the rollout, such as inference or training."),
        Column("task_instruction", Text, nullable=False, comment="Instruction presented for this attempt."),
        Column("task_system_prompt", Text, nullable=True, comment="Optional task-level system prompt."),
        Column(
            "task_metadata",
            JSON,
            nullable=False,
            comment="Dataset identity, task configuration, and extra task fields.",
        ),
        Column("env_ref", Text, nullable=False, comment="Environment implementation or registered name."),
        Column(
            "env_metadata", JSON, nullable=False, comment="Environment configuration and reproducibility information."
        ),
        Column("running_dir", Text, nullable=True, comment="Optional local execution directory."),
        Column("injected_skills", JSON, nullable=False, comment="Ordered Skill configurations injected into the run."),
        Column("skill_bindings", JSON, nullable=False, comment="Local Skill/version references and usage information."),
        Column("agent_type", Text, nullable=False, comment="Agent implementation identifier."),
        Column("agent_profile", JSON, nullable=False, comment="Agent configuration snapshot without credentials."),
        Column("status", Text, nullable=False, comment="Execution outcome recorded by the rollout owner."),
        Column("trajectory", JSON, nullable=False, comment="Ordered interaction records, including tool calls."),
        Column("reward_score", Float, nullable=True, comment="Optional finite environment-defined score."),
        Column("reward_detail", Text, nullable=True, comment="Optional explanation of the reward."),
        Column(
            "reward_metadata",
            JSON,
            nullable=False,
            comment="Extra evaluation metrics, including success when available.",
        ),
        Column("started_at", UTCDateTime(), nullable=True, comment="Execution start time when known."),
        Column("finished_at", UTCDateTime(), nullable=True, comment="Execution completion time; absent while running."),
        Column("n_turn", Integer, nullable=False, comment="Number of agent interaction turns."),
        Column("error_info", Text, nullable=True, comment="Optional execution failure details."),
        Column("metadata", JSON, nullable=False, comment="Additional local execution annotations."),
        Column("created_at", UTCDateTime(), nullable=False, comment="Time this trajectory record was stored."),
        Column("user_id", Text, nullable=True, comment="Business user associated with a reported attempt."),
        Column("agent_id", Text, nullable=True, comment="Agent associated with a reported attempt."),
        Column("session_id", Text, nullable=True, comment="Logical session of a reported attempt."),
        Column("received_at", UTCDateTime(), nullable=True, comment="Server receipt time of a reported attempt."),
        UniqueConstraint("project_id", "rollout_id", "attempt_no"),
        UniqueConstraint("project_id", "trajectory_id"),
        CheckConstraint("attempt_no >= 0 AND n_turn >= 0"),
        Index("trajectories_task_idx", "task_id"),
    )
    versions = Table(
        TRAJECTORY_VERSIONS_TABLE,
        metadata,
        Column("project_id", Text, primary_key=True, comment="Project owning the execution."),
        Column("trajectory_id", Text, primary_key=True, comment="Physical attempt with the shared payload."),
        Column("skill_version_id", Text, primary_key=True, comment="Exact version bound to the attempt."),
        Column("skill_id", Text, nullable=True, comment="Skill family for family-level lookups."),
        Column("first_for_family", Boolean, nullable=False, comment="Representative binding for a family page."),
        Column("task_id", Text, nullable=False, comment="Task identity used in summaries."),
        Column("source", Text, nullable=False, comment="Whether the attempt was reported or local."),
        Column("user_id", Text, nullable=True, comment="Business user for actor filtering."),
        Column("agent_id", Text, nullable=True, comment="Executing agent for actor filtering."),
        Column("status", Text, nullable=False, comment="Outcome for summary filtering."),
        Column("started_at", UTCDateTime(), nullable=True, comment="Execution start time."),
        Column("finished_at", UTCDateTime(), nullable=True, comment="Execution finish time."),
        Column("created_at", UTCDateTime(), nullable=False, comment="Persistence time for evidence and statistics."),
        Column("received_at", UTCDateTime(), nullable=True, comment="Receipt time for reported attempts."),
        ForeignKeyConstraint(
            ["project_id", "trajectory_id"],
            [f"{TRAJECTORIES_TABLE}.project_id", f"{TRAJECTORIES_TABLE}.trajectory_id"],
            ondelete="CASCADE",
        ),
        Index(TRAJECTORY_REPORT_CURSOR_INDEX, "project_id", "skill_id", "received_at", "trajectory_id"),
        Index(TRAJECTORY_REPORT_ACTOR_INDEX, "project_id", "skill_id", "user_id", "agent_id"),
        Index("trajectories_version_created_idx", "project_id", "skill_version_id", "created_at", "trajectory_id"),
    )
    consumption = Table(
        TRAJECTORY_CONSUMPTIONS_TABLE,
        metadata,
        Column("project_id", Text, primary_key=True, comment="Project owning the consumed trajectory."),
        Column("trajectory_id", Text, primary_key=True, comment="Distinct trajectory successfully processed."),
        Column("skill_version_id", Text, primary_key=True, comment="Exact input Skill version."),
        Column("run_id", Text, nullable=False, comment="First successful evolution run."),
        Column("consumed_at", UTCDateTime(), nullable=False, comment="Time successful use was recorded."),
        ForeignKeyConstraint(
            ["project_id", "trajectory_id", "skill_version_id"],
            [
                f"{TRAJECTORY_VERSIONS_TABLE}.project_id",
                f"{TRAJECTORY_VERSIONS_TABLE}.trajectory_id",
                f"{TRAJECTORY_VERSIONS_TABLE}.skill_version_id",
            ],
            ondelete="CASCADE",
        ),
        Index("trajectory_consumptions_version_idx", "project_id", "skill_version_id"),
    )
    return (
        TableSpec(name=TRAJECTORIES_TABLE, table=table, insert_model=TrajectoryRecord),
        TableSpec(name=TRAJECTORY_VERSIONS_TABLE, table=versions, insert_model=TrajectoryVersionIndex),
        TableSpec(name=TRAJECTORY_CONSUMPTIONS_TABLE, table=consumption, insert_model=TrajectoryConsumption),
    )
