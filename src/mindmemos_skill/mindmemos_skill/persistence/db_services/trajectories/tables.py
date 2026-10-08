"""Business tables owned by the trajectories data service."""

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    Float,
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
from ...constants import TRAJECTORIES_TABLE, TRAJECTORY_REPORT_ACTOR_INDEX, TRAJECTORY_REPORT_CURSOR_INDEX
from .typing import TrajectoryRecord


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
        Column("skill_id", Text, nullable=True, comment="Family associated with a reported attempt."),
        Column("skill_version_id", Text, nullable=True, comment="Exact loaded version of a reported attempt."),
        Column("user_id", Text, nullable=True, comment="Business user associated with a reported attempt."),
        Column("agent_id", Text, nullable=True, comment="Agent associated with a reported attempt."),
        Column("session_id", Text, nullable=True, comment="Logical session of a reported attempt."),
        Column("received_at", UTCDateTime(), nullable=True, comment="Server receipt time of a reported attempt."),
        UniqueConstraint("project_id", "rollout_id", "attempt_no"),
        UniqueConstraint("project_id", "trajectory_id"),
        CheckConstraint("attempt_no >= 0 AND n_turn >= 0"),
        Index("trajectories_task_idx", "task_id"),
        Index(TRAJECTORY_REPORT_CURSOR_INDEX, "project_id", "skill_id", "received_at", "trajectory_id"),
        Index(TRAJECTORY_REPORT_ACTOR_INDEX, "project_id", "skill_id", "user_id", "agent_id"),
    )
    return (TableSpec(name=TRAJECTORIES_TABLE, table=table, insert_model=TrajectoryRecord),)
