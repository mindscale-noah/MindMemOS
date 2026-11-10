"""Independent evaluation result table in the Skill database."""

from sqlalchemy import JSON, Boolean, Column, Float, Index, MetaData, Table, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB

from ....infra.database import TableSpec
from ....infra.database.typing import UTCDateTime
from ...constants import SKILL_EVALUATIONS_TABLE
from .typing import EvaluationRecord


def build_tables(metadata: MetaData) -> tuple[TableSpec, ...]:
    """Register evaluation storage without foreign keys to Skill or graph tables."""

    json_type = JSON().with_variant(JSONB(), "postgresql")
    evaluations = Table(
        SKILL_EVALUATIONS_TABLE,
        metadata,
        Column("evaluation_id", Text, primary_key=True, comment="Globally unique evaluation identity."),
        Column("project_id", Text, nullable=False, comment="Owning project identity."),
        Column("evaluation_key", Text, nullable=False, comment="Stable result-affecting input digest."),
        Column("judge_type", Text, nullable=False, comment="Static Skill or trajectory analyzer."),
        Column("skill_id", Text, comment="Optional Skill family business identity."),
        Column("skill_version_id", Text, comment="Optional exact Skill version business identity."),
        Column("trajectory_id", Text, comment="Optional exact trajectory business identity."),
        Column("algorithm", Text, nullable=False, comment="Analyzer implementation name."),
        Column("algorithm_version", Text, nullable=False, comment="Analyzer behavior version."),
        Column("prompt_version", Text, nullable=False, comment="Prompt family revision."),
        Column("status", Text, nullable=False, comment="completed or failed."),
        Column("summary", Text, comment="Successful assessment summary."),
        Column("score_mean", Float, comment="Mean static score."),
        Column("scores", json_type, nullable=False, comment="Named static scores."),
        Column("success", Boolean, comment="Trajectory success verdict."),
        Column("confidence", Text, comment="Trajectory verdict confidence."),
        Column("skill_usage_rate", Float, comment="Trajectory Skill usage ratio."),
        Column("metrics", json_type, nullable=False, comment="Complete analyzer metrics."),
        Column("findings", json_type, nullable=False, comment="Analyzer findings."),
        Column("artifacts", json_type, nullable=False, comment="Analyzer artifacts."),
        Column("metadata", json_type, nullable=False, comment="Sanitized provenance metadata."),
        Column("error", Text, comment="Failure description without credentials."),
        Column("created_at", UTCDateTime(), nullable=False, comment="Initial result time."),
        Column("updated_at", UTCDateTime(), nullable=False, comment="Latest result update time."),
        UniqueConstraint("project_id", "evaluation_key", name="skill_evaluations_project_key_uq"),
        Index("skill_evaluations_version_idx", "project_id", "skill_version_id", "judge_type", "updated_at"),
        Index("skill_evaluations_trajectory_idx", "project_id", "trajectory_id", "updated_at"),
        Index("skill_evaluations_status_idx", "project_id", "status", "updated_at"),
    )
    return (TableSpec(name=SKILL_EVALUATIONS_TABLE, table=evaluations, insert_model=EvaluationRecord),)


__all__ = ["build_tables"]
