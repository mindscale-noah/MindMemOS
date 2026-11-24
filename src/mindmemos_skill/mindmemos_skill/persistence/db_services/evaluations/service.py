"""Independent evaluation result persistence."""

from ....infra.database import Database
from ....infra.database.scoped import ScopedDatabase
from ....infra.database.typing import Filter
from ...constants import SKILL_EVALUATIONS_TABLE
from .typing import EvaluationRecord

_UPDATE_FIELDS = (
    "judge_type",
    "skill_id",
    "skill_version_id",
    "trajectory_id",
    "algorithm",
    "algorithm_version",
    "prompt_version",
    "status",
    "summary",
    "score_mean",
    "scores",
    "success",
    "confidence",
    "skill_usage_rate",
    "metrics",
    "findings",
    "artifacts",
    "metadata",
    "error",
    "updated_at",
)


class EvaluationDataService:
    """Persist and query judge results through business identities only."""

    def __init__(self, database: Database | ScopedDatabase) -> None:
        """Borrow a database with the independent evaluation table registered."""

        self.database = database

    async def save(self, record: EvaluationRecord) -> EvaluationRecord:
        """Idempotently write one result for its stable project and evaluation key."""

        await self.database.upsert(
            SKILL_EVALUATIONS_TABLE,
            [record.model_dump(mode="python")],
            conflict_fields={"project_id", "evaluation_key"},
            update_fields=list(_UPDATE_FIELDS),
        )
        stored = await self.get(record.evaluation_key)
        if stored is None:
            raise RuntimeError("Evaluation upsert did not produce a readable record")
        return stored

    async def get(self, evaluation_key: str, *, completed_only: bool = False) -> EvaluationRecord | None:
        """Read the latest result state for one key inside the active project."""

        filters: list[Filter] = [Filter(field="evaluation_key", value=evaluation_key)]
        if completed_only:
            filters.append(Filter(field="status", value="completed"))
        rows = await self.database.select(SKILL_EVALUATIONS_TABLE, filters=filters, limit=1)
        return EvaluationRecord.model_validate(rows[0]) if rows else None

    async def get_completed(self, evaluation_key: str) -> EvaluationRecord | None:
        """Read only a reusable completed cache entry."""

        return await self.get(evaluation_key, completed_only=True)

    async def list_for_versions(
        self,
        skill_version_ids: list[str],
        *,
        judge_types: list[str] | None = None,
        completed_only: bool = True,
        limit: int = 200,
    ) -> list[EvaluationRecord]:
        """Read results for exact versions with stable version and timestamp ordering."""

        if not skill_version_ids:
            return []
        filters: list[Filter] = [Filter(field="skill_version_id", op="in", value=skill_version_ids)]
        if judge_types:
            filters.append(Filter(field="judge_type", op="in", value=judge_types))
        if completed_only:
            filters.append(Filter(field="status", value="completed"))
        rows = await self.database.select(
            SKILL_EVALUATIONS_TABLE,
            filters=filters,
            order_by=["skill_version_id", "judge_type", "-updated_at"],
            limit=limit,
        )
        return [EvaluationRecord.model_validate(row) for row in rows]


__all__ = ["EvaluationDataService"]
