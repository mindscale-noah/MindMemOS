"""Cache-first orchestration around pure Skill analysis algorithms."""

import math
from datetime import UTC, datetime
from typing import Awaitable, Callable, Literal, TypeVar, cast

from pydantic import BaseModel, JsonValue

from mindmemos_skill.common.ids import generate_id

from ....algos import Algorithm
from ....algos.analyze import (
    SkillAnalysisResult,
    SkillAnalyzer,
    SkillIdentity,
    SkillJudgeRequest,
    TrajectoryAnalysisInput,
    TrajectoryJudgeRequest,
    TrajectorySkillReference,
)
from ....api.typing import (
    EvaluationQueryRequest,
    EvaluationQueryResult,
    SkillAnalyzeRequest,
    TrajectoryAnalyzeRequest,
)
from ....common.checkpoint import fingerprint
from ....common.secrets import redact_credentials
from ....common.skill_versions import snapshot_from_version
from ....errors import TrajectoryNotFoundError
from ....logging import get_logger
from ....persistence import EvaluationRecord, SkillVersionRecord, TrajectoryRecord
from ....persistence.store import SkillPersistence
from ....skill import Skill
from ...analyze import AnalyzePipeline
from ...base import Pipeline
from .config import AnalyzePipelineConfig

RequestT = TypeVar("RequestT", bound=BaseModel)


def _number(value: JsonValue | None) -> float | None:
    """Return a finite numeric value without treating booleans as scores."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


@Pipeline.register("analyze")
class BuiltinAnalyzePipeline(AnalyzePipeline[AnalyzePipelineConfig]):
    """Reuse completed evaluations and persist every execution outcome."""

    config_type = AnalyzePipelineConfig
    """Selected pure analyzers and bounded recommendation query settings."""

    def __init__(self, config: AnalyzePipelineConfig, *, persistence: SkillPersistence):
        """Build analyzers while borrowing the service-owned persistence."""

        if not isinstance(persistence, SkillPersistence):
            raise TypeError("Analyze pipeline requires SkillPersistence")
        super().__init__(config, persistence=persistence)
        self.skill_judge: SkillAnalyzer | None = None
        self.trajectory_judge: SkillAnalyzer | None = None
        if config.skill_judge is not None:
            self.skill_judge = cast(
                SkillAnalyzer,
                Algorithm.build(config.skill_judge.class_name, config.skill_judge.config),
            )
        if config.trajectory_judge is not None:
            self.trajectory_judge = cast(
                SkillAnalyzer,
                Algorithm.build(config.trajectory_judge.class_name, config.trajectory_judge.config),
            )

    @property
    def evaluation_query_limit(self) -> int:
        """Return the configured bound for recommendation evaluation reads."""

        return self.config.evaluation_query_limit

    async def analyze_skill(self, request: SkillAnalyzeRequest) -> EvaluationRecord:
        """Return a cached static result or execute and persist the exact version."""

        analyzer = self.skill_judge
        if analyzer is None:
            raise NotImplementedError("Static Skill Judge is not configured")
        version = await self.persistence.skills.get(request.skill_version_id)
        identity = self._skill_identity(version)
        key = self._skill_key(version, identity, analyzer)
        if not request.force_refresh:
            cached = await self.persistence.evaluations.get_completed(key)
            if cached is not None:
                return cached
        judge_request = SkillJudgeRequest(
            skill=Skill.from_snapshot(snapshot_from_version(version)),
            identity=identity,
            request_id=request.request_id,
        )
        return await self._execute(
            analyzer=analyzer,
            judge_type="static_skill",
            evaluation_key=key,
            skill_id=version.skill_id,
            skill_version_id=version.version_id,
            trajectory_id=None,
            callback=analyzer.analyze,
            request=judge_request,
        )

    async def analyze_trajectory(self, request: TrajectoryAnalyzeRequest) -> EvaluationRecord:
        """Return a cached trajectory result or execute and persist the exact attempt."""

        analyzer = self.trajectory_judge
        if analyzer is None:
            raise NotImplementedError("Trajectory Judge is not configured")
        record = await self.persistence.trajectories.get(request.trajectory_id)
        if record is None:
            raise TrajectoryNotFoundError(f"trajectory {request.trajectory_id!r} was not found")
        trajectory = self._trajectory_input(record)
        verifier_score = request.verifier_score if request.verifier_score is not None else trajectory.reward_score
        key = self._trajectory_key(trajectory, request.attribute, verifier_score, analyzer)
        if not request.force_refresh:
            cached = await self.persistence.evaluations.get_completed(key)
            if cached is not None:
                return cached
        judge_request = TrajectoryJudgeRequest(
            trajectory=trajectory,
            attribute=request.attribute,
            verifier_score=verifier_score,
            request_id=request.request_id,
        )
        return await self._execute(
            analyzer=analyzer,
            judge_type="trajectory",
            evaluation_key=key,
            skill_id=record.skill_id,
            skill_version_id=record.skill_version_id,
            trajectory_id=record.trajectory_id,
            callback=analyzer.analyze,
            request=judge_request,
        )

    async def query(self, request: EvaluationQueryRequest) -> EvaluationQueryResult:
        """Read completed evaluations for exact versions in the active project."""

        rows = await self.persistence.evaluations.list_for_versions(
            request.skill_version_ids,
            judge_types=list(request.judge_types) if request.judge_types is not None else None,
            completed_only=True,
            limit=request.limit,
        )
        order = {version_id: index for index, version_id in enumerate(request.skill_version_ids)}

        def sort_key(row: EvaluationRecord) -> tuple[int, str, float]:
            return (
                order.get(row.skill_version_id or "", len(order)),
                row.judge_type,
                -row.updated_at.timestamp(),
            )

        rows.sort(key=sort_key)
        return EvaluationQueryResult(items=rows)

    async def _execute(
        self,
        *,
        analyzer: SkillAnalyzer,
        judge_type: Literal["static_skill", "trajectory"],
        evaluation_key: str,
        skill_id: str | None,
        skill_version_id: str | None,
        trajectory_id: str | None,
        callback: Callable[[RequestT], Awaitable[SkillAnalysisResult]],
        request: RequestT,
    ) -> EvaluationRecord:
        """Execute one analyzer outside storage transactions and persist its outcome."""

        now = datetime.now(UTC)
        algorithm = analyzer.algorithm_name
        algorithm_version = str(analyzer.config.model_dump().get("algorithm_version", "1"))
        prompt_version = str(analyzer.config.model_dump().get("prompt_version", "1"))
        try:
            result = await callback(request)
        except Exception as exc:
            failed = EvaluationRecord(
                evaluation_id=generate_id("eval_"),
                project_id=self.persistence.project_id,
                evaluation_key=evaluation_key,
                judge_type=judge_type,
                skill_id=skill_id,
                skill_version_id=skill_version_id,
                trajectory_id=trajectory_id,
                algorithm=algorithm,
                algorithm_version=algorithm_version,
                prompt_version=prompt_version,
                status="failed",
                error=f"{type(exc).__name__}: {exc}"[:2000],
                created_at=now,
                updated_at=datetime.now(UTC),
            )
            try:
                await self.persistence.evaluations.save(failed)
            except Exception:
                get_logger(__name__).exception("analyze.failed_result_persist_failed")
            raise
        record = self._success_record(
            result=result,
            judge_type=judge_type,
            evaluation_key=evaluation_key,
            skill_id=skill_id,
            skill_version_id=skill_version_id,
            trajectory_id=trajectory_id,
            algorithm=algorithm,
            algorithm_version=algorithm_version,
            prompt_version=prompt_version,
            created_at=now,
        )
        return await self.persistence.evaluations.save(record)

    def _skill_identity(self, version: SkillVersionRecord) -> SkillIdentity:
        """Build immutable algorithm identity from one stored version."""

        return SkillIdentity(
            project_id=self.persistence.project_id,
            skill_id=version.skill_id,
            version_id=version.version_id,
            name=version.name,
            description=version.description,
            content_hash=version.content_hash,
            skill_hash=version.skill_hash,
        )

    def _skill_key(self, version: SkillVersionRecord, identity: SkillIdentity, analyzer: SkillAnalyzer) -> str:
        """Hash all static result-affecting inputs while excluding request identity."""

        return fingerprint(
            {
                "judge_type": "static_skill",
                "project_id": self.persistence.project_id,
                "skill_version_id": version.version_id,
                "content_hash": identity.content_hash,
                "skill_hash": identity.skill_hash,
                "algorithm": analyzer.algorithm_name,
                "config": redact_credentials(analyzer.config.model_dump(mode="json"), preserve_execution_settings=True),
            }
        )

    def _trajectory_key(
        self,
        trajectory: TrajectoryAnalysisInput,
        attribute: bool | None,
        verifier_score: float | None,
        analyzer: SkillAnalyzer,
    ) -> str:
        """Hash trajectory overrides and analyzer settings but not request identity."""

        return fingerprint(
            {
                "judge_type": "trajectory",
                "project_id": self.persistence.project_id,
                "trajectory_id": trajectory.trajectory_id,
                "attribute": attribute,
                "verifier_score": verifier_score,
                "algorithm": analyzer.algorithm_name,
                "config": redact_credentials(analyzer.config.model_dump(mode="json"), preserve_execution_settings=True),
            }
        )

    def _trajectory_input(self, record: TrajectoryRecord) -> TrajectoryAnalysisInput:
        """Project a stored trajectory into the algorithm-owned input boundary."""

        return TrajectoryAnalysisInput(
            trajectory_id=record.trajectory_id,
            status=record.status,
            task_instruction=record.task_instruction,
            events=[event for event in record.trajectory if isinstance(event, dict)],
            injected_skills=[
                TrajectorySkillReference.model_validate(item)
                for item in record.injected_skills
                if isinstance(item, dict)
            ],
            skill_bindings=[
                TrajectorySkillReference.model_validate(item)
                for item in record.skill_bindings
                if isinstance(item, dict)
            ],
            skill_id=record.skill_id,
            skill_version_id=record.skill_version_id,
            reward_score=record.reward_score,
        )

    def _success_record(
        self,
        *,
        result: SkillAnalysisResult,
        judge_type: Literal["static_skill", "trajectory"],
        evaluation_key: str,
        skill_id: str | None,
        skill_version_id: str | None,
        trajectory_id: str | None,
        algorithm: str,
        algorithm_version: str,
        prompt_version: str,
        created_at: datetime,
    ) -> EvaluationRecord:
        """Project one transport-neutral result into the independent storage model."""

        metrics = result.metrics
        scores = metrics.get("scores")
        success = metrics.get("success")
        confidence = metrics.get("confidence")
        return EvaluationRecord(
            evaluation_id=generate_id("eval_"),
            project_id=self.persistence.project_id,
            evaluation_key=evaluation_key,
            judge_type=judge_type,
            skill_id=skill_id,
            skill_version_id=skill_version_id,
            trajectory_id=trajectory_id,
            algorithm=algorithm,
            algorithm_version=algorithm_version,
            prompt_version=prompt_version,
            status="completed",
            summary=result.summary,
            score_mean=_number(metrics.get("score_mean")),
            scores=cast(dict[str, JsonValue], scores) if isinstance(scores, dict) else {},
            success=success if isinstance(success, bool) else None,
            confidence=confidence if isinstance(confidence, str) else None,
            skill_usage_rate=_number(metrics.get("skill_usage_rate")),
            metrics=metrics,
            findings=[finding.model_dump(mode="json") for finding in result.findings],
            artifacts=result.artifacts,
            metadata=result.metadata,
            created_at=created_at,
            updated_at=datetime.now(UTC),
        )


__all__ = ["BuiltinAnalyzePipeline"]
