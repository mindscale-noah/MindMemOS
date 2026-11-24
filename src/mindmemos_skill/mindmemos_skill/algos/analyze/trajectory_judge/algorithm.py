"""Dynamic LLM-based analysis of package-native trajectories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import JsonValue

from ....agents import AgentResult, OpenAIChatAgent
from ...base import Algorithm
from .._json_parse import extract_json_object
from ..base import SkillAnalyzer
from ..typing import SkillAnalysisResult, SkillFinding, TrajectoryAnalysisInput, TrajectoryJudgeRequest
from .config import TrajectoryJudgeConfig
from .models import AttributionVerdict, SuccessVerdict
from .prompts import build_attribution_prompt, build_success_prompt
from .segments import (
    SkillSegment,
    build_skill_segments,
    detect_skill_invocations,
    event_tokens,
    normalize_segment_label,
    render_attribution_trace,
    render_event_text,
)


@Algorithm.register("trajectory_judge")
class TrajectoryJudge(SkillAnalyzer[TrajectoryJudgeConfig, TrajectoryJudgeRequest]):
    """Evaluate completed trajectories and optionally attribute failures."""

    config_type = TrajectoryJudgeConfig
    algorithm_name: ClassVar[str] = "trajectory_judge"

    def __init__(self, config: TrajectoryJudgeConfig) -> None:
        """Configure the structured-output Agent used for success evaluation."""

        super().__init__(config)
        self._success_agent = OpenAIChatAgent(config.judge, output_parser=self._parse_success)

    async def analyze(self, request: TrajectoryJudgeRequest) -> SkillAnalysisResult:
        """Evaluate one completed trajectory without persistence."""

        trajectory = request.trajectory
        if trajectory.status == "running":
            raise ValueError(f"trajectory {trajectory.trajectory_id!r} is still running")

        verifier_score = self._verifier_score(request)
        invocations = detect_skill_invocations(trajectory)
        segments = build_skill_segments(trajectory, invocations)
        trace = self._render_trajectory(trajectory, self.config.max_trajectory_chars)
        prompt = build_success_prompt(
            trace=trace,
            trajectory_count=1,
            verifier_score=verifier_score,
        )
        response = await self._success_agent.run(messages=[{"role": "user", "content": prompt}])
        verdict = response.output
        if not isinstance(verdict, SuccessVerdict):
            raise TypeError("trajectory_judge returned an invalid parsed verdict")

        attribution: dict[str, JsonValue] | None = None
        should_attribute = (
            request.attribute if request.attribute is not None else self.config.attribute_failures
        )
        if self._should_attribute(verdict, verifier_score, should_attribute):
            attribution = await self._attribute(
                trajectory=trajectory,
                segments=segments,
                verdict=verdict,
                verifier_score=verifier_score,
            )
        return self._to_result(request, verdict, attribution, verifier_score, response, segments)

    def _should_attribute(
        self,
        verdict: SuccessVerdict,
        verifier_score: float | None,
        should_attribute: bool,
    ) -> bool:
        """Run attribution for judge-declared failures or configured partial successes."""

        if not should_attribute:
            return False
        if not verdict.success:
            return True
        return (
            self.config.attribute_on_partial_success
            and verifier_score is not None
            and verifier_score < 1.0
        )

    @staticmethod
    def _parse_success(content: str) -> SuccessVerdict:
        """Parse one success verdict from model text."""

        return SuccessVerdict.model_validate(extract_json_object(content))

    def _make_attribution_parser(self, allowed_labels: set[str]) -> Callable[[str], AttributionVerdict]:
        """Build a parser that normalizes segment labels reported by the model."""

        def parse(content: str) -> AttributionVerdict:
            verdict = AttributionVerdict.model_validate(extract_json_object(content))
            for segment in verdict.segments:
                segment.skill = normalize_segment_label(segment.skill, allowed_labels)
            if verdict.primary_cause.skill is not None:
                verdict.primary_cause.skill = normalize_segment_label(
                    verdict.primary_cause.skill,
                    allowed_labels,
                )
            return verdict

        return parse

    @staticmethod
    def _verifier_score(request: TrajectoryJudgeRequest) -> float | None:
        """Resolve the optional verifier override or stored trajectory reward."""

        value = request.verifier_score
        if value is None:
            value = request.trajectory.reward_score
        if value is None:
            return None
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("trajectory verifier score must be between 0 and 1")
        return float(value)

    def _render_trajectory(self, trajectory: TrajectoryAnalysisInput, cap: int) -> str:
        """Render trajectory events with bounded head and tail context."""

        events = [self._event_text(event) for event in trajectory.events]
        if not events:
            return "[no events recorded]"
        joined = "\n".join(events)
        if len(joined) <= cap:
            return joined
        budget = max(128, cap - 80)
        head = budget // 3
        tail = budget - head
        return joined[:head] + "\n[events truncated]\n" + joined[-tail:]

    def _event_text(self, event: dict[str, Any]) -> str:
        """Render one event within the configured per-event character budget."""

        return render_event_text(event, self.config.max_event_chars)

    async def _attribute(
        self,
        *,
        trajectory: TrajectoryAnalysisInput,
        segments: list[SkillSegment],
        verdict: SuccessVerdict,
        verifier_score: float | None,
    ) -> dict[str, JsonValue]:
        """Attribute a failed run to its detected Skill segments."""

        segment_text = render_attribution_trace(
            trajectory,
            segments,
            max_chars=self.config.max_trajectory_chars,
            render_event=self._event_text,
        )
        allowed = {segment.label for segment in segments}
        agent = OpenAIChatAgent(
            self.config.judge,
            output_parser=self._make_attribution_parser(allowed),
        )
        prompt = build_attribution_prompt(
            task=trajectory.task_instruction,
            segments=segment_text,
            verifier_score=verifier_score,
            verdict=verdict,
        )
        response = await agent.run(messages=[{"role": "user", "content": prompt}])
        parsed = response.output
        if not isinstance(parsed, AttributionVerdict):
            raise TypeError("trajectory_judge returned an invalid attribution verdict")
        if any(segment.skill not in allowed for segment in parsed.segments):
            raise ValueError("trajectory_judge attribution contains an unknown segment label")
        if parsed.primary_cause.skill is not None and parsed.primary_cause.skill not in allowed:
            raise ValueError("trajectory_judge primary cause contains an unknown segment label")
        return parsed.model_dump(mode="json")

    def _to_result(
        self,
        request: TrajectoryJudgeRequest,
        verdict: SuccessVerdict,
        attribution: dict[str, JsonValue] | None,
        verifier_score: float | None,
        response: AgentResult,
        segments: list[SkillSegment],
    ) -> SkillAnalysisResult:
        """Convert judge verdicts and deterministic usage metrics into one result."""

        trajectory = request.trajectory
        injected_skills = sorted(
            {
                *(reference.name for reference in trajectory.skill_bindings),
                *(reference.name for reference in trajectory.injected_skills),
            },
            key=str.lower,
        )
        injected_norm = {name.lower() for name in injected_skills}
        used_norm = {name.lower() for name in self._used_skill_labels(trajectory)}

        # Runtime event coverage varies by Agent: merge binding usage with
        # invocation segments so explicit calls still count when bindings are absent.
        display_by_lower = {name.lower(): name for name in injected_skills}
        for segment in segments:
            if segment.name is None:
                continue
            lowered = segment.name.lower()
            display_by_lower.setdefault(lowered, segment.name)
            used_norm.add(lowered)
        skills_used = sorted((display_by_lower[name] for name in used_norm), key=str.lower)

        findings: list[SkillFinding] = []
        if not verdict.success:
            findings.append(
                SkillFinding(
                    category="trajectory.execution",
                    message=verdict.reason,
                    severity="error",
                    evidence=verdict.evidence,
                )
            )
        elif verdict.confidence != "high":
            findings.append(
                SkillFinding(
                    category="trajectory.confidence",
                    message=verdict.reason,
                    severity="warning",
                    evidence=verdict.evidence,
                )
            )
        if attribution is not None:
            primary = attribution["primary_cause"]
            if not isinstance(primary, dict):
                raise TypeError("trajectory_judge attribution has an invalid primary cause")
            primary_skill = primary.get("skill")
            findings.append(
                SkillFinding(
                    category="trajectory.attribution",
                    message=str(primary["reason"]),
                    severity="warning" if primary_skill is not None else "info",
                    evidence=[
                        str(segment["reason"])
                        for segment in attribution["segments"]
                        if isinstance(segment, dict)
                    ],
                    metadata={"primary_cause": primary_skill},
                )
            )

        metrics: dict[str, JsonValue] = {
            "success": verdict.success,
            "confidence": verdict.confidence,
            "trajectory_count": 1,
            "skills_injected": injected_skills,
            "skills_used": skills_used,
            "skill_usage_rate": (
                len(used_norm) / len(injected_norm) if injected_norm else 0.0
            ),
            "skill_segments": [
                {
                    "label": segment.label,
                    "events": [segment.start + 1, segment.end],
                    "event_count": segment.end - segment.start,
                    "tokens": dict(segment.tokens),
                }
                for segment in segments
            ],
            "tokens": self._token_totals(trajectory),
        }
        if verifier_score is not None:
            metrics["verifier_score"] = verifier_score
        if response.usage:
            metrics["judge_tokens"] = {
                "input": response.usage.get("input_tokens"),
                "output": response.usage.get("output_tokens"),
                "total": response.usage.get("total_tokens"),
                "parse_attempts": response.usage.get("parse_attempts"),
            }
        metadata: dict[str, JsonValue] = {
            "judge_type": "trajectory",
            "algorithm": self.algorithm_name,
            "algorithm_version": self.config.algorithm_version,
            "prompt_version": self.config.prompt_version,
            "request_id": request.request_id,
            "skill_id": trajectory.skill_id,
            "version_id": trajectory.skill_version_id,
            "trajectory_ids": [trajectory.trajectory_id],
            "attribution_included": attribution is not None,
        }
        if attribution is not None:
            metrics["attribution"] = attribution
        return SkillAnalysisResult(
            summary=verdict.reason,
            findings=findings,
            metrics=metrics,
            metadata=metadata,
        )

    @staticmethod
    def _used_skill_labels(trajectory: TrajectoryAnalysisInput) -> list[str]:
        """Return bindings with resolved usage other than ``unused``."""

        return [
            binding.name
            for binding in trajectory.skill_bindings
            if binding.usage is not None and binding.usage != "unused"
        ]

    @staticmethod
    def _token_totals(trajectory: TrajectoryAnalysisInput) -> dict[str, int]:
        """Sum recorded model tokens across the trajectory."""

        totals = {"input": 0, "output": 0, "cached": 0, "total": 0}
        for event in trajectory.events:
            bucket = event_tokens(event)
            totals["input"] += bucket["input"]
            totals["output"] += bucket["output"]
            totals["cached"] += bucket["cached"]
            totals["total"] += bucket["input"] + bucket["output"]
        return totals


__all__ = ["TrajectoryJudge"]
