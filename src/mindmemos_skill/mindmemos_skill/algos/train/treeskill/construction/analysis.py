"""Outcome-aware trajectory analysis for TreeSkill."""

from __future__ import annotations

import json
from typing import Any

from .....agents import OpenAIChatAgent
from .....common.json import extract_json_object, read_json
from ..config import TreeSkillTrainerConfig
from ..env_adapter import TreeSkillEnvAdapter
from ..typing import AnalysisOutcome, AnnotationMode, TraceEvidence, TrajectoryAnalysisRecord, TreeSkillAnalysisContext
from ..typing import TrainingRollout as Trajectory


def parse_analysis(text: str) -> TrajectoryAnalysisRecord:
    """Validate the model's analysis JSON without repairing its contents.

    Args:
        text: Raw model output.

    Returns:
        Strictly validated analysis record.
    """
    return TrajectoryAnalysisRecord.model_validate(extract_json_object(text, repair=False))


async def analyze_trajectory(
    inputs: AnalysisInput,
    *,
    agent: OpenAIChatAgent,
    success_score_threshold: float,
    env_adapter: TreeSkillEnvAdapter,
) -> AnalysisOutcome:
    """Analyze one trajectory without writing experiment files.

    Args:
        inputs: Trainer-scheduled evidence; private tool inputs are not read here.
        agent: Direct chat agent configured with the analysis parser.
        success_score_threshold: Cutoff separating success from failure.
        env_adapter: Environment-specific public message renderer.

    Returns:
        Validated analysis and model evidence, or an explicit item failure.
    """
    evidence = inputs.evidence
    outcome = AnalysisOutcome()
    source = (
        "unlabeled" if evidence.score is None else ("success" if evidence.score >= success_score_threshold else "error")
    )
    try:
        messages = env_adapter.build_analysis_messages(TreeSkillAnalysisContext(evidence=evidence, source=source))
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome
    try:
        response = await agent.run(messages=messages)
    except Exception as exc:
        outcome.failed_calls = 1
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome
    outcome.responses.append(response)
    try:
        record = response.output
        if not isinstance(record, TrajectoryAnalysisRecord):
            raise TypeError("Analysis agent must return TrajectoryAnalysisRecord")
        if record.instance_id != evidence.trajectory_id:
            raise ValueError("analysis instance_id does not match the source trajectory")
        if record.task_id != evidence.task_id or record.record_source != source:
            raise ValueError("analysis task or outcome source does not match the source trajectory")
        outcome.record = record
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


def render_trajectory(events: list[dict[str, Any]], max_chars: int) -> str:
    """Render ordered trajectory events into the compact LLM transcript format.

    Args:
        events: Original ordered environment events.
        max_chars: Maximum retained characters per event under the reference truncation rule.
    """

    lines: list[str] = []
    for event in events:
        if "role" in event and "content" in event:
            rendered = f"[{event.get('role', '?')}] {event.get('content', '')}"
        elif "text" in event:
            rendered = f"[text] {event.get('text', '')}"
        elif "url" in event:
            rendered = f"[url] {event.get('url', '')}"
        elif "file_name" in event:
            rendered = f"[file] {event.get('file_name', '')}"
        else:
            rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
        lines.append(_truncate(rendered, max_chars))
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n…[{len(text) - max_chars} chars elided]…\n{tail}"


def prepare_evidence(
    records: list[Trajectory], config: TreeSkillTrainerConfig, expected_skill_hash: str
) -> list[TraceEvidence]:
    """Extract reference-formatted evidence from successful physical executions.

    Args:
        records: Completed executions; failed slots must be excluded by the trainer.
        config: Annotation policy, transcript budget and source matching requirement.
        expected_skill_hash: Complete initial Skill identity, including resources.

    Returns:
        Evidence in physical completion order, with private labels only when allowed.

    Raises:
        ValueError: An execution used a different initial Skill.
    """
    evidence = []
    for record in records:
        if config.require_skill_match:
            request = read_json(record.work_dir / "audit/request.json")
            if request["skills"][0]["skill_hash"] != expected_skill_hash:
                raise ValueError("Sampled trajectory does not match the initial Skill")
        labeled = config.annotation_mode != AnnotationMode.IGNORE
        evidence.append(
            TraceEvidence(
                trajectory_id=record.trajectory_id,
                task_id=record.sample.task.id,
                transcript=render_trajectory(record.result.trajectory, config.transcript_max_chars),
                score=record.result.reward.score if labeled else None,
                annotation_detail=record.result.reward.detail if labeled else None,
                annotation_metadata=(record.result.reward.model_extra or {}) if labeled else {},
            )
        )
    return evidence
