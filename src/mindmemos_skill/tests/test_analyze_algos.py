"""Contracts and no-network behavior for the analysis algorithms."""

import json

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent
from mindmemos_skill.algos import Algorithm
from mindmemos_skill.algos.analyze._json_parse import extract_json_object
from mindmemos_skill.algos.analyze.skill_judge.algorithm import SkillJudge
from mindmemos_skill.algos.analyze.trajectory_judge.algorithm import TrajectoryJudge
from mindmemos_skill.algos.analyze.typing import (
    SkillJudgeRequest,
    TrajectoryAnalysisInput,
    TrajectoryJudgeRequest,
    TrajectorySkillReference,
)
from mindmemos_skill.skill import Skill, SkillContent


def _agent_options(**overrides):
    """Return an inline OpenAI-compatible judge config for offline tests."""

    return {
        "class_name": "openai_chat",
        "model": "judge-test",
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "parse_max_attempts": 2,
        **overrides,
    }


def _install_model_outputs(monkeypatch, outputs):
    """Patch model transport while retaining the Agent parsing and retry path."""

    pending = iter(outputs)

    async def fake_run_messages(_self, **_kwargs):
        return AgentResult(
            output=next(pending),
            usage={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )

    monkeypatch.setattr(OpenAIChatAgent, "run_messages", fake_run_messages)


def _verdict(**overrides):
    """Return one valid static Judge verdict as model text."""

    value = {
        "scores": {
            "safety": {"score": 4, "reason": "safe"},
            "completeness": {"score": 5, "reason": "complete"},
            "executability": {"score": 4, "reason": "executable"},
            "maintainability": {"score": 3, "reason": "maintainable"},
        },
        "overall_summary": "The Skill is usable.",
        "concerns": ["One maintainability concern."],
        "evidence": ["SKILL.md lines 10-12"],
    }
    value.update(overrides)
    return json.dumps(value)


def test_extract_json_object_accepts_fenced_model_output():
    """Parse the markdown fencing emitted by several judge models."""

    assert extract_json_object('```json\n{"result": true}\n```') == {"result": True}


async def test_skill_judge_registers_and_runs_offline(monkeypatch):
    """Exercise the adapted static Judge through the current registry."""

    _install_model_outputs(monkeypatch, [f"```json\n{_verdict()}\n```"])
    judge = Algorithm.build("skill_judge", {"judge": _agent_options()})
    assert isinstance(judge, SkillJudge)

    skill = Skill(
        SkillContent(
            content="# Skill\n\nA useful procedure.\n",
            name="demo",
            description="A demo Skill.",
            standard_format=False,
        )
    )
    result = await judge.analyze(SkillJudgeRequest(skill=skill, request_id="request-1"))

    assert result.metrics["scores"] == {
        "safety": 4,
        "completeness": 5,
        "executability": 4,
        "maintainability": 3,
    }
    assert result.metrics["score_reasons"] == {
        "safety": "safe",
        "completeness": "complete",
        "executability": "executable",
        "maintainability": "maintainable",
    }
    assert result.metrics["score_mean"] == 4.0
    assert result.metadata["judge_type"] == "static_skill"
    assert result.metadata["algorithm"] == "skill_judge"
    assert result.metadata["request_id"] == "request-1"


def test_skill_judge_sliding_windows_overlap_and_cap():
    """Cover a long SKILL.md with bounded overlapping windows."""

    judge = Algorithm.build(
        "skill_judge",
        {
            "judge": _agent_options(),
            "per_file_cap": 256,
            "window_overlap": 64,
            "max_windows": 10,
        },
    )
    skill = Skill(
        SkillContent(
            content="x" * 700,
            name="long",
            description="Long Skill.",
            standard_format=False,
        )
    )

    windows, _manifest, metadata = judge._build_windows(skill)

    assert [window.coverage for window in windows] == [
        "SKILL.md@0-256",
        "SKILL.md@192-448",
        "SKILL.md@384-640",
        "SKILL.md@576-700",
    ]
    assert all(len(window.text) <= 256 for window in windows)
    assert metadata["windows"] == 4


def test_skill_judge_rejects_when_window_cap_would_omit_coverage():
    """Fail rather than silently auditing only a prefix of a large Skill."""

    judge = Algorithm.build(
        "skill_judge",
        {"judge": _agent_options(), "per_file_cap": 256, "window_overlap": 0, "max_windows": 1},
    )
    skill = Skill(
        SkillContent(
            content="x" * 600,
            name="too-large",
            description="Large Skill.",
            standard_format=False,
        )
    )

    with pytest.raises(ValueError, match="max_windows"):
        judge._build_windows(skill)


def test_skill_judge_skips_sensitive_binary_and_non_utf8_resources():
    """Keep unsafe or undecodable bytes out of the judge prompt."""

    judge = Algorithm.build("skill_judge", {"judge": _agent_options()})
    skill = Skill(
        SkillContent(
            content="# Skill\n",
            name="resources",
            description="Resource handling.",
            standard_format=False,
            resources={
                ".env": b"TOKEN=secret",
                "assets/blob.bin": b"\x00\x01",
                "notes/readme.txt": b"safe text",
                "notes/legacy.txt": b"\xff\xfe",
            },
        )
    )

    windows, manifest, metadata = judge._build_windows(skill)

    assert any("safe text" in window.text for window in windows)
    assert not any("TOKEN=secret" in window.text for window in windows)
    assert metadata["skipped_files"] == [
        ".env: sensitive_path",
        "assets/blob.bin: binary_file",
        "notes/legacy.txt: non_utf8",
    ]
    assert any("skipped | sensitive_path" in line for line in manifest)


async def test_trajectory_judge_registers_and_reports_usage_metrics(monkeypatch):
    """Exercise trajectory success evaluation and deterministic usage metrics."""

    success = {
        "success": False,
        "confidence": "high",
        "reason": "The task failed after using alpha.",
        "evidence": ["tool result mismatch"],
    }
    _install_model_outputs(monkeypatch, [json.dumps(success)])
    judge = Algorithm.build("trajectory_judge", {"judge": _agent_options()})
    assert isinstance(judge, TrajectoryJudge)

    trajectory = TrajectoryAnalysisInput(
        trajectory_id="trajectory-1",
        status="completed",
        task_instruction="Complete the task",
        injected_skills=[
            TrajectorySkillReference(name="alpha"),
            TrajectorySkillReference(name="beta"),
        ],
        skill_bindings=[
            TrajectorySkillReference(name="alpha", usage="unused"),
            TrajectorySkillReference(name="beta", usage="unused"),
        ],
        events=[
            {
                "role": "assistant",
                "content": "Calling alpha",
                "tool_calls": [
                    {
                        "function": {
                            "name": "Skill",
                            "arguments": json.dumps({"skill": "alpha"}),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "alpha failed"},
        ],
        reward_score=0.25,
    )

    result = await judge.analyze(TrajectoryJudgeRequest(trajectory=trajectory, request_id="trajectory-request"))

    assert result.metrics["success"] is False
    assert result.metrics["skills_used"] == ["alpha"]
    assert result.metrics["skill_usage_rate"] == 0.5
    assert result.metadata["judge_type"] == "trajectory"
    assert result.metadata["trajectory_ids"] == ["trajectory-1"]
    assert result.findings[0].category == "trajectory.execution"


async def test_trajectory_judge_attributes_failure_to_detected_skill(monkeypatch):
    """Run and normalize the optional attribution pass for a failed trajectory."""

    success = {
        "success": False,
        "confidence": "high",
        "reason": "alpha failed.",
        "evidence": ["tool result mismatch"],
    }
    attribution = {
        "segments": [
            {
                "skill": "alpha",
                "suspicion": 0.9,
                "reason": "The failing event belongs to alpha.",
                "evidence": ["alpha failed"],
            }
        ],
        "primary_cause": {
            "skill": "alpha",
            "reason": "alpha produced the failing behavior.",
            "confidence": "high",
        },
    }
    _install_model_outputs(monkeypatch, [json.dumps(success), json.dumps(attribution)])
    judge = Algorithm.build(
        "trajectory_judge",
        {"judge": _agent_options(), "attribute_failures": True},
    )
    trajectory = TrajectoryAnalysisInput(
        trajectory_id="trajectory-2",
        status="completed",
        task_instruction="Complete the task",
        injected_skills=[TrajectorySkillReference(name="alpha")],
        events=[
            {
                "role": "assistant",
                "content": "Calling alpha",
                "tool_calls": [
                    {
                        "function": {
                            "name": "Skill",
                            "arguments": json.dumps({"skill": "alpha"}),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "alpha failed"},
        ],
    )

    result = await judge.analyze(TrajectoryJudgeRequest(trajectory=trajectory))

    assert result.metrics["attribution"]["primary_cause"]["skill"] == "alpha"
    assert result.metadata["attribution_included"] is True
    assert result.findings[-1].category == "trajectory.attribution"
