from __future__ import annotations

from datetime import UTC, datetime

import pytest
from mindmemos_skill.agents import Agent, AgentConfig
from mindmemos_skill.service import (
    SkillAlgorithms,
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillCapabilityUnavailableError,
    SkillFinding,
    SkillOptimizationRequest,
    SkillOptimizationResult,
)
from mindmemos_skill.typing import AgentExecutionRequest, Skill, Trajectory


def make_skill(content: str = "Use the API carefully.") -> Skill:
    return Skill(
        skill_id="skill-1",
        version_id="version-1",
        version_label="1.0.0",
        content_hash="sha256:api-helper",
        name="api-helper",
        description="Helps call an API",
        blob={"SKILL.md": content},
        created_at=datetime(2026, 8, 4, tzinfo=UTC),
    )


class FakeAnalyzer:
    def __init__(self) -> None:
        self.requests: list[SkillAnalysisRequest] = []

    async def analyze(self, request: SkillAnalysisRequest) -> SkillAnalysisResult:
        self.requests.append(request)
        return SkillAnalysisResult(
            summary="One ambiguous instruction",
            findings=[SkillFinding(category="clarity", message="Specify the API response contract")],
        )


class FakeOptimizer:
    def __init__(self) -> None:
        self.requests: list[SkillOptimizationRequest] = []

    async def optimize(self, request: SkillOptimizationRequest) -> SkillOptimizationResult:
        self.requests.append(request)
        optimized = request.skill.model_copy(
            update={
                "blob": {
                    **request.skill.blob,
                    "SKILL.md": request.skill.blob["SKILL.md"] + "\nValidate the response schema.",
                }
            }
        )
        return SkillOptimizationResult(skill=optimized, changed=True, analysis=request.analysis)


class FakeAgent(Agent[AgentConfig]):
    async def execute(self, request: AgentExecutionRequest) -> Trajectory:
        raise AssertionError(f"unexpected execution: {request.trajectory_id}")


@pytest.mark.asyncio
async def test_algorithm_api_delegates_analyze_and_optimize() -> None:
    analyzer = FakeAnalyzer()
    optimizer = FakeOptimizer()
    algorithms = SkillAlgorithms(analyzer=analyzer, optimizer=optimizer)
    skill = make_skill()

    analysis = await algorithms.analyze(SkillAnalysisRequest(skill=skill))
    result = await algorithms.optimize(SkillOptimizationRequest(skill=skill, analysis=analysis))

    assert analyzer.requests[0].skill == skill
    assert optimizer.requests[0].analysis == analysis
    assert result.changed is True
    assert "Validate the response schema." in result.skill.blob["SKILL.md"]
    assert algorithms.capabilities == frozenset({"analyze", "optimize"})


def test_algorithm_api_owns_an_immutable_agent_registry() -> None:
    agent = FakeAgent(AgentConfig())
    source = {"executor": agent}

    algorithms = SkillAlgorithms(analyzer=FakeAnalyzer(), agents=source)
    source.clear()

    assert dict(algorithms.agents) == {"executor": agent}
    with pytest.raises(TypeError):
        algorithms.agents["other"] = agent  # type: ignore[index]


@pytest.mark.asyncio
async def test_missing_capability_raises_clear_error() -> None:
    algorithms = SkillAlgorithms(analyzer=FakeAnalyzer())

    with pytest.raises(SkillCapabilityUnavailableError, match="optimization"):
        await algorithms.optimize(SkillOptimizationRequest(skill=make_skill()))


def test_algorithm_api_requires_at_least_one_capability() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SkillAlgorithms()


def test_removed_runtime_facade_is_not_public() -> None:
    import mindmemos_skill

    assert not hasattr(mindmemos_skill, "MindMemosSkill")
