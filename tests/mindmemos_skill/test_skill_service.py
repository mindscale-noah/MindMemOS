from __future__ import annotations

from datetime import UTC, datetime

import pytest
from mindmemos_skill.agents import Agent, AgentConfig
from mindmemos_skill.service import (
    SkillAlgorithms,
    SkillAnalysisRequest,
    SkillAnalysisResult,
    SkillCandidate,
    SkillCapabilityUnavailableError,
    SkillFinding,
    Trace2SkillInput,
    Trace2SkillOutput,
)
from mindmemos_skill.typing import (
    AgentExecutionRequest,
    ExecutionInfo,
    Rollout,
    Skill,
    Task,
    Trajectory,
    TrajectoryStatus,
)


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


def make_trajectory(skill: Skill) -> Trajectory:
    return Trajectory(
        trajectory_id="trajectory-1",
        task=Task(task_id="task-1", instruction="Call the API"),
        rollout=Rollout(rollout_id="rollout-1"),
        injected_skills=[skill],
        events=[{"role": "user", "content": "Call the API"}],
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=datetime(2026, 8, 4, tzinfo=UTC),
            finished_at=datetime(2026, 8, 4, 0, 0, 1, tzinfo=UTC),
        ),
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
        self.requests: list[Trace2SkillInput] = []

    async def optimize(self, request: Trace2SkillInput) -> Trace2SkillOutput[dict[str, str]]:
        self.requests.append(request)
        candidate = SkillCandidate(
            blob={
                **request.base_skill.blob,
                "SKILL.md": request.base_skill.blob["SKILL.md"] + "\nValidate the response schema.",
            },
            resources=request.base_skill.resources,
        )
        return Trace2SkillOutput(candidate=candidate, report={"status": "optimized"})


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
    result = await algorithms.optimize(Trace2SkillInput(base_skill=skill, trajectories=[make_trajectory(skill)]))

    assert analyzer.requests[0].skill == skill
    assert analysis.summary == "One ambiguous instruction"
    assert optimizer.requests[0].base_skill == skill
    assert result.changed is True
    assert result.candidate is not None
    assert "Validate the response schema." in result.candidate.blob["SKILL.md"]
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
    skill = make_skill()

    with pytest.raises(SkillCapabilityUnavailableError, match="optimization"):
        await algorithms.optimize(Trace2SkillInput(base_skill=skill, trajectories=[make_trajectory(skill)]))


def test_algorithm_api_requires_at_least_one_capability() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SkillAlgorithms()


def test_removed_runtime_facade_is_not_public() -> None:
    import mindmemos_skill

    assert not hasattr(mindmemos_skill, "MindMemosSkill")
