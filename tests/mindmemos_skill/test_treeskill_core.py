from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.trace2skill.treeskill import (
    TreeSkill,
    TreeSkillConfig,
    TreeSkillRouter,
    compile_tree_metadata,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
)
from mindmemos_skill.algos.trace2skill.treeskill.models import TreeRoutingResult
from mindmemos_skill.llm import ChatResponse
from mindmemos_skill.registry import ComponentType, get_component
from mindmemos_skill.skill_runtime import SkillRuntimeCoordinator, SkillRuntimeRegistry
from mindmemos_skill.skill_runtime.runtimes import TreeSkillRuntime, TreeSkillRuntimeMetadata
from mindmemos_skill.typing import (
    AgentExecutionRequest,
    Environment,
    ExecutionInfo,
    Reward,
    Rollout,
    Skill,
    SkillInjectionMode,
    Task,
    Trace2SkillInput,
    Trajectory,
    TrajectoryStatus,
    compute_skill_content_hash,
)


class ScriptedChatModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls.append({"task": task, "messages": messages, **kwargs})
        content = self.responses.pop(0)
        parser = kwargs.get("format_parser")
        parsed = parser(content) if parser is not None else None
        return ChatResponse(finish_reason="stop", content=content, model="fake", parsed=parsed)


class FixedRouter:
    def __init__(self, selected_ids: tuple[str, ...]) -> None:
        self.selected_ids = selected_ids
        self.calls: list[dict[str, Any]] = []

    async def route(self, **kwargs: Any) -> TreeRoutingResult:
        self.calls.append(kwargs)
        skill = kwargs["skill"]
        return TreeRoutingResult(
            selected_node_ids=self.selected_ids,
            content_node_ids=self.selected_ids,
            ancestor_node_ids=(),
            skill_content="ignored by runtime",
            full_char_count=len(skill.content),
            routed_char_count=0,
        )


def make_skill(
    content: str,
    *,
    runtime_type: str = "static",
    runtime_metadata: dict[str, Any] | None = None,
) -> Skill:
    blob = {"SKILL.md": content}
    return Skill(
        skill_id="skill-tree",
        version_id="version-tree-1",
        version_label="1.0.0",
        content_hash=compute_skill_content_hash(blob),
        name="demo",
        description="Demo guidance",
        blob=blob,
        runtime_type=runtime_type,
        runtime_schema_version=1,
        runtime_metadata=runtime_metadata or {},
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )


def test_markdown_tree_metadata_round_trip_and_ordered_subtree_rendering() -> None:
    content = """# Workbook

Inspect the workbook.

## Formulas

Write formulas.

## Saving

Save and verify the output.
"""
    tree = parse_skill_markdown(content)

    assert [(node.node_id, node.heading) for node in tree.nodes] == [
        ("001", "Workbook"),
        ("002", "Formulas"),
        ("003", "Saving"),
    ]
    metadata = compile_tree_metadata(tree)
    restored = parse_tree_with_metadata(content, metadata)
    assert restored.node_by_id["002"].heading == "Formulas"

    routed = render_selected_subtrees(restored, ("002",))
    assert "# Workbook" in routed
    assert "## Formulas" in routed
    assert "## Saving" not in routed


@pytest.mark.asyncio
async def test_treeskill_runtime_routes_then_reuses_normal_skill_projection() -> None:
    content = """# Workbook

Inspect the workbook.

## Formulas

Write formulas.

## Saving

Save and verify the output.
"""
    metadata = compile_tree_metadata(parse_skill_markdown(content))
    skill = make_skill(content, runtime_type="treeskill", runtime_metadata=metadata)
    router = FixedRouter(("002",))
    coordinator = SkillRuntimeCoordinator(SkillRuntimeRegistry([TreeSkillRuntime(router=router)]))
    task = Task(task_id="task-1", instruction="Add a formula.")

    async with coordinator.on_task(task=task, skills=[skill], context={"env_ref": "demo"}) as scope:
        projected = await scope.projected_skills()
        trace = scope.trace()["skills"][0]

    assert len(router.calls) == 1
    assert router.calls[0]["task"] == task
    assert "# Workbook" in projected[0].content
    assert "## Formulas" in projected[0].content
    assert "## Saving" not in projected[0].content
    assert projected[0].runtime_metadata == metadata
    assert trace["runtime_type"] == "treeskill"
    assert trace["metadata"]["content_node_ids"] == ["002"]
    assert trace["metadata"]["routed_char_count"] < trace["metadata"]["full_char_count"]


@pytest.mark.asyncio
async def test_react_agent_uses_treeskill_callback_before_existing_system_prompt_injection() -> None:
    content = """# Workbook

Inspect the workbook.

## Formulas

Write formulas.

## Saving

Save and verify the output.
"""
    metadata = compile_tree_metadata(parse_skill_markdown(content))
    skill = make_skill(content, runtime_type="treeskill", runtime_metadata=metadata)
    policy_model = ScriptedChatModel(["done"])
    router = FixedRouter(("002",))
    agent = ReactAgent(
        {"skill_injection_mode": SkillInjectionMode.SYSTEM_PROMPT},
        llm=policy_model,
    )
    agent.register_skill_runtime(TreeSkillRuntime(router=router))
    request = AgentExecutionRequest(
        trajectory_id="trajectory-runtime-1",
        task=Task(task_id="task-runtime-1", instruction="Add a formula.", system_prompt="Use the skill."),
        rollout=Rollout(rollout_id="rollout-runtime-1"),
        skills=[skill],
    )

    trajectory = await agent.execute(request)

    injected_prompt = policy_model.calls[0]["messages"][0]["content"]
    assert "## Formulas" in injected_prompt
    assert "## Saving" not in injected_prompt
    assert trajectory.metadata["skill_runtime"]["skills"][0]["metadata"]["content_node_ids"] == ["002"]


def test_treeskill_runtime_metadata_rejects_inconsistent_edges() -> None:
    metadata = compile_tree_metadata(parse_skill_markdown("# Root\n\nRoot text.\n\n## Child\n\nChild text.\n"))
    metadata["nodes"][0]["child_ids"] = ["missing"]

    with pytest.raises(ValueError, match="unknown child IDs"):
        TreeSkillRuntimeMetadata.model_validate(metadata)


def test_treeskill_registry_rejects_metadata_that_does_not_match_skill_content() -> None:
    original = "# Root\n\nOriginal guidance.\n"
    metadata = compile_tree_metadata(parse_skill_markdown(original))
    stale = make_skill("# Root\n\nChanged guidance.\n", runtime_type="treeskill", runtime_metadata=metadata)
    registry = SkillRuntimeRegistry([TreeSkillRuntime(router=FixedRouter(("001",)))])

    with pytest.raises(ValueError, match="content hash"):
        registry.validate(stale)


def test_treeskill_is_registered_as_an_optimize_algorithm() -> None:
    component = get_component(type=ComponentType.ALGO, name="treeskill")

    assert component.factory is TreeSkill
    assert component.config_model is TreeSkillConfig
    assert component.capabilities == frozenset({"optimize"})


@dataclass(frozen=True)
class AlgorithmContext:
    models: dict[str, Any]
    agents: dict[str, Any]
    config_hash: str = "config-hash"


@pytest.mark.asyncio
async def test_treeskill_candidate_persists_tree_as_runtime_metadata() -> None:
    base = make_skill("# Workbook\n\nInspect the workbook.\n")
    now = datetime(2026, 8, 12, tzinfo=UTC)
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        task=Task(task_id="task-1", instruction="Update the workbook."),
        rollout=Rollout(rollout_id="rollout-1"),
        environment=Environment(env_ref="spreadsheetbench"),
        injected_skills=[base],
        events=[{"role": "assistant", "content": "Saved output.xlsx after verification."}],
        reward=Reward(score=1.0, detail="passed"),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )
    model = ScriptedChatModel(
        [
            json.dumps(
                {
                    "instance_id": "trajectory-1",
                    "task_id": "task-1",
                    "record_source": "success",
                    "items": [
                        {
                            "item_id": "i1",
                            "kind": "success_memory",
                            "title": "Verify output",
                            "description": "The output was saved and verified.",
                            "content": "Verify the saved workbook before finishing.",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "instance_id": "trajectory-1",
                    "evidence": [
                        {
                            "evidence_id": "e1",
                            "reusable_lesson": "Verify the saved workbook before finishing.",
                            "target_node_id": "001",
                            "rationale": "This is general workbook guidance.",
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "rationale": "Add supported verification guidance.",
                    "edits": [
                        {
                            "operation": "update_node",
                            "content": "Inspect the workbook and verify the saved output.",
                            "rationale": "The trajectory supports this workflow.",
                        }
                    ],
                }
            ),
        ]
    )

    result = await TreeSkill(
        config=TreeSkillConfig(min_trajectories=1, max_trajectories=1),
        context=AlgorithmContext(models={"chat": model}, agents={}),
    ).optimize(Trace2SkillInput(run_id="run-1", base_skill=base, trajectories=[trajectory]))

    assert result.candidate is not None
    assert result.candidate.runtime_type == "treeskill"
    assert result.candidate.runtime_schema_version == 1
    TreeSkillRuntimeMetadata.model_validate(result.candidate.runtime_metadata)
    assert result.candidate.metadata["treeskill_evolution"]["run_id"] == "run-1"
    assert "verify the saved output" in result.candidate.blob["SKILL.md"]
    assert base.runtime_type == "static"
    assert [call["task"] for call in model.calls] == [
        "treeskill_trajectory_analysis",
        "treeskill_evidence_localization",
        "treeskill_node_fusion",
    ]


@pytest.mark.asyncio
async def test_llm_router_reads_runtime_metadata() -> None:
    content = "# Workbook\n\nInspect first.\n"
    skill = make_skill(
        content,
        runtime_type="treeskill",
        runtime_metadata=compile_tree_metadata(parse_skill_markdown(content)),
    )
    model = ScriptedChatModel(
        [json.dumps({"selected_subtree_ids": ["001"], "rationale": "Workbook guidance is applicable."})]
    )
    router = TreeSkillRouter(chat_model=model)
    result = await router.route(skill=skill, task=Task(task_id="task-1", instruction="Inspect the workbook."))

    assert result.content_node_ids == ("001",)
    assert result.fallback_used is False
    assert parse_tree_with_metadata(skill.content, skill.runtime_metadata).nodes
