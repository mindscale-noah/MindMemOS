from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from mindmemos_skill.agents.react import ReactAgent
from mindmemos_skill.algos.trace2skill.treeskill import (
    TreeMetadataError,
    TreeSkill,
    TreeSkillConfig,
    compile_tree_metadata,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
)
from mindmemos_skill.algos.trace2skill.treeskill.models import (
    AnalysisItem,
    LocatedEvidence,
    TrajectoryAnalysisRecord,
)
from mindmemos_skill.algos.trace2skill.treeskill.prompts import (
    LOCALIZATION_SYSTEM_PROMPT,
    NODE_FUSION_SYSTEM_PROMPT,
    ROUTING_SYSTEM_PROMPT,
    fusion_user_prompt,
    localization_user_prompt,
    routing_user_prompt,
)
from mindmemos_skill.algos.trace2skill.treeskill.tree import (
    NewTreeNode,
    create_child_subtree,
    tree_prompt_payload,
    update_node_content,
)
from mindmemos_skill.envs import EnvRolloutContext, SpreadsheetBenchEnv
from mindmemos_skill.llm import ChatResponse
from mindmemos_skill.registry import ComponentType, get_component
from mindmemos_skill.typing import (
    AgentExecutionRequest,
    Environment,
    ExecutionInfo,
    Reward,
    Rollout,
    Skill,
    SkillInjectionMode,
    SkillUsageType,
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


def make_skill(content: str, *, metadata: dict[str, Any] | None = None) -> Skill:
    blob = {"SKILL.md": content}
    return Skill(
        skill_id="skill-tree",
        version_id="version-tree-1",
        version_label="1.0.0",
        content_hash=compute_skill_content_hash(blob),
        name="spreadsheet",
        description="Spreadsheet guidance",
        blob=blob,
        resources={"references/helper.py": "HELPER = True\n"},
        metadata=metadata or {},
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
    )


def test_treeskill_prompt_resources_match_reference_files() -> None:
    prompt_package = "mindmemos_skill.algos.trace2skill.treeskill.prompt_templates"
    expected_hashes = {
        "locating_system_prompt.txt": "40433e54931be334a84fc5366bac7432d93cd0a719e41f058cca7b389384ac6a",
        "node_fusion_system_prompt.txt": "bbecdd8e368cb4d92c96e64b2409ccb37d876ac3af892ed4fb832bf75dbeb246",
        "tree_only_skill_routing_system_prompt.txt": (
            "04c69b98c278fb92db5d4df9554b7d037860e26116e4531ed75dcc7023e4f613"
        ),
    }
    resources = files(prompt_package)
    for name, expected in expected_hashes.items():
        assert hashlib.sha256(resources.joinpath(name).read_bytes()).hexdigest() == expected

    assert LOCALIZATION_SYSTEM_PROMPT == resources.joinpath("locating_system_prompt.txt").read_text()
    assert NODE_FUSION_SYSTEM_PROMPT == resources.joinpath("node_fusion_system_prompt.txt").read_text()
    assert ROUTING_SYSTEM_PROMPT == resources.joinpath("tree_only_skill_routing_system_prompt.txt").read_text().rstrip()


def test_treeskill_user_prompts_match_reference_payloads() -> None:
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n\n## Formulas\n\nWrite formulas.\n")
    tree_payload = tree_prompt_payload(tree)
    record = TrajectoryAnalysisRecord(
        instance_id="trajectory-1",
        task_id="task-1",
        record_source="success",
        items=(
            AnalysisItem(
                item_id="i1",
                kind="success_memory",
                title="Verify formulas",
                description="Verification supported the successful result.",
                content="Recalculate and verify formula outputs.",
            ),
        ),
    )
    expected_localization = {
        "analysis_record": record.model_dump(mode="json"),
        "skill_tree": tree_payload,
    }
    assert localization_user_prompt(tree, record) == (
        "Locate trajectory-derived evidence into the Markdown skill tree.\n"
        "Use the recursive skill tree to choose existing fusion target nodes.\n\n"
        + json.dumps(expected_localization, ensure_ascii=False, indent=2)
    )

    evidence = [
        LocatedEvidence(
            instance_id="trajectory-1",
            evidence_id="e1",
            record_source="success",
            reusable_lesson="Recalculate and verify formula outputs.",
            target_node_id="002",
            rationale="Formula guidance owns this lesson.",
        )
    ]
    expected_fusion = {
        "target_node_id": "002",
        "skill_tree": tree_payload,
        "located_evidence": [
            {
                "instance_id": "trajectory-1",
                "evidence_id": "e1",
                "record_source": "success",
                "reusable_lesson": "Recalculate and verify formula outputs.",
                "target_node_id": "002",
            }
        ],
    }
    assert fusion_user_prompt(tree, "002", evidence) == (
        "Fuse the located evidence for this target Markdown node.\n"
        "Use the complete recursive skill tree to avoid redundant or misplaced edits.\n\n"
        + json.dumps(expected_fusion, ensure_ascii=False, indent=2)
    )

    task = Task(task_id="sheet-1", instruction="Write a formula.")
    routing_context = {
        "instance_id": "sheet-1",
        "instruction_type": "cell",
        "answer_position": "Sheet1!C2",
        "instruction": "Write a formula.",
        "spreadsheet_content": "('Revenue', 'Cost')",
    }
    expected_routing = {"task": routing_context, "skill_tree": tree_payload}
    assert routing_user_prompt(tree, task, routing_context) == (
        "Route skill subtrees for this spreadsheet task.\n\n"
        + json.dumps(expected_routing, ensure_ascii=False, indent=2)
    )


def test_markdown_tree_round_trip_mutation_and_ordered_rendering() -> None:
    content = """---
name: demo
---
# Workbook Operations

## Loading

Load the workbook first.

## Editing

### Formulas

Write formulas carefully.

## Empty Leaf

```python
# Not a heading
```
"""
    tree = parse_skill_markdown(content)

    assert [(node.node_id, node.heading) for node in tree.nodes] == [
        ("001", "Workbook Operations"),
        ("002", "Loading"),
        ("003", "Editing"),
        ("004", "Formulas"),
        ("005", "Empty Leaf"),
    ]
    assert tree.node_by_id["003"].local_content == ""
    metadata = compile_tree_metadata(tree)
    assert parse_tree_with_metadata(content, metadata).node_by_id["004"].heading == "Formulas"

    updated = update_node_content(tree, "002", "Load and inspect every relevant sheet.")
    assert updated.node_by_id["002"].heading == "Loading"
    assert updated.node_by_id["004"].heading == "Formulas"
    expanded = create_child_subtree(
        updated,
        "003",
        NewTreeNode(heading="Saving", content="Save to the requested output path."),
    )
    assert expanded.node_by_id["002"].node_id == "002"
    saving = next(node for node in expanded.nodes if node.heading == "Saving")
    assert saving.node_id == "006"
    assert saving.parent_id == "003"

    rendered = render_selected_subtrees(expanded, ("004", "002"))
    assert rendered.index("## Loading") < rendered.index("### Formulas")
    assert "### Saving" not in rendered

    stale = {**metadata, "skill_content_hash": "sha256:stale"}
    with pytest.raises(TreeMetadataError, match="content hash"):
        parse_tree_with_metadata(content, stale)


def test_treeskill_is_registered_as_an_optimize_algorithm() -> None:
    component = get_component(type=ComponentType.ALGO, name="treeskill")

    assert component.factory is TreeSkill
    assert component.config_model is TreeSkillConfig
    assert component.capabilities == frozenset({"optimize"})
    assert component.requirements.required_model_roles == frozenset({"chat"})


@dataclass(frozen=True)
class AlgorithmContext:
    models: dict[str, Any]
    agents: dict[str, Any]
    config_hash: str = "config-hash"


@pytest.mark.asyncio
async def test_treeskill_evolves_candidate_without_mutating_base_skill() -> None:
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
                            "rationale": "This is a general workbook operation.",
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
                            "content": "Inspect the workbook, save the result, and verify the saved output.",
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

    assert result.changed is True
    assert result.candidate is not None
    assert "verify the saved output" in result.candidate.blob["SKILL.md"]
    assert result.candidate.resources == base.resources
    assert result.candidate.metadata["treeskill"]["enabled"] is True
    assert base.content == "# Workbook\n\nInspect the workbook.\n"
    assert [call["task"] for call in model.calls] == [
        "treeskill_trajectory_analysis",
        "treeskill_evidence_localization",
        "treeskill_node_fusion",
    ]


@pytest.mark.asyncio
async def test_react_tree_routing_is_query_aware_ephemeral_and_auditable(tmp_path: Path) -> None:
    content = "# Skill\n\n## Keep\n\nSelected rule.\n\n## Omit\n\nOther rule.\n"
    tree = parse_skill_markdown(content)
    skill = make_skill(content, metadata={"treeskill": compile_tree_metadata(tree)})
    llm = ScriptedChatModel(
        [
            json.dumps({"selected_subtree_ids": ["002"], "rationale": "The task needs Keep."}),
            "done",
        ]
    )
    agent = ReactAgent(
        {"skill_injection_mode": "tree_routed_system_prompt"},
        llm=llm,
    )
    request = AgentExecutionRequest(
        trajectory_id="route-trajectory",
        task=Task(
            task_id="route-task",
            instruction="Apply the selected rule.",
            metadata={"benchmark": "SpreadsheetBench", "src_dir": "/private/data/task"},
        ),
        rollout=Rollout(rollout_id="route-rollout"),
        environment=Environment(env_ref="spreadsheetbench", running_dir=str(tmp_path)),
        skills=[skill],
    )

    trajectory = await agent.execute(request)

    assert len(llm.calls) == 2
    router_prompt = llm.calls[0]["messages"][1]["content"]
    assert "/private/data/task" not in router_prompt
    policy_prompt = llm.calls[1]["messages"][0]["content"]
    assert "Selected rule." in policy_prompt
    assert "Other rule." not in policy_prompt
    assert (tmp_path / "treeskill_routed_skills" / "spreadsheet" / "references" / "helper.py").is_file()
    routing = trajectory.metadata["treeskill_routing"]
    assert routing["skills"]["spreadsheet"]["content_node_ids"] == ["002"]
    assert "skill_content" not in routing["skills"]["spreadsheet"]
    assert trajectory.injected_skills[0].version_id == "version-tree-1"
    assert trajectory.skill_bindings[0].usage is SkillUsageType.INJECTED
    assert trajectory.skill_bindings[0].injection_mode is SkillInjectionMode.TREE_ROUTED_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_spreadsheetbench_tree_routing_bypasses_legacy_skill_tool(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source"
    source.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = 1
    workbook.save(source / "case_init.xlsx")
    workbook.save(source / "case_golden.xlsx")
    workbook.close()

    content = "# Skill\n\n## Keep\n\nSelected rule.\n\n## Omit\n\nOther rule.\n"
    skill = make_skill(content, metadata={"treeskill": compile_tree_metadata(parse_skill_markdown(content))})
    llm = ScriptedChatModel(
        [
            json.dumps({"selected_subtree_ids": ["002"], "rationale": "Use Keep."}),
            "done",
        ]
    )
    agent = ReactAgent({"skill_injection_mode": "tree_routed_system_prompt"}, llm=llm)
    task = Task(
        task_id="sheet-task",
        instruction="Preserve A1.",
        metadata={
            "benchmark": "SpreadsheetBench",
            "src_dir": str(source),
            "answer_position": "A1",
            "answer_sheet": "Sheet",
            "instruction_type": "cell",
        },
    )

    trajectory = await SpreadsheetBenchEnv({"max_turns": 1}).rollout(
        agent,
        task,
        [skill],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="sheet-rollout"),
            workspace_root=tmp_path / "runs",
            env_ref="spreadsheetbench",
        ),
    )

    assert len(llm.calls) == 2
    router_call = llm.calls[0]
    assert router_call["messages"][0]["content"] == ROUTING_SYSTEM_PROMPT
    router_payload = json.loads(
        router_call["messages"][1]["content"].removeprefix("Route skill subtrees for this spreadsheet task.\n\n")
    )
    assert router_payload["task"] == {
        "instance_id": "sheet-task",
        "instruction_type": "cell",
        "answer_position": "A1",
        "instruction": "Preserve A1.",
        "spreadsheet_content": "('1',)",
    }
    policy_call = llm.calls[1]
    assert "Selected rule." in policy_call["messages"][0]["content"]
    assert "Other rule." not in policy_call["messages"][0]["content"]
    assert all(tool["function"]["name"] != "skill" for tool in policy_call["tools"])
    workspace = Path(trajectory.environment.running_dir or "")
    assert not (workspace / "skills").exists()
    assert (workspace / "treeskill_routed_skills" / "spreadsheet" / "SKILL.md").is_file()
    assert "Selected rule." in (workspace / "target_system_prompt.txt").read_text(encoding="utf-8")
    assert trajectory.agent.skill_injection_mode is SkillInjectionMode.TREE_ROUTED_SYSTEM_PROMPT
