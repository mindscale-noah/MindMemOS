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
from mindmemos_skill.algos.trace2skill.contracts import TraceEvidence
from mindmemos_skill.algos.trace2skill.treeskill import (
    TreeMetadataError,
    TreeSkill,
    TreeSkillConfig,
    TreeSkillModelRequestError,
    compile_tree_metadata,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
)
from mindmemos_skill.algos.trace2skill.treeskill.fusion import TreeSkillNodeFuser
from mindmemos_skill.algos.trace2skill.treeskill.localization import TreeSkillEvidenceLocator
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
from mindmemos_skill.algos.trace2skill.treeskill.routing import TreeSkillRouter
from mindmemos_skill.algos.trace2skill.treeskill.tree import (
    NewTreeNode,
    create_child_subtree,
    tree_prompt_payload,
    update_node_content,
)
from mindmemos_skill.envs import EnvRolloutContext, SpreadsheetBenchEnv
from mindmemos_skill.envs.registered_envs.spreadsheetbench.analysis import SpreadsheetBenchReferenceAnalyzer
from mindmemos_skill.envs.registered_envs.spreadsheetbench.evaluator import compare_workbooks
from mindmemos_skill.envs.registered_envs.spreadsheetbench.trace2skill_compat import (
    FORMAT_ERROR_MESSAGE,
    PolicyResponseType,
    format_reference_observation,
    parse_policy_response,
)
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


class FailingChatModel:
    def __init__(self, message: str = "endpoint unavailable") -> None:
        self.message = message
        self.calls = 0

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        del task, messages, kwargs
        self.calls += 1
        raise RuntimeError(self.message)


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


def test_spreadsheetbench_reference_prompt_resources_are_pinned() -> None:
    prompt_package = "mindmemos_skill.envs.registered_envs.spreadsheetbench.prompt_templates"
    expected_hashes = {
        "trace2skill_preloaded_system_prompt.txt": ("8db86cc7c8c881d082f50fcf0efe2da828529c14b783706445f76211e261f4bc"),
        "trace2skill_no_skill_system_prompt.txt": ("d78f50a08e928e1a134b87e1ed36384be62354b6a630375ca4abfd09ff08b7ff"),
        "error_analysis_system.txt": "744654d1956a6cf4fdf8fb7be5065855a182a28319ec8e9b33fa0cfe81742d3c",
        "error_analysis_user.txt": "dfbb3b7e34a6647f6ce362104fa645061cfb00c02e0a38e438c0733a66c5e5b4",
        "success_analysis_system_llm.txt": ("0ba6a3a530e1cd50f4afdadd8af3448bf7f363a82fade2499b57e49655777f70"),
        "success_analysis_user_llm.txt": ("3c91275e61cf55df71fd37dd9c8be5026638f0c8edc60e033e18a45f6bbcb318"),
    }
    resources = files(prompt_package)

    for name, expected in expected_hashes.items():
        assert hashlib.sha256(resources.joinpath(name).read_bytes()).hexdigest() == expected


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
        "analysis_record": {
            "record_source": "success",
            "instance_id": "trajectory-1",
            "source_file": "",
            "items": [
                {
                    "type": "success_memory",
                    "number": 1,
                    "title": "Verify formulas",
                    "description": "Verification supported the successful result.",
                    "content": "Recalculate and verify formula outputs.",
                }
            ],
        },
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


def test_trace2skill_policy_parser_ignores_braces_in_strings_and_repairs_outer_brace() -> None:
    parsed = parse_policy_response(
        """Thought: inspect the row.
Action:
{
  "name": "bash",
  "arguments": {"command": "python -c \\"print(f'Row {row}: {row_data}')\\""}
"""
    )

    assert parsed.response_type is PolicyResponseType.ACTION
    assert parsed.action is not None
    assert parsed.action.name == "bash"
    assert "{row_data}" in parsed.action.arguments["command"]


def test_trace2skill_reference_feedback_uses_observation_protocol() -> None:
    assert format_reference_observation(FORMAT_ERROR_MESSAGE).startswith("Observation: Failed to parse your action.")


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
    assert model.calls[1]["response_format"]["type"] == "json_schema"
    assert model.calls[2]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_treeskill_locator_retries_with_larger_budget_and_keeps_valid_items() -> None:
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n")
    record = TrajectoryAnalysisRecord(
        instance_id="trajectory-1",
        task_id="task-1",
        record_source="success",
        items=(
            AnalysisItem(
                item_id="i1",
                kind="success_memory",
                title="Inspect",
                description="Inspection supported execution.",
                content="Inspect the workbook before editing.",
            ),
        ),
    )
    model = ScriptedChatModel(
        [
            "not json",
            json.dumps(
                {
                    "instance_id": "trajectory-1",
                    "evidence": [
                        {
                            "evidence_id": "e1",
                            "reusable_lesson": "Inspect the workbook before editing.",
                            "target_node_id": "001",
                            "rationale": "Workbook-level guidance.",
                        },
                        {
                            "evidence_id": "e2",
                            "reusable_lesson": "Unknown placement must not discard e1.",
                            "target_node_id": "999",
                            "rationale": "Invalid target for regression coverage.",
                        },
                        {
                            "evidence_id": "e3",
                            "reusable_lesson": "Compare the result with the ground truth workbook.",
                            "target_node_id": "001",
                            "rationale": "Analysis-only guidance must not become skill content.",
                        },
                    ],
                }
            ),
        ]
    )

    located, failures = await TreeSkillEvidenceLocator(
        chat_model=model,
        task="locate",
        concurrency=1,
        temperature=0.0,
        max_tokens=2048,
    ).locate(tree, [record])

    assert [item.evidence_id for item in located] == ["e1"]
    assert len(failures) == 1
    assert "unknown target_node_id" in failures[0].error
    assert "ground-truth or gold-answer reference" in failures[0].error
    assert [call["max_tokens"] for call in model.calls] == [2048, 4096]
    assert all(call["response_format"]["type"] == "json_schema" for call in model.calls)
    assert model.calls[0]["messages"] == model.calls[1]["messages"]


@pytest.mark.asyncio
async def test_spreadsheetbench_reference_analyzer_matches_success_and_agentic_error_paths(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    for path, value in (
        (workspace / "input.xlsx", 1),
        (workspace / "output.xlsx", 0),
        (workspace / "gold.xlsx", 1),
        (source / "case_golden.xlsx", 1),
    ):
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = value
        workbook.save(path)
        workbook.close()
    now = datetime(2026, 8, 12, tzinfo=UTC)
    success = Trajectory(
        trajectory_id="success-1",
        task=Task(task_id="task-success", instruction="Preserve A1."),
        rollout=Rollout(rollout_id="rollout-success"),
        environment=Environment(env_ref="spreadsheetbench", running_dir=str(workspace)),
        events=[{"role": "assistant", "content": "Inspected and saved the workbook."}],
        reward=Reward(score=1.0),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )
    failure = Trajectory(
        trajectory_id="failure-1",
        task=Task(
            task_id="task-failure",
            instruction="Preserve A1.",
            metadata={"src_dir": str(source), "answer_position": "A1", "answer_sheet": "Sheet"},
        ),
        rollout=Rollout(rollout_id="rollout-failure"),
        environment=Environment(env_ref="spreadsheetbench", running_dir=str(workspace)),
        events=[{"role": "assistant", "content": "Wrote the wrong value."}],
        reward=Reward(score=0.0),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )
    success_model = ScriptedChatModel(
        [
            """# Success Memory Item 1

## Title
Inspect before saving

## Description
Workbook inspection supported the successful edit.

## Content
Inspect the target cells before saving the output.
"""
        ]
    )
    failure_model = ScriptedChatModel(
        [
            "Action:\n{not valid JSON}",
            "Action:\n"
            + json.dumps(
                {
                    "name": "bash",
                    "arguments": {"command": "cp agent_work/gold.xlsx agent_work/output_fixed.xlsx"},
                }
            ),
            (
                "Action:\n"
                + json.dumps(
                    {
                        "name": "evaluate_output",
                        "arguments": {
                            "output_file": "agent_work/output_fixed.xlsx",
                            "ground_truth": "agent_work/gold.xlsx",
                            "answer_position": "Sheet!A1",
                        },
                    }
                )
            ),
            """# Failure Cause Item 1

## Title
Wrong target value

## Description
The edit wrote a value inconsistent with the requested preservation.

## Content
The agent overwrote a cell that should have been preserved.

## Relation to Skill
The existing preservation guidance was not followed.

# Failure Memory Item 1

## Title
Verify preserved cells

## Description
Verify cells that must remain unchanged before completion.

## Content
Compare preserved cells before and after editing and correct accidental overwrites.

## Skill Reflection
The skill should emphasize verification after saving.

ACTION: TASK_COMPLETE
""",
        ]
    )
    analyzer = SpreadsheetBenchReferenceAnalyzer(
        chat_model=success_model,
        failure_chat_model=failure_model,
        task="analysis",
        output_root=tmp_path / "analysis",
        concurrency=1,
        success_score_threshold=1.0,
        temperature=1.0,
        max_tokens=16384,
        max_turns=5,
    )

    records, failures = await analyzer.analyze(
        [
            TraceEvidence(trajectory_id="success-1", task_id="task-success", transcript="", score=1.0),
            TraceEvidence(trajectory_id="failure-1", task_id="task-failure", transcript="", score=0.0),
        ],
        trajectories_by_id={"success-1": success, "failure-1": failure},
    )

    assert failures == []
    assert [record.record_source for record in records] == ["error", "success"]
    assert [item.kind for item in records[0].items] == ["failure_cause", "failure_memory"]
    assert records[0].source_file == "analysis_report.md"
    assert records[1].source_file == "success_analysis.md"
    assert records[0].items[0].relation_to_skill == "The existing preservation guidance was not followed."
    assert records[0].items[1].skill_reflection == "The skill should emphasize verification after saving."
    assert [call["task"] for call in success_model.calls] == ["analysis:success"]
    assert [call["task"] for call in failure_model.calls] == ["analysis:error"] * 4
    assert failure_model.calls[1]["messages"][-1]["content"].startswith("Observation: Failed to parse your action.")
    failure_dir = tmp_path / "analysis" / "error" / "failure-1"
    assert (failure_dir / "evaluate_passed.flag").read_text(encoding="utf-8") == "PASS\n"
    assert (failure_dir / "agent_work" / "output_fixed.xlsx").is_file()


@pytest.mark.asyncio
async def test_spreadsheetbench_reference_analysis_propagates_model_request_failures(tmp_path: Path) -> None:
    now = datetime(2026, 8, 12, tzinfo=UTC)
    trajectory = Trajectory(
        trajectory_id="success-1",
        task=Task(task_id="task-success", instruction="Preserve A1."),
        rollout=Rollout(rollout_id="rollout-success"),
        environment=Environment(env_ref="spreadsheetbench"),
        events=[{"role": "assistant", "content": "Saved the workbook."}],
        reward=Reward(score=1.0),
        execution=ExecutionInfo(
            status=TrajectoryStatus.SUCCEEDED,
            started_at=now,
            finished_at=now,
            n_turn=1,
        ),
    )
    model = FailingChatModel()
    analyzer = SpreadsheetBenchReferenceAnalyzer(
        chat_model=model,
        task="analysis",
        output_root=tmp_path / "analysis",
        concurrency=128,
        success_score_threshold=1.0,
        temperature=1.0,
        max_tokens=16384,
    )

    with pytest.raises(TreeSkillModelRequestError, match="success analysis model request failed"):
        await analyzer.analyze(
            [TraceEvidence(trajectory_id="success-1", task_id="task-success", transcript="", score=1.0)],
            trajectories_by_id={"success-1": trajectory},
        )


@pytest.mark.asyncio
async def test_treeskill_locator_and_fuser_propagate_model_request_failures() -> None:
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n")
    record = TrajectoryAnalysisRecord(
        instance_id="trajectory-1",
        task_id="task-1",
        record_source="success",
        items=(
            AnalysisItem(
                item_id="i1",
                kind="success_memory",
                content="Inspect the workbook before editing.",
            ),
        ),
    )
    locator_model = FailingChatModel()
    with pytest.raises(TreeSkillModelRequestError, match="evidence localization model request failed"):
        await TreeSkillEvidenceLocator(
            chat_model=locator_model,
            task="locate",
            concurrency=16,
            temperature=0.0,
            max_tokens=2048,
        ).locate(tree, [record])
    assert locator_model.calls == 2

    fuser_model = FailingChatModel()
    with pytest.raises(TreeSkillModelRequestError, match="node fusion model request failed"):
        await TreeSkillNodeFuser(
            chat_model=fuser_model,
            task="fuse",
            temperature=0.0,
            max_tokens=4096,
        ).fuse(
            tree,
            [
                LocatedEvidence(
                    instance_id="trajectory-1",
                    evidence_id="e1",
                    record_source="success",
                    reusable_lesson="Inspect the workbook before editing.",
                    target_node_id="001",
                    rationale="Workbook-level guidance.",
                )
            ],
        )


def test_spreadsheetbench_reference_evaluator_uses_full_workbook_when_position_is_empty(
    tmp_path: Path,
) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    for name, value in (("gold.xlsx", 1), ("output.xlsx", 0)):
        workbook = openpyxl.Workbook()
        workbook.active["A1"] = value
        workbook.save(analysis_dir / name)
        workbook.close()

    observation, passed = SpreadsheetBenchReferenceAnalyzer._evaluate_output(
        analysis_dir,
        {"output_file": "output.xlsx", "ground_truth": "gold.xlsx"},
        "",
    )

    assert passed is False
    assert observation.startswith("Result: FAIL")


def test_spreadsheetbench_reference_evaluator_supports_whole_column_ranges(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    golden_path = tmp_path / "golden.xlsx"
    output_path = tmp_path / "output.xlsx"
    for path, sheet4_value in ((golden_path, 42), (output_path, 0)):
        workbook = openpyxl.Workbook()
        workbook.active.title = "Sheet3"
        workbook.active["G3"] = 7
        workbook.create_sheet("Sheet4")["G3"] = sheet4_value
        workbook.save(path)
        workbook.close()

    passed, detail = compare_workbooks(
        golden_path,
        output_path,
        "Sheet3'!A:G,'Sheet4'!A:G",
    )

    assert passed is False
    assert "Sheet4!G3" in detail


@pytest.mark.asyncio
async def test_node_fusion_rejects_analysis_artifacts_in_updates_and_new_subtrees() -> None:
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n")
    model = ScriptedChatModel(
        [
            json.dumps(
                {
                    "rationale": "Attempt analysis-only edits.",
                    "edits": [
                        {
                            "operation": "update_node",
                            "content": "Compare the workbook with the ground truth output.",
                            "rationale": "Evaluator-derived guidance.",
                        },
                        {
                            "operation": "create_child",
                            "new_child": {
                                "heading": "Verification",
                                "content": "",
                                "children": [
                                    {
                                        "heading": "Gold answer workflow",
                                        "content": "Use evaluator data.",
                                        "children": [],
                                    }
                                ],
                            },
                            "rationale": "Evaluator-derived subtree.",
                        },
                    ],
                }
            )
        ]
    )
    evidence = [
        LocatedEvidence(
            instance_id="trajectory-1",
            evidence_id="e1",
            record_source="error",
            reusable_lesson="Verify the workbook.",
            target_node_id="001",
            rationale="Workbook-level guidance.",
        )
    ]

    final_tree, edits, failures = await TreeSkillNodeFuser(
        chat_model=model,
        task="fuse",
        temperature=0.0,
        max_tokens=4096,
    ).fuse(tree, evidence)

    assert final_tree.full_content == tree.full_content
    assert failures == []
    assert [edit.accepted for edit in edits] == [False, False]
    assert all("ground-truth or gold-answer reference" in edit.message for edit in edits)


@pytest.mark.asyncio
async def test_spreadsheetbench_reference_policy_preloads_skill_and_exposes_only_bash(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source"
    source.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = 1
    workbook.save(source / "case_init.xlsx")
    workbook.save(source / "case_golden.xlsx")
    workbook.close()

    skill = make_skill("# Spreadsheet\n\nInspect before editing.\n")
    llm = ScriptedChatModel(
        [
            'Action:\n{"name":"bash","arguments":{"command":"cp input.xlsx output.xlsx"}}',
            "ACTION: TASK_COMPLETE",
        ]
    )
    agent = ReactAgent(
        {"model": "fake", "max_turns": 2, "skill_injection_mode": "system_prompt"},
        llm=llm,
    )
    task = Task(
        task_id="sheet-reference",
        instruction="Preserve A1.",
        metadata={
            "src_dir": str(source),
            "answer_position": "A1",
            "answer_sheet": "Sheet",
            "instruction_type": "Cell-Level Manipulation",
        },
    )

    trajectory = await SpreadsheetBenchEnv({"max_turns": 2, "trace2skill_reference_mode": True}).rollout(
        agent,
        task,
        [skill],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="reference-rollout"),
            workspace_root=tmp_path / "runs",
            env_ref="spreadsheetbench",
        ),
    )

    assert trajectory.reward.score == 1.0
    assert trajectory.agent.skill_injection_mode is SkillInjectionMode.SYSTEM_PROMPT
    assert len(llm.calls) == 2
    assert all("tools" not in call for call in llm.calls)
    assert "Inspect before editing." in llm.calls[0]["messages"][0]["content"]
    assert llm.calls[0]["messages"][1]["content"].startswith(
        "Task: Below is the spreadsheet manipulation question you need to solve:"
    )
    assert "### answer_position\nA1" in llm.calls[0]["messages"][1]["content"]
    workspace = Path(trajectory.environment.running_dir or "")
    assert (workspace / "preloaded_skills" / "spreadsheet" / "references" / "helper.py").is_file()
    assert (workspace / "gold.xlsx").is_file()


@pytest.mark.asyncio
async def test_spreadsheetbench_reference_policy_formats_parse_errors_as_observations(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source"
    source.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = 1
    workbook.save(source / "case_init.xlsx")
    workbook.save(source / "case_golden.xlsx")
    workbook.close()

    llm = ScriptedChatModel(
        [
            "Action:\n{not valid JSON}",
            'Action:\n{"name":"bash","arguments":{"command":"cp input.xlsx output.xlsx"}}',
            "ACTION: TASK_COMPLETE",
        ]
    )
    agent = ReactAgent(
        {"model": "fake", "max_turns": 3, "skill_injection_mode": "system_prompt"},
        llm=llm,
    )
    task = Task(
        task_id="sheet-format-retry",
        instruction="Preserve A1.",
        metadata={"src_dir": str(source), "answer_position": "A1"},
    )

    trajectory = await SpreadsheetBenchEnv({"max_turns": 3, "trace2skill_reference_mode": True}).rollout(
        agent,
        task,
        [make_skill("# Spreadsheet\n")],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="format-retry-rollout"),
            workspace_root=tmp_path / "runs",
            env_ref="spreadsheetbench",
        ),
    )

    assert trajectory.reward.score == 1.0
    assert llm.calls[1]["messages"][-1]["content"].startswith("Observation: Failed to parse your action.")


@pytest.mark.asyncio
async def test_spreadsheetbench_reference_policy_stops_identical_action_result_loops(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source"
    source.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = 1
    workbook.save(source / "case_init.xlsx")
    workbook.save(source / "case_golden.xlsx")
    workbook.close()

    repeated_action = 'Action:\n{"name":"bash","arguments":{"command":"false"}}'
    llm = ScriptedChatModel([repeated_action] * 5)
    agent = ReactAgent(
        {"model": "fake", "max_turns": 100, "skill_injection_mode": "system_prompt"},
        llm=llm,
    )
    task = Task(
        task_id="sheet-stagnation",
        instruction="Preserve A1.",
        metadata={"src_dir": str(source), "answer_position": "A1"},
    )

    trajectory = await SpreadsheetBenchEnv(
        {"max_turns": 100, "trace2skill_reference_mode": True, "reference_stagnation_limit": 5}
    ).rollout(
        agent,
        task,
        [make_skill("# Spreadsheet\n")],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="stagnation-rollout"),
            workspace_root=tmp_path / "runs",
            env_ref="spreadsheetbench",
        ),
    )

    assert len(llm.calls) == 5
    assert trajectory.metadata["stagnation_detected"] is True
    assert trajectory.execution.error_info is not None
    assert "identical action/result pair 5 consecutive times" in trajectory.execution.error_info


@pytest.mark.asyncio
async def test_spreadsheetbench_reference_policy_retries_missing_output_once(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source"
    source.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = 1
    workbook.save(source / "case_init.xlsx")
    workbook.save(source / "case_golden.xlsx")
    workbook.close()

    llm = ScriptedChatModel(["ACTION: TASK_COMPLETE", "ACTION: TASK_COMPLETE"])
    agent = ReactAgent(
        {"model": "fake", "max_turns": 4, "skill_injection_mode": "system_prompt"},
        llm=llm,
    )
    task = Task(
        task_id="sheet-missing-output",
        instruction="Preserve A1.",
        metadata={"src_dir": str(source), "answer_position": "A1"},
    )

    trajectory = await SpreadsheetBenchEnv({"max_turns": 4, "trace2skill_reference_mode": True}).rollout(
        agent,
        task,
        [make_skill("# Spreadsheet\n")],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="missing-output-rollout"),
            workspace_root=tmp_path / "runs",
            env_ref="spreadsheetbench",
        ),
    )

    assert len(llm.calls) == 2
    assert trajectory.reward.score == 0.0
    assert trajectory.execution.error_info is not None
    assert "Output file was not created" in trajectory.execution.error_info
    assert "execution_exception_type" not in trajectory.metadata


@pytest.mark.asyncio
async def test_spreadsheetbench_reference_policy_marks_model_request_exceptions(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "source"
    source.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = 1
    workbook.save(source / "case_init.xlsx")
    workbook.save(source / "case_golden.xlsx")
    workbook.close()

    agent = ReactAgent(
        {"model": "fake", "max_turns": 2, "skill_injection_mode": "system_prompt"},
        llm=FailingChatModel(),
    )
    task = Task(
        task_id="sheet-model-failure",
        instruction="Preserve A1.",
        metadata={"src_dir": str(source), "answer_position": "A1"},
    )

    trajectory = await SpreadsheetBenchEnv({"max_turns": 2, "trace2skill_reference_mode": True}).rollout(
        agent,
        task,
        [make_skill("# Spreadsheet\n")],
        context=EnvRolloutContext(
            rollout=Rollout(rollout_id="model-failure-rollout"),
            workspace_root=tmp_path / "runs",
            env_ref="spreadsheetbench",
        ),
    )

    assert trajectory.execution.status is TrajectoryStatus.FAILED
    assert trajectory.metadata["execution_exception_type"] == "RuntimeError"
    assert trajectory.execution.error_info == "RuntimeError: endpoint unavailable"


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
async def test_tree_router_invalid_json_uses_one_call_then_full_skill_fallback() -> None:
    content = "# Skill\n\n## Keep\n\nSelected rule.\n"
    tree = parse_skill_markdown(content)
    skill = make_skill(content, metadata={"treeskill": compile_tree_metadata(tree)})
    model = ScriptedChatModel(["not json"])

    result = await TreeSkillRouter(chat_model=model).route(
        skill=skill,
        task=Task(task_id="route-task", instruction="Apply the rule."),
    )

    assert len(model.calls) == 1
    assert "format_parser" not in model.calls[0]
    assert result.fallback_used is True
    assert result.skill_content == content


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
