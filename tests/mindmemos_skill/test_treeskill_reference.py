"""Pinned upstream regression cases adapted only to current package interfaces."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench import (
    SpreadsheetBenchReferenceAnalyzer,
    routing_user_prompt,
)
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_policy import (
    FORMAT_ERROR_MESSAGE,
    PolicyResponseType,
    format_reference_observation,
    parse_policy_response,
)
from mindmemos_skill.algos.train.treeskill.fusion import TreeSkillNodeFuser
from mindmemos_skill.algos.train.treeskill.fusion_tree import (
    TreeMetadataError,
    compile_tree_metadata,
    create_child_subtree,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
    tree_prompt_payload,
    update_node_content,
)
from mindmemos_skill.algos.train.treeskill.localization import TreeSkillEvidenceLocator
from mindmemos_skill.algos.train.treeskill.prompts import (
    LOCALIZATION_SYSTEM_PROMPT,
    NODE_FUSION_SYSTEM_PROMPT,
    ROUTING_SYSTEM_PROMPT,
    fusion_user_prompt,
    localization_user_prompt,
)
from mindmemos_skill.algos.train.treeskill.typing import (
    AnalysisItem,
    ChatResponse,
    LocatedEvidence,
    NewTreeNode,
    TrajectoryAnalysisRecord,
)
from mindmemos_skill.envs import Task
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.reference_reward import compare_workbooks


class ScriptedChatModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        self.calls.append({"task": task, "messages": messages, **kwargs})
        content = self.responses.pop(0)
        parser = kwargs.get("format_parser")
        parsed = parser(content) if parser is not None else None
        return ChatResponse(content=content, parsed=parsed)


class FailingChatModel:
    def __init__(self, message: str = "endpoint unavailable") -> None:
        self.message = message
        self.calls = 0

    async def chat(self, task: str, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResponse:
        del task, messages, kwargs
        self.calls += 1
        raise RuntimeError(self.message)


def test_treeskill_prompt_resources_match_reference_files() -> None:
    prompt_package = "mindmemos_skill.algos.train.treeskill.prompt_templates"
    expected_hashes = {
        "locating_system_prompt.txt": "40433e54931be334a84fc5366bac7432d93cd0a719e41f058cca7b389384ac6a",
        "node_fusion_system_prompt.txt": "bbecdd8e368cb4d92c96e64b2409ccb37d876ac3af892ed4fb832bf75dbeb246",
    }
    resources = files(prompt_package)
    for name, expected in expected_hashes.items():
        assert hashlib.sha256(resources.joinpath(name).read_bytes()).hexdigest() == expected

    assert LOCALIZATION_SYSTEM_PROMPT == resources.joinpath("locating_system_prompt.txt").read_text()
    assert NODE_FUSION_SYSTEM_PROMPT == resources.joinpath("node_fusion_system_prompt.txt").read_text()
    assert (
        ROUTING_SYSTEM_PROMPT
        == files("mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_prompts")
        .joinpath("tree_only_skill_routing_system_prompt.txt")
        .read_text()
        .rstrip()
    )


def test_spreadsheetbench_reference_prompt_resources_are_pinned() -> None:
    prompt_package = "mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_prompts"
    expected_hashes = {
        "tree_only_skill_routing_system_prompt.txt": (
            "04c69b98c278fb92db5d4df9554b7d037860e26116e4531ed75dcc7023e4f613"
        ),
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

    task = Task(id="sheet-1", instruction="Write a formula.")
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
async def test_treeskill_locator_and_fuser_record_failed_items_without_aborting() -> None:
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n")
    record = TrajectoryAnalysisRecord(
        instance_id="trajectory-1",
        task_id="task-1",
        record_source="success",
        items=(AnalysisItem(item_id="i1", kind="success_memory", content="Inspect the workbook."),),
    )
    locator_model = FailingChatModel()
    located, failures = await TreeSkillEvidenceLocator(
        chat_model=locator_model, task="locate", concurrency=16, temperature=0.0, max_tokens=2048
    ).locate(tree, [record])
    assert not located and len(failures) == 1
    assert "endpoint unavailable" in failures[0].error
    assert locator_model.calls == 2
    model = FailingChatModel()
    final, edits, failures = await TreeSkillNodeFuser(
        chat_model=model, task="fuse", temperature=0.0, max_tokens=4096
    ).fuse(
        tree,
        [
            LocatedEvidence(
                instance_id="trajectory-1",
                evidence_id="e1",
                record_source="success",
                reusable_lesson="Inspect the workbook.",
                target_node_id="001",
                rationale="Workbook guidance.",
            )
        ],
    )
    assert final == tree and not edits and len(failures) == 1
    assert "endpoint unavailable" in failures[0].error


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
