"""Pinned upstream regression cases adapted only to current package interfaces."""

import copy
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from mindmemos_skill.agents import AgentResult, OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.algos.train.treeskill import TreeSkillEnvAdapter, TreeSkillTrainer, TreeSkillTrainerConfig
from mindmemos_skill.algos.train.treeskill.construction.fusion import (
    NODE_FUSION_SYSTEM_PROMPT,
    fusion_user_prompt,
)
from mindmemos_skill.algos.train.treeskill.construction.localization import (
    LOCALIZATION_SYSTEM_PROMPT,
    localization_user_prompt,
)
from mindmemos_skill.algos.train.treeskill.construction.markdown import (
    TreeMetadataError,
    compile_tree_metadata,
    create_child_subtree,
    parse_skill_markdown,
    parse_tree_with_metadata,
    render_selected_subtrees,
    tree_prompt_payload,
    update_node_content,
)
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_prompts import ROUTING_SYSTEM_PROMPT
from mindmemos_skill.algos.train.treeskill.env_specific.spreadsheetbench_text_action import (
    SpreadsheetBenchReferenceAnalyzer,
)
from mindmemos_skill.algos.train.treeskill.typing import (
    AnalysisItem,
    LocatedEvidence,
    NewTreeNode,
    TrajectoryAnalysisRecord,
    TreeSkillRoutingContext,
)
from mindmemos_skill.envs import Task
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.policy import (
    FORMAT_ERROR_MESSAGE,
    PolicyResponseType,
    format_reference_observation,
    parse_policy_response,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.reference_reward import compare_workbooks


class ScriptedAgent(OpenAIChatAgent):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(OpenAIChatConfig(model="offline"))
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def run_messages(self, *, messages, tools=None, response_format=None) -> AgentResult:
        self.calls.append(
            {"messages": messages, "max_tokens": self.config.max_completion_tokens, "response_format": response_format}
        )
        content = self.responses.pop(0)
        return AgentResult(output=content)


class FailingAgent(OpenAIChatAgent):
    def __init__(self, message: str = "endpoint unavailable") -> None:
        super().__init__(OpenAIChatConfig(model="offline"))
        self.message = message
        self.calls = []

    async def run_messages(self, *, messages, tools=None, response_format=None) -> AgentResult:
        # Match the chat model interface while simulating an endpoint failure.
        self.calls.append(messages)
        raise RuntimeError(self.message)


def make_stage_trainer(agent, **options):
    def factory(config):
        configured = copy.copy(agent)
        configured.config = config
        return configured

    return TreeSkillTrainer(
        TreeSkillTrainerConfig(optimizer=agent.config, router=agent.config, **options), agent_factory=factory
    )


async def run_localization(tree, records, *, agent, concurrency, temperature, max_tokens):
    trainer = make_stage_trainer(
        agent,
        localization_concurrency=concurrency,
        localization_temperature=temperature,
        localization_max_tokens=max_tokens,
    )
    with TemporaryDirectory() as directory:
        return await trainer._localize(tree, records, Path(directory))


async def run_fusion(tree, evidence, *, agent, temperature, max_tokens, audit_dir=None):
    trainer = make_stage_trainer(agent, fusion_temperature=temperature, fusion_max_tokens=max_tokens)
    if audit_dir is not None:
        return await trainer._fuse(tree, evidence, audit_dir)
    with TemporaryDirectory() as directory:
        return await trainer._fuse(tree, evidence, Path(directory))


def test_treeskill_prompt_resources_match_reference_files() -> None:
    prompt_package = "mindmemos_skill.algos.train.treeskill.construction.prompts"
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


def test_spreadsheetbench_text_action_prompt_resources_are_pinned() -> None:
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
    for name, expected in expected_hashes.items():
        resources = files(
            "mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.prompts"
            if name.startswith("trace2skill_")
            else prompt_package
        )
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
    assert TreeSkillEnvAdapter.build("spreadsheetbench_text_action", {}).build_routing_messages(
        TreeSkillRoutingContext(task=task, tree=tree_payload, public_context=routing_context)
    )[1]["content"] == (
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
@pytest.mark.parametrize(("max_tokens", "retry_tokens"), [(2048, 4096), (16384, 16384)])
async def test_treeskill_locator_retries_with_bounded_budget_and_keeps_valid_items(max_tokens, retry_tokens) -> None:
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
    model = ScriptedAgent(
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

    located, failures = await run_localization(
        tree, [record], agent=model, concurrency=1, temperature=0.0, max_tokens=max_tokens
    )

    assert [item.evidence_id for item in located] == ["e1"]
    assert len(failures) == 1
    assert "unknown target_node_id" in failures[0].error
    assert "ground-truth or gold-answer reference" in failures[0].error
    assert [call["max_tokens"] for call in model.calls] == [max_tokens, retry_tokens]
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
    locator_model = FailingAgent()
    located, failures = await run_localization(
        tree, [record], agent=locator_model, concurrency=16, temperature=0.0, max_tokens=2048
    )
    assert not located and len(failures) == 1
    assert "endpoint unavailable" in failures[0].error
    assert len(locator_model.calls) == 2
    model = FailingAgent()
    final, edits, failures = await run_fusion(
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
        agent=model,
        temperature=0.0,
        max_tokens=4096,
    )
    assert final == tree and not edits and len(failures) == 1
    assert "endpoint unavailable" in failures[0].error


async def test_spreadsheetbench_text_action_evaluator_uses_full_workbook_when_position_is_empty(
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

    observation, passed = await SpreadsheetBenchReferenceAnalyzer._evaluate_output(
        analysis_dir,
        {"output_file": "output.xlsx", "ground_truth": "gold.xlsx"},
        "",
    )

    assert passed is False
    assert observation.startswith("Result: FAIL")


def test_spreadsheetbench_text_action_evaluator_supports_whole_column_ranges(tmp_path: Path) -> None:
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
    model = ScriptedAgent(
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

    final_tree, edits, failures = await run_fusion(tree, evidence, agent=model, temperature=0.0, max_tokens=4096)

    assert final_tree.full_content == tree.full_content
    assert failures == []
    assert [edit.accepted for edit in edits] == [False, False]
    assert all("ground-truth or gold-answer reference" in edit.message for edit in edits)


@pytest.mark.parametrize(
    "edits",
    [
        [],
        [{"operation": "reject", "rationale": "No lesson"}] * 3,
        [None],
        [{"operation": "delete", "rationale": "Invalid operation"}],
        [
            {"operation": "reject", "rationale": "No lesson"},
            {"operation": "update_node", "content": "x", "rationale": "Update"},
        ],
        [{"operation": "update_node", "content": "x", "rationale": "Update"}] * 2,
        [{"operation": "update_node", "content": 1, "rationale": "Update"}],
        [
            {
                "operation": "create_child",
                "new_child": {"heading": "Child", "extra": "Not in contract"},
                "rationale": "Add child",
            }
        ],
        [{"operation": " UPDATE ", "content": "x", "rationale": "Update"}],
    ],
)
async def test_invalid_fusion_response_is_a_failure_without_applying_edits(edits):
    tree = parse_skill_markdown("# Workbook\nInspect first.\n")
    model = ScriptedAgent([json.dumps({"rationale": "Proposed change", "edits": edits})] * 2)
    evidence = [
        LocatedEvidence(
            instance_id="t1",
            evidence_id="e1",
            record_source="success",
            reusable_lesson="Verify saved workbooks.",
            target_node_id="001",
            rationale="Relevant",
        )
    ]
    final, applied, failures = await run_fusion(tree, evidence, agent=model, temperature=0, max_tokens=4096)
    assert final == tree and applied == [] and len(failures) == 1
    assert len(model.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_child", [False, True])
async def test_fusion_applies_update_before_child_and_preserves_partial_success(tmp_path, invalid_child):
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n")
    child = {
        "heading": "## Verification",
        "content": "",
        "children": [
            {
                "heading": "Reload",
                "content": "   " if invalid_child else "Reload after saving.",
                "children": [],
            }
        ],
    }
    model = ScriptedAgent(
        [
            json.dumps(
                {
                    "rationale": "Add verification",
                    "edits": [
                        {"operation": "create_child", "new_child": child, "rationale": "Add verification"},
                        {
                            "operation": "update_node",
                            "content": "Inspect sheet names before editing.",
                            "rationale": "Inspect first",
                        },
                    ],
                }
            )
        ]
    )
    evidence = [
        LocatedEvidence(
            instance_id="t1",
            evidence_id="e1",
            record_source="success",
            reusable_lesson="Verify saved workbooks.",
            target_node_id="001",
            rationale="Relevant.",
        )
    ]
    final, edits, failures = await run_fusion(
        tree, evidence, agent=model, temperature=0.0, max_tokens=4096, audit_dir=tmp_path
    )
    assert failures == []
    assert [edit.operation for edit in edits] == ["update_node", "create_child"]
    assert [edit.accepted for edit in edits] == [True, not invalid_child]
    assert "Inspect sheet names before editing." in final.full_content
    assert "Audit-only explanation" not in final.full_content
    assert len(final.nodes) == (1 if invalid_child else 3)
    assert "Add verification" in "\n".join(path.read_text() for path in tmp_path.rglob("*.json"))
    assert len(model.calls) == 1
    assert model.calls[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_truncated_fusion_response_remains_a_failure_after_one_retry():
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n")
    model = ScriptedAgent(['{"edits": [{"operation": "update_node", "content": "unfinished'] * 2)
    evidence = [
        LocatedEvidence(
            instance_id="t1",
            evidence_id="e1",
            record_source="success",
            reusable_lesson="Verify saved workbooks.",
            target_node_id="001",
            rationale="Relevant.",
        )
    ]
    final, edits, failures = await run_fusion(tree, evidence, agent=model, temperature=0.0, max_tokens=4096)
    assert final == tree
    assert edits == []
    assert len(failures) == 1
    assert len(model.calls) == 2


def test_reference_tree_checks_reject_empty_leaf_update_and_invalid_new_leaf():
    tree = parse_skill_markdown("# Workbook\n\nInspect first.\n")
    with pytest.raises(ValueError, match="leaf node cannot be updated to empty"):
        update_node_content(tree, "001", "")
    with pytest.raises(ValueError, match="lost its stable identity"):
        update_node_content(tree, "001", "   ")
    assert tree.node_by_id["001"].local_content == "Inspect first."
    for heading, content in [("###", "Valid content"), ("Valid heading", "   ")]:
        with pytest.raises(ValueError):
            create_child_subtree(tree, "001", NewTreeNode(heading=heading, content=content))


def test_empty_parent_update_preserves_identity_and_allows_new_children():
    tree = parse_skill_markdown("# Workbook\nInspect first.\n\n## Existing\nExisting instructions.\n")
    updated = update_node_content(tree, "001", "")
    assert list(updated.node_by_id) == ["001", "002"]
    assert updated.node_by_id["001"].local_content == ""
    expanded = create_child_subtree(updated, "001", NewTreeNode(heading="New child", content="New instructions."))
    assert list(expanded.node_by_id) == ["001", "002", "003"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_output",
    [
        '{"edits": [',
        json.dumps(
            {
                "rationale": "Invalid",
                "edits": [
                    {
                        "operation": "create_child",
                        "rationale": "Add",
                        "new_child": {"heading": "Check", "content": "Verify.", "children": [], "rationale": "Extra"},
                    }
                ],
            }
        ),
    ],
)
async def test_fusion_parse_retry_keeps_inputs_responses_and_applies_only_valid_output(first_output):
    from mindmemos_skill.algos.train.treeskill.construction.fusion import fuse_node

    tree = parse_skill_markdown("# Workbook\nInspect first.\n")
    valid = json.dumps(
        {
            "rationale": "Improve",
            "edits": [{"operation": "update_node", "content": "Verify saved workbook.", "rationale": "Useful"}],
        }
    )
    agent = ScriptedAgent([first_output, valid])
    outcome = await fuse_node(tree, "001", [], agent=agent)
    assert agent.calls[0] == agent.calls[1]
    assert [response.output for response in outcome.responses] == [first_output, valid]
    assert not outcome.error
    assert len(outcome.edits) == len(outcome.patches) == 1
    assert outcome.edits[0].accepted
    assert "Verify saved workbook." in outcome.content
