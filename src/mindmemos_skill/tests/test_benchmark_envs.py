"""Exercise benchmark protocols, exact feedback, privacy, and rollout recovery."""

import asyncio
import json
import subprocess
import sys

import openpyxl
import pytest
from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
from mindmemos_skill.agents.buitin_agents.openai_chat.callbacks import parse_action
from mindmemos_skill.agents.typing import AgentResult
from mindmemos_skill.envs import Environment
from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history import ALFWorldConfig, ALFWorldEnvironment
from mindmemos_skill.envs.builtin_envs.officeqa import OfficeQAConfig, OfficeQAEnvironment
from mindmemos_skill.envs.builtin_envs.officeqa.scoring import score_answer
from mindmemos_skill.envs.builtin_envs.officeqa.tools import run_tool
from mindmemos_skill.envs.builtin_envs.searchqa import SearchQAConfig, SearchQAEnvironment
from mindmemos_skill.envs.builtin_envs.searchqa.reward import evaluate as search_reward
from mindmemos_skill.envs.builtin_envs.searchqa.typing import SearchQARewardData, SearchQATask
from mindmemos_skill.envs.builtin_envs.spreadsheetbench import (
    SpreadsheetBenchTextActionConfig,
    SpreadsheetBenchTextActionEnvironment,
    SpreadsheetBenchToolCallingConfig,
    SpreadsheetBenchToolCallingEnvironment,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.reward import compare_workbooks, verification_report
from mindmemos_skill.envs.data import Data
from mindmemos_skill.infra.execution import ExecutionConfig


@pytest.mark.parametrize(
    ("environment_type", "config_type"),
    [
        (SpreadsheetBenchTextActionEnvironment, SpreadsheetBenchTextActionConfig),
        (SpreadsheetBenchToolCallingEnvironment, SpreadsheetBenchToolCallingConfig),
    ],
)
def test_spreadsheet_dependency_failure_stops_before_model(spreadsheet, tmp_path, environment_type, config_type):
    agent = ScriptedAgent([])
    from unittest.mock import patch

    with (
        patch(
            "mindmemos_skill.envs.builtin_envs.spreadsheetbench.base.prepare_execution",
            return_value=ExecutionConfig(python_command=[sys.executable, "-S"]),
        ),
        pytest.raises(RuntimeError, match="before model execution"),
    ):
        environment_type(config_type(agent=agent.config), agent=agent)
    assert agent.requests == []


class ScriptedAgent(OpenAIChatAgent):
    """Record public requests and return deterministic scripted responses."""

    def __init__(self, outputs):
        super().__init__(OpenAIChatConfig(model="test-model"))
        self.outputs = iter(outputs)
        self.requests = []
        self.tool_requests = []

    def run(self, *, system=None, user=None, messages=None, tools=None):
        return self.run_messages(
            messages=messages
            if messages is not None
            else [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=tools,
        )

    def run_messages(self, *, messages, tools=None):
        self.tool_requests.append(tools)
        self.requests.append(json.loads(json.dumps(messages)))
        output = next(self.outputs)
        event = {"role": "assistant", "content": output} if isinstance(output, str) else output
        return AgentResult(
            output=event["content"],
            trajectory=[*messages, event],
            usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


def workbook(path, values):
    book = openpyxl.Workbook()
    for coordinate, value in values.items():
        book.active[coordinate] = value
    book.save(path)
    book.close()


@pytest.fixture
def spreadsheet(tmp_path):
    directory = tmp_path / "case"
    directory.mkdir()
    workbook(directory / "1_case_init.xlsx", {"A1": 1})
    workbook(directory / "1_case_golden.xlsx", {"A1": 2})
    source = tmp_path / "dataset.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "case",
                    "instruction": "Increment A1",
                    "spreadsheet_path": "case",
                    "answer_position": "Sheet!A1:A2",
                }
            ]
        )
    )
    return source


@pytest.mark.parametrize(
    "name",
    [
        "searchqa",
        "officeqa",
        "spreadsheetbench_tool_calling",
        "spreadsheetbench_text_action",
        "alfworld_bounded_history",
    ],
)
def test_builtin_registration(name):
    assert Environment.resolve(name)
    assert Data.resolve("spreadsheetbench" if name.startswith("spreadsheetbench_") else name)


def test_no_fast_registration():
    with pytest.raises(ValueError, match="Unknown"):
        Environment.resolve("alfworld_fast")


@pytest.mark.parametrize(
    "won,done,expected",
    [
        (True, True, ""),
        (False, True, "Episode ended without completing the task"),
        (False, False, "Timeout after 17 steps"),
    ],
)
def test_alfworld_exact_terminal_feedback(won, done, expected):
    from mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.reward import evaluate

    reward = evaluate(won=won, done=done, max_steps=17)
    assert reward.detail == expected
    assert reward.score == float(won)


def test_search_feedback_is_exact_and_private(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "question": "Who?", "context": "Public passage", "answers": ["Secret O'Brien"]})
    )
    sample = Data.build("searchqa", {"data_path": path}).load()[0]
    agent = ScriptedAgent(["<answer>Alice</answer>"])
    env = SearchQAEnvironment(SearchQAConfig(agent=agent.config), agent=agent)
    result = asyncio.run(env.rollout(sample.task, tmp_path / "rollout", reward_data=sample.reward_data))
    expected = "[EVALUATION RESULT]\nQuestion: Who?\nPredicted answer: 'Alice'\nGold answers: [\"Secret O'Brien\"]\nExact Match: 0.0\nF1: 0.0000"
    assert result.reward.detail == expected
    assert result.trajectory[-1] == {"role": "system", "content": expected}
    assert "Secret" not in json.dumps(agent.requests)
    assert result.usage.total_tokens == 12
    assert asyncio.run(env.rollout(sample.task, tmp_path / "rollout", reward_data=sample.reward_data)) == result
    assert len(agent.requests) == 1


def test_search_normalization_empty_semantics_and_last_tag():
    task = SearchQATask(id="q", instruction="Who?")
    data = SearchQARewardData(answers=["The red fox"])
    reward = search_reward("<answer>wrong</answer><ANSWER>RED, FOX!</ANSWER>", task, data)
    assert (reward.em, reward.f1, reward.sub_em) == (1, 1, 1)
    assert search_reward("", task, data).sub_em == 1
    assert search_reward("<answer>red</answer>", task, data).f1 == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "gold,prediction,expected",
    [
        ("113864", "113,864 thousand dollars", 1),
        ("73", "<FINAL_ANSWER>73%</FINAL_ANSWER>", 1),
        ("30", "The inputs were 10 and 20.\nThe final result is 30.", 0),
        ("2.66", "2.66 percentage points", 1),
        ("[8, 152260]", "[8,152,260]", 1),
        ("[1, 2]", "[2, 1]", 0),
        ("543 million", "543 billion", 0),
        ("March 1977", "April 1977", 0),
        ("100", "100 in 2023", 1),
        ("100", "100.01", 0),
    ],
)
def test_office_official_scoring(gold, prediction, expected):
    assert score_answer(gold, prediction) == expected


def test_office_tools_score_and_private_reference(tmp_path):
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "bulletin.txt").write_text("Revenue: 42\nOther context")
    source = tmp_path / "questions.csv"
    source.write_text("uid,question,ground_truth,source_files\nq1,Revenue?,42,bulletin.txt\n")
    sample = Data.build("officeqa", {"data_path": source, "docs_root": documents}).load()[0]
    tool = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "read1", "type": "function", "function": {"name": "read", "arguments": '{"path":"bulletin.txt"}'}}
        ],
    }
    agent = ScriptedAgent([tool, "<answer>41</answer>"])
    env = OfficeQAEnvironment(OfficeQAConfig(agent=agent.config), agent=agent)
    result = asyncio.run(env.rollout(sample.task, tmp_path / "rollout", reward_data=sample.reward_data))
    assert result.reward.detail == "predicted '41' but expected '42'"
    assert result.usage.total_tokens == 24
    assert "Revenue: 42" in json.dumps(agent.requests[-1])
    assert "ground_truth" not in json.dumps(agent.requests)
    assert run_tool("read", {"path": str(source)}, sample.task.documents, 1000).startswith("[read error:")
    assert run_tool("read", {"path": "../questions.csv"}, sample.task.documents, 1000).startswith("[read error:")


@pytest.mark.parametrize(
    "input_name,reference_name",
    [
        ("1_case_init.xlsx", "1_case_golden.xlsx"),
        ("initial.xlsx", "golden.xlsx"),
        ("1_42930_init.xlsx", "1_43930_golden.xlsx"),
    ],
)
def test_shared_case_discovery(spreadsheet, input_name, reference_name):
    directory = spreadsheet.parent / "case"
    (directory / "1_case_init.xlsx").rename(directory / input_name)
    (directory / "1_case_golden.xlsx").rename(directory / reference_name)
    forest = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    assert len(forest.task.cases) == 1
    assert forest.task.cases[0].path.name == input_name
    assert forest.reward_data.cases[0].path.name == reference_name
    assert forest.task.answer_position == "Sheet!A1:A2"
    assert "golden" not in forest.task.model_dump_json()


def test_spreadsheet_sheet_list_preserves_unqualified_scoring_range(spreadsheet):
    rows = json.loads(spreadsheet.read_text())
    rows[0]["answer_sheet"] = "Sheet,Existing Task,Additions,Retired"
    rows[0]["answer_position"] = "A1:A2"
    spreadsheet.write_text(json.dumps(rows))
    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    assert sample.task.answer_position == "A1:A2"
    assert sample.reward_data.answer_position == "A1:A2"
    reference = sample.reward_data.cases[0].path
    assert compare_workbooks(
        reference, reference, sample.reward_data.answer_position, answer_sheet=sample.reward_data.answer_sheet
    ) == (True, "")


@pytest.mark.parametrize("name", ["spreadsheetbench"])
@pytest.mark.parametrize("kind", ["init", "golden"])
@pytest.mark.parametrize("count", [0, 2])
def test_case_discovery_rejects_missing_or_ambiguous_files(spreadsheet, name, kind, count):
    directory = spreadsheet.parent / "case"
    if count == 0:
        (directory / f"1_case_{kind}.xlsx").unlink()
    else:
        workbook(directory / f"2_case_{kind}.xlsx", {"A1": 3})
    with pytest.raises(ValueError, match="Expected one init and one golden workbook"):
        Data.build(name, {"data_path": spreadsheet}).load()


def test_workbook_feedback_golden_and_order(tmp_path):
    gold, output = tmp_path / "gold.xlsx", tmp_path / "out.xlsx"
    workbook(gold, {"A1": 2, "A2": None, "B1": "secret", "B2": "5"})
    workbook(output, {"A1": 1, "A2": None, "B1": "wrong", "B2": 5})
    assert compare_workbooks(gold, output, "Sheet!A1:B2") == (False, "value@Sheet!A1: gt=2 pred=1")
    assert verification_report(output, gold, "Sheet!A1:B2") == (
        "## Output Verification\n  Sheet!A1: got=1, expected=2 ✗\n"
        "  Sheet!B1: got='wrong', expected='secret' ✗\n  Sheet!B2: got=5, expected='5' ✓\n"
        "  (+1 empty cells correct, omitted)"
    )


def test_workbook_report_truncation(tmp_path):
    gold, output = tmp_path / "gold.xlsx", tmp_path / "out.xlsx"
    workbook(gold, {f"A{index}": index for index in range(1, 500)})
    workbook(output, {f"A{index}": -index for index in range(1, 500)})
    report = verification_report(output, gold, "Sheet!A1:A499")
    full = "## Output Verification\n" + "\n".join(
        f"  Sheet!A{index}: got={-index}, expected={index} ✗" for index in range(1, 500)
    )
    assert (
        report == full[:6000] + f"\n  ...[verification report truncated, {len(full)} chars total]...\n" + full[-6000:]
    )


def test_shared_ranges_and_missing_sheet(tmp_path):
    gold, output = tmp_path / "gold.xlsx", tmp_path / "out.xlsx"
    workbook(gold, {"A1": 2, "A2": 3})
    workbook(output, {"A1": 2, "A2": 3})
    assert compare_workbooks(gold, output, "Sheet'!A:A") == (True, "")
    assert compare_workbooks(gold, output, "Missing!A1") == (False, "worksheet not found: Missing")


@pytest.mark.parametrize("changed_case", ["input", "reference"])
def test_spreadsheet_full_rollout_feedback_and_resume(spreadsheet, tmp_path, monkeypatch, changed_case):
    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    agent = ScriptedAgent(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"commands": ["python solution.py"]}'},
                    }
                ],
            },
            "Done",
        ]
    )
    executed = []

    def controlled_execution(name, arguments, directory, config):
        executed.append(directory)
        book = openpyxl.load_workbook(directory / "input.xlsx")
        book.active["A1"] = book.active["A1"].value + 1
        book.save(directory / "output.xlsx")
        book.close()
        return "exit_code: 0"

    monkeypatch.setattr(
        "mindmemos_skill.envs.builtin_envs.spreadsheetbench.workspace_tools.run_tool", controlled_execution
    )
    env = SpreadsheetBenchToolCallingEnvironment(SpreadsheetBenchToolCallingConfig(agent=agent.config), agent=agent)
    result = asyncio.run(env.rollout(sample.task, tmp_path / "rollout", reward_data=sample.reward_data))
    reports = [
        f"## Eval Result (case {number}): PASS\n\n\n## Output Verification\n  Sheet!A1: got={number + 1}, expected={number + 1} ✓\n  (+1 empty cells correct, omitted)"
        for number in (1,)
    ]
    assert result.reward.detail == "[POST-EXECUTION VERIFICATION]\n\n" + "\n\n---\n\n".join(reports)
    assert result.reward.score == 1 and result.reward.success
    assert "golden" not in json.dumps(agent.requests)
    assert len(executed) == 1
    assert asyncio.run(env.rollout(sample.task, tmp_path / "rollout", reward_data=sample.reward_data)) == result
    assert len(executed) == 1
    case = sample.task.cases[0] if changed_case == "input" else sample.reward_data.cases[0]
    workbook(case.path, {"A1": 99})
    with pytest.raises(ValueError, match="inputs changed"):
        asyncio.run(env.rollout(sample.task, tmp_path / "rollout", reward_data=sample.reward_data))
    assert len(executed) == 1


def test_text_action_rollout(spreadsheet, tmp_path, monkeypatch):
    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    action = 'Action:\n{"name":"bash","arguments":{"command":"edit workbook"}}'
    agent = ScriptedAgent([action, "ACTION: TASK_COMPLETE"])

    def controlled_action(command, *, working_dir, timeout_seconds, execution):
        workbook(working_dir / "output.xlsx", {"A1": 2})
        return "done"

    monkeypatch.setattr(
        "mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action.run_reference_bash", controlled_action
    )
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(
            max_turns=15,
            agent=agent.config,
        ),
        agent=agent,
    )
    result = asyncio.run(env.rollout(sample.task, tmp_path / "tree", reward_data=sample.reward_data))
    assert "[POST-EXECUTION VERIFICATION]" in result.reward.detail and result.reward.success
    assert result.trajectory[-2] == {"role": "user", "content": "Observation: done"}
    assert "golden" not in json.dumps(agent.requests)
    assert result.usage.total_tokens == 24
    assert agent.tool_requests == [None, None]


@pytest.mark.parametrize(
    "text",
    [
        'Action:\n{"name":"bash","arguments":{"command":"echo }"}}',
        "Action:\n{'name':'bash','arguments':{'command':'echo hi'}}",
        'Action:\n{"name":"bash","arguments":{"command":"echo hi"}',
    ],
)
def test_text_action_parser(text):
    assert parse_action(text)["name"] == "bash"


def test_alfworld_history_success_and_cleanup(tmp_path, monkeypatch):
    game = tmp_path / "game.json"
    game.write_text("{}")
    source = tmp_path / "items.json"
    source.write_text(json.dumps([{"id": "game", "gamefile": str(game)}]))
    sample = Data.build("alfworld_bounded_history", {"data_path": source}).load()[0]

    class Simulator:
        closed = False
        steps = 0

        def reset(self):
            return ["Your task is to: put apple in bowl"], {"admissible_commands": [["look", "help"]]}

        def step(self, actions):
            self.steps += 1
            return (
                [f"observation {self.steps}"],
                [0],
                [self.steps == 3],
                {"admissible_commands": [["look"]], "won": [self.steps == 3]},
            )

        def close(self):
            self.closed = True

    simulator = Simulator()
    monkeypatch.setattr(
        "mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.rollout._create_simulator", lambda *args: simulator
    )
    agent = ScriptedAgent(["<think>inspect</think><action>look</action>"] * 3)
    env = ALFWorldEnvironment(ALFWorldConfig(agent=agent.config, simulator={}, history_length=1), agent=agent)
    result = asyncio.run(env.rollout(sample.task, tmp_path / "alf", reward_data=sample.reward_data))
    assert result.reward.success and result.reward.detail == ""
    assert simulator.closed and result.usage.total_tokens == 36
    final_prompt = agent.requests[-1][-1]["content"]
    assert "Observation 2:" in final_prompt and "Observation 1:" not in final_prompt
    assert result.trajectory[-1]["reward"] == 10
    assert result.reward.model_extra["task_description"] == "put apple in bowl"
    assert result.reward.model_extra["task_type"] == "other"


def test_alfworld_provider_failure_is_not_cached(tmp_path, monkeypatch):
    game = tmp_path / "game.json"
    game.write_text("{}")
    source = tmp_path / "items.json"
    source.write_text(json.dumps([{"id": "game", "gamefile": str(game)}]))
    sample = Data.build("alfworld_bounded_history", {"data_path": source}).load()[0]

    class Simulator:
        closed = False

        def reset(self):
            return ["Your task is to: inspect"], {"admissible_commands": [["look"]]}

        def close(self):
            self.closed = True

    simulator = Simulator()
    monkeypatch.setattr(
        "mindmemos_skill.envs.builtin_envs.alfworld_bounded_history.rollout._create_simulator", lambda *args: simulator
    )
    agent = ScriptedAgent([])
    env = ALFWorldEnvironment(ALFWorldConfig(agent=agent.config, simulator={}), agent=agent)
    with pytest.raises(RuntimeError, match="Synchronous operation raised StopIteration"):
        asyncio.run(env.rollout(sample.task, tmp_path / "failed", reward_data=sample.reward_data))
    assert simulator.closed
    assert not (tmp_path / "failed/audit/result.json").exists()


def test_office_parsed_pages_keep_only_referenced_evidence(tmp_path):
    root = tmp_path / "docs"
    (root / "jsons").mkdir(parents=True)
    (root / "jsons/bulletin.json").write_text(
        json.dumps(
            {
                "document": {
                    "elements": [
                        {"bbox": [{"page_id": 2}], "content": "<table><tr><td>Revenue</td><td>42</td></tr></table>"},
                        {"bbox": [{"page_id": 3}], "content": "UNRELATED_PAGE"},
                    ]
                }
            }
        )
    )
    source = tmp_path / "items.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "q",
                    "question": "Revenue?",
                    "answer": "SECRET_REFERENCE",
                    "source_files": ["bulletin.pdf"],
                    "source_docs": ["https://example.invalid/bulletin?page=2"],
                }
            ]
        )
    )
    sample = Data.build("officeqa", {"data_path": source, "docs_root": root}).load()[0]
    assert "Revenue | 42" in sample.task.oracle_context
    assert "UNRELATED_PAGE" not in sample.task.oracle_context
    assert "SECRET_REFERENCE" not in sample.task.model_dump_json()


def test_agent_tool_call_sdk_boundary(monkeypatch):
    import httpx
    import openai

    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "tool-response",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": '{"path":"public.txt"}'},
                                }
                            ],
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(
        openai, "DefaultHttpxClient", lambda **kwargs: httpx.Client(transport=httpx.MockTransport(respond), **kwargs)
    )
    agent = OpenAIChatAgent(OpenAIChatConfig(model="test", api_key="test-key", base_url="https://test.invalid/v1"))
    messages = [{"role": "user", "content": "Read public.txt"}]
    tools = [
        {
            "type": "function",
            "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
        }
    ]
    result = agent.run_messages(messages=messages, tools=tools)
    assert result.output == ""
    assert result.trajectory[-1]["tool_calls"][0]["id"] == "call-1"
    assert payloads[0]["tools"] == tools
    with pytest.raises(RuntimeError):
        agent.run_messages(messages=messages)


def test_spreadsheet_native_skill_tool(spreadsheet, tmp_path, monkeypatch):
    from mindmemos_skill.skill import Skill, SkillContent, SkillSet

    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    actions = [
        {"name": "skill", "arguments": {"name": "xlsx"}},
        {"name": "shell", "arguments": {"commands": ["python solution.py"]}},
    ]
    outputs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": str(index),
                    "type": "function",
                    "function": {"name": action["name"], "arguments": json.dumps(action["arguments"])},
                }
            ],
        }
        for index, action in enumerate(actions)
    ] + ["Done"]
    agent = ScriptedAgent(outputs)

    def execute(command, directory, config):
        assert (directory / "skills/xlsx/references/guide.md").read_bytes() == b"Original reference\r\n"
        workbook(directory / "output.xlsx", {"A1": 2})
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("mindmemos_skill.envs.builtin_envs.spreadsheetbench.tools.execute", execute)
    env = SpreadsheetBenchToolCallingEnvironment(SpreadsheetBenchToolCallingConfig(agent=agent.config), agent=agent)
    result = asyncio.run(
        env.rollout(
            sample.task,
            tmp_path / "result",
            reward_data=sample.reward_data,
            skill_set=SkillSet(
                {
                    "0": Skill(
                        source=SkillContent(
                            content="Use Python literal values.",
                            resources={"references/guide.md": b"Original reference\r\n"},
                            name="test-skill",
                            description="Task instructions",
                            standard_format=False,
                        )
                    )
                },
                max_concurrency=1,
            ),
        )
    )
    assert result.reward.success and result.reward.skill_loaded
    assert "Use Python literal values." not in json.dumps(agent.requests[0])
    assert "Use Python literal values." in json.dumps(agent.requests[1])
    assert "[POST-EXECUTION VERIFICATION]" not in json.dumps(agent.requests)
    assert result.reward.n_turns == 3
    assert all(agent.tool_requests)
    assert any(event.get("role") == "tool" and "tool_call_id" in event for event in result.trajectory)


@pytest.mark.parametrize(
    "sheet,position,cell",
    [
        ("Sheet3", "Sheet3'!A:G", "G3"),
        ("Sheet1", "'Sheet1'!BD2:308", "BD308"),
        ("b2b, sez, de", "'b2b, sez, de'!A5:V10", "V10"),
    ],
)
def test_shared_spreadsheet_special_ranges(tmp_path, sheet, position, cell):
    gold, output = tmp_path / "gold.xlsx", tmp_path / "output.xlsx"
    book = openpyxl.Workbook()
    book.active.title = sheet
    book.active[cell] = 42
    book.save(gold)
    book.save(output)
    assert compare_workbooks(gold, output, position) == (True, "")
    assert f"{sheet}!{cell}: got=42, expected=42 ✓" in verification_report(output, gold, position)
    book.active[cell] = 43
    book.save(output)
    book.close()
    assert compare_workbooks(gold, output, position)[0] is False
    assert f"{sheet}!{cell}: got=43, expected=42 ✗" in verification_report(output, gold, position)


@pytest.mark.parametrize("outcome", ["correct", "wrong", "missing", "formula", "corrupt"])
def test_spreadsheet_environments_share_reward(spreadsheet, tmp_path, monkeypatch, outcome):
    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]

    def produce(directory):
        if outcome == "corrupt":
            (directory / "output.xlsx").write_text("not a workbook")
        elif outcome != "missing":
            workbook(directory / "output.xlsx", {"A1": {"correct": 2, "wrong": 3, "formula": "=1+1"}[outcome]})
        return "done"

    monkeypatch.setattr(
        "mindmemos_skill.envs.builtin_envs.spreadsheetbench.workspace_tools.run_tool",
        lambda name, arguments, directory, config: produce(directory),
    )
    monkeypatch.setattr(
        "mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action.run_reference_bash",
        lambda command, *, working_dir, timeout_seconds, execution: produce(working_dir),
    )
    forest_agent = ScriptedAgent(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "edit",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"commands":["edit"]}'},
                    }
                ],
            },
            "Done",
        ]
    )
    tree_agent = ScriptedAgent(
        ['Action: {"name":"bash","arguments":{"command":"edit"}}', "ACTION: TASK_COMPLETE", "ACTION: TASK_COMPLETE"]
    )
    forest = SpreadsheetBenchToolCallingEnvironment(
        SpreadsheetBenchToolCallingConfig(agent=forest_agent.config), agent=forest_agent
    )
    tree = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(
            max_turns=15,
            agent=tree_agent.config,
        ),
        agent=tree_agent,
    )
    left = asyncio.run(forest.rollout(sample.task, tmp_path / "forest", reward_data=sample.reward_data))
    right = asyncio.run(tree.rollout(sample.task, tmp_path / "tree", reward_data=sample.reward_data))
    left_reward = left.reward.model_dump(
        exclude={"n_turns", "skill_loaded", "skill_loading_mode", "skill_tool_exposed"}
    )
    right_reward = right.reward.model_dump(exclude={"n_turns", "finished", "stop_reason"})
    assert left_reward == right_reward
    assert left.reward.score == float(outcome == "correct")
    for result in (left, right):
        assert result.trajectory[-1]["role"] == "assistant"
        assert "[POST-EXECUTION VERIFICATION]" not in json.dumps(result.trajectory)
        assert "[POST-EXECUTION VERIFICATION]" in result.reward.detail
    assert all(
        "[POST-EXECUTION VERIFICATION]" not in json.dumps(agent.requests) for agent in (forest_agent, tree_agent)
    )


def test_spreadsheet_empty_range_is_not_a_pass(tmp_path):
    gold = tmp_path / "gold.xlsx"
    workbook(gold, {"A1": 2})
    with pytest.raises(ValueError, match="answer_position"):
        compare_workbooks(gold, gold, " , ")


def test_spreadsheet_default_sheet_and_row_major_report(tmp_path):
    gold, output = tmp_path / "gold.xlsx", tmp_path / "output.xlsx"
    book = openpyxl.Workbook()
    target = book.create_sheet("Target")
    target["A2"], target["B1"] = 2, 3
    book.save(gold)
    target["A2"], target["B1"] = 20, 30
    book.save(output)
    book.close()
    assert compare_workbooks(gold, output, "A1:B2", answer_sheet="Target") == (False, "value@Target!B1: gt=3 pred=30")
    report = verification_report(output, gold, "A1:B2", answer_sheet="Target")
    assert report.index("Target!B1") < report.index("Target!A2")


def test_spreadsheet_shared_reward_aggregates_cases(spreadsheet, tmp_path):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.reward import evaluate

    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    reference = sample.reward_data.cases[0]
    reward_data = sample.reward_data.model_copy(
        update={"cases": [reference, reference.model_copy(update={"number": "2"})]}
    )
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workbook(first / "output.xlsx", {"A1": 2})
    result = evaluate(reward_data, [first, second])
    assert result.score == result.soft == 0.5
    assert result.success is False and result.hard == 0
    assert result.n_cases == 2 and result.n_pass == result.n_exec_pass == 1
    assert result.fail_reason == "output-not-found"


@pytest.mark.parametrize(
    "options,error",
    [
        ({"interaction_mode": "bash_only"}, "interaction_mode"),
        ({"tool_agent": "text_action"}, "tool_agent"),
        ({"interaction_mode": "unknown"}, "interaction_mode"),
        ({"execution": {"python_command": ["/external/python"]}}, "execution"),
    ],
)
@pytest.mark.parametrize("config_type", [SpreadsheetBenchToolCallingConfig, SpreadsheetBenchTextActionConfig])
def test_spreadsheet_mode_validation(options, error, config_type):
    with pytest.raises(ValueError, match=error):
        config_type(agent=OpenAIChatConfig(model="test-model"), **options)


@pytest.mark.parametrize("response", ['Action: {"name":"bash","arguments":{"command":"echo same"}}', "Action: broken"])
def test_text_action_stagnation(spreadsheet, tmp_path, monkeypatch, response):
    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    agent = ScriptedAgent([response] * 3)
    monkeypatch.setattr(
        "mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action.run_reference_bash",
        lambda command, *, working_dir, timeout_seconds, execution: "same output",
    )
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(
            agent=agent.config,
            max_turns=15,
            stagnation_limit=3,
        ),
        agent=agent,
    )
    result = asyncio.run(env.rollout(sample.task, tmp_path / "stagnation", reward_data=sample.reward_data))
    assert len(agent.requests) == 3
    assert result.reward.finished is False
    assert "3 consecutive times" in result.reward.stop_reason
    assert result.reward.fail_reason == "output-not-found"


def test_text_action_scores_output_at_turn_limit(spreadsheet, tmp_path, monkeypatch):
    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    agent = ScriptedAgent(['Action: {"name":"bash","arguments":{"command":"edit"}}'])

    def edit(command, *, working_dir, timeout_seconds, execution):
        workbook(working_dir / "output.xlsx", {"A1": 2})
        return "done"

    monkeypatch.setattr("mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action.run_reference_bash", edit)
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(
            agent=agent.config,
            max_turns=1,
        ),
        agent=agent,
    )
    result = asyncio.run(env.rollout(sample.task, tmp_path / "limit", reward_data=sample.reward_data))
    assert result.reward.success and result.reward.score == 1
    assert result.reward.finished is False
    assert result.reward.fail_reason == ""
    assert result.reward.stop_reason == "Max turns exceeded"


def test_subclass_hooks_share_scoring_and_isolate_concurrent_rollouts(spreadsheet, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import SpreadsheetObservation

    barrier = Barrier(2)
    schema = {"type": "function", "function": {"name": "custom", "parameters": {"type": "object"}}}

    class Model(OpenAIChatAgent):
        def run_messages(self, *, messages, tools=None):
            assert tools == [schema]
            assert messages[0]["content"] == "Custom policy"
            return AgentResult(
                output=messages[1]["content"],
                trajectory=[{"role": "assistant", "content": messages[1]["content"]}],
                usage={"total_tokens": 3},
            )

    class CustomEnvironment(SpreadsheetBenchToolCallingEnvironment):
        def _system_prompt(self, context):
            return "Custom policy"

        def _user_prompt(self, context):
            assert "reward_data" not in type(context).model_fields
            return context.task.id

        def _get_observation(self, response, context):
            barrier.wait(timeout=5)
            assert response.output.text == context.task.id
            workbook(context.directory / "output.xlsx", {"A1": 2})
            return SpreadsheetObservation(
                assistant={"role": "assistant", "content": context.task.id},
                output=context.task.id,
                complete=True,
            )

    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    config = SpreadsheetBenchToolCallingConfig(agent=OpenAIChatConfig(model="offline"), tools=[schema])
    env = CustomEnvironment(config, agent=Model(config.agent))

    def run(identifier):
        return asyncio.run(
            env.rollout(
                sample.task.model_copy(update={"id": identifier}),
                tmp_path / identifier,
                reward_data=sample.reward_data,
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ["first", "second"]))
    assert [result.output for result in results] == ["first", "second"]
    assert all(result.reward.success for result in results)
    assert all(result.trajectory[-1] == {"role": "assistant", "content": result.output} for result in results)
    assert results[0].reward == results[1].reward
    assert all(result.usage.calls == [{"total_tokens": 3}] for result in results)
    assert all(result.usage.reported for result in results)
    assert CustomEnvironment._run is SpreadsheetBenchToolCallingEnvironment._run
    assert CustomEnvironment._interact is SpreadsheetBenchToolCallingEnvironment._interact
    # The base class still renders its own prompts and cannot inherit custom hooks.
    base_agent = ScriptedAgent(["Done"])
    base = SpreadsheetBenchToolCallingEnvironment(
        SpreadsheetBenchToolCallingConfig(agent=base_agent.config), agent=base_agent
    )
    asyncio.run(base.rollout(sample.task, tmp_path / "base", reward_data=sample.reward_data))
    assert base_agent.requests[0][0]["content"] != "Custom policy"


def test_spreadsheet_resume_checks_tool_schemas(spreadsheet, tmp_path):
    sample = Data.build("spreadsheetbench", {"data_path": spreadsheet}).load()[0]
    agent = ScriptedAgent(["Done"])
    config = SpreadsheetBenchToolCallingConfig(agent=agent.config)
    env = SpreadsheetBenchToolCallingEnvironment(config, agent=agent)
    root = tmp_path / "run"
    original = asyncio.run(env.rollout(sample.task, root, reward_data=sample.reward_data))
    assert asyncio.run(env.rollout(sample.task, root, reward_data=sample.reward_data)) == original
    assert len(agent.requests) == 1
    changed = SpreadsheetBenchToolCallingEnvironment(config.model_copy(update={"tools": []}), agent=agent)
    with pytest.raises(ValueError, match="inputs changed"):
        asyncio.run(changed.rollout(sample.task, root, reward_data=sample.reward_data))
    assert len(agent.requests) == 1


@pytest.mark.parametrize("name", ["spreadsheetbench", "spreadsheetbench_reference"])
def test_obsolete_spreadsheet_environment_names_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown Environment"):
        Environment.resolve(name)
    assert Data.resolve("spreadsheetbench")


def test_spreadsheet_protocols_share_lifecycle_without_inheriting_each_other():
    native = SpreadsheetBenchToolCallingEnvironment
    textual = SpreadsheetBenchTextActionEnvironment
    assert not issubclass(native, textual)
    assert not issubclass(textual, native)
    assert native._run is textual._run
    assert native._interact is textual._interact
    assert native.config_type is not textual.config_type
