"""Compare actor messages and tools against pinned source checkouts without LLM calls."""

import argparse
import ast
import importlib
import json
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src/mindmemos_skill"))


def main() -> None:
    """Compare public actor behavior, excluding the deliberately unified reward.

    Raises:
        AssertionError: A prompt, message, tool result, or completion output differs.
    """
    import openpyxl
    from mindmemos_skill.agents import OpenAIChatAgent, OpenAIChatConfig
    from mindmemos_skill.agents.buitin_agents.openai_chat.callbacks import (
        native_observation,
        parse_native_tools,
        parse_text_action,
        text_format_error,
        text_observation,
    )
    from mindmemos_skill.agents.typing import AgentResult
    from mindmemos_skill.envs.builtin_envs.common import export_skills
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import SpreadsheetBenchConfig, SpreadsheetBenchTask
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.bash_only import run_bash_only
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.prompts.bash_only import NO_SKILL, WITH_SKILL
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.prompts.workspace_tools import SYSTEM_PROMPT
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.tools import TOOLS, run_tool
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import WorkbookInput
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.workspace_tools import run_workspace_tools
    from mindmemos_skill.infra.execution import ExecutionConfig
    from mindmemos_skill.skill import Skill, SkillBundle, SkillResources

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-source", type=Path, required=True, help="Skill-Forest repository root")
    parser.add_argument("--bash-source", type=Path, required=True, help="Original TreeSkill repository root")
    args = parser.parse_args()
    forest_file = args.workspace_source / "skillopt/envs/spreadsheetbench/react_agent.py"
    module = ast.parse(forest_file.read_text())
    module.body = [
        node for node in module.body if not (isinstance(node, ast.ImportFrom) and node.module == "skillopt.model")
    ]
    forest = {}
    exec(compile(module, str(forest_file), "exec"), forest)
    assert SYSTEM_PROMPT == forest["SYSTEM_PROMPT"]
    assert TOOLS == forest["TOOLS"]
    for actual, filename in [
        (NO_SKILL, "cli_only_full_system_v1.txt"),
        (WITH_SKILL, "cli_skill_preloaded_full_system_v1.txt"),
    ]:
        assert actual == (args.bash_source / "spreadsheet_agent/system_prompt" / filename).read_text()

    sys.path[:0] = [str(args.bash_source), str(args.bash_source / "src")]
    from react_agent import Tool
    from spreadsheet_agent.agents.base import AgentContext
    from spreadsheet_agent.agents.cli_only_agent import CLIOnlyAgent
    from spreadsheet_agent.agents.cli_skill_preloaded_agent import CLISkillPreloadedAgent

    class ScriptedModel(OpenAIChatAgent):
        """Replay responses while recording model-visible conversations."""

        def __init__(self, events, config):
            super().__init__(config)
            self.events = iter(events)
            self.requests = []

        def run_messages(self, *, messages, tools=None):
            self.requests.append(json.loads(json.dumps(messages)))
            event = next(self.events)
            if isinstance(event, str):
                event = {"role": "assistant", "content": event}
            return AgentResult(output=event.get("content", ""), trajectory=[*messages, event])

    class ReferenceClient:
        """Replay upstream text responses without constructing a provider client."""

        def __init__(self, events):
            self.events = iter(events)
            self.requests = []

        async def chat_async(self, messages, settings):
            assert settings.stop == ["Observation:"]
            self.requests.append([{"role": m.role, "content": m.content} for m in messages])
            return next(self.events)

    counts = {"prompt_and_schema": 4, "workspace_conversations": 0, "bash_conversations": 0, "tool_observations": 0}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        case = root / "case"
        case.mkdir()
        book = openpyxl.Workbook()
        book.active["A1"] = 1
        book.save(case / "input.xlsx")
        book.close()
        task = SpreadsheetBenchTask(
            id="fixture",
            instruction="Increment A1",
            answer_position="Sheet!A1",
            cases=[WorkbookInput(number="1", path=case / "input.xlsx", sha256="fixture")],
        )
        for name, arguments in [
            ("write", {"path": "a.txt", "content": "same"}),
            ("read", {"path": "a.txt"}),
            ("edit", {"path": "a.txt", "original_text": "absent", "replacement_text": "x"}),
            ("shell", {"commands": ["printf hello", "printf error >&2; exit 2"]}),
            ("shell", {"commands": ["pip install blocked"]}),
            ("shell", {"commands": [1]}),
        ]:
            assert run_tool(name, arguments, case, ExecutionConfig()) == forest["_execute_tool"](
                name, arguments, str(case)
            )
            counts["tool_observations"] += 1

        def native(name, arguments):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call", "type": "function", "function": {"name": name, "arguments": arguments}}],
            }

        prepared_skills = [
            Skill(
                bundle=SkillBundle(files={"SKILL.md": text}),
                resources=SkillResources(files={f"references/{index}.md": b"Reference"}),
            )
            for index, text in enumerate(("Use values.", "Verify saved cells."))
        ]
        exported_text = export_skills(prepared_skills, case / "skills/xlsx")
        assert exported_text == "Use values.\n\nVerify saved cells."
        workspace_cases = [
            ("tool", "", ["Done"]),
            ("system_prompt", "", ["Done"]),
            ("system_prompt", "Use values.", ["Done"]),
            ("tool", "Use values.", [native("skill", '{"name":"xlsx"}'), "Done"]),
            ("tool", "", [native("shell", "{"), "Done"]),
            ("system_prompt", exported_text, ["Done"]),
            ("tool", exported_text, [native("skill", '{"name":"xlsx"}'), "Done"]),
        ]
        for loading, text, events in workspace_cases:
            config = OpenAIChatConfig(model="fixture", invalid_tool_arguments="empty")
            model = ScriptedModel(events, config)
            settings = SpreadsheetBenchConfig(agent=config, skill_loading_mode=loading)
            adapter = OpenAIChatAgent(
                config, model=model, parse_response=parse_native_tools, observation=native_observation
            )
            run_workspace_tools(task, text, [case], root, settings, adapter)
            source_events = iter(events)
            requests = []

            def replay(**kwargs):
                requests.append(json.loads(json.dumps(kwargs["messages"])))
                event = next(source_events)
                if isinstance(event, str):
                    event = {"role": "assistant", "content": event}
                return SimpleNamespace(content=event["content"], tool_calls=event.get("tool_calls", [])), {}, 0

            forest["_chat_target_with_backoff"] = replay
            forest["run_react"](
                task.instruction,
                str(case / "input.xlsx"),
                str(case / "output.xlsx"),
                str(case),
                answer_position=task.answer_position,
                skill_content=text,
                skill_loading_mode=loading,
            )
            if model.requests != requests:
                print("request counts", len(model.requests), len(requests))
                for index, (actual, expected) in enumerate(zip(model.requests, requests)):
                    for a, e in zip(actual, expected):
                        if a != e:
                            print("difference", index, a, e)
                raise AssertionError((loading, text, "workspace conversations differ"))
            counts["workspace_conversations"] += 1

        skill_dir = root / "skills" / "xlsx"
        skill_dir.mkdir(parents=True)
        raw_skill = "---\nname: xlsx\ndescription: Test\n---\n\nUse values."
        (skill_dir / "SKILL.md").write_text(raw_skill)
        action = 'Action: {"name":"bash","arguments":{"command":"write-output"}}'
        scenarios = [
            (False, [action, "ACTION: TASK_COMPLETE"]),
            (True, [action, "ACTION: TASK_COMPLETE"]),
            (False, ["Done", action, "Done"]),
            (False, ["Done", "Done"]),
            (False, ["Action: broken", action, "Done"]),
            (False, [action]),
            (False, ['Action: {"name":"bash","arguments":{"command":"echo"}}'] * 6),
        ]
        for with_skill, events in scenarios:
            (case / "output.xlsx").unlink(missing_ok=True)
            budget = len(events)
            config = OpenAIChatConfig(model="fixture")
            model = ScriptedModel(events, config)
            settings = SpreadsheetBenchConfig(
                agent=config,
                interaction_mode="bash_only",
                tool_agent="text_action",
                skill_loading_mode="system_prompt",
                max_turns=budget,
            )
            adapter = OpenAIChatAgent(
                config,
                model=model,
                parse_response=parse_text_action,
                format_error=text_format_error,
                observation=text_observation,
                native_tools=False,
            )

            def execute(command):
                if command == "write-output":
                    (case / "output.xlsx").write_text("fixture")
                return "done"

            implementation = importlib.import_module("mindmemos_skill.envs.builtin_envs.spreadsheetbench.bash_only")
            with patch.object(
                implementation,
                "run_bash_action",
                lambda action, directory, config: execute(action["arguments"]["command"]),
            ):
                result = run_bash_only(task, raw_skill if with_skill else "", case, root, settings, adapter)
            (case / "output.xlsx").unlink(missing_ok=True)
            client = ReferenceClient(events)
            if with_skill:
                reference = CLISkillPreloadedAgent(
                    client, skills_dir=str(root / "skills"), max_turns=budget, verbose=False
                )
                # Align only the public resource location; content and rendering remain upstream-owned.
                reference._skills[0][0].file_path = str(case / "SKILL.md")
            else:
                reference = CLIOnlyAgent(client, max_turns=budget, verbose=False)
            reference.create_tools = lambda working_dir: [Tool(name="bash", description="", func=execute)]
            context = AgentContext(
                instruction=task.instruction,
                working_dir=str(case),
                input_file=str(case / "input.xlsx"),
                output_file=str(case / "output.xlsx"),
                spreadsheet_content="('1',)",
                instruction_type="",
                answer_position=task.answer_position,
            )
            expected = reference.run(context)
            assert model.requests == client.requests, (with_skill, events, "model conversations differ")
            if expected["success"]:
                assert result.output == expected["answer"]
            counts["bash_conversations"] += 1
    print(json.dumps({"passed": counts}, indent=2))
    for file in [
        forest_file,
        args.bash_source / "src/react_agent/agent.py",
        args.bash_source / "src/react_agent/models.py",
    ]:
        print(f"sha256 {file}: {sha256(file.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
