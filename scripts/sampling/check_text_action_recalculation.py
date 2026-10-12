"""Exercise the text-action prompt's real bash/Python/LibreOffice chain without a model."""

import argparse
import asyncio
import hashlib
import json
import shlex
from pathlib import Path

from mindmemos_skill.agents.buitin_agents.openai_chat.config import OpenAIChatConfig
from mindmemos_skill.agents.typing import AgentResult
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.config import SpreadsheetBenchTextActionConfig
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.environment import (
    SpreadsheetBenchTextActionEnvironment,
)
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.policy import run_reference_bash
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import (
    SpreadsheetBenchTask,
    SpreadsheetContext,
    WorkbookInput,
)
from openpyxl import Workbook, load_workbook
from pydantic import JsonValue


async def check(directory: Path) -> dict[str, JsonValue]:
    """Run prompt-provided commands and verify caches, statuses, and interpreter selection.

    Args:
        directory: New output directory for case workspaces and a JSON summary.

    Returns:
        JSON summary with python (managed interpreter), libreoffice_version (version text),
        and cases (case name, command, full recalculation status, and cached_value).

    Raises:
        AssertionError: A real command or workbook result differs from the contract.
    """
    directory.mkdir(parents=True, exist_ok=False)
    env = SpreadsheetBenchTextActionEnvironment(
        SpreadsheetBenchTextActionConfig(agent=OpenAIChatConfig(model="offline"))
    )
    await env.initialize()
    probe = await run_reference_bash(
        "python -c " + shlex.quote("import sys, openpyxl; print(sys.executable)"),
        working_dir=directory,
        timeout_seconds=10,
        execution=env.execution,
    )
    assert Path(probe).absolute() == Path(env.execution.python_command[0]).absolute(), probe
    results = []
    for name, expression, expected_status, expected_value in (
        ("sum with spaces", "=SUM(A1:A2)", "success", 3),
        ("literal", 3, "not_needed", 3),
        ("formula error", "=1/0", "errors_found", "#DIV/0!"),
    ):
        workspace = directory / name
        workspace.mkdir()
        output = workspace / "output.xlsx"
        workbook = Workbook()
        workbook.active["A1"] = 1
        workbook.active["A2"] = 2
        workbook.active["B1"] = expression
        workbook.save(workspace / "input.xlsx")
        workbook.save(output)
        workbook.close()
        before = hashlib.sha256(output.read_bytes()).hexdigest()
        task = SpreadsheetBenchTask(
            id=name,
            instruction="Save the requested result in B1.",
            cases=[WorkbookInput(number="1", path=workspace / "input.xlsx")],
        )
        context = SpreadsheetContext(task=task, directory=workspace, skill_directory=workspace, tools=[], skill_text="")
        prompt = env._user_prompt(context)
        command = prompt.split("run this exact command before signaling completion:\n", 1)[1].split("\n", 1)[0]
        action = "Action:\n" + json.dumps({"name": "bash", "arguments": {"command": command}})
        result = await env._get_observation(AgentResult(output=action), context)
        observation = result.messages[0]["content"]
        (workspace / "task.md").write_text(prompt, encoding="utf-8")
        (workspace / "observation.txt").write_text(observation, encoding="utf-8")
        status = json.loads((workspace / "recalc_status.json").read_text())
        assert status["status"] == expected_status, observation
        cached = load_workbook(output, data_only=True)
        try:
            assert cached.active["B1"].value == expected_value, cached.active["B1"].value
        finally:
            cached.close()
        assert not (workspace / "output.recalc.xlsx").exists()
        assert not [path for path in workspace.glob(".spreadsheetbench_recalc_*") if path.is_dir()]
        if expected_status == "not_needed":
            assert hashlib.sha256(output.read_bytes()).hexdigest() == before
        else:
            assert status["replaced_output"]
            assert status["formula_count_before"] == status["formula_count_after"] == 1
            assert status["cached_formula_value_count_after"] == 1
        assert ("[Exit code: 1]" in observation) == (expected_status == "errors_found")
        results.append({"case": name, "command": command, "status": status, "cached_value": expected_value})
    version = await run_reference_bash(
        shlex.join([results[0]["status"]["soffice"], "--version"]),
        working_dir=directory,
        timeout_seconds=10,
        execution=env.execution,
    )
    summary = {"python": probe, "libreoffice_version": version, "cases": results}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    """Verify a configured text-action runtime and print the resulting summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(check(args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
