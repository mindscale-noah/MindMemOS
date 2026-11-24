"""Verify real text-action Python selection and dependency failure gates."""

import json
import os
import shlex
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.execution import check_execution, prepare_execution
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_mode.policy import run_reference_bash
from mindmemos_skill.infra.execution import ExecutionConfig
from mindmemos_skill.infra.execution.constants import PATH_ENV


@pytest.mark.skipif(os.name == "nt", reason="This regression uses POSIX shell syntax")
async def test_text_actions_use_configured_python_with_unrelated_path(tmp_path):
    launcher = tmp_path / "selected launcher.py"
    launcher.write_text(
        'import os, subprocess, sys\nos.environ["SELECTED_RUNTIME"] = "yes"\n'
        "raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))\n"
    )
    config = ExecutionConfig(python_command=[sys.executable, str(launcher)], environment={PATH_ENV: os.defpath})
    probe = 'import os, sys, json, openpyxl; print(json.dumps([os.environ["SELECTED_RUNTIME"], sys.executable, os.getcwd()]))'
    for name in ("python", "python3"):
        observation = await run_reference_bash(
            f"{name} -c {shlex.quote(probe)}", working_dir=tmp_path, timeout_seconds=10, execution=config
        )
        selected, interpreter, workspace = json.loads(observation)
        assert selected == "yes"
        assert interpreter == sys.executable
        assert Path(workspace) == tmp_path.resolve()
    assert "SELECTED_RUNTIME" not in os.environ


async def test_dependency_probe_and_missing_packages(tmp_path):
    (await check_execution(ExecutionConfig(python_command=[sys.executable]), tmp_path))
    with pytest.raises(RuntimeError, match="before model execution"):
        (await check_execution(ExecutionConfig(python_command=[sys.executable, "-S"]), tmp_path))


async def test_protocol_selects_packaged_requirements_and_fixed_name(tmp_path, monkeypatch):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import execution

    calls = []

    def prepare(name, requirements):
        calls.append((name, requirements))
        assert requirements.is_file()
        return tmp_path / "envs" / name / "bin/python"

    monkeypatch.setattr(execution, "prepare_python_environment", AsyncMock(side_effect=prepare))
    for protocol in ("text_action", "tool_calling"):
        config = await prepare_execution(protocol)
        assert config.python_command == [str(tmp_path / "envs" / f"spreadsheetbench_{protocol}" / "bin/python")]
    assert calls[0][1] != calls[1][1]


@pytest.mark.parametrize("recalculation", [False, True])
def test_native_prompt_matches_cached_value_grading(recalculation, tmp_path):
    """Only cached-value grading instructs the model to avoid unevaluated formulas."""
    from mindmemos_skill.agents import OpenAIChatConfig
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.config import SpreadsheetBenchToolCallingConfig
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.tool_call_mode.workspace_tools import system_prompt
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench.typing import (
        SpreadsheetBenchTask,
        SpreadsheetContext,
        WorkbookInput,
    )

    config = SpreadsheetBenchToolCallingConfig(
        agent=OpenAIChatConfig(model="offline"), transactional_recalculation=recalculation
    )
    context = SpreadsheetContext(
        task=SpreadsheetBenchTask(
            id="fixture", instruction="Compute totals", cases=[WorkbookInput(number="1", path=tmp_path / "input.xlsx")]
        ),
        directory=tmp_path,
        skill_directory=tmp_path / "skills",
        skill_text="",
        tools=[],
    )
    prompt = system_prompt(context, config)
    assert ("Write the final computed values into the target cells" in prompt) is (not recalculation)
    assert ("no formula recalculation" in prompt) is (not recalculation)
