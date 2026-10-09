"""Verify real text-action Python selection and dependency failure gates."""

import json
import os
import shlex
import sys
from pathlib import Path

import pytest
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.execution import check_execution, prepare_execution
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.recalculation import append_recalculation_instructions
from mindmemos_skill.envs.builtin_envs.spreadsheetbench.text_action_policy import run_reference_bash
from mindmemos_skill.infra.execution import ExecutionConfig
from mindmemos_skill.infra.execution.constants import PATH_ENV


@pytest.mark.skipif(os.name == "nt", reason="This regression uses POSIX shell syntax")
def test_text_actions_use_configured_python_with_unrelated_path(tmp_path):
    launcher = tmp_path / "selected launcher.py"
    launcher.write_text(
        'import os, subprocess, sys\nos.environ["SELECTED_RUNTIME"] = "yes"\n'
        "raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))\n"
    )
    config = ExecutionConfig(python_command=[sys.executable, str(launcher)], environment={PATH_ENV: os.defpath})
    probe = 'import os, sys, json, openpyxl; print(json.dumps([os.environ["SELECTED_RUNTIME"], sys.executable, os.getcwd()]))'
    for name in ("python", "python3"):
        observation = run_reference_bash(
            f"{name} -c {shlex.quote(probe)}", working_dir=tmp_path, timeout_seconds=10, execution=config
        )
        selected, interpreter, workspace = json.loads(observation)
        assert selected == "yes"
        assert interpreter == sys.executable
        assert Path(workspace) == tmp_path.resolve()
    assert "SELECTED_RUNTIME" not in os.environ


def test_dependency_probe_and_missing_packages(tmp_path):
    check_execution(ExecutionConfig(python_command=[sys.executable]), tmp_path)
    with pytest.raises(RuntimeError, match="before model execution"):
        check_execution(ExecutionConfig(python_command=[sys.executable, "-S"]), tmp_path)


def test_protocol_selects_packaged_requirements_and_fixed_name(tmp_path, monkeypatch):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import execution

    calls = []

    def prepare(name, requirements):
        calls.append((name, requirements))
        assert requirements.is_file()
        return tmp_path / "envs" / name / "bin/python"

    monkeypatch.setattr(execution, "prepare_python_environment", prepare)
    for protocol in ("text_action", "tool_calling"):
        config = prepare_execution(protocol)
        assert config.python_command == [str(tmp_path / "envs" / f"spreadsheetbench_{protocol}" / "bin/python")]
    assert calls[0][1] != calls[1][1]


def test_recalculation_retains_virtual_environment_interpreter(tmp_path, monkeypatch):
    from mindmemos_skill.envs.builtin_envs.spreadsheetbench import recalculation

    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setattr(recalculation.sys, "executable", str(interpreter))
    prompt = append_recalculation_instructions("Task", working_dir=tmp_path, output_file=tmp_path / "output.xlsx")
    assert str(interpreter) in prompt
