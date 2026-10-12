"""Exercise real local Python selection, persistent files, and timeout cleanup."""

import json
import os
import shutil
import subprocess
import sys

import pytest
from mindmemos_skill.infra.execution import ExecutionConfig, execute
from pydantic import ValidationError


async def test_python_selection_and_workspace_persistence(tmp_path):
    config = ExecutionConfig(python_command=[sys.executable])
    (tmp_path / "save.py").write_text(
        'from pathlib import Path\nPath("saved.txt").write_text("跨轮", encoding="utf-8")\n', encoding="utf-8"
    )
    result = await execute([*config.shell_command, "python save.py"], tmp_path, config)
    assert result.returncode == 0, result.stderr
    (tmp_path / "read.py").write_text(
        'import sys, json\nfrom pathlib import Path\nprint(json.dumps([sys.executable, Path("saved.txt").read_text(encoding="utf-8")]))\n',
        encoding="utf-8",
    )
    result = await execute([*config.shell_command, "python3 read.py"], tmp_path, config)
    assert result.returncode == 0, result.stderr
    interpreter, content = json.loads(result.stdout)
    assert os.path.samefile(interpreter, sys.executable)
    assert content == "跨轮"


async def test_launcher_prefix_and_paths_with_spaces(tmp_path):
    launcher = tmp_path / "prefix launcher.py"
    launcher.write_text(
        'import subprocess, sys\nassert sys.argv[1] == "literal argument"\nraise SystemExit(subprocess.call([sys.executable, *sys.argv[2:]]))\n'
    )
    config = ExecutionConfig(python_command=[sys.executable, str(launcher), "literal argument"])
    (tmp_path / "script.py").write_text('print("chosen launcher")\n')
    result = await execute([*config.shell_command, "python script.py"], tmp_path, config)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "chosen launcher"


async def test_local_timeout(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        (await execute(["python", "-c", "import time; time.sleep(20)"], tmp_path, ExecutionConfig(timeout_s=0.1)))


async def test_missing_python_launcher(tmp_path):
    with pytest.raises(FileNotFoundError, match="launcher not found"):
        (await execute(["python", "-V"], tmp_path, ExecutionConfig(python_command=["not-a-real-python-987"])))


@pytest.mark.parametrize("value", [[], [""], ["python", " "]])
def test_invalid_python_command(value):
    with pytest.raises(ValidationError):
        ExecutionConfig(python_command=value)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
async def test_uv_launcher_offline(tmp_path):
    config = ExecutionConfig(
        python_command=[
            "uv",
            "--cache-dir",
            str(tmp_path / "uv-cache"),
            "run",
            "--offline",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
        ]
    )
    (tmp_path / "script.py").write_text('print("uv selected")\n')
    result = await execute([*config.shell_command, "python script.py"], tmp_path, config)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "uv selected"


async def test_windows_timeout_uses_process_tree_cleanup(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from mindmemos_skill.infra.execution import executor

    events = []
    process = SimpleNamespace(pid=123, returncode=None, kill=lambda: events.append("kill"), wait=AsyncMock())
    killer = SimpleNamespace(wait=AsyncMock())
    launch = AsyncMock(return_value=killer)
    monkeypatch.setattr(executor, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(executor.asyncio, "create_subprocess_exec", launch)
    await executor._terminate(process)
    assert launch.call_args.args == ("taskkill", "/PID", "123", "/T", "/F")
    assert events == ["kill"]
    killer.wait.assert_awaited_once()
    process.wait.assert_awaited_once()


def test_windows_launchers_quote_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mindmemos_skill.infra.execution import executor

    monkeypatch.setattr(executor, "os", SimpleNamespace(name="nt"))
    executor._python_launchers(tmp_path, ["C:/Python Env/python.exe"], "C:/Windows")
    assert (tmp_path / "python.cmd").is_file()
    assert (tmp_path / "python3.cmd").is_file()
    assert "%*" in (tmp_path / "python.cmd").read_text()
    assert "C:/Python Env/python.exe" in (tmp_path / "launch.py").read_text()
