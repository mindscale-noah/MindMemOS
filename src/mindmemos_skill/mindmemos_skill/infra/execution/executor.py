"""Run local commands with a selected Python environment and bounded lifetime."""

import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import ExecutionConfig
from .constants import PATH_ENV, PYTHON_IO_ENCODING_ENV, PYTHON_UNBUFFERED_ENV


def _resolve_executable(command: list[str]) -> list[str]:
    """Resolve the launcher before changing the working directory."""
    name = os.path.expanduser(command[0])
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"Execution launcher not found: {name}")
    return [str(Path(executable).absolute()), *command[1:]]


def _python_launchers(directory: Path, python: list[str], search_path: str) -> None:
    """Expose python/python3 to shells without rewriting generated source."""
    launcher = directory / "launch.py"
    launcher.write_text(
        "import os, subprocess, sys\n"
        f"os.environ[{PATH_ENV!r}] = {search_path!r}\n"
        f"raise SystemExit(subprocess.call({python!r} + sys.argv[1:]))\n",
        encoding="utf-8",
    )
    for name in ("python", "python3"):
        if os.name == "nt":
            path = directory / f"{name}.cmd"
            invocation = subprocess.list2cmdline([sys.executable, str(launcher)]).replace("%", "%%")
            path.write_text(f"@echo off\nchcp 65001 >nul\n{invocation} %*\n", encoding="utf-8")
        else:
            path = directory / name
            path.write_text(f'#!/bin/sh\nexec {shlex.join([sys.executable, str(launcher)])} "$@"\n', encoding="utf-8")
            path.chmod(0o700)


def _terminate(process: subprocess.Popen) -> None:
    """Stop the command's process tree using the host platform's mechanism."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.communicate()


def execute(
    command: list[str], workspace: Path, config: ExecutionConfig, *, input_text: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute locally with persistent workspace files and the chosen Python.

    Args:
        command: Literal process argv; shells receive their command as one argument.
        workspace: Current directory for the subprocess, not an OS sandbox.
        config: Python launcher, shell, and timeout settings.
        input_text: Optional UTF-8 content supplied on stdin.

    Returns:
        Captured UTF-8 stdout, stderr, and exit status.

    Raises:
        FileNotFoundError: The configured launcher or command is unavailable.
        subprocess.TimeoutExpired: The command exceeded its execution budget.
    """
    python = _resolve_executable(config.python_command)
    invocation = [*python, *command[1:]] if command[0] in {"python", "python3"} else command
    env = {name: value for name, value in os.environ.items() if name not in config.excluded_environment}
    env.update(config.environment)
    search_path = env.get(PATH_ENV, os.defpath)
    with tempfile.TemporaryDirectory(prefix="python-launcher-") as temporary:
        _python_launchers(Path(temporary), python, search_path)
        env[PATH_ENV] = temporary + os.pathsep + search_path
        env[PYTHON_UNBUFFERED_ENV] = "1"
        env[PYTHON_IO_ENCODING_ENV] = "utf-8"
        options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        )
        process = subprocess.Popen(
            invocation,
            cwd=workspace.resolve(),
            env=env,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **options,
        )
        try:
            stdout, stderr = process.communicate(input=input_text, timeout=config.timeout_s)
        except BaseException:
            _terminate(process)
            raise
    return subprocess.CompletedProcess(invocation, process.returncode, stdout, stderr)
