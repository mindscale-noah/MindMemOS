"""Verify managed environment reuse, interrupted setup, and concurrent initialization."""

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from mindmemos_sdk.config.constants import MINDMEMOS_HOME_ENV
from mindmemos_skill.infra.execution import python_env


@pytest.fixture
def installation(tmp_path, monkeypatch):
    home = tmp_path / "mindmemos"
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("openpyxl==3.1.5\n")
    calls = []
    monkeypatch.setenv(MINDMEMOS_HOME_ENV, str(home))
    monkeypatch.setattr(python_env.shutil, "which", lambda name: "/installed/uv")

    def run(argv, **kwargs):
        calls.append(argv)
        if "venv" in argv:
            directory = home / "envs" / "spreadsheetbench_text_action"
            executable = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
            (directory / "pyvenv.cfg").touch()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(python_env.subprocess, "run", run)
    return home, requirements, calls, run


def test_environment_is_created_under_home_and_reused_without_uv(installation, monkeypatch):
    home, requirements, calls, _ = installation
    first = python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    assert first.is_relative_to(home / "envs")
    assert len(calls) == 2
    monkeypatch.setattr(python_env.shutil, "which", lambda name: None)
    assert python_env.prepare_python_environment("spreadsheetbench_text_action", requirements) == first
    assert len(calls) == 2


def test_interrupted_installation_is_not_marked_ready_and_recovers(installation, monkeypatch):
    home, requirements, calls, run = installation

    def fail_install(argv, **kwargs):
        if "sync" in argv:
            raise subprocess.CalledProcessError(1, argv)
        return run(argv, **kwargs)

    monkeypatch.setattr(python_env.subprocess, "run", fail_install)
    with pytest.raises(subprocess.CalledProcessError):
        python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    assert not (home / "envs/spreadsheetbench_text_action/.ready.json").exists()
    monkeypatch.setattr(python_env.subprocess, "run", run)
    python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    assert len([argv for argv in calls if "venv" in argv]) == 1
    assert (home / "envs/spreadsheetbench_text_action/.ready.json").is_file()


def test_concurrent_initialization_installs_only_once(installation):
    _, requirements, calls, _ = installation
    barrier = threading.Barrier(2)

    def prepare():
        barrier.wait()
        return python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare) for _ in range(2)]
        paths = [future.result() for future in futures]
    assert paths[0] == paths[1]
    assert len(calls) == 2
