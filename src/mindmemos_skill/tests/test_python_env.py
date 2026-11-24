"""Verify managed environment reuse, interrupted setup, and concurrent initialization."""

import asyncio
import hashlib
import json
import os
import subprocess
from unittest.mock import AsyncMock

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

    monkeypatch.setattr(python_env, "_install", AsyncMock(side_effect=run))
    return home, requirements, calls, run


async def test_environment_is_created_under_home_and_reused_without_uv(installation, monkeypatch):
    home, requirements, calls, _ = installation
    first = await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    assert first.is_relative_to(home / "envs")
    assert len(calls) == 2
    monkeypatch.setattr(python_env.shutil, "which", lambda name: None)
    assert (await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)) == first
    assert len(calls) == 2


async def test_interrupted_installation_is_not_marked_ready_and_recovers(installation, monkeypatch):
    home, requirements, calls, run = installation

    def fail_install(argv, **kwargs):
        if "sync" in argv:
            raise subprocess.CalledProcessError(1, argv)
        return run(argv, **kwargs)

    monkeypatch.setattr(python_env, "_install", AsyncMock(side_effect=fail_install))
    with pytest.raises(subprocess.CalledProcessError):
        (await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements))
    assert not (home / "envs/spreadsheetbench_text_action/.ready.json").exists()
    monkeypatch.setattr(python_env, "_install", AsyncMock(side_effect=run))
    (await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements))
    assert len([argv for argv in calls if "venv" in argv]) == 1
    assert (home / "envs/spreadsheetbench_text_action/.ready.json").is_file()


async def test_concurrent_initialization_installs_only_once(installation):
    _, requirements, calls, _ = installation
    barrier = asyncio.Barrier(2)

    async def prepare():
        await barrier.wait()
        return await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)

    paths = await asyncio.gather(prepare(), prepare())
    assert paths[0] == paths[1]
    assert len(calls) == 2


async def test_changed_requirements_are_synchronized_once_for_concurrent_callers(installation):
    home, requirements, calls, _ = installation
    first = await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    requirements.write_text("openpyxl==3.1.5\npandas==2.3.3\n")
    paths = await asyncio.gather(
        python_env.prepare_python_environment("spreadsheetbench_text_action", requirements),
        python_env.prepare_python_environment("spreadsheetbench_text_action", requirements),
    )
    assert paths == [first, first]
    assert len(calls) == 3
    assert "sync" in calls[-1]
    marker = home / "envs/spreadsheetbench_text_action/.ready.json"
    assert (
        json.loads(marker.read_text())["requirements_sha256"] == hashlib.sha256(requirements.read_bytes()).hexdigest()
    )


async def test_failed_dependency_update_invalidates_ready_marker_and_recovers(installation, monkeypatch):
    home, requirements, calls, run = installation
    await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    requirements.write_text("openpyxl==3.1.5\npandas==2.3.3\n")
    monkeypatch.setattr(python_env, "_install", AsyncMock(side_effect=subprocess.CalledProcessError(1, ["uv", "sync"])))
    with pytest.raises(subprocess.CalledProcessError):
        await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    marker = home / "envs/spreadsheetbench_text_action/.ready.json"
    assert not marker.exists()
    monkeypatch.setattr(python_env, "_install", AsyncMock(side_effect=run))
    await python_env.prepare_python_environment("spreadsheetbench_text_action", requirements)
    assert marker.is_file()
    assert len([argv for argv in calls if "venv" in argv]) == 1
