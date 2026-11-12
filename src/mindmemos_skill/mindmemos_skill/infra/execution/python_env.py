"""Prepare reusable Python environments inside the selected MindMemOS home."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from filelock import AsyncFileLock


async def prepare_python_environment(name: str, requirements: Path) -> Path:
    """Create and install one named environment, then reuse completed installations.

    Args:
        name: Fixed environment directory name beneath MindMemOS home/envs.
        requirements: Packaged requirements file with locked dependency versions.

    Returns:
        Absolute path to the installed Python executable.

    Raises:
        RuntimeError: uv is unavailable when installation or dependency synchronization is needed.
        subprocess.CalledProcessError: Environment creation or dependency installation fails.
    """
    from mindmemos_sdk.config.paths import get_mindmemos_home

    home = get_mindmemos_home().expanduser().absolute()
    directory = home / "envs" / name
    directory.parent.mkdir(parents=True, exist_ok=True)
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = directory / ".ready.json"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    async with AsyncFileLock(str(directory.parent / f"{name}.lock"), run_in_executor=False):
        if python.is_file() and marker.is_file():
            if json.loads(marker.read_text(encoding="utf-8"))["requirements_sha256"] == digest:
                return python
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required to initialize or synchronize the managed Python environment")
        marker.unlink(missing_ok=True)
        launcher = [uv, "--cache-dir", str(home / "cache" / "uv")]
        if not (directory / "pyvenv.cfg").is_file():
            await _install([*launcher, "venv", "--python", "3.13", str(directory)])
        await _install([*launcher, "pip", "sync", "--python", str(python), str(requirements)])
        marker.write_text(json.dumps({"requirements_sha256": digest}) + "\n", encoding="utf-8")
    return python


async def _install(command: list[str]) -> None:
    """Run one managed environment installer with bounded lifetime."""
    from .config import ExecutionConfig
    from .executor import execute

    result = await execute(command, Path.cwd(), ExecutionConfig(timeout_s=300))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
