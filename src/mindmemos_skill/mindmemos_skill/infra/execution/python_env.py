"""Prepare reusable Python environments inside the selected MindMemOS home."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from filelock import FileLock


def prepare_python_environment(name: str, requirements: Path) -> Path:
    """Create and install one named environment, then reuse completed installations.

    Args:
        name: Fixed environment directory name beneath MindMemOS home/envs.
        requirements: Packaged requirements file with locked dependency versions.

    Returns:
        Absolute path to the installed Python executable.

    Raises:
        RuntimeError: uv is unavailable or an existing environment has different requirements.
        subprocess.CalledProcessError: Environment creation or dependency installation fails.
    """
    from mindmemos_sdk.config.paths import get_mindmemos_home

    home = get_mindmemos_home().expanduser().absolute()
    directory = home / "envs" / name
    directory.parent.mkdir(parents=True, exist_ok=True)
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = directory / ".ready.json"
    digest = hashlib.sha256(requirements.read_bytes()).hexdigest()
    with FileLock(str(directory.parent / f"{name}.lock")):
        if python.is_file() and marker.is_file():
            if json.loads(marker.read_text(encoding="utf-8"))["requirements_sha256"] != digest:
                raise RuntimeError(f"Installed requirements changed; recreate the managed environment: {directory}")
            return python
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv is required for the first Python environment initialization")
        launcher = [uv, "--cache-dir", str(home / "cache" / "uv")]
        if not (directory / "pyvenv.cfg").is_file():
            subprocess.run([*launcher, "venv", "--python", "3.13", str(directory)], check=True, timeout=300)
        subprocess.run([*launcher, "pip", "sync", "--python", str(python), str(requirements)], check=True, timeout=300)
        marker.write_text(json.dumps({"requirements_sha256": digest}) + "\n", encoding="utf-8")
    return python
