"""Compose operations shared by startup, status, stop, and explicit data cleanup."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from . import constants as env
from .control import request_sessions
from .typing import DependencyPlan


def compose(root: Path, *arguments: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Run Compose with the already resolved process environment.

    Args:
        root: Repository containing the shared Compose definition.
        arguments: Compose operation and its arguments.
        capture: Capture output when it contains private configuration values.

    Returns:
        Successful Compose result; failures raise CalledProcessError.
    """
    return subprocess.run(
        ["docker", "compose", "--env-file", "/dev/null", "-f", "dockers/compose.yaml", *arguments],
        cwd=root,
        check=True,
        capture_output=capture,
        text=True,
    )


def _runtime_directory(root: Path, project: str) -> Path:
    """Isolate generated mounts by workspace and resolved Compose project."""
    return root / ".local" / "docker" / hashlib.sha256(project.encode()).hexdigest()[:16]


def prepare_collector(root: Path, plan: DependencyPlan) -> None:
    """Materialize a durable Collector mount only when starting local telemetry.

    Args:
        root: Workspace containing the Collector template and generated mounts.
        plan: Final dependency selection, including optional Kafka exporter.
    """
    if "otel-collector" not in plan.services:
        return
    document = yaml.safe_load((root / "dockers" / "otel" / "collector.yaml").read_text())
    if "kafka-exporter" not in plan.services:
        document["receivers"].pop("prometheus", None)
        document["service"]["pipelines"].pop("metrics", None)
    content = yaml.safe_dump(document, sort_keys=False)
    project = json.loads(compose(root, "config", "--format", "json", capture=True).stdout)["name"]
    directory = _runtime_directory(root, project)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{hashlib.sha256(content.encode()).hexdigest()}.yaml"
    if not target.exists():
        # The file contains environment references, not resolved credentials.
        # Atomic publication prevents partially written bind mounts.
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as file:
            temporary = Path(file.name)
            file.write(content)
        try:
            temporary.chmod(0o644)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    os.environ[env.OTEL_CONFIG] = str(target.resolve())


def manage(root: Path, action: str) -> int:
    """Inspect or stop managed applications and the current Compose project.

    Args:
        root: Workspace whose processes and Compose project are selected.
        action: status, stop, or clean. Cleanup requires interactive confirmation.

    Returns:
        Zero after successful management, or one when cleanup is cancelled.
    """
    if action == "status":
        responses = request_sessions(root, "status")
        if not responses:
            print("No managed application processes are running.")
        for response in responses:
            print(("READY" if response["ready"] else "STARTING") + "\n" + response["summary"])
        compose(root, "ps", "--all")
        return 0
    config = json.loads(compose(root, "config", "--format", "json", capture=True).stdout)
    if action == "clean":
        print("This deletes the following Docker volumes and their stored data:")
        for name, volume in config.get("volumes", {}).items():
            if not volume.get("external"):
                print("  " + volume.get("name", name))
        print("YAML, dotenv, SDK configuration and host SQLite files are not removed.")
        try:
            confirmed = input("Type DELETE to stop services and delete these volumes: ") == "DELETE"
        except EOFError:
            confirmed = False
        if not confirmed:
            print("Cleanup cancelled; no services or data were changed.")
            return 1
    request_sessions(root, "stop")
    compose(root, "down", *(["--volumes"] if action == "clean" else []))
    shutil.rmtree(_runtime_directory(root, config["name"]), ignore_errors=True)
    return 0
