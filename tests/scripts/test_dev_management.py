"""Verify scoped process control and destructive cleanup boundaries."""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.dev import control, docker


def test_cancel_clean_changes_nothing(tmp_path, monkeypatch, capsys):
    compose = Mock(
        return_value=subprocess.CompletedProcess([], 0, json.dumps({"volumes": {"data": {"name": "test_data"}}}))
    )
    monkeypatch.setattr(docker, "compose", compose)
    monkeypatch.setattr(docker, "request_sessions", lambda *args: pytest.fail("Must not stop before confirmation"))
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert docker.manage(tmp_path, "clean") == 1
    assert compose.call_count == 1
    assert "test_data" in capsys.readouterr().out


def test_confirmed_clean_stops_apps_before_deleting_volumes(tmp_path, monkeypatch):
    operations = []

    def compose(root, *args, **kwargs):
        operations.append(args)
        return subprocess.CompletedProcess([], 0, '{"name": "test", "volumes": {}}')

    monkeypatch.setattr(docker, "compose", compose)
    monkeypatch.setattr(docker, "request_sessions", lambda root, command: operations.append((command,)))
    monkeypatch.setattr("builtins.input", lambda prompt: "DELETE")
    assert docker.manage(tmp_path, "clean") == 0
    assert operations[-2:] == [("stop",), ("down", "--volumes")]


def test_stop_preserves_volumes(tmp_path, monkeypatch):
    operations = []

    def compose(root, *args, **kwargs):
        operations.append(args)
        return subprocess.CompletedProcess([], 0, '{"name": "test"}')

    monkeypatch.setattr(docker, "compose", compose)
    monkeypatch.setattr(docker, "request_sessions", lambda root, command: operations.append((command,)))
    assert docker.manage(tmp_path, "stop") == 0
    assert operations == [("config", "--format", "json"), ("stop",), ("down",)]


def test_control_sessions_are_workspace_scoped(tmp_path):
    session = control.ControlSession(tmp_path / "owner", "Skill API http://127.0.0.1:18001")

    def serve():
        try:
            while not session.poll(True):
                time.sleep(0.01)
        finally:
            session.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert control.request_sessions(tmp_path / "other", "stop") == []
        responses = control.request_sessions(tmp_path / "owner", "status")
        assert responses[0]["ready"] is True
        assert "18001" in responses[0]["summary"]
        control.request_sessions(tmp_path / "owner", "stop")
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert not session.path.exists()
    finally:
        if thread.is_alive():
            control.request_sessions(tmp_path / "owner", "stop")
            thread.join(timeout=2)


def test_launcher_control_stops_child_in_another_process(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = (
        "from pathlib import Path; import sys; from scripts.dev import run; "
        f"run.ROOT=Path({str(tmp_path)!r}); "
        'raise SystemExit(run._run_apis([[sys.executable,"-c","import time; time.sleep(60)"]], summary="test runner"))'
    )
    child = subprocess.Popen([sys.executable, "-c", code], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 5
        while not list(control._directory(tmp_path).glob("*.sock")):
            assert child.poll() is None
            if time.monotonic() > deadline:
                pytest.fail("Launcher did not publish control socket")
            time.sleep(0.05)
        assert control.request_sessions(tmp_path, "status")[0]["summary"] == "test runner"
        control.request_sessions(tmp_path, "stop")
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("kafka", [False, True])
def test_collector_mount_is_durable_and_matches_dependencies(tmp_path, monkeypatch, kafka):
    import os
    import shutil

    import yaml

    from scripts.dev import constants as env
    from scripts.dev.typing import DependencyPlan

    source = Path(__file__).resolve().parents[2] / "dockers" / "otel" / "collector.yaml"
    target = tmp_path / "dockers" / "otel" / "collector.yaml"
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    original = target.read_bytes()
    monkeypatch.setenv(env.OTEL_CONFIG, "unused")
    monkeypatch.setattr(
        docker, "compose", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, '{"name": "test"}')
    )
    services = {"otel-collector", "clickhouse", "grafana"}
    if kafka:
        services.add("kafka-exporter")
    docker.prepare_collector(tmp_path, DependencyPlan(services=services))
    mount = Path(os.environ[env.OTEL_CONFIG])
    generated = yaml.safe_load(mount.read_text())
    assert ("prometheus" in generated["receivers"]) == kafka
    assert ("metrics" in generated["service"]["pipelines"]) == kafka
    assert generated["exporters"]["clickhouse"]["password"] == "${env:CLICKHOUSE_PASSWORD}"
    assert mount.is_file() and target.read_bytes() == original
    modified = mount.stat().st_mtime_ns
    docker.prepare_collector(tmp_path, DependencyPlan(services=services))
    assert mount.stat().st_mtime_ns == modified
    services.symmetric_difference_update({"kafka-exporter"})
    docker.prepare_collector(tmp_path, DependencyPlan(services=services))
    assert Path(os.environ[env.OTEL_CONFIG]) != mount
    assert mount.exists()
    monkeypatch.setattr(docker, "request_sessions", lambda *args: [])
    other_project = docker._runtime_directory(tmp_path, "other")
    other_project.mkdir(parents=True)
    (other_project / "keep.yaml").write_text("keep")
    assert docker.manage(tmp_path, "stop") == 0
    assert not mount.exists()
    assert (other_project / "keep.yaml").exists()


def test_collector_disabled_does_not_generate_or_query_compose(tmp_path, monkeypatch):
    from scripts.dev.typing import DependencyPlan

    monkeypatch.setattr(docker, "compose", lambda *args, **kwargs: pytest.fail("No Collector is required"))
    docker.prepare_collector(tmp_path, DependencyPlan())
    assert not (tmp_path / ".local").exists()


def test_failed_down_keeps_container_mounts(tmp_path, monkeypatch):
    directory = docker._runtime_directory(tmp_path, "test")
    directory.mkdir(parents=True)
    mount = directory / "collector.yaml"
    mount.write_text("keep")

    def compose(root, *args, **kwargs):
        if args[0] == "down":
            raise subprocess.CalledProcessError(1, "docker compose down")
        return subprocess.CompletedProcess([], 0, '{"name": "test"}')

    monkeypatch.setattr(docker, "compose", compose)
    monkeypatch.setattr(docker, "request_sessions", lambda *args: [])
    with pytest.raises(subprocess.CalledProcessError):
        docker.manage(tmp_path, "stop")
    assert mount.read_text() == "keep"
