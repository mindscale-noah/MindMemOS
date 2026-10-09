"""Regression coverage for configuration-driven local development startup."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from mindmemos_skill.api.constants import SKILL_CONFIG_PATH_ENV
from mindmemos_skill.config.loader import read_skill_config
from sqlalchemy.engine import make_url

from scripts.dev import constants as env
from scripts.dev import run
from scripts.dev.config import finish_plan, memory_document, prepare_environment, skill_document
from scripts.dev.typing import DependencyPlan


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Prevent private repository dotenv values from entering test configurations."""
    for key in list(os.environ):
        if key.startswith("MINDMEMOS_"):
            monkeypatch.delenv(key)
    from mindmemos.config import app

    monkeypatch.setattr(app, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "load_dotenv", lambda *args, **kwargs: None)
    project = _write(
        tmp_path / "project.yaml",
        {
            "default_project": "local",
            "projects": {
                "local": {
                    "project_id": "test-project",
                    "api_key": "test-local-key",
                    "memory_algorithm": "vanilla",
                    "scopes": ["memory:read", "memory:write", "skill:read", "skill:write"],
                }
            },
        },
    )
    monkeypatch.setattr(run, "DEFAULT_PROJECT_CONFIG", project)
    monkeypatch.setattr(run, "DEFAULT_UI_HOME", tmp_path / "sdk")


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _skill(tmp_path, *, postgres=False, kafka=False, otlp=False):
    service = {
        "home": "./home",
        "persistence": {"database": {"class_name": "sqlite", "config": {}}},
    }
    if postgres:
        service["persistence"]["database"] = {
            "class_name": "postgres",
            "config": {"url": "postgresql+psycopg://old:old@localhost:55432/old"},
        }
    if kafka:
        service["task_manager"] = {"mode": "kafka", "bootstrap_servers": "localhost:9092", "worker": True}
    if otlp:
        service["observability"] = {"otlp": {"endpoint": "http://localhost:4318"}}
    return _write(
        tmp_path / "skill.yaml",
        {
            "service": service,
            "fastapi": {"auth": {"mode": "none", "default_project_id": "test-project"}},
        },
    )


def test_dotenv_ports_override_old_endpoints_and_shell_wins(tmp_path, monkeypatch):
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        f"{env.QDRANT_HTTP_PORT}=16333\n{env.QDRANT_URL}=http://localhost:6333\n"
        f"{env.KAFKA_PORT}=19092\n{env.KAFKA_SERVERS}=localhost:9092\n"
        f"{env.OTLP_PORT}=14318\n{env.NEO4J_PORT}=17687\n"
    )
    monkeypatch.setenv(env.QDRANT_HTTP_PORT, "26333")
    prepare_environment(dotenv)
    assert os.environ[env.QDRANT_URL] == "http://localhost:26333"
    assert os.environ[env.KAFKA_SERVERS] == "localhost:19092"
    assert os.environ[env.OTLP_ENDPOINT] == "http://localhost:14318"
    assert os.environ[env.NEO4J_URI] == "bolt://localhost:17687"


def test_memory_snapshot_and_compose_share_connections(tmp_path, monkeypatch):
    source = _write(tmp_path / "memory.yaml", {"kafka": {"enabled": True}, "telemetry": {"enabled": True}})
    original = source.read_bytes()
    for key, value in {
        env.QDRANT_HTTP_PORT: "16333",
        env.QDRANT_GRPC_PORT: "16334",
        env.NEO4J_PORT: "17687",
        env.KAFKA_PORT: "19092",
        env.OTLP_PORT: "14318",
        env.NEO4J_PASSWORD: "",
    }.items():
        monkeypatch.setenv(key, value)
    prepare_environment(tmp_path / "absent")
    plan = DependencyPlan()
    document = memory_document(source, plan, async_tasks=True, otel=True)
    finish_plan(plan)
    assert document["database"]["qdrant"]["url"] == "http://localhost:16333"
    assert document["database"]["qdrant"]["grpc_port"] == 16334
    assert document["database"]["neo4j"]["uri"] == "bolt://localhost:17687"
    assert document["database"]["neo4j"]["password"] == env.LOCAL_PASSWORD
    assert document["kafka"]["bootstrap_servers"] == "localhost:19092"
    assert document["telemetry"]["telemetry_endpoint"] == "http://localhost:14318"
    assert plan.services == {
        "qdrant",
        "neo4j",
        "kafka",
        "kafka-ui",
        "kafka-exporter",
        "otel-collector",
        "clickhouse",
        "grafana",
    }
    from mindmemos.config import build_config

    runtime = run._write_config(tmp_path, "runtime", document)
    config = build_config(config_path=runtime)
    assert config.database.qdrant.url == "http://localhost:16333"
    assert config.auth.api_key_file == str(tmp_path / "api_keys.yaml")
    assert source.read_bytes() == original


def test_ports_do_not_enable_disabled_features_or_replace_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv(env.POSTGRES_PORT, "55433")
    monkeypatch.setenv(env.KAFKA_PORT, "19092")
    monkeypatch.setenv(env.OTLP_PORT, "14318")
    prepare_environment(tmp_path / "absent")
    plan = DependencyPlan()
    document = skill_document(_skill(tmp_path), plan, "127.0.0.1")
    finish_plan(plan)
    assert not plan.services
    assert document["service"]["persistence"]["database"]["class_name"] == "sqlite"


def test_skill_postgres_credentials_ports_and_original_paths(tmp_path, monkeypatch):
    source = _skill(tmp_path, postgres=True, kafka=True, otlp=True)
    original = source.read_bytes()
    for key, value in {
        env.POSTGRES_PORT: "55433",
        env.POSTGRES_DB: "skill_dev",
        env.POSTGRES_USER: "local_user",
        env.POSTGRES_PASSWORD: "p@ss$word:/#",
        env.KAFKA_PORT: "19092",
        env.OTLP_PORT: "14318",
    }.items():
        monkeypatch.setenv(key, value)
    prepare_environment(tmp_path / "absent")
    plan = DependencyPlan()
    document = skill_document(source, plan, "127.0.0.1", async_tasks=True, otel=True)
    runtime = run._write_config(tmp_path, "runtime", document)
    config = read_skill_config(runtime)
    url = make_url(config.service.persistence.database.config.url)
    assert (url.port, url.database, url.username, url.password) == (55433, "skill_dev", "local_user", "p@ss$word:/#")
    assert config.service.home == tmp_path / "home"
    assert config.service.task_manager.bootstrap_servers == "localhost:19092"
    assert config.service.task_manager.worker is True
    assert str(config.service.observability.otlp.endpoint) == "http://localhost:14318/"
    assert runtime.stat().st_mode & 0o777 == 0o600
    assert source.read_bytes() == original


def test_telemetry_does_not_pull_in_disabled_kafka(tmp_path):
    plan = DependencyPlan()
    skill_document(_skill(tmp_path, otlp=True), plan, "127.0.0.1", otel=True)
    finish_plan(plan)
    assert plan.services == {"otel-collector", "clickhouse", "grafana"}


def test_external_endpoints_never_provision_local_containers(tmp_path, monkeypatch):
    source = _skill(tmp_path, postgres=True, kafka=True, otlp=True)
    data = yaml.safe_load(source.read_text())
    data["service"]["persistence"]["database"]["config"]["url"] = "postgresql+psycopg://user:secret@db.example:5432/app"
    _write(source, data)
    monkeypatch.setenv(env.POSTGRES_PORT, "55433")
    monkeypatch.setenv(env.KAFKA_SERVERS, "broker.example:9092,broker2.example:9092")
    monkeypatch.setenv(env.OTLP_ENDPOINT, "https://otel.example:443")
    monkeypatch.setenv(env.KAFKA_PORT, "19092")
    monkeypatch.setenv(env.OTLP_PORT, "14318")
    prepare_environment(tmp_path / "absent")
    plan = DependencyPlan()
    skill_document(source, plan, "127.0.0.1", async_tasks=True, otel=True)
    finish_plan(plan)
    assert not plan.services
    assert ("otel.example", 443) in plan.connections
    assert ("db.example", 5432) in plan.connections


def test_authentication_secrets_survive_runtime_snapshot(tmp_path):
    source = _skill(tmp_path)
    data = yaml.safe_load(source.read_text())
    data["fastapi"]["auth"] = {"mode": "gateway_jwt", "gateway_jwt_secret": "private-value"}
    _write(source, data)
    document = skill_document(source, DependencyPlan(), "127.0.0.1")
    runtime = run._write_config(tmp_path, "runtime", document)
    assert read_skill_config(runtime).fastapi.auth.gateway_jwt_secret.get_secret_value() == "private-value"


def test_invalid_config_fails_before_docker(tmp_path, monkeypatch):
    source = _write(tmp_path / "invalid.yaml", {"service": {"home": "./home"}})
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)
    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: pytest.fail("Docker must not start"))
    with pytest.raises(ValueError):
        run.main(["dev", "--service", "skill", "--feature", "core", "--ui", "0", "--skill-config", str(source)])


def test_runner_passes_same_effective_config_and_env_to_children(tmp_path, monkeypatch):
    source = _skill(tmp_path, postgres=True, kafka=True, otlp=True)
    monkeypatch.setenv(env.POSTGRES_PORT, "55433")
    monkeypatch.setenv(env.KAFKA_PORT, "19092")
    monkeypatch.setenv(env.OTLP_PORT, "14318")
    prepare_environment(tmp_path / "absent")
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)
    calls = []
    monkeypatch.setattr(run, "prepare_collector", lambda root, plan: None)
    monkeypatch.setattr(run.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(run, "_wait_connections", lambda plan: None)

    def check_children(commands, **kwargs):
        assert Path(os.environ[run.MINDMEMOS_HOME_ENV]) == run.DEFAULT_UI_HOME.absolute()
        config = read_skill_config(os.environ[SKILL_CONFIG_PATH_ENV])
        assert config.service.task_manager.bootstrap_servers == "localhost:19092"
        assert make_url(config.service.persistence.database.config.url).port == 55433
        assert "--port" in commands[0]
        assert commands[0][-1] == "18001"
        assert "--remove-orphans" not in calls[0]
        return 0

    monkeypatch.setattr(run, "_run_apis", check_children)
    assert (
        run.main(["dev", "--service", "skill", "--ui", "0", "--skill-config", str(source), "--skill-port", "18001"])
        == 0
    )
    assert "skill-postgres" in calls[0]
    assert not Path(os.environ[SKILL_CONFIG_PATH_ENV]).exists()


def test_compose_failure_prevents_api_start(tmp_path, monkeypatch):
    source = _skill(tmp_path, postgres=True)
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(run.subprocess, "run", fail)
    monkeypatch.setattr(run, "_run_apis", lambda commands: pytest.fail("API must not start"))
    with pytest.raises(subprocess.CalledProcessError):
        run.main(["dev", "--service", "skill", "--feature", "core", "--ui", "0", "--skill-config", str(source)])


def test_unreachable_endpoint_fails_quickly():
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="unreachable"):
            run._wait_connections(DependencyPlan(connections={("127.0.0.1", port)}), timeout=0)


def test_child_failure_stops_sibling_process(tmp_path):
    pidfile = tmp_path / "pid"
    sleeper = [
        sys.executable,
        "-c",
        f"import os,time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(30)",
    ]
    failure = [sys.executable, "-c", "import time; time.sleep(0.3); raise SystemExit(7)"]
    assert run._run_apis([sleeper, failure]) == 7
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_unauthenticated_public_bind_rejected_before_startup(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        skill_document(_skill(tmp_path), DependencyPlan(), "0.0.0.0")


@pytest.mark.parametrize("async_tasks,otel", [(False, False), (True, False), (True, True)])
def test_feature_presets_override_yaml_and_preserve_task_options(tmp_path, async_tasks, otel):
    source = _skill(tmp_path, kafka=True, otlp=True)
    data = yaml.safe_load(source.read_text())
    data["service"]["task_manager"].update(topic="custom.jobs", max_concurrency=3, worker=False)
    data["service"]["observability"]["backend"] = {"class_name": "sqlite", "config": {}}
    _write(source, data)
    original = source.read_bytes()
    plan = DependencyPlan()
    document = skill_document(source, plan, "127.0.0.1", async_tasks=async_tasks, otel=otel)
    runtime = run._write_config(tmp_path, "runtime", document)
    config = read_skill_config(runtime)
    assert (config.service.task_manager is not None) == async_tasks
    if async_tasks:
        assert config.service.task_manager.worker is True
        assert config.service.task_manager.mode == "kafka"
        assert config.service.task_manager.topic == "custom.jobs"
        assert config.service.task_manager.max_concurrency == 3
    assert config.service.observability.backend.class_name == "sqlite"
    assert (config.service.observability.otlp is not None) == otel
    finish_plan(plan)
    assert ("kafka" in plan.services) == async_tasks
    assert ("otel-collector" in plan.services) == otel
    assert source.read_bytes() == original


@pytest.mark.parametrize("async_tasks,otel", [(False, False), (True, False), (True, True)])
def test_memory_presets_override_source_switches(tmp_path, async_tasks, otel):
    source = _write(
        tmp_path / "memory.yaml", {"kafka": {"enabled": not async_tasks}, "telemetry": {"enabled": not otel}}
    )
    plan = DependencyPlan()
    document = memory_document(source, plan, async_tasks=async_tasks, otel=otel)
    assert document["kafka"]["enabled"] == async_tasks
    assert document["telemetry"]["enabled"] == otel
    assert ("kafka" in plan.services) == async_tasks
    assert ("otel-collector" in plan.services) == otel


@pytest.mark.parametrize("action", ["dev", "docker", "api"])
@pytest.mark.parametrize("service", ["all", "memory", "skill"])
@pytest.mark.parametrize("feature", ["all", "core", "async"])
def test_make_command_matrix(action, service, feature):
    result = subprocess.run(
        ["make", "-n", action, f"SERVICE={service}", f"FEATURE={feature}", "PLAN=1"],
        cwd=run.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    command = result.stdout
    assert f"scripts.dev.run {action}" in command
    assert f'--service "{service}"' in command
    assert f'--feature "{feature}"' in command
    assert "--plan" in command
    assert "docker compose" not in command


def test_full_preset_creates_omitted_optional_sections(tmp_path):
    source = _skill(tmp_path)
    document = skill_document(source, DependencyPlan(), "127.0.0.1", async_tasks=True, otel=True)
    runtime = run._write_config(tmp_path, "runtime", document)
    config = read_skill_config(runtime)
    assert config.service.task_manager.worker
    assert config.service.observability.otlp is not None


def test_invalid_feature_rejected():
    with pytest.raises(SystemExit):
        run.main(["api", "--feature", "otel", "--plan"])


def test_ui_and_feature_summary_uses_actual_ports(tmp_path, monkeypatch, capsys):
    source = _skill(tmp_path)
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)
    assert (
        run.main(
            [
                "dev",
                "--service",
                "skill",
                "--feature",
                "core",
                "--skill-config",
                str(source),
                "--skill-port",
                "18001",
                "--ui",
                "1",
                "--ui-port",
                "18765",
                "--plan",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Skill API        http://127.0.0.1:18001" in output
    assert "UI               http://127.0.0.1:18765" in output
    assert "Memory API       NOT STARTED" in output
    assert "Kafka tasks:     DISABLED" in output
    assert "OTel monitoring: DISABLED" in output
    assert "Development services ready" not in output


def test_ui_port_collision_fails_before_starting(tmp_path, monkeypatch):
    source = _skill(tmp_path)
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)
    with pytest.raises(ValueError, match="different development ports"):
        run.main(
            [
                "dev",
                "--service",
                "skill",
                "--feature",
                "core",
                "--skill-config",
                str(source),
                "--skill-port",
                "18765",
                "--ui",
                "1",
                "--ui-port",
                "18765",
                "--plan",
            ]
        )


def test_ready_summary_waits_for_every_health_check(monkeypatch, capsys):
    from unittest.mock import MagicMock

    child = MagicMock(pid=99123)
    child.poll.side_effect = [None, None, 7]
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(run.os, "killpg", lambda *args: None)
    monkeypatch.setattr(run.time, "sleep", lambda *args: None)
    opener = MagicMock()
    response = MagicMock()
    response.__enter__.return_value.status = 200
    opener.open.side_effect = [OSError("not yet"), response]
    monkeypatch.setattr(run, "build_opener", lambda *args: opener)
    assert (
        run._run_apis(
            [["server"]], ready_urls={"Skill API": "http://localhost:8001/health"}, summary="Skill and UI summary"
        )
        == 7
    )
    output = capsys.readouterr().out
    assert output.count("Development services ready") == 1
    assert "Skill API exited with status 7" in output
    assert opener.open.call_count == 2


def test_failure_before_health_never_reports_ready(monkeypatch, capsys):
    from unittest.mock import MagicMock

    child = MagicMock(pid=99123)
    child.poll.return_value = 7
    monkeypatch.setattr(run.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(run.os, "killpg", lambda *args: None)
    assert run._run_apis([["server"]], ready_urls={"UI": "http://localhost:8765/"}, summary="Ready") == 7
    assert "Development services ready" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "action,uses_docker,uses_api", [("dev", True, True), ("docker", True, False), ("api", False, True)]
)
def test_actions_share_plan_but_start_only_requested_processes(tmp_path, monkeypatch, action, uses_docker, uses_api):
    source = _skill(tmp_path, postgres=True)
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)
    docker_calls, api_calls = [], []
    monkeypatch.setattr(run, "compose", lambda *args: docker_calls.append(args))
    monkeypatch.setattr(run, "_wait_connections", lambda plan: None)

    def launch(commands, **kwargs):
        api_calls.append(commands)
        return 0

    monkeypatch.setattr(run, "_run_apis", launch)
    assert (
        run.main([action, "--service", "skill", "--feature", "core", "--ui", "0", "--skill-config", str(source)]) == 0
    )
    assert bool(docker_calls) == uses_docker
    assert bool(api_calls) == uses_api
    if uses_docker:
        assert "skill-postgres" in docker_calls[0]


def test_standalone_ui_does_not_require_service_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)
    monkeypatch.setattr(run, "compose", lambda *args: pytest.fail("UI must not start Docker"))

    def launch(commands, **kwargs):
        calls.extend(commands)
        return 0

    monkeypatch.setattr(run, "_run_apis", launch)
    assert run.main(["ui", "--ui-port", "18765"]) == 0
    assert len(calls) == 1 and "mindmemos_ui.cli" in calls[0]


def test_api_missing_dependency_never_starts_docker_or_apps(tmp_path, monkeypatch):
    source = _skill(tmp_path, postgres=True)
    monkeypatch.setattr(run, "prepare_environment", lambda path: None)
    monkeypatch.setattr(run, "compose", lambda *args: pytest.fail("API must not start Docker"))
    monkeypatch.setattr(run, "_run_apis", lambda *args, **kwargs: pytest.fail("Missing dependency must block startup"))

    def unreachable(plan):
        raise RuntimeError("Dependency unreachable; start make docker")

    monkeypatch.setattr(run, "_wait_connections", unreachable)
    with pytest.raises(RuntimeError, match="unreachable"):
        run.main(["api", "--service", "skill", "--feature", "core", "--skill-config", str(source)])


def test_occupied_port_rejected_before_process_start(monkeypatch):
    import socket

    monkeypatch.setattr(
        run.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Port conflict must block startup")
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match=f"Cannot bind 127.0.0.1:{port}"):
            run._run_apis([["server", "--host", "127.0.0.1", "--port", str(port)]])


@pytest.mark.parametrize("service", ["all", "memory", "skill"])
@pytest.mark.parametrize("feature", ["all", "core", "async"])
def test_service_feature_matrix_selects_expected_containers(tmp_path, service, feature):
    prepare_environment(tmp_path / "absent")
    plan = DependencyPlan()
    expected = set()
    if service in {"all", "memory"}:
        memory_document(
            _write(tmp_path / "memory.yaml", {}), plan, async_tasks=feature != "core", otel=feature == "all"
        )
        expected.update({"qdrant", "neo4j"})
    if service in {"all", "skill"}:
        skill_document(
            _skill(tmp_path, postgres=True), plan, "127.0.0.1", async_tasks=feature != "core", otel=feature == "all"
        )
        expected.add("skill-postgres")
    if feature != "core":
        expected.update({"kafka", "kafka-ui"})
    if feature == "all":
        expected.update({"kafka-exporter", "otel-collector", "clickhouse", "grafana"})
    finish_plan(plan)
    assert plan.services == expected
